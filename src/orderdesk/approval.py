"""Stage 6: human approval. The only code path that writes to the ERP.

    uv run python -m orderdesk.approval list
    uv run python -m orderdesk.approval show <document_id>
    uv run python -m orderdesk.approval approve <document_id> --by you@example.com
    uv run python -m orderdesk.approval reject <document_id> --by you@example.com --reason "..."

Design rules:
  1. A human decides. The pipeline and the agent only suggest.
  2. Exactly once: the ERP call carries an idempotency key derived from the document, and the
     document row is locked while deciding. Double-clicks, two reviewers, or a crash and retry
     all produce ONE draft order.
  3. Every correction is recorded, and a corrected product match is learned as a customer alias,
     so the same wording matches correctly next time.
"""

import argparse
import sys
from decimal import Decimal

import psycopg
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from orderdesk.erp_client import ERPError, ERPWriter
from orderdesk.erp_sim.app import DraftOrderIn, DraftOrderLineIn

DECIDABLE = ("ready_for_approval", "needs_investigation", "investigated")


class ApprovalError(RuntimeError):
    pass


class ReviewLine(BaseModel):
    line_number: int
    description: str
    customer_sku: str | None
    customer_quantity: Decimal
    customer_unit: str | None
    sku: str | None  # suggested: the agent's pick if it investigated, else the match
    quantity: int | None  # suggested, in our selling unit
    unit_price: Decimal | None
    source: str  # 'agent' or 'match'
    agent_reasoning: str | None = None


class ReviewItem(BaseModel):
    document_id: str
    filename: str
    status: str
    customer_id: str | None
    customer_name: str | None
    customer_po_number: str | None
    issues: list[dict]
    agent_action: str | None
    agent_summary: str | None
    agent_risk_flags: list[str]
    lines: list[ReviewLine]


class FinalLine(BaseModel):
    line_number: int
    sku: str
    quantity: int = Field(gt=0)


def suggest_quantity(quantity: Decimal) -> int | None:
    whole = quantity.to_integral_value()
    return int(whole) if quantity == whole and whole > 0 else None


class ApprovalService:
    def __init__(self, conn: psycopg.Connection, erp: ERPWriter) -> None:
        self.conn = conn
        self.erp = erp

    # --- reading -------------------------------------------------------------------------

    def queue(self) -> list[tuple[str, str, str, str | None]]:
        """(document_id, filename, status, customer) for everything awaiting a decision."""
        return self.conn.execute(
            """
            SELECT d.document_id, d.filename, d.status, c.name
            FROM documents d LEFT JOIN order_proposals p USING (document_id)
            LEFT JOIN customers c ON c.customer_id = p.customer_id
            WHERE d.status = ANY(%s)
            ORDER BY (d.status = 'ready_for_approval') DESC, d.received_at NULLS LAST, d.filename
            """,
            (list(DECIDABLE),),
        ).fetchall()

    def review(self, document_id: str) -> ReviewItem:
        row = self.conn.execute(
            """
            SELECT d.filename, d.status, p.customer_id, c.name, p.extracted, p.lines, p.issues,
                   (SELECT findings FROM investigations i
                    WHERE i.document_id = d.document_id AND findings IS NOT NULL
                    ORDER BY created_at DESC LIMIT 1)
            FROM documents d JOIN order_proposals p USING (document_id)
            LEFT JOIN customers c ON c.customer_id = p.customer_id
            WHERE d.document_id = %s
            """,
            (document_id,),
        ).fetchone()
        if row is None:
            raise ApprovalError(f"no proposal for document {document_id}")
        filename, status, customer_id, name, extracted, lines, issues, findings = row
        by_line = {f["line_number"]: f for f in (findings or {}).get("lines", [])}

        review_lines = []
        for ln in lines:
            agent = by_line.get(ln["line_number"])
            if agent:
                sku, qty, source = agent["recommended_sku"], agent["recommended_quantity"], "agent"
            else:
                sku = ln["chosen_sku"]
                qty, source = suggest_quantity(Decimal(ln["quantity"])), "match"
            review_lines.append(
                ReviewLine(
                    line_number=ln["line_number"],
                    description=ln["description"],
                    customer_sku=ln["customer_sku"],
                    customer_quantity=ln["quantity"],
                    customer_unit=ln["customer_unit"],
                    sku=sku,
                    quantity=qty,
                    unit_price=ln["unit_price"],
                    source=source,
                    agent_reasoning=agent["reasoning"] if agent else None,
                )
            )
        return ReviewItem(
            document_id=document_id,
            filename=filename,
            status=status,
            customer_id=customer_id,
            customer_name=name,
            customer_po_number=extracted.get("customer_po_number"),
            issues=issues,
            agent_action=findings["recommended_action"] if findings else None,
            agent_summary=findings["summary"] if findings else None,
            agent_risk_flags=findings["risk_flags"] if findings else [],
            lines=review_lines,
        )

    # --- deciding ------------------------------------------------------------------------

    def approve(self, document_id: str, reviewer: str, lines: list[FinalLine] | None = None) -> str:
        """Create the ERP draft order. `lines` overrides the suggestion; None accepts it as is.
        Returns the ERP order id."""
        self._lock_decidable(document_id)
        try:
            item = self.review(document_id)
            if item.customer_id is None:
                raise ApprovalError("can't approve: no customer identified. Reject it instead.")
            final = lines if lines is not None else self._accept_suggestion(item)
            if not final:
                raise ApprovalError("can't approve an order with no lines")
            extracted = self.conn.execute(
                "SELECT extracted FROM order_proposals WHERE document_id = %s", (document_id,)
            ).fetchone()[0]

            order = DraftOrderIn(
                customer_id=item.customer_id,
                customer_po_number=extracted.get("customer_po_number"),
                requested_delivery_date=extracted.get("requested_delivery_date"),
                delivery_address=extracted.get("delivery_address"),
                notes=f"orderdesk {document_id}, approved by {reviewer}",
                lines=[DraftOrderLineIn(sku=ln.sku, quantity=ln.quantity) for ln in final],
            )
            # Same document -> same key -> the ERP returns the original order on any retry.
            draft = self.erp.create_draft_order(order, idempotency_key=f"orderdesk-{document_id}")
        except (ApprovalError, ERPError) as exc:
            self.conn.rollback()  # releases the lock; nothing was decided
            self._audit(document_id, reviewer, "approval_failed", {"error": str(exc)})
            self.conn.commit()
            raise ApprovalError(str(exc)) from exc

        corrections = self._corrections(item, final)
        self.conn.execute(
            """
            INSERT INTO decisions (document_id, decision, decided_by, suggested, final_lines,
                                   corrections, erp_order_id, order_total)
            VALUES (%s, 'approved', %s, %s, %s, %s, %s, %s)
            """,
            (
                document_id,
                reviewer,
                Jsonb([ln.model_dump(mode="json") for ln in item.lines]),
                Jsonb([ln.model_dump() for ln in final]),
                Jsonb(corrections),
                draft.order_id,
                draft.order_total,
            ),
        )
        self._learn_aliases(item, corrections)
        self._set_status(document_id, "approved")
        self._audit(
            document_id,
            reviewer,
            "approved",
            {
                "erp_order_id": draft.order_id,
                "order_total": str(draft.order_total),
                "corrections": len(corrections),
            },
        )
        self.conn.commit()
        return draft.order_id

    def reject(self, document_id: str, reviewer: str, reason: str) -> None:
        if not reason.strip():
            raise ApprovalError("a rejection needs a reason")
        self._lock_decidable(document_id)
        self.conn.execute(
            "INSERT INTO decisions (document_id, decision, decided_by, reason) "
            "VALUES (%s, 'rejected', %s, %s)",
            (document_id, reviewer, reason),
        )
        self._set_status(document_id, "rejected")
        self._audit(document_id, reviewer, "rejected", {"reason": reason})
        self.conn.commit()

    # --- helpers -------------------------------------------------------------------------

    def _lock_decidable(self, document_id: str) -> None:
        """Lock the row until commit, so two reviewers can't both decide the same order."""
        row = self.conn.execute(
            "SELECT status FROM documents WHERE document_id = %s FOR UPDATE", (document_id,)
        ).fetchone()
        if row is None:
            self.conn.rollback()
            raise ApprovalError(f"unknown document {document_id}")
        if row[0] not in DECIDABLE:
            self.conn.rollback()
            raise ApprovalError(f"document {document_id} is '{row[0]}', not awaiting a decision")

    @staticmethod
    def _accept_suggestion(item: ReviewItem) -> list[FinalLine]:
        incomplete = [ln.line_number for ln in item.lines if not ln.sku or not ln.quantity]
        if incomplete:
            raise ApprovalError(
                f"lines {incomplete} have no suggested product or quantity: fill them in "
                "(approve with explicit lines) or reject the order"
            )
        return [
            FinalLine(line_number=ln.line_number, sku=ln.sku, quantity=ln.quantity)
            for ln in item.lines
        ]

    @staticmethod
    def _corrections(item: ReviewItem, final: list[FinalLine]) -> list[dict]:
        suggested = {ln.line_number: ln for ln in item.lines}
        changes = []
        for ln in final:
            s = suggested.get(ln.line_number)
            if s is None or s.sku != ln.sku or s.quantity != ln.quantity:
                changes.append(
                    {
                        "line_number": ln.line_number,
                        "description": s.description if s else None,
                        "customer_sku": s.customer_sku if s else None,
                        "suggested_sku": s.sku if s else None,
                        "final_sku": ln.sku,
                        "suggested_quantity": s.quantity if s else None,
                        "final_quantity": ln.quantity,
                    }
                )
        kept = {ln.line_number for ln in final}
        changes += [{"line_number": n, "removed": True} for n in suggested if n not in kept]
        return changes

    def _learn_aliases(self, item: ReviewItem, corrections: list[dict]) -> None:
        """A reviewer changing the product is the system being taught: remember it."""
        for c in corrections:
            if c.get("removed") or c["suggested_sku"] == c["final_sku"]:
                continue
            code, text = c["customer_sku"], c["description"]
            self.conn.execute(
                "DELETE FROM customer_aliases WHERE customer_id = %s AND "
                "(lower(customer_sku) = lower(%s) OR lower(customer_text) = lower(%s))",
                (item.customer_id, code or "", text or ""),
            )
            self.conn.execute(
                "INSERT INTO customer_aliases (customer_id, customer_sku, customer_text, sku, "
                "source) VALUES (%s, %s, %s, %s, 'learned')",
                (item.customer_id, code, None if code else text, c["final_sku"]),
            )

    def _set_status(self, document_id: str, status: str) -> None:
        self.conn.execute(
            "UPDATE documents SET status = %s, updated_at = now() WHERE document_id = %s",
            (status, document_id),
        )

    def _audit(self, document_id: str, actor: str, event: str, detail: dict) -> None:
        self.conn.execute(
            "INSERT INTO audit_log (document_id, actor, event, detail) VALUES (%s, %s, %s, %s)",
            (document_id, actor, event, Jsonb(detail)),
        )


# --- command line ------------------------------------------------------------------------


def _print_review(item: ReviewItem) -> None:
    print(
        f"{item.filename}  [{item.status}]  customer: {item.customer_name or 'UNKNOWN'}"
        f"  PO: {item.customer_po_number or '-'}"
    )
    for issue in item.issues:
        line = f" (line {issue['line_number']})" if issue.get("line_number") else ""
        print(f"  ! {issue['code']}{line}: {issue['message']}")
    if item.agent_action:
        flags = ", ".join(item.agent_risk_flags) or "none"
        print(f"  agent: {item.agent_action} (risks: {flags})\n  {item.agent_summary}")
    print("\n  line  customer wrote                         suggested           qty    price")
    for ln in item.lines:
        wrote = f"{ln.customer_quantity} {ln.customer_unit or ''} {ln.description}"[:38]
        print(
            f"  {ln.line_number:<5} {wrote:<38} {ln.sku or '???':<19} "
            f"{ln.quantity or '?':<6} {ln.unit_price or '-'}  ({ln.source})"
        )


def main(argv: list[str] | None = None) -> int:
    from orderdesk.config import get_settings
    from orderdesk.db import connect, migrate
    from orderdesk.erp_client import HttpERPClient

    parser = argparse.ArgumentParser(prog="orderdesk.approval")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    show = sub.add_parser("show")
    show.add_argument("document_id")
    approve = sub.add_parser("approve")
    approve.add_argument("document_id")
    approve.add_argument("--by", required=True, help="reviewer's email")
    approve.add_argument(
        "--line",
        action="append",
        default=[],
        metavar="N=SKUxQTY",
        help="override a line, e.g. --line 4=HH-CUT-WD-Kx1 (repeatable)",
    )
    reject = sub.add_parser("reject")
    reject.add_argument("document_id")
    reject.add_argument("--by", required=True)
    reject.add_argument("--reason", required=True)
    args = parser.parse_args(argv)

    settings = get_settings()
    erp = HttpERPClient(settings.erp_base_url, settings.erp_write_key.get_secret_value())
    with connect(settings.database_url) as conn:
        migrate(conn)
        service = ApprovalService(conn, erp)
        try:
            if args.command == "list":
                for doc_id, filename, status, customer in service.queue():
                    print(f"{doc_id}  {status:<20} {customer or 'UNKNOWN':<24} {filename}")
            elif args.command == "show":
                _print_review(service.review(args.document_id))
            elif args.command == "approve":
                lines = None
                if args.line:
                    overrides = {}
                    for spec in args.line:
                        number, rest = spec.split("=", 1)
                        sku, qty = rest.rsplit("x", 1)
                        overrides[int(number)] = (sku, int(qty))
                    item = service.review(args.document_id)
                    lines = []
                    for ln in item.lines:
                        sku, qty = overrides.get(ln.line_number, (ln.sku, ln.quantity))
                        if not sku or not qty:
                            print(f"line {ln.line_number} needs --line {ln.line_number}=SKUxQTY")
                            return 1
                        lines.append(FinalLine(line_number=ln.line_number, sku=sku, quantity=qty))
                order_id = service.approve(args.document_id, args.by, lines)
                print(f"Approved. ERP draft order {order_id}")
            elif args.command == "reject":
                service.reject(args.document_id, args.by, args.reason)
                print("Rejected.")
        except ApprovalError as exc:
            print(f"Not done: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
