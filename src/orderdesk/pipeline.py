"""Stage 4: the end-to-end pipeline.

    uv run python -m orderdesk.pipeline [inbox_folder]

For each unique document: register -> extract -> identify customer -> match lines ->
price and stock -> look for issues -> route to 'ready_for_approval' or 'needs_investigation'.

Safe to re-run: a document already routed is skipped, and a failed one is retried.
Nothing here writes to the ERP. That only happens after a human approves (milestone 5).
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from orderdesk.customers import Customer, identify_customer
from orderdesk.embeddings import Embedder
from orderdesk.erp_client import ERPError, ERPReader
from orderdesk.ingestion import scan_inbox
from orderdesk.matching import MatchResult, match_line
from orderdesk.models import ExtractedOrder, RawDocument
from orderdesk.proposals import Issue, OrderProposal, ProposedLine

Extractor = Callable[[RawDocument], ExtractedOrder]
RETRYABLE = {"received", "failed"}


@dataclass(frozen=True)
class Thresholds:
    min_similarity: float = 0.75
    min_margin: float = 0.02


@dataclass
class Outcome:
    document_id: str
    filename: str
    status: str
    issues: list[str]
    skipped: bool = False


def is_confident(match: MatchResult, thresholds: Thresholds) -> bool:
    best = match.best
    if best is None:
        return False
    if best.source != "hybrid":
        return True  # alias or exact SKU: the customer told us exactly what they meant
    if best.similarity is None or best.similarity < thresholds.min_similarity:
        return False
    return match.margin is None or match.margin >= thresholds.min_margin


def unit_mismatch(customer_unit: str | None, product_unit: str) -> bool:
    """'box' vs 'case' is a mismatch; 'pack of 6' vs 'pack' is fine; no unit given is fine."""
    return bool(customer_unit) and not customer_unit.startswith(product_unit)


class Pipeline:
    def __init__(
        self,
        conn: psycopg.Connection,
        extract: Extractor,
        embedder: Embedder,
        erp: ERPReader,
        thresholds: Thresholds | None = None,
    ) -> None:
        self.conn = conn
        self.extract = extract
        self.embedder = embedder
        self.erp = erp
        self.thresholds = thresholds or Thresholds()

    # --- public ----------------------------------------------------------------------------

    def run_inbox(self, folder: Path) -> list[Outcome]:
        return [self.process(doc) for doc in scan_inbox(folder)]

    def process(self, doc: RawDocument) -> Outcome:
        row = self.conn.execute(
            "SELECT status FROM documents WHERE document_id = %s", (doc.document_id,)
        ).fetchone()
        if row and row[0] not in RETRYABLE:
            return Outcome(doc.document_id, doc.filename, row[0], [], skipped=True)
        if row is None:
            self._register(doc)

        try:
            order = self.extract(doc)
            proposal = self.build_proposal(doc, order)
            self._save(proposal)
            self.conn.commit()
            return Outcome(
                doc.document_id, doc.filename, proposal.status, [i.code for i in proposal.issues]
            )
        except Exception as exc:  # noqa: BLE001 - record any failure on the document, keep going
            self.conn.rollback()
            self._set_status(doc.document_id, "failed", error=repr(exc))
            self._audit(doc.document_id, "failed", {"error": repr(exc)})
            self.conn.commit()
            return Outcome(doc.document_id, doc.filename, "failed", [type(exc).__name__])

    def build_proposal(self, doc: RawDocument, order: ExtractedOrder) -> OrderProposal:
        customer = identify_customer(self.conn, order.customer_email or doc.sender)
        issues = self._order_issues(doc, order, customer)
        lines = []
        for order_line in order.lines:
            match = match_line(
                self.conn, self.embedder, order_line, customer.customer_id if customer else None
            )
            line, line_issues = self._assess_line(order_line, match, customer)
            lines.append(line)
            issues.extend(line_issues)
        return OrderProposal(
            document_id=doc.document_id,
            customer_id=customer.customer_id if customer else None,
            customer_name=customer.name if customer else None,
            extracted=order,
            lines=lines,
            issues=issues,
        )

    # --- checks ----------------------------------------------------------------------------

    def _order_issues(
        self, doc: RawDocument, order: ExtractedOrder, customer: Customer | None
    ) -> list[Issue]:
        issues = []
        if order.is_amendment:
            issues.append(Issue(code="amendment", message="Changes an earlier order."))
        if customer is None:
            issues.append(
                Issue(
                    code="unknown_customer",
                    message=f"No customer account for {order.customer_email or doc.sender}.",
                )
            )
            return issues
        if customer.account_status != "active":
            issues.append(
                Issue(
                    code="customer_on_hold",
                    message=f"{customer.name} account is {customer.account_status}.",
                )
            )
        if order.customer_po_number and not order.is_amendment:
            earlier = self.conn.execute(
                """
                SELECT document_id FROM order_proposals
                WHERE customer_id = %s AND extracted->>'customer_po_number' = %s
                  AND NOT (extracted->>'is_amendment')::boolean AND document_id <> %s
                LIMIT 1
                """,
                (customer.customer_id, order.customer_po_number, doc.document_id),
            ).fetchone()
            if earlier:
                issues.append(
                    Issue(
                        code="possible_duplicate",
                        message=f"PO {order.customer_po_number} already received "
                        f"(document {earlier[0]}).",
                    )
                )
        return issues

    def _assess_line(self, order_line, match: MatchResult, customer: Customer | None):
        n = order_line.line_number
        issues: list[Issue] = []
        best = match.best
        line = ProposedLine(
            line_number=n,
            description=order_line.description,
            quantity=order_line.quantity,
            customer_unit=order_line.unit_of_measure,
            customer_sku=order_line.customer_sku,
            candidates=match.candidates,
            chosen_sku=best.sku if best else None,
            similarity=best.similarity if best else None,
            margin=match.margin,
        )
        if best is None:
            issues.append(
                Issue(
                    code="no_match",
                    line_number=n,
                    message=f"Nothing in the catalogue matches '{line.description}'.",
                )
            )
            return line, issues
        if not is_confident(match, self.thresholds):
            issues.append(
                Issue(
                    code="low_match_confidence",
                    line_number=n,
                    message=f"'{line.description}' best guess {best.sku} is uncertain.",
                )
            )
        if unit_mismatch(order_line.unit_of_measure, best.unit_of_measure):
            issues.append(
                Issue(
                    code="unit_mismatch",
                    line_number=n,
                    message=f"Customer ordered by '{order_line.unit_of_measure}', we sell "
                    f"{best.sku} by the {best.unit_of_measure} of {best.pack_size}.",
                )
            )
        if order_line.quantity != order_line.quantity.to_integral_value():
            issues.append(
                Issue(
                    code="non_integer_quantity",
                    line_number=n,
                    message=f"Quantity {order_line.quantity} is not a whole number.",
                )
            )
        if customer is not None:
            try:
                line.unit_price = self.erp.get_price(customer.customer_id, best.sku).unit_price
                line.available_stock = self.erp.get_stock(best.sku).total_available
            except ERPError as exc:
                issues.append(Issue(code="erp_lookup_failed", line_number=n, message=str(exc)))
            else:
                if line.available_stock < order_line.quantity:
                    issues.append(
                        Issue(
                            code="out_of_stock",
                            line_number=n,
                            message=f"{best.sku}: {line.available_stock} available, "
                            f"{order_line.quantity} ordered.",
                        )
                    )
        return line, issues

    # --- storage ---------------------------------------------------------------------------

    def _register(self, doc: RawDocument) -> None:
        self.conn.execute(
            """
            INSERT INTO documents (document_id, content_sha256, filename, source_type, sender,
                                   subject, received_at, raw)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (document_id) DO NOTHING
            """,
            (
                doc.document_id,
                doc.content_sha256,
                doc.filename,
                doc.source_type,
                doc.sender,
                doc.subject,
                doc.received_at,
                Jsonb(doc.model_dump(mode="json")),
            ),
        )
        self._audit(doc.document_id, "received", {"filename": doc.filename})
        self.conn.commit()

    def _save(self, proposal: OrderProposal) -> None:
        self.conn.execute(
            """
            INSERT INTO order_proposals (document_id, customer_id, extracted, lines, issues)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (document_id) DO UPDATE SET
                customer_id = EXCLUDED.customer_id, extracted = EXCLUDED.extracted,
                lines = EXCLUDED.lines, issues = EXCLUDED.issues, updated_at = now()
            """,
            (
                proposal.document_id,
                proposal.customer_id,
                Jsonb(proposal.extracted.model_dump(mode="json")),
                Jsonb([line.model_dump(mode="json") for line in proposal.lines]),
                Jsonb([issue.model_dump(mode="json") for issue in proposal.issues]),
            ),
        )
        self._set_status(proposal.document_id, proposal.status)
        self._audit(
            proposal.document_id,
            "routed",
            {"status": proposal.status, "issues": [i.code for i in proposal.issues]},
        )

    def _set_status(self, document_id: str, status: str, error: str | None = None) -> None:
        self.conn.execute(
            "UPDATE documents SET status = %s, error = %s, updated_at = now() "
            "WHERE document_id = %s",
            (status, error, document_id),
        )

    def _audit(self, document_id: str, event: str, detail: dict, actor: str = "pipeline") -> None:
        self.conn.execute(
            "INSERT INTO audit_log (document_id, actor, event, detail) VALUES (%s, %s, %s, %s)",
            (document_id, actor, event, Jsonb(detail)),
        )


def main() -> None:
    from orderdesk.config import get_settings
    from orderdesk.db import connect, migrate
    from orderdesk.embeddings import FastEmbedEmbedder
    from orderdesk.erp_client import HttpERPClient
    from orderdesk.extraction import extract_order
    from orderdesk.llm import LLMClient

    settings = get_settings()
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample_data/inbox")
    llm = LLMClient(settings)
    erp = HttpERPClient(settings.erp_base_url, settings.erp_read_key.get_secret_value())
    with connect(settings.database_url) as conn:
        migrate(conn)
        pipeline = Pipeline(
            conn,
            extract=lambda doc: extract_order(doc, llm),
            embedder=FastEmbedEmbedder(settings.embedding_model),
            erp=erp,
            thresholds=Thresholds(settings.match_min_similarity, settings.match_min_margin),
        )
        outcomes = pipeline.run_inbox(folder)

    for o in outcomes:
        note = "(already processed)" if o.skipped else ", ".join(o.issues)
        print(f"{o.status:<20} {o.filename:<46} {note}")
    print(f"\nLLM: {llm.budget.calls} calls, ${llm.budget.cost_usd:.4f}")


if __name__ == "__main__":
    main()
