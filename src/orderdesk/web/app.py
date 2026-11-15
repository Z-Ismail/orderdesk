"""Stage 7: the reviewer's web app.

    uv run uvicorn orderdesk.web.app:create_app --factory --port 8000

Server-rendered HTML (Jinja templates) with a little HTMX for in-place updates, so there's no
JavaScript build step. All business rules live in ApprovalService; this file only translates
between HTTP forms and that service, which is why the UI and the command line can't disagree.
"""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated

import psycopg
from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from orderdesk.approval import ApprovalError, ApprovalService, FinalLine
from orderdesk.erp_client import ERPError, ERPWriter

TEMPLATES = Jinja2Templates(directory=Path(__file__).parent / "templates")

STATUS_LABELS = {
    "ready_for_approval": ("Ready", "ok"),
    "investigated": ("Investigated", "warn"),
    "needs_investigation": ("Needs investigation", "bad"),
    "approved": ("Approved", "done"),
    "rejected": ("Rejected", "muted"),
}
TEMPLATES.env.globals["status_label"] = lambda s: STATUS_LABELS.get(s, (s, "muted"))


def create_app(
    connect: Callable[[], psycopg.Connection] | None = None,
    erp: ERPWriter | None = None,
    reviewer: str | None = None,
) -> FastAPI:
    """Everything external is injectable so tests can run the real app against test services."""
    if connect is None or erp is None or reviewer is None:
        from orderdesk.config import get_settings
        from orderdesk.db import connect as db_connect
        from orderdesk.db import migrate
        from orderdesk.erp_client import HttpERPClient

        settings = get_settings()
        connect = connect or (lambda: db_connect(settings.database_url))
        erp = erp or HttpERPClient(settings.erp_base_url, settings.erp_write_key.get_secret_value())
        reviewer = reviewer or settings.reviewer_email
        with connect() as conn:
            migrate(conn)

    app = FastAPI(title="orderdesk", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

    def service() -> Iterator[ApprovalService]:
        with connect() as conn:  # one connection per request, closed afterwards
            yield ApprovalService(conn, erp)

    Service = Annotated[ApprovalService, Depends(service)]

    def page(request: Request, name: str, status_code: int = 200, **context) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request, name, {"reviewer": reviewer, **context}, status_code=status_code
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def queue(request: Request, svc: Service, done: str | None = None) -> HTMLResponse:
        recent = svc.conn.execute(
            """
            SELECT d.document_id, d.filename, d.status, c.name, x.decided_by, x.erp_order_id,
                   x.order_total, x.decided_at
            FROM decisions x JOIN documents d USING (document_id)
            LEFT JOIN order_proposals p USING (document_id)
            LEFT JOIN customers c ON c.customer_id = p.customer_id
            ORDER BY x.decided_at DESC LIMIT 10
            """
        ).fetchall()
        return page(
            request,
            "queue.html",
            pending=[svc.review(r[0]) for r in svc.queue()],
            recent=recent,
            done=done,
        )

    @app.get("/orders/{document_id}", response_class=HTMLResponse)
    def order(request: Request, document_id: str, svc: Service) -> HTMLResponse:
        return _order_page(request, svc, document_id)

    def _order_page(request, svc, document_id, error=None, status_code=200) -> HTMLResponse:
        try:
            item = svc.review(document_id)
        except ApprovalError:
            return page(request, "missing.html", status_code=404, document_id=document_id)
        candidates = svc.conn.execute(
            "SELECT lines FROM order_proposals WHERE document_id = %s", (document_id,)
        ).fetchone()[0]
        options = {ln["line_number"]: ln["candidates"][:5] for ln in candidates}
        decision = svc.conn.execute(
            "SELECT decision, decided_by, reason, erp_order_id, order_total, decided_at "
            "FROM decisions WHERE document_id = %s",
            (document_id,),
        ).fetchone()
        return page(
            request,
            "order.html",
            status_code=status_code,
            item=item,
            options=options,
            decision=decision,
            error=error,
        )

    @app.post("/orders/{document_id}/approve")
    async def approve(request: Request, document_id: str, svc: Service):
        form = await request.form()
        item = svc.review(document_id)
        lines, problems = [], []
        for ln in item.lines:
            if form.get(f"remove_{ln.line_number}"):
                continue
            sku = str(form.get(f"sku_{ln.line_number}", "")).strip()
            qty = str(form.get(f"qty_{ln.line_number}", "")).strip()
            if not sku or not qty.isdigit() or int(qty) < 1:
                problems.append(ln.line_number)
                continue
            lines.append(FinalLine(line_number=ln.line_number, sku=sku, quantity=int(qty)))
        if problems:
            return _order_page(
                request,
                svc,
                document_id,
                status_code=422,
                error=f"Lines {problems} need a product and a whole-number "
                "quantity, or tick Remove.",
            )
        try:
            order_id = svc.approve(document_id, reviewer, lines)
        except ApprovalError as exc:
            return _order_page(request, svc, document_id, error=str(exc), status_code=422)
        # Post/Redirect/Get: refreshing the next page can't resubmit the approval.
        return RedirectResponse(
            f"/?done=Approved+{item.filename}+as+ERP+order+{order_id}", status_code=303
        )

    @app.post("/orders/{document_id}/reject")
    def reject(
        request: Request, document_id: str, svc: Service, reason: Annotated[str, Form()] = ""
    ):
        try:
            svc.reject(document_id, reviewer, reason)
        except ApprovalError as exc:
            return _order_page(request, svc, document_id, error=str(exc), status_code=422)
        return RedirectResponse("/?done=Rejected", status_code=303)

    # --- HTMX fragments: small pieces of HTML swapped into the page ---------------------------

    @app.get("/orders/{document_id}/lines/{line_number}/check", response_class=HTMLResponse)
    def check_line(
        request: Request, document_id: str, line_number: int, svc: Service
    ) -> HTMLResponse:
        """Live feedback while the reviewer edits a line: product name, price, stock.

        HTMX sends the row's own inputs, which are named sku_<n> and qty_<n>.
        """
        sku = request.query_params.get(f"sku_{line_number}", "")
        qty = request.query_params.get(f"qty_{line_number}", "")
        item = svc.review(document_id)
        info, problem = None, None
        if sku.strip():
            try:
                product = erp.get_product(sku.strip())
                price = erp.get_price(item.customer_id, product.sku) if item.customer_id else None
                stock = erp.get_stock(product.sku).total_available
                info = {"product": product, "price": price, "stock": stock}
                if not product.active:
                    problem = "discontinued"
                elif qty.isdigit() and int(qty) > stock:
                    problem = f"only {stock} in stock"
            except ERPError as exc:
                problem = "unknown SKU" if exc.status_code == 404 else str(exc)
        return page(request, "_line_check.html", info=info, problem=problem)

    @app.get("/orders/{document_id}/history", response_class=HTMLResponse)
    def history(request: Request, document_id: str, svc: Service) -> HTMLResponse:
        events = svc.conn.execute(
            "SELECT at, actor, event, detail FROM audit_log WHERE document_id = %s ORDER BY id",
            (document_id,),
        ).fetchall()
        investigation = svc.conn.execute(
            "SELECT transcript, steps, cost_usd, model FROM investigations "
            "WHERE document_id = %s ORDER BY created_at DESC LIMIT 1",
            (document_id,),
        ).fetchone()
        return page(request, "_history.html", events=events, investigation=investigation)

    @app.get("/orders/{document_id}/original", response_class=HTMLResponse)
    def original(request: Request, document_id: str, svc: Service) -> HTMLResponse:
        row = svc.conn.execute(
            "SELECT raw FROM documents WHERE document_id = %s", (document_id,)
        ).fetchone()
        return page(request, "_original.html", raw=row[0] if row else None)

    return app
