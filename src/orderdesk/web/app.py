"""Stage 7: the reviewer's web app.

    uv run uvicorn orderdesk.web.app:create_app --factory --port 8000

Server-rendered HTML (Jinja templates) with a little HTMX for in-place updates, so there's no
JavaScript build step. All business rules live in ApprovalService; this file only translates
between HTTP and that service, which is why the UI and the command line can't disagree.

Security layers, outermost first:
  1. Security headers on every response (CSP blocks injected scripts, no framing, no sniffing).
  2. Login required for everything except /login, /health and /static.
  3. Every form POST must carry the session's CSRF token.
  4. Templates auto-escape, so customer text is shown as text, never run as HTML.
"""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

import psycopg
from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from orderdesk import auth
from orderdesk.approval import ApprovalError, ApprovalService, FinalLine
from orderdesk.erp_client import ERPError, ERPWriter

TEMPLATES = Jinja2Templates(directory=Path(__file__).parent / "templates")
COOKIE = "orderdesk_session"
STATUS_LABELS = {
    "ready_for_approval": ("Ready", "ok"),
    "investigated": ("Investigated", "warn"),
    "needs_investigation": ("Needs investigation", "bad"),
    "approved": ("Approved", "done"),
    "rejected": ("Rejected", "muted"),
}
TEMPLATES.env.globals["status_label"] = lambda s: STATUS_LABELS.get(s, (s, "muted"))

SECURITY_HEADERS = {
    # Only our own scripts may run: an injected <script> or onclick= is blocked by the browser.
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'; "
    "form-action 'self'; base-uri 'self'",
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
}


class LoginRequired(Exception):
    pass


class CSRFFailed(Exception):
    pass


def create_app(
    connect: Callable[[], psycopg.Connection] | None = None,
    erp: ERPWriter | None = None,
    secure_cookies: bool | None = None,
    session_hours: int | None = None,
    demo: dict | None = None,
    demo_reset: Callable[[psycopg.Connection], None] | None = None,
) -> FastAPI:
    """Everything external is injectable so tests can run the real app against test services."""
    if connect is None or erp is None or secure_cookies is None or session_hours is None:
        from orderdesk.config import get_settings
        from orderdesk.db import connect as db_connect
        from orderdesk.db import migrate
        from orderdesk.erp_client import HttpERPClient

        settings = get_settings()
        connect = connect or (lambda: db_connect(settings.database_url))
        erp = erp or HttpERPClient(settings.erp_base_url, settings.erp_write_key.get_secret_value())
        secure_cookies = settings.secure_cookies if secure_cookies is None else secure_cookies
        session_hours = session_hours or settings.session_hours
        if demo is None and settings.demo_mode:
            demo = {
                "email": settings.demo_email,
                "password": settings.demo_password.get_secret_value(),
            }
            demo_reset = demo_reset or _default_demo_reset(settings, erp)
        with connect() as conn:
            migrate(conn)

    app = FastAPI(title="orderdesk", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        if secure_cookies:
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    @app.exception_handler(LoginRequired)
    def to_login(request: Request, _exc: LoginRequired) -> Response:
        if request.headers.get("HX-Request"):  # an HTMX fragment: send the whole page to login
            return Response(status_code=401, headers={"HX-Redirect": "/login"})
        return RedirectResponse(f"/login?next={quote(request.url.path)}", status_code=303)

    @app.exception_handler(CSRFFailed)
    def csrf_failed(_request: Request, _exc: CSRFFailed) -> Response:
        return HTMLResponse("Form expired or invalid. Go back, refresh and try again.", 403)

    # --- dependencies ------------------------------------------------------------------------

    def db() -> Iterator[psycopg.Connection]:
        with connect() as conn:  # one connection per request, closed afterwards
            yield conn

    DB = Annotated[psycopg.Connection, Depends(db)]

    def current_user(request: Request, conn: DB) -> auth.Session:
        session = auth.session_for(conn, request.cookies.get(COOKIE))
        if session is None:
            raise LoginRequired
        return session

    User = Annotated[auth.Session, Depends(current_user)]

    def service(conn: DB) -> ApprovalService:
        return ApprovalService(conn, erp)

    Service = Annotated[ApprovalService, Depends(service)]

    def check_csrf(user: auth.Session, submitted) -> None:
        if not auth.csrf_ok(user, str(submitted) if submitted else None):
            raise CSRFFailed

    def page(
        request: Request,
        name: str,
        user: auth.Session | None = None,
        status_code: int = 200,
        **context,
    ) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request,
            name,
            {"user": user, "csrf": user.csrf_token if user else "", "demo": demo, **context},
            status_code=status_code,
        )

    # --- public ------------------------------------------------------------------------------

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/login", response_class=HTMLResponse)
    def login_form(request: Request, next: str = "/") -> HTMLResponse:
        return page(request, "login.html", next=next)

    @app.post("/login")
    def login(
        request: Request,
        conn: DB,
        email: Annotated[str, Form()] = "",
        password: Annotated[str, Form()] = "",
        next: Annotated[str, Form()] = "/",
    ):
        try:
            token = auth.login(conn, email, password, hours=session_hours)
        except auth.AuthError as exc:
            return page(
                request, "login.html", status_code=401, error=str(exc), email=email, next=next
            )
        # Only redirect within this site: an open redirect would help phishing.
        target = next if next.startswith("/") and not next.startswith("//") else "/"
        response = RedirectResponse(target, status_code=303)
        response.set_cookie(
            COOKIE,
            token,
            max_age=session_hours * 3600,
            httponly=True,
            secure=secure_cookies,
            samesite="lax",
        )
        return response

    @app.post("/logout")
    async def logout(request: Request, conn: DB, user: User):
        check_csrf(user, (await request.form()).get("csrf"))
        auth.logout(conn, request.cookies.get(COOKIE))
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(COOKIE)
        return response

    # --- the review queue ----------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def queue(request: Request, svc: Service, user: User, done: str | None = None):
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
            user,
            pending=[svc.review(r[0]) for r in svc.queue()],
            recent=recent,
            done=done,
        )

    @app.get("/orders/{document_id}", response_class=HTMLResponse)
    def order(request: Request, document_id: str, svc: Service, user: User):
        return _order_page(request, svc, user, document_id)

    def _order_page(request, svc, user, document_id, error=None, status_code=200):
        try:
            item = svc.review(document_id)
        except ApprovalError:
            return page(request, "missing.html", user, status_code=404, document_id=document_id)
        proposal = svc.conn.execute(
            "SELECT lines FROM order_proposals WHERE document_id = %s", (document_id,)
        ).fetchone()[0]
        options = {ln["line_number"]: ln["candidates"][:5] for ln in proposal}
        decision = svc.conn.execute(
            "SELECT decision, decided_by, reason, erp_order_id, order_total, decided_at "
            "FROM decisions WHERE document_id = %s",
            (document_id,),
        ).fetchone()
        return page(
            request,
            "order.html",
            user,
            status_code=status_code,
            item=item,
            options=options,
            decision=decision,
            error=error,
        )

    @app.post("/orders/{document_id}/approve")
    async def approve(request: Request, document_id: str, svc: Service, user: User):
        form = await request.form()
        check_csrf(user, form.get("csrf"))
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
                user,
                document_id,
                status_code=422,
                error=f"Lines {problems} need a product and a whole-number "
                "quantity, or tick Remove.",
            )
        try:
            order_id = svc.approve(document_id, user.email, lines)
        except ApprovalError as exc:
            return _order_page(request, svc, user, document_id, error=str(exc), status_code=422)
        # Post/Redirect/Get: refreshing the next page can't resubmit the approval.
        return RedirectResponse(
            f"/?done=Approved+{item.filename}+as+ERP+order+{order_id}", status_code=303
        )

    @app.post("/orders/{document_id}/reject")
    def reject(
        request: Request,
        document_id: str,
        svc: Service,
        user: User,
        reason: Annotated[str, Form()] = "",
        csrf: Annotated[str, Form()] = "",
    ):
        check_csrf(user, csrf)
        try:
            svc.reject(document_id, user.email, reason)
        except ApprovalError as exc:
            return _order_page(request, svc, user, document_id, error=str(exc), status_code=422)
        return RedirectResponse("/?done=Rejected", status_code=303)

    # --- HTMX fragments: small pieces of HTML swapped into the page -----------------------------

    @app.get("/orders/{document_id}/lines/{line_number}/check", response_class=HTMLResponse)
    def check_line(
        request: Request, document_id: str, line_number: int, svc: Service, user: User
    ) -> HTMLResponse:
        """Live feedback while the reviewer edits a line: product name, price, stock.

        HTMX sends the row's own inputs, which are named sku_<n> and qty_<n>.
        """
        sku = request.query_params.get(f"sku_{line_number}", "").strip()
        qty = request.query_params.get(f"qty_{line_number}", "")
        item = svc.review(document_id)
        info, problem = None, None
        if sku:
            try:
                product = erp.get_product(sku)
                price = erp.get_price(item.customer_id, product.sku) if item.customer_id else None
                stock = erp.get_stock(product.sku).total_available
                info = {"product": product, "price": price, "stock": stock}
                if not product.active:
                    problem = "discontinued"
                elif qty.isdigit() and int(qty) > stock:
                    problem = f"only {stock} in stock"
            except ERPError as exc:
                problem = "unknown SKU" if exc.status_code == 404 else str(exc)
        return page(request, "_line_check.html", user, info=info, problem=problem)

    @app.get("/orders/{document_id}/history", response_class=HTMLResponse)
    def history(request: Request, document_id: str, conn: DB, user: User) -> HTMLResponse:
        events = conn.execute(
            "SELECT at, actor, event, detail FROM audit_log WHERE document_id = %s ORDER BY id",
            (document_id,),
        ).fetchall()
        investigation = conn.execute(
            "SELECT transcript, steps, cost_usd, model FROM investigations "
            "WHERE document_id = %s ORDER BY created_at DESC LIMIT 1",
            (document_id,),
        ).fetchone()
        calls = conn.execute(
            "SELECT at, purpose, model, input_tokens, output_tokens, cost_usd, latency_ms, error "
            "FROM llm_calls WHERE document_id = %s ORDER BY at",
            (document_id,),
        ).fetchall()
        return page(
            request, "_history.html", user, events=events, investigation=investigation, calls=calls
        )

    @app.get("/orders/{document_id}/original", response_class=HTMLResponse)
    def original(request: Request, document_id: str, conn: DB, user: User) -> HTMLResponse:
        row = conn.execute(
            "SELECT raw FROM documents WHERE document_id = %s", (document_id,)
        ).fetchone()
        return page(request, "_original.html", user, raw=row[0] if row else None)

    # --- demo -----------------------------------------------------------------------------------

    @app.post("/demo/reset")
    async def reset_demo(request: Request, conn: DB, user: User):
        if demo is None or demo_reset is None:
            return Response(status_code=404)
        check_csrf(user, (await request.form()).get("csrf"))
        demo_reset(conn)
        return RedirectResponse(
            "/?done=Demo+reset:+every+order+is+back+in+the+queue", status_code=303
        )

    # --- operations: what the AI is costing and how it's performing ----------------------------

    @app.get("/ops", response_class=HTMLResponse)
    def ops(request: Request, conn: DB, user: User) -> HTMLResponse:
        by_purpose = conn.execute(
            """
            SELECT purpose, model, count(*), sum(cost_usd), avg(latency_ms)::int,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)::int,
                   count(*) FILTER (WHERE error IS NOT NULL)
            FROM llm_calls GROUP BY purpose, model ORDER BY sum(cost_usd) DESC
            """
        ).fetchall()
        by_day = conn.execute(
            "SELECT date_trunc('day', at)::date, count(*), sum(cost_usd) FROM llm_calls "
            "GROUP BY 1 ORDER BY 1 DESC LIMIT 14"
        ).fetchall()
        outcomes = conn.execute(
            "SELECT status, count(*) FROM documents GROUP BY status ORDER BY count(*) DESC"
        ).fetchall()
        corrections = conn.execute(
            "SELECT count(*) FILTER (WHERE decision = 'approved'), "
            "count(*) FILTER (WHERE decision = 'approved' AND corrections <> '[]'::jsonb) "
            "FROM decisions"
        ).fetchone()
        return page(
            request,
            "ops.html",
            user,
            by_purpose=by_purpose,
            by_day=by_day,
            outcomes=outcomes,
            corrections=corrections,
        )

    return app


def _default_demo_reset(settings, erp) -> Callable[[psycopg.Connection], None]:
    """Re-run the demo seed. The embedding model loads on first use, not at startup."""
    from orderdesk.pipeline import Thresholds

    state = {}

    def reset(conn: psycopg.Connection) -> None:
        from orderdesk.demo import seed
        from orderdesk.embeddings import FastEmbedEmbedder

        state.setdefault("embedder", FastEmbedEmbedder(settings.embedding_model))
        seed(
            conn,
            state["embedder"],
            erp,
            Thresholds(settings.match_min_similarity, settings.match_min_margin),
        )

    return reset
