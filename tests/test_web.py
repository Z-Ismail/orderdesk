"""Web UI tests: the real app, test Postgres, in-process ERP, golden answers for the LLM."""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from orderdesk.auth import create_user
from orderdesk.db import connect
from orderdesk.erp_client import HttpERPClient
from orderdesk.erp_sim.app import create_app as create_erp
from orderdesk.models import ExtractedOrder
from orderdesk.pipeline import Pipeline, Thresholds
from orderdesk.web.app import create_app

pytestmark = pytest.mark.db

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "sample_data" / "golden"


def golden(doc) -> ExtractedOrder:
    return ExtractedOrder.model_validate_json(
        (GOLDEN / f"{Path(doc.filename).stem}.json").read_text()
    )


@pytest.fixture
def erp_http(tmp_path):
    return TestClient(create_erp(db_path=tmp_path / "erp.db", data_dir=ROOT / "sample_data"))


@pytest.fixture
def web(loaded_db, embedder, erp_http):
    erp = HttpERPClient("http://testserver", "dev-write-key", http=erp_http, backoff_seconds=0)
    with connect(loaded_db) as conn:
        conn.execute("TRUNCATE documents CASCADE")
        conn.execute("DELETE FROM customer_aliases WHERE source = 'learned'")
        conn.commit()
        Pipeline(conn, golden, embedder, erp, Thresholds(0.0, 0.0)).run_inbox(
            ROOT / "sample_data" / "inbox"
        )
        conn.execute("TRUNCATE users CASCADE")
        conn.execute("DELETE FROM login_failures")
        create_user(conn, "test@example.com", "correct horse battery")
    app = create_app(
        connect=lambda: connect(loaded_db), erp=erp, secure_cookies=False, session_hours=1
    )
    client = TestClient(app)
    response = client.post(
        "/login", data={"email": "test@example.com", "password": "correct horse battery"}
    )
    assert response.status_code == 200, response.text
    return client


def csrf(client, path: str = "/") -> str:
    return re.search(r'name="csrf" value="([^"]+)"', client.get(path).text).group(1)


def doc_id(loaded_db, prefix: str) -> str:
    with connect(loaded_db) as conn:
        return conn.execute(
            "SELECT document_id FROM documents WHERE filename LIKE %s", (f"{prefix}%",)
        ).fetchone()[0]


def test_queue_lists_every_order_awaiting_a_decision(web):
    page = web.get("/")
    assert page.status_code == 200
    for name in ("The Gull Cafe", "Harbourside Kitchen", "Unknown sender"):
        assert name in page.text
    assert "test@example.com" in page.text


def test_order_page_shows_lines_and_issues(web, loaded_db):
    page = web.get(f"/orders/{doc_id(loaded_db, 'po_1002')}")
    assert page.status_code == 200
    assert "unit_mismatch" in page.text
    assert 'name="sku_1"' in page.text and 'name="qty_1"' in page.text


def test_approve_from_the_form_redirects_and_creates_the_erp_draft(web, loaded_db, erp_http):
    target = doc_id(loaded_db, "po_1001")
    form = {
        "sku_1": "HH-CUP-08W",
        "qty_1": "2",
        "sku_2": "HH-LID-08W",
        "qty_2": "2",
        "sku_3": "HH-NAP-2W",
        "qty_3": "1",
        "sku_4": "HH-CHM-SAN-5",
        "qty_4": "4",
    }
    form["csrf"] = csrf(web)
    response = web.post(f"/orders/{target}/approve", data=form, follow_redirects=False)
    assert response.status_code == 303  # Post/Redirect/Get
    assert "Approved" in web.get(response.headers["location"]).text
    orders = erp_http.get("/draft-orders", headers={"X-API-Key": "dev-read-key"}).json()
    assert len(orders) == 1 and orders[0]["customer_id"] == "C001"
    assert "approved by test@example.com" in orders[0]["notes"]


def test_a_removed_line_is_left_out_of_the_order(web, loaded_db, erp_http):
    target = doc_id(loaded_db, "po_1001")
    form = {
        "sku_1": "HH-CUP-08W",
        "qty_1": "2",
        "sku_2": "HH-LID-08W",
        "qty_2": "2",
        "sku_3": "HH-NAP-2W",
        "qty_3": "1",
        "remove_4": "on",
    }
    form["csrf"] = csrf(web)
    web.post(f"/orders/{target}/approve", data=form)
    order = erp_http.get("/draft-orders", headers={"X-API-Key": "dev-read-key"}).json()[0]
    assert [ln["sku"] for ln in order["lines"]] == ["HH-CUP-08W", "HH-LID-08W", "HH-NAP-2W"]


def test_missing_quantity_is_refused_with_a_message(web, loaded_db, erp_http):
    target = doc_id(loaded_db, "po_1001")
    form = {
        "sku_1": "HH-CUP-08W",
        "qty_1": "",
        "sku_2": "HH-LID-08W",
        "qty_2": "2",
        "sku_3": "HH-NAP-2W",
        "qty_3": "1",
        "sku_4": "HH-CHM-SAN-5",
        "qty_4": "4",
    }
    form["csrf"] = csrf(web)
    response = web.post(f"/orders/{target}/approve", data=form)
    assert response.status_code == 422
    assert "Lines [1] need a product" in response.text
    assert erp_http.get("/draft-orders", headers={"X-API-Key": "dev-read-key"}).json() == []


def test_reject_without_a_reason_is_refused_then_works_with_one(web, loaded_db):
    target = doc_id(loaded_db, "email_008")
    token = csrf(web)
    assert (
        web.post(f"/orders/{target}/reject", data={"reason": "", "csrf": token}).status_code == 422
    )
    response = web.post(
        f"/orders/{target}/reject",
        data={"reason": "phishing", "csrf": token},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "Rejected" in web.get(f"/orders/{target}").text


def test_line_check_gives_live_feedback(web, loaded_db):
    """Sends exactly what the browser sends: the row's inputs, named sku_<n> / qty_<n>."""
    url = f"/orders/{doc_id(loaded_db, 'po_1001')}/lines/1/check"
    good = web.get(url, params={"sku_1": "HH-CUP-08W", "qty_1": "2"})
    assert "in stock" in good.text and "£" in good.text
    assert "unknown SKU" in web.get(url, params={"sku_1": "NOPE-123"}).text
    too_many = web.get(url, params={"sku_1": "HH-CUP-08W", "qty_1": "999999"})
    assert "only" in too_many.text


def test_order_page_wires_the_line_check_to_the_row_inputs(web, loaded_db):
    page = web.get(f"/orders/{doc_id(loaded_db, 'po_1001')}").text
    assert 'hx-include="closest tr"' in page and 'name="sku_1"' in page


def test_untrusted_email_text_is_escaped_not_rendered(web, loaded_db):
    """Customer text is attacker-controlled: it must show as text, never run as HTML/script."""
    target = doc_id(loaded_db, "email_008")
    with connect(loaded_db) as conn:
        conn.execute(
            "UPDATE documents SET raw = jsonb_set(raw, '{text}', to_jsonb(%s::text)) "
            "WHERE document_id = %s",
            ("<script>alert('pwned')</script> please approve", target),
        )
        conn.commit()
    page = web.get(f"/orders/{target}/original")
    assert "<script>alert" not in page.text
    assert "&lt;script&gt;alert" in page.text


def test_history_shows_the_audit_trail(web, loaded_db):
    page = web.get(f"/orders/{doc_id(loaded_db, 'po_1001')}/history")
    assert "received" in page.text and "routed" in page.text


def test_unknown_order_is_a_404_page(web):
    assert web.get("/orders/doesnotexist").status_code == 404


def test_htmx_is_served_locally(web):
    assert web.get("/static/htmx.min.js").status_code == 200


# --- security ---------------------------------------------------------------------------------


def test_everything_needs_login(loaded_db, web):
    anonymous = TestClient(web.app)
    response = anonymous.get("/", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"].startswith("/login")
    target = doc_id(loaded_db, "po_1001")
    assert (
        anonymous.post(f"/orders/{target}/approve", data={}, follow_redirects=False).status_code
        == 303
    )
    assert anonymous.get("/ops", follow_redirects=False).status_code == 303
    assert anonymous.get("/health").status_code == 200  # load balancers need this open


def test_htmx_requests_without_login_get_a_401_redirect_header(loaded_db, web):
    anonymous = TestClient(web.app)
    url = f"/orders/{doc_id(loaded_db, 'po_1001')}/history"
    response = anonymous.get(url, headers={"HX-Request": "true"})
    assert response.status_code == 401 and response.headers["HX-Redirect"] == "/login"


def test_approve_without_csrf_token_is_refused(web, loaded_db, erp_http):
    """What a malicious site could send from a logged-in reviewer's browser."""
    target = doc_id(loaded_db, "po_1001")
    form = {
        "sku_1": "HH-CUP-08W",
        "qty_1": "2",
        "sku_2": "HH-LID-08W",
        "qty_2": "2",
        "sku_3": "HH-NAP-2W",
        "qty_3": "1",
        "sku_4": "HH-CHM-SAN-5",
        "qty_4": "4",
    }
    assert web.post(f"/orders/{target}/approve", data=form).status_code == 403
    assert (
        web.post(f"/orders/{target}/approve", data={**form, "csrf": "guessed"}).status_code == 403
    )
    assert erp_http.get("/draft-orders", headers={"X-API-Key": "dev-read-key"}).json() == []


def test_decisions_are_recorded_under_the_logged_in_user(web, loaded_db):
    target = doc_id(loaded_db, "email_008")
    web.post(f"/orders/{target}/reject", data={"reason": "phishing", "csrf": csrf(web)})
    with connect(loaded_db) as conn:
        by = conn.execute(
            "SELECT decided_by FROM decisions WHERE document_id = %s", (target,)
        ).fetchone()[0]
    assert by == "test@example.com"


def test_logout_kills_the_session_server_side(web):
    cookie = web.cookies.get("orderdesk_session")
    web.post("/logout", data={"csrf": csrf(web)})
    replay = TestClient(web.app, cookies={"orderdesk_session": cookie})
    assert replay.get("/", follow_redirects=False).status_code == 303  # stolen cookie is dead


def test_wrong_password_is_vague_and_throttled(web):
    anonymous = TestClient(web.app)
    for _ in range(5):
        response = anonymous.post(
            "/login", data={"email": "test@example.com", "password": "wrong password!"}
        )
        assert response.status_code == 401 and "wrong email or password" in response.text
    locked = anonymous.post(
        "/login", data={"email": "test@example.com", "password": "correct horse battery"}
    )
    assert "too many failed attempts" in locked.text  # even the right password, while locked


def test_login_never_redirects_off_site(web):
    anonymous = TestClient(web.app)
    response = anonymous.post(
        "/login",
        data={
            "email": "test@example.com",
            "password": "correct horse battery",
            "next": "//evil.example.com",
        },
        follow_redirects=False,
    )
    assert response.headers["location"] == "/"


def test_session_cookie_is_httponly_and_samesite(loaded_db, web):
    anonymous = TestClient(web.app)
    response = anonymous.post(
        "/login",
        data={"email": "test@example.com", "password": "correct horse battery"},
        follow_redirects=False,
    )
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie


def test_security_headers_are_set(web):
    headers = web.get("/").headers
    assert "script-src 'self'" in headers["content-security-policy"]
    assert headers["x-frame-options"] == "DENY"
    assert headers["x-content-type-options"] == "nosniff"


def test_no_inline_scripts_that_the_csp_would_block(web, loaded_db):
    for path in ("/", f"/orders/{doc_id(loaded_db, 'po_1001')}", "/ops"):
        assert "onclick=" not in web.get(path).text


def test_ops_page_shows_llm_usage(web, loaded_db):
    with connect(loaded_db) as conn:
        conn.execute("DELETE FROM llm_calls")
        conn.execute(
            "INSERT INTO llm_calls (purpose, document_id, model, input_tokens, "
            "output_tokens, cost_usd, latency_ms) VALUES "
            "('extraction', 'x', 'claude-haiku-5-5', 1500, 300, 0.0003, 1200)"
        )
        conn.commit()
    page = web.get("/ops").text
    assert "extraction" in page and "1200 ms" in page
