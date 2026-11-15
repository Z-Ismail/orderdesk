"""Web UI tests: the real app, test Postgres, in-process ERP, golden answers for the LLM."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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
    app = create_app(connect=lambda: connect(loaded_db), erp=erp, reviewer="test@example.com")
    return TestClient(app)


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
    assert "Reviewing as test@example.com" in page.text


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
    response = web.post(f"/orders/{target}/approve", data=form)
    assert response.status_code == 422
    assert "Lines [1] need a product" in response.text
    assert erp_http.get("/draft-orders", headers={"X-API-Key": "dev-read-key"}).json() == []


def test_reject_without_a_reason_is_refused_then_works_with_one(web, loaded_db):
    target = doc_id(loaded_db, "email_008")
    assert web.post(f"/orders/{target}/reject", data={"reason": ""}).status_code == 422
    response = web.post(
        f"/orders/{target}/reject", data={"reason": "phishing"}, follow_redirects=False
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
