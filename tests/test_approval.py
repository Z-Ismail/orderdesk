"""Approval tests: real Postgres, real ERP app in-process, golden answers in place of the LLM."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from psycopg.types.json import Jsonb

from orderdesk.approval import ApprovalError, ApprovalService, FinalLine
from orderdesk.catalogue import load_catalogue
from orderdesk.db import connect
from orderdesk.erp_client import HttpERPClient
from orderdesk.erp_sim.app import DraftOrderIn, DraftOrderLineIn, create_app
from orderdesk.matching import match_line
from orderdesk.models import ExtractedOrder
from orderdesk.pipeline import Pipeline, Thresholds

pytestmark = pytest.mark.db

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "sample_data" / "golden"
REVIEWER = "reviewer@example.com"


def golden(doc) -> ExtractedOrder:
    return ExtractedOrder.model_validate_json(
        (GOLDEN / f"{Path(doc.filename).stem}.json").read_text()
    )


@pytest.fixture
def erp_http(tmp_path):
    return TestClient(create_app(db_path=tmp_path / "erp.db", data_dir=ROOT / "sample_data"))


@pytest.fixture
def erp(erp_http):
    return HttpERPClient("http://testserver", "dev-write-key", http=erp_http, backoff_seconds=0)


@pytest.fixture
def db(loaded_db, embedder, erp):
    with connect(loaded_db) as conn:
        conn.execute("TRUNCATE documents CASCADE")
        conn.execute("DELETE FROM customer_aliases WHERE source = 'learned'")
        conn.commit()
        Pipeline(conn, golden, embedder, erp, Thresholds(0.0, 0.0)).run_inbox(
            ROOT / "sample_data" / "inbox"
        )
        yield conn


@pytest.fixture
def service(db, erp):
    return ApprovalService(db, erp)


def doc_id(conn, prefix: str) -> str:
    return conn.execute(
        "SELECT document_id FROM documents WHERE filename LIKE %s", (f"{prefix}%",)
    ).fetchone()[0]


def status(conn, document_id: str) -> str:
    return conn.execute(
        "SELECT status FROM documents WHERE document_id = %s", (document_id,)
    ).fetchone()[0]


def erp_orders(erp_http) -> list:
    return erp_http.get("/draft-orders", headers={"X-API-Key": "dev-read-key"}).json()


def test_approving_a_clean_order_creates_one_erp_draft(service, db, erp_http):
    target = doc_id(db, "po_1001")
    order_id = service.approve(target, REVIEWER)

    assert status(db, target) == "approved"
    orders = erp_orders(erp_http)
    assert [o["order_id"] for o in orders] == [order_id]
    assert orders[0]["customer_po_number"] == "GC-24117"
    decision = db.execute(
        "SELECT decision, decided_by, erp_order_id, corrections "
        "FROM decisions WHERE document_id = %s",
        (target,),
    ).fetchone()
    assert decision == ("approved", REVIEWER, order_id, [])
    last_event = db.execute(
        "SELECT actor, event FROM audit_log WHERE document_id = %s ORDER BY id DESC LIMIT 1",
        (target,),
    ).fetchone()
    assert last_event == (REVIEWER, "approved")


def test_approving_twice_is_refused_and_still_one_order(service, db, erp_http):
    target = doc_id(db, "po_1001")
    service.approve(target, REVIEWER)
    with pytest.raises(ApprovalError, match="'approved'"):
        service.approve(target, "someone-else@example.com")
    assert len(erp_orders(erp_http)) == 1


def test_crash_after_erp_call_then_retry_reuses_the_same_order(service, db, erp, erp_http):
    """Simulate: ERP accepted the order, then we crashed before saving the decision."""
    target = doc_id(db, "po_1001")
    ghost = erp.create_draft_order(
        DraftOrderIn(customer_id="C001", lines=[DraftOrderLineIn(sku="HH-CUP-08W", quantity=1)]),
        idempotency_key=f"orderdesk-{target}",
    )
    assert service.approve(target, REVIEWER) == ghost.order_id
    assert len(erp_orders(erp_http)) == 1


def test_unknown_customer_cannot_be_approved(service, db):
    target = doc_id(db, "email_008")
    with pytest.raises(ApprovalError, match="no customer"):
        service.approve(target, REVIEWER)
    assert status(db, target) == "needs_investigation"
    event = db.execute(
        "SELECT event FROM audit_log WHERE document_id = %s ORDER BY id DESC", (target,)
    ).fetchone()[0]
    assert event == "approval_failed"


def test_erp_refusal_leaves_the_order_undecided(service, db):
    target = doc_id(db, "po_1005")  # Exe Valley: account on hold, the ERP says 409
    with pytest.raises(ApprovalError, match="hold"):
        service.approve(
            target, REVIEWER, [FinalLine(line_number=1, sku="HH-GLV-NIT-M", quantity=1)]
        )
    assert status(db, target) == "needs_investigation"
    assert db.execute("SELECT count(*) FROM decisions").fetchone()[0] == 0


def test_reject_needs_a_reason(service, db):
    target = doc_id(db, "email_008")
    with pytest.raises(ApprovalError, match="reason"):
        service.reject(target, REVIEWER, "  ")
    service.reject(target, REVIEWER, "Not a customer; looks like phishing")
    assert status(db, target) == "rejected"


def test_agent_findings_become_the_suggestion(service, db):
    target = doc_id(db, "email_004")
    findings = {
        "recommended_action": "approve_with_suggestions",
        "customer_id": "C004",
        "summary": "Line 4 is knives.",
        "risk_flags": [],
        "lines": [
            {
                "line_number": 4,
                "recommended_sku": "HH-CUT-WD-K",
                "recommended_quantity": 1,
                "confidence": "high",
                "reasoning": "Customer wrote knives.",
            }
        ],
    }
    db.execute(
        "INSERT INTO investigations (document_id, findings, transcript, steps, cost_usd, "
        "model) VALUES (%s, %s, '[]', 1, 0, 'test')",
        (target, Jsonb(findings)),
    )
    line4 = next(ln for ln in service.review(target).lines if ln.line_number == 4)
    assert (line4.sku, line4.source, line4.agent_reasoning) == (
        "HH-CUT-WD-K",
        "agent",
        "Customer wrote knives.",
    )


def test_a_correction_is_recorded_and_learned(service, db, embedder):
    target = doc_id(db, "email_004")
    item = service.review(target)
    final = [
        FinalLine(line_number=ln.line_number, sku=ln.sku, quantity=ln.quantity or 1)
        for ln in item.lines
    ]
    knives = next(ln for ln in final if ln.line_number == 4)
    knives.sku = "HH-CUT-WD-K" if knives.sku != "HH-CUT-WD-K" else "HH-CUT-WD-F"
    service.approve(target, REVIEWER, final)

    corrections = db.execute(
        "SELECT corrections FROM decisions WHERE document_id = %s", (target,)
    ).fetchone()[0]
    assert any(c["line_number"] == 4 and c["final_sku"] == knives.sku for c in corrections)

    # Next time this customer writes the same words, the alias wins over search.
    order = golden(type("Doc", (), {"filename": "email_004_harbourside_kitchen.eml"}))
    line = next(ln for ln in order.lines if ln.line_number == 4)
    best = match_line(db, embedder, line, "C004").best
    assert (best.source, best.sku) == ("alias", knives.sku)


def test_reloading_the_catalogue_keeps_learned_aliases(service, db, embedder):
    db.execute(
        "INSERT INTO customer_aliases (customer_id, customer_text, sku, source) "
        "VALUES ('C004', 'the wooden knives', 'HH-CUT-WD-K', 'learned')"
    )
    db.commit()
    load_catalogue(db, embedder, ROOT / "sample_data")
    learned = db.execute("SELECT count(*) FROM customer_aliases WHERE source = 'learned'")
    assert learned.fetchone()[0] == 1


def test_missing_suggestions_must_be_filled_in(service, db):
    target = doc_id(db, "po_1001")
    db.execute(
        "UPDATE order_proposals SET lines = jsonb_set(lines, '{0,chosen_sku}', 'null') "
        "WHERE document_id = %s",
        (target,),
    )
    with pytest.raises(ApprovalError, match="no suggested product"):
        service.approve(target, REVIEWER)
