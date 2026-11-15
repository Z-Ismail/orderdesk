"""End-to-end pipeline tests: real Postgres, real ERP app (in-process), golden answers standing
in for the LLM. Free, deterministic, and they exercise every routing rule."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from orderdesk.db import connect
from orderdesk.erp_client import HttpERPClient
from orderdesk.erp_sim.app import create_app
from orderdesk.ingestion import load_document
from orderdesk.models import ExtractedOrder, RawDocument
from orderdesk.pipeline import Pipeline, Thresholds

pytestmark = pytest.mark.db

ROOT = Path(__file__).resolve().parents[1]
INBOX = ROOT / "sample_data" / "inbox"
GOLDEN = ROOT / "sample_data" / "golden"
# The offline test embedder has no real notion of meaning, so don't judge its confidence here.
LENIENT = Thresholds(min_similarity=0.0, min_margin=0.0)


def golden_extractor(doc: RawDocument) -> ExtractedOrder:
    stem = Path(doc.filename).stem.replace("_redelivered", "")
    return ExtractedOrder.model_validate_json((GOLDEN / f"{stem}.json").read_text())


@pytest.fixture
def erp(tmp_path) -> HttpERPClient:
    http = TestClient(create_app(db_path=tmp_path / "erp.db", data_dir=ROOT / "sample_data"))
    return HttpERPClient("http://testserver", "dev-read-key", http=http, backoff_seconds=0)


@pytest.fixture
def pipeline(loaded_db, embedder, erp):
    with connect(loaded_db) as conn:
        conn.execute("TRUNCATE documents CASCADE")
        conn.commit()
        yield Pipeline(conn, golden_extractor, embedder, erp, LENIENT)


def run(pipeline, filename: str):
    return pipeline.process(load_document(INBOX / filename))


def test_clean_order_is_ready_for_approval(pipeline):
    outcome = run(pipeline, "po_1001_gull_cafe.pdf")
    assert (outcome.status, outcome.issues) == ("ready_for_approval", [])


def test_clean_order_has_prices_and_stock(pipeline):
    run(pipeline, "po_1001_gull_cafe.pdf")
    lines = pipeline.conn.execute("SELECT lines FROM order_proposals").fetchone()[0]
    assert lines[0]["chosen_sku"] == "HH-CUP-08W"
    assert lines[0]["unit_price"] == "42.00"
    assert lines[0]["available_stock"] > 0


@pytest.mark.parametrize(
    ("filename", "expected_issue"),
    [
        ("po_1005_exe_valley_scanned.pdf", "customer_on_hold"),
        ("email_008_unknown_sender.eml", "unknown_customer"),
        ("email_007_clifton_grand_amendment.eml", "amendment"),
        ("po_1002_clifton_grand.pdf", "unit_mismatch"),  # 2000 'each' napkins vs case
        ("email_003_redland_primary.eml", "unit_mismatch"),  # 1 'box' of gloves vs case
    ],
)
def test_problem_orders_are_sent_for_investigation(pipeline, filename, expected_issue):
    outcome = run(pipeline, filename)
    assert outcome.status == "needs_investigation"
    assert expected_issue in outcome.issues


def test_unknown_customer_gets_no_prices(pipeline):
    run(pipeline, "email_008_unknown_sender.eml")
    lines = pipeline.conn.execute("SELECT lines FROM order_proposals").fetchone()[0]
    assert all(line["unit_price"] is None for line in lines)


def test_rerun_is_idempotent(pipeline):
    first = run(pipeline, "po_1001_gull_cafe.pdf")
    second = run(pipeline, "po_1001_gull_cafe.pdf")
    assert second.skipped and second.status == first.status
    assert pipeline.conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 1
    assert pipeline.conn.execute("SELECT count(*) FROM order_proposals").fetchone()[0] == 1


def test_whole_inbox_skips_the_redelivered_email(pipeline):
    outcomes = pipeline.run_inbox(INBOX)
    assert len(outcomes) == 7
    statuses = {o.filename: o.status for o in outcomes}
    assert statuses["po_1001_gull_cafe.pdf"] == "ready_for_approval"


def test_retyped_resend_is_flagged_as_possible_duplicate(pipeline):
    run(pipeline, "po_1001_gull_cafe.pdf")
    # Same order, different bytes: the content hash differs, the business key doesn't.
    original = load_document(INBOX / "po_1001_gull_cafe.pdf")
    retyped = original.model_copy(
        update={
            "document_id": "retyped00000001",
            "content_sha256": "f" * 64,
            "filename": "resend.pdf",
        }
    )
    pipeline.extract = lambda doc: golden_extractor(original)
    assert "possible_duplicate" in pipeline.process(retyped).issues


def test_extraction_failure_is_recorded_and_retried(pipeline):
    def broken(doc):
        raise RuntimeError("LLM unavailable")

    pipeline.extract = broken
    failed = run(pipeline, "po_1001_gull_cafe.pdf")
    assert failed.status == "failed"
    error = pipeline.conn.execute("SELECT error FROM documents").fetchone()[0]
    assert "LLM unavailable" in error

    pipeline.extract = golden_extractor  # the LLM is back
    assert run(pipeline, "po_1001_gull_cafe.pdf").status == "ready_for_approval"


def test_every_step_is_audited(pipeline):
    run(pipeline, "po_1001_gull_cafe.pdf")
    events = [r[0] for r in pipeline.conn.execute("SELECT event FROM audit_log ORDER BY id")]
    assert events == ["received", "routed"]


def test_strict_thresholds_flag_searched_matches(pipeline):
    pipeline.thresholds = Thresholds(min_similarity=0.999, min_margin=0.5)
    outcome = run(pipeline, "email_004_harbourside_kitchen.eml")  # no aliases, all searched
    assert "low_match_confidence" in outcome.issues


def test_scan_without_email_is_identified_by_name_and_flagged(pipeline):
    def no_email(doc):
        return golden_extractor(doc).model_copy(update={"customer_email": None})

    pipeline.extract = no_email
    outcome = run(pipeline, "po_1005_exe_valley_scanned.pdf")
    assert "customer_on_hold" in outcome.issues  # the real problem is now visible
    assert "customer_matched_by_name" in outcome.issues  # and the weaker evidence is flagged
