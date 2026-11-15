from datetime import date
from pathlib import Path

from orderdesk.ingestion.mail import load_email
from orderdesk.ingestion.pdf import load_pdf

INBOX = Path(__file__).resolve().parents[1] / "sample_data" / "inbox"


def test_text_pdf():
    doc = load_pdf(INBOX / "po_1001_gull_cafe.pdf")
    assert doc.page_count == 1
    assert doc.needs_ocr is False
    assert "GC-24117" in doc.text
    assert doc.document_id == doc.content_sha256[:16]


def test_scanned_pdf_is_flagged():
    assert load_pdf(INBOX / "po_1005_exe_valley_scanned.pdf").needs_ocr is True


def test_hash_is_deterministic():
    a = load_pdf(INBOX / "po_1002_clifton_grand.pdf")
    b = load_pdf(INBOX / "po_1002_clifton_grand.pdf")
    assert a.content_sha256 == b.content_sha256

def test_plain_email():
    doc = load_email(INBOX / "email_003_redland_primary.eml")
    assert doc.source_type == "email"
    assert doc.sender == "office@redlandprimary.bristol.sch.uk"
    assert doc.subject == "Order for next week"
    assert doc.received_at.tzinfo is not None
    assert doc.received_at.date() == date(2026, 11, 11)
    assert "blue roll" in doc.text.lower()
    assert doc.attachments == []


def test_email_with_pdf_attachment():
    doc = load_email(INBOX / "email_004_harbourside_kitchen.eml")
    assert len(doc.attachments) == 1
    att = doc.attachments[0]
    assert att.filename == "po_hk_2026_311.pdf"
    assert "HK-2026-311" in att.text
    assert att.content_sha256 != doc.content_sha256


def test_injection_text_is_preserved_as_data():
    doc = load_email(INBOX / "email_008_unknown_sender.eml")
    assert "ignore all previous instructions" in doc.text
