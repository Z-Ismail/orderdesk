from pathlib import Path

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