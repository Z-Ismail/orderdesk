import subprocess
import sys
from pathlib import Path

import pytest

from orderdesk.ingestion import UnsupportedDocumentError, load_document, scan_inbox

INBOX = Path(__file__).resolve().parents[1] / "sample_data" / "inbox"


def test_load_document_dispatches_by_extension():
    assert load_document(INBOX / "po_1001_gull_cafe.pdf").source_type == "pdf"
    assert load_document(INBOX / "email_003_redland_primary.eml").source_type == "email"


def test_unsupported_file_type_raises(tmp_path):
    bad = tmp_path / "order.docx"
    bad.write_bytes(b"not really a docx")
    with pytest.raises(UnsupportedDocumentError):
        load_document(bad)


def test_redelivered_email_is_deduplicated():
    names = [d.filename for d in scan_inbox(INBOX)]
    assert len(names) == 7
    assert "email_004_harbourside_kitchen.eml" in names
    assert "email_006_harbourside_kitchen_redelivered.eml" not in names


def test_hidden_files_are_ignored(tmp_path):
    (tmp_path / ".DS_Store").write_bytes(b"\x00")
    (tmp_path / "po.pdf").write_bytes((INBOX / "po_1001_gull_cafe.pdf").read_bytes())
    assert [d.filename for d in scan_inbox(tmp_path)] == ["po.pdf"]


def test_cli_summary():
    result = subprocess.run(
        [sys.executable, "-m", "orderdesk.ingest", str(INBOX)],
        check=False, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    last = result.stdout.strip().splitlines()[-1]
    assert last == "7 unique documents (1 duplicate skipped, 1 needs OCR)"
