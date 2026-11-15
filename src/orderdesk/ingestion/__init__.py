"""Turn files from the inbox into RawDocuments. The rest of the system only calls this module."""

from pathlib import Path

from orderdesk.ingestion.mail import email_from_bytes
from orderdesk.ingestion.pdf import pdf_from_bytes
from orderdesk.models import RawDocument

__all__ = ["UnsupportedDocumentError", "load_document", "scan_inbox"]

LOADERS = {".pdf": pdf_from_bytes, ".eml": email_from_bytes}


class UnsupportedDocumentError(ValueError):
    """Raised for file types we don't know how to read."""


def load_document(path: Path) -> RawDocument:
    path = Path(path)
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise UnsupportedDocumentError(f"Unsupported file type: {path.name}")
    return loader(path.read_bytes(), path.name)


def scan_inbox(folder: Path) -> list[RawDocument]:
    """Load every document in the folder, skipping hidden files and exact duplicates."""
    seen_hashes: set[str] = set()
    documents: list[RawDocument] = []
    for path in sorted(Path(folder).iterdir()):
        if path.name.startswith(".") or not path.is_file():
            continue
        doc = load_document(path)
        if doc.content_sha256 in seen_hashes:
            continue  # byte-identical re-delivery: first file wins
        seen_hashes.add(doc.content_sha256)
        documents.append(doc)
    return documents
