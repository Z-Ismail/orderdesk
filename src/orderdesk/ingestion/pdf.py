import base64
import hashlib
import io
from pathlib import Path

from pypdf import PdfReader

from orderdesk.models import RawDocument

MIN_CHARS_PER_PAGE = 50  # below this we assume a scan with no text layer


def pdf_from_bytes(data: bytes, filename: str) -> RawDocument:
    sha = hashlib.sha256(data).hexdigest()
    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages)
    chars_per_page = len(text.strip()) / max(len(pages), 1)
    needs_ocr = chars_per_page < MIN_CHARS_PER_PAGE
    return RawDocument(
        document_id=sha[:16],
        content_sha256=sha,
        source_type="pdf",
        filename=filename,
        text=text,
        page_count=len(pages),
        needs_ocr=needs_ocr,
        ocr_source_b64=base64.b64encode(data).decode() if needs_ocr else None,
    )


def load_pdf(path: Path) -> RawDocument:
    return pdf_from_bytes(path.read_bytes(), path.name)
