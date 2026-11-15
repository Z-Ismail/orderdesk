import hashlib
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr
from pathlib import Path

from orderdesk.ingestion.pdf import pdf_from_bytes
from orderdesk.models import RawDocument


def email_from_bytes(data: bytes, filename: str) -> RawDocument:
    sha = hashlib.sha256(data).hexdigest()
    msg = BytesParser(policy=policy.default).parsebytes(data)

    body = msg.get_body(preferencelist=("plain",))
    text = body.get_content() if body else ""

    attachments = [
        pdf_from_bytes(part.get_content(), part.get_filename() or "attachment.pdf")
        for part in msg.iter_attachments()
        if part.get_content_type() == "application/pdf"
    ]

    date_header = msg["Date"]
    return RawDocument(
        document_id=sha[:16],
        content_sha256=sha,
        source_type="email",
        filename=filename,
        text=text,
        sender=parseaddr(msg["From"] or "")[1].lower() or None,
        subject=msg["Subject"],
        received_at=date_header.datetime if date_header else None,
        message_id=msg["Message-ID"],
        attachments=attachments,
    )


def load_email(path: Path) -> RawDocument:
    return email_from_bytes(path.read_bytes(), path.name)
