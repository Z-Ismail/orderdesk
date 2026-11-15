"""CLI: python -m orderdesk.ingest <folder>"""

import sys
from pathlib import Path

from orderdesk.ingestion import scan_inbox


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample_data/inbox")
    files = [p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")]
    docs = scan_inbox(folder)

    needs_ocr = 0
    for doc in docs:
        flagged = doc.needs_ocr or any(a.needs_ocr for a in doc.attachments)
        needs_ocr += flagged
        extra = f" +{len(doc.attachments)} attachment(s)" if doc.attachments else ""
        flag = "  [needs OCR]" if flagged else ""
        print(f"{doc.document_id}  {doc.source_type:<5}  {doc.filename}{extra}{flag}")

    duplicates = len(files) - len(docs)
    print(f"{len(docs)} unique documents ({duplicates} duplicate skipped, {needs_ocr} needs OCR)")


if __name__ == "__main__":
    main()
