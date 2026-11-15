"""Run real extraction over the inbox. Saves each result to scratch/out/<file>.json.

Clears old results first, so the scorer can never grade a stale run, and exits
with an error code if any document failed.
"""

import sys
from pathlib import Path

from orderdesk.config import get_settings
from orderdesk.extraction import NeedsOCRError, extract_order
from orderdesk.ingestion import scan_inbox
from orderdesk.llm import LLMClient

OUT = Path("scratch/out")
OUT.mkdir(parents=True, exist_ok=True)
for old in OUT.iterdir():
    old.unlink()  # never let a previous run's results masquerade as this run's

llm = LLMClient(get_settings())
failures = 0

for doc in scan_inbox(Path("sample_data/inbox")):
    stem = Path(doc.filename).stem
    try:
        order = extract_order(doc, llm)
        (OUT / f"{stem}.json").write_text(order.model_dump_json(indent=2))
        print(f"OK       {doc.filename}  ({len(order.lines)} lines)")
    except NeedsOCRError:
        print(f"SKIPPED  {doc.filename}  (needs OCR)")
    except Exception as exc:  # noqa: BLE001 - batch runner: record any failure, keep going
        failures += 1
        (OUT / f"{stem}.error.txt").write_text(repr(exc))
        print(f"FAILED   {doc.filename}  {exc!r}")

print(f"\nTOTAL: {llm.budget.calls} calls, ${llm.budget.cost_usd:.4f}, {failures} failed")
sys.exit(1 if failures else 0)
