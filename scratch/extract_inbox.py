"""Run real extraction over the whole inbox and print the results and total cost."""

from orderdesk.config import get_settings
from orderdesk.extraction import NeedsOCRError, extract_order
from orderdesk.ingestion import scan_inbox
from orderdesk.llm import LLMClient

llm = LLMClient(get_settings())

for doc in scan_inbox("sample_data/inbox"):
    print(f"\n=== {doc.filename}")
    try:
        order = extract_order(doc, llm)
        print(order.model_dump_json(indent=2, exclude_none=True))
    except NeedsOCRError as exc:
        print(f"SKIPPED: {exc}")

print(f"\nTOTAL: {llm.budget.calls} calls, ${llm.budget.cost_usd:.4f}")
