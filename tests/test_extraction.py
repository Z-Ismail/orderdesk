"""Extraction tests with a fake LLM: they check our plumbing, not Claude's accuracy."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr, ValidationError

from orderdesk.config import Settings
from orderdesk.extraction import NeedsOCRError, build_document_text, extract_order
from orderdesk.ingestion import load_document
from orderdesk.llm import LLMClient

INBOX = Path(__file__).resolve().parents[1] / "sample_data" / "inbox"


class FakeMessages:
    def __init__(self, tool_input):
        self.tool_input = tool_input
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", name="record_order", input=self.tool_input)],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
        )


def make_llm(tool_input):
    settings = Settings(anthropic_api_key=SecretStr("test"), _env_file=None)
    fake = FakeMessages(tool_input)
    return LLMClient(settings, client=SimpleNamespace(messages=fake)), fake


GOOD = {
    "customer_name": "Harbourside Kitchen",
    "lines": [
        {
            "line_number": 1,
            "description": "Kraft box No.1",
            "quantity": 2,
            "unit_of_measure": "case",
        }
    ],
}


def test_prompt_contains_body_and_attachment():
    text = build_document_text(load_document(INBOX / "email_004_harbourside_kitchen.eml"))
    assert "before 10am" in text  # from the email body
    assert "HK-2026-311" in text  # from the PDF attachment
    assert "<sent>Wednesday 11 November 2026</sent>" in text


def test_extract_returns_validated_order_and_forces_the_tool():
    llm, fake = make_llm(GOOD)
    order = extract_order(load_document(INBOX / "email_004_harbourside_kitchen.eml"), llm)
    assert order.lines[0].description == "Kraft box No.1"
    assert fake.requests[0]["tool_choice"] == {"type": "tool", "name": "record_order"}


def test_scanned_pdf_never_reaches_the_llm():
    llm, fake = make_llm(GOOD)
    with pytest.raises(NeedsOCRError):
        extract_order(load_document(INBOX / "po_1005_exe_valley_scanned.pdf"), llm)
    assert fake.requests == []


def test_invalid_model_output_is_rejected():
    bad = {**GOOD, "discount_pct": 90}  # a hallucinated field
    llm, _ = make_llm(bad)
    with pytest.raises(ValidationError):
        extract_order(load_document(INBOX / "email_004_harbourside_kitchen.eml"), llm)
