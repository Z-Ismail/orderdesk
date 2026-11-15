"""Investigator agent tests. A scripted fake plays Claude, so we test the loop and guardrails,
not the model's judgement (that's what running the real agent is for)."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from orderdesk.agent import TOOLS, AgentTools, InvestigationFailedError, Investigator, build_case
from orderdesk.config import Settings
from orderdesk.db import connect
from orderdesk.erp_client import HttpERPClient
from orderdesk.erp_sim.app import create_app
from orderdesk.llm import LLMClient
from orderdesk.models import ExtractedOrder
from orderdesk.pipeline import Pipeline, Thresholds

pytestmark = pytest.mark.db

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "sample_data" / "golden"

GOOD_FINDINGS = {
    "recommended_action": "approve_with_suggestions",
    "customer_id": "C004",
    "summary": "Line 4 is clearly wooden knives, not forks.",
    "risk_flags": [],
    "lines": [
        {
            "line_number": 4,
            "recommended_sku": "HH-CUT-WD-K",
            "recommended_quantity": 1,
            "confidence": "high",
            "reasoning": "Customer wrote knives.",
        }
    ],
}


def tool_use(name: str, args: dict, call_id: str = "t1"):
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=args)


class ScriptedClaude:
    """Returns the next scripted response each time; records every request."""

    def __init__(self, script: list[list]) -> None:
        self.script, self.requests = list(script), []

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))  # snapshot: the agent mutates its list
        content = (
            self.script.pop(0) if self.script else [tool_use("search_catalogue", {"query": "loop"})]
        )
        return SimpleNamespace(
            content=content, usage=SimpleNamespace(input_tokens=1000, output_tokens=100)
        )


@pytest.fixture
def erp(tmp_path):
    http = TestClient(create_app(db_path=tmp_path / "erp.db", data_dir=ROOT / "sample_data"))
    return HttpERPClient("http://testserver", "dev-read-key", http=http, backoff_seconds=0)


@pytest.fixture
def conn(loaded_db, embedder, erp):
    """A database with the whole sample inbox already run through the pipeline."""

    def golden(doc):
        stem = Path(doc.filename).stem
        return ExtractedOrder.model_validate_json((GOLDEN / f"{stem}.json").read_text())

    with connect(loaded_db) as connection:
        connection.execute("TRUNCATE documents CASCADE")
        connection.commit()
        Pipeline(connection, golden, embedder, erp, Thresholds(0.999, 0.5)).run_inbox(
            ROOT / "sample_data" / "inbox"
        )
        yield connection


def doc_id(conn, filename_prefix: str) -> str:
    return conn.execute(
        "SELECT document_id FROM documents WHERE filename LIKE %s", (f"{filename_prefix}%",)
    ).fetchone()[0]


def investigator(conn, embedder, erp, script, max_steps=8):
    claude = ScriptedClaude(script)
    settings = Settings(anthropic_api_key=SecretStr("test"), _env_file=None)
    llm = LLMClient(settings, client=SimpleNamespace(messages=claude))
    return Investigator(conn, llm, embedder, erp, "claude-sonnet-5-5", max_steps), claude


def test_investigation_uses_tools_then_saves_findings(conn, embedder, erp):
    target = doc_id(conn, "email_004")
    agent, claude = investigator(
        conn,
        embedder,
        erp,
        [
            [tool_use("search_catalogue", {"query": "wooden knife"})],
            [tool_use("submit_findings", GOOD_FINDINGS, "t2")],
        ],
    )
    findings = agent.investigate(target)

    assert findings.lines[0].recommended_sku == "HH-CUT-WD-K"
    status = conn.execute(
        "SELECT status FROM documents WHERE document_id = %s", (target,)
    ).fetchone()[0]
    assert status == "investigated"
    steps, transcript = conn.execute(
        "SELECT steps, transcript FROM investigations WHERE document_id = %s", (target,)
    ).fetchone()
    assert steps == 2
    assert transcript[0]["tool"] == "search_catalogue"
    assert "HH-CUT-WD-K" in str(transcript[0]["output"])  # the tool really searched
    # The tool result went back to the model on the second request.
    second = claude.requests[1]["messages"]
    assert second[-1]["content"][0]["type"] == "tool_result"


def test_step_limit_stops_a_looping_agent(conn, embedder, erp):
    target = doc_id(conn, "email_004")
    agent, claude = investigator(conn, embedder, erp, [], max_steps=3)  # loops forever
    with pytest.raises(InvestigationFailedError, match="3 steps"):
        agent.investigate(target)
    assert len(claude.requests) == 3
    status = conn.execute(
        "SELECT status FROM documents WHERE document_id = %s", (target,)
    ).fetchone()[0]
    assert status == "needs_investigation"  # unchanged: a human will see it as uninvestigated
    event = conn.execute(
        "SELECT event FROM audit_log WHERE document_id = %s ORDER BY id DESC", (target,)
    ).fetchone()[0]
    assert event == "investigation_failed"


def test_invalid_findings_are_returned_to_the_model_to_fix(conn, embedder, erp):
    target = doc_id(conn, "email_004")
    bad = {**GOOD_FINDINGS, "recommended_action": "approve_and_ship_now"}  # not allowed
    agent, claude = investigator(
        conn,
        embedder,
        erp,
        [
            [tool_use("submit_findings", bad)],
            [tool_use("submit_findings", GOOD_FINDINGS, "t2")],
        ],
    )
    assert agent.investigate(target).recommended_action == "approve_with_suggestions"
    error_result = claude.requests[1]["messages"][-1]["content"][0]
    assert error_result["is_error"] is True


def test_agent_has_no_tools_that_write():
    names = {t["name"] for t in TOOLS}
    assert names == {
        "search_catalogue",
        "get_product",
        "check_stock",
        "get_customer_price",
        "find_customer",
        "get_order_history",
        "submit_findings",
    }
    assert not any(
        word in name for name in names for word in ("create", "approve", "update", "delete", "send")
    )


def test_tool_errors_become_messages_not_crashes(conn, embedder, erp):
    tools = AgentTools(conn, embedder, erp, "C001", "x")
    assert "error" in tools.run("get_product", {"sku": "NOT-A-SKU"})
    assert "error" in tools.run("get_product", {"wrong_arg": 1})
    assert "error" in tools.run("delete_everything", {})


def test_prices_only_for_this_orders_customer(conn, embedder, erp):
    assert "error" in AgentTools(conn, embedder, erp, None, "x").run(
        "get_customer_price", {"sku": "HH-CUP-08W"}
    )


def test_order_history_finds_the_original_for_an_amendment(conn, embedder, erp):
    tools = AgentTools(conn, embedder, erp, "C002", doc_id(conn, "email_007"))
    pos = [
        o["po_number"] for o in tools.run("get_order_history", {"customer_id": "C002"})["orders"]
    ]
    assert "CGH-PO-88341" in pos


def test_case_briefing_wraps_the_document_as_untrusted(conn):
    case, customer_id = build_case(conn, doc_id(conn, "email_008"))
    assert customer_id is None
    assert "NOT IDENTIFIED" in case
    assert "unknown_customer" in case
    start, end = case.index("<untrusted_document>"), case.index("</untrusted_document>")
    assert start < case.index("ignore all previous instructions") < end


def test_empty_text_blocks_are_not_sent_back(conn, embedder, erp):
    """Real Claude can reply with an empty text block beside a tool call; the API rejects
    empty text in the history, so it must be dropped."""
    target = doc_id(conn, "email_004")
    empty = SimpleNamespace(type="text", text="")
    agent, claude = investigator(
        conn,
        embedder,
        erp,
        [
            [empty, tool_use("search_catalogue", {"query": "wooden knife"})],
            [tool_use("submit_findings", GOOD_FINDINGS, "t2")],
        ],
    )
    agent.investigate(target)
    history = claude.requests[1]["messages"]
    sent = [b for m in history if m["role"] == "assistant" for b in m["content"]]
    assert all(b.get("text", "x").strip() for b in sent if b["type"] == "text")
    assert sent[0]["type"] == "tool_use"
