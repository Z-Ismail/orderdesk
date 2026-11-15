"""Every LLM call is traced: successes with tokens and cost, failures with the error."""

from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from orderdesk.config import Settings
from orderdesk.db import connect
from orderdesk.llm import LLMClient
from orderdesk.tracing import PostgresRecorder

SETTINGS = Settings(anthropic_api_key=SecretStr("test"), _env_file=None)


class FakeMessages:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def create(self, **kwargs):
        if self.fail:
            raise TimeoutError("API timed out")
        return SimpleNamespace(
            stop_reason="end_turn", usage=SimpleNamespace(input_tokens=1000, output_tokens=200)
        )


def test_successful_call_is_recorded_with_cost():
    records = []
    llm = LLMClient(
        SETTINGS, client=SimpleNamespace(messages=FakeMessages()), recorder=records.append
    )
    llm.create(purpose="extraction", document_id="doc1", messages=[], model="claude-haiku-5-5")
    (r,) = records
    assert (r.purpose, r.document_id, r.input_tokens, r.output_tokens) == (
        "extraction",
        "doc1",
        1000,
        200,
    )
    assert r.cost_usd == pytest.approx((1000 * 0.10 + 200 * 0.50) / 1_000_000)
    assert r.error is None and r.latency_ms >= 0


def test_failed_call_is_recorded_then_reraised():
    records = []
    llm = LLMClient(
        SETTINGS, client=SimpleNamespace(messages=FakeMessages(fail=True)), recorder=records.append
    )
    with pytest.raises(TimeoutError):
        llm.create(purpose="agent", messages=[], model="claude-haiku-5-5")
    assert "timed out" in records[0].error


def test_a_broken_recorder_never_breaks_the_call():
    def broken(_record):
        raise RuntimeError("tracing database down")

    llm = LLMClient(SETTINGS, client=SimpleNamespace(messages=FakeMessages()), recorder=broken)
    assert llm.create(messages=[], model="claude-haiku-5-5").stop_reason == "end_turn"


def test_tracing_kwargs_are_not_sent_to_the_api():
    sent = {}

    class Spy(FakeMessages):
        def create(self, **kwargs):
            sent.update(kwargs)
            return super().create(**kwargs)

    LLMClient(SETTINGS, client=SimpleNamespace(messages=Spy())).create(
        purpose="x", document_id="y", messages=[], model="claude-haiku-5-5"
    )
    assert "purpose" not in sent and "document_id" not in sent


@pytest.mark.db
def test_postgres_recorder_survives_a_rollback(loaded_db):
    recorder = PostgresRecorder(loaded_db)
    llm = LLMClient(SETTINGS, client=SimpleNamespace(messages=FakeMessages()), recorder=recorder)
    with connect(loaded_db) as conn:
        conn.execute("DELETE FROM llm_calls")
        conn.commit()
        conn.execute("SELECT 1")  # the pipeline's own transaction...
        llm.create(
            purpose="extraction", document_id="rolled-back", messages=[], model="claude-haiku-5-5"
        )
        conn.rollback()  # ...fails and rolls back, but the trace was committed separately
        count = conn.execute("SELECT count(*) FROM llm_calls WHERE document_id = 'rolled-back'")
        assert count.fetchone()[0] == 1
    recorder.close()
