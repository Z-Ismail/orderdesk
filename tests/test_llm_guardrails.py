"""Guardrail tests. These use a fake client, so they never call the real API or cost money."""

from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from orderdesk.config import Settings
from orderdesk.llm import BudgetExceededError, LLMClient


class FakeMessages:
    """Pretends every call used 1M input tokens: $0.10 on Haiku."""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(input_tokens=1_000_000, output_tokens=0))


def make_client(**overrides):
    settings = Settings(
        anthropic_api_key=SecretStr("sk-ant-test-secret"), _env_file=None, **overrides
    )
    fake = SimpleNamespace(messages=FakeMessages())
    return LLMClient(settings, client=fake), fake.messages


def ask(llm):
    return llm.create(messages=[{"role": "user", "content": "hi"}])


def test_call_limit_stops_runaway_loops():
    llm, fake = make_client(max_calls_per_run=2)
    ask(llm)
    ask(llm)
    with pytest.raises(BudgetExceededError):
        ask(llm)
    assert len(fake.requests) == 2  # the third call never reached the API


def test_cost_limit():
    llm, _fake = make_client(max_cost_usd_per_run=0.15)
    ask(llm)  # $0.10
    ask(llm)  # $0.20: allowed because the check happens before the call
    with pytest.raises(BudgetExceededError):
        ask(llm)
    assert llm.budget.cost_usd == pytest.approx(0.20)


def test_max_tokens_is_clamped():
    llm, fake = make_client(max_output_tokens=2000)
    llm.create(messages=[{"role": "user", "content": "hi"}], max_tokens=100_000)
    assert fake.requests[0]["max_tokens"] == 2000


def test_unpriced_model_is_refused():
    llm, fake = make_client()
    with pytest.raises(ValueError):
        llm.create(model="some-unknown-model", messages=[])
    assert fake.requests == []


def test_api_key_never_appears_in_output():
    settings = Settings(anthropic_api_key=SecretStr("sk-ant-test-secret"), _env_file=None)
    assert "sk-ant-test-secret" not in repr(settings)
    assert "sk-ant-test-secret" not in str(settings.model_dump())
