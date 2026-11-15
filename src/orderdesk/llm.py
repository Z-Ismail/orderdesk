"""The only place the app talks to an LLM. Every call goes through a budget check."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anthropic

from orderdesk.config import Settings

# USD per million tokens (input, output). Source: platform.claude.com/docs/en/about-claude/pricing
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}


class BudgetExceededError(RuntimeError):
    """Raised before a call that would break the per-run budget."""


@dataclass
class Budget:
    max_calls: int
    max_cost_usd: float
    calls: int = 0
    cost_usd: float = 0.0

    def check(self) -> None:
        if self.calls >= self.max_calls:
            raise BudgetExceededError(f"call limit reached ({self.max_calls})")
        if self.cost_usd >= self.max_cost_usd:
            raise BudgetExceededError(
                f"cost limit reached (${self.cost_usd:.4f} of ${self.max_cost_usd:.2f})"
            )

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        price_in, price_out = PRICES_PER_MTOK[model]
        cost = (input_tokens * price_in + output_tokens * price_out) / 1_000_000
        self.calls += 1
        self.cost_usd += cost
        return cost


@dataclass(frozen=True)
class CallRecord:
    """One LLM call, successful or not. What tracing stores."""

    purpose: str
    document_id: str | None
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    stop_reason: str | None
    error: str | None


Recorder = Callable[[CallRecord], None]


class LLMClient:
    def __init__(
        self, settings: Settings, client: Any = None, recorder: Recorder | None = None
    ) -> None:
        self.settings = settings
        self.recorder = recorder
        self.budget = Budget(settings.max_calls_per_run, settings.max_cost_usd_per_run)
        if client is None and not settings.anthropic_api_key.get_secret_value():
            raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example)")
        self._client = client or anthropic.Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            max_retries=2,  # SDK retries rate limits and server errors with backoff
            timeout=60.0,  # never hang forever
        )

    def create(
        self, *, purpose: str = "unspecified", document_id: str | None = None, **kwargs: Any
    ) -> Any:
        """Call the Messages API. `purpose` and `document_id` are for tracing only."""
        model = kwargs.setdefault("model", self.settings.extraction_model)
        if model not in PRICES_PER_MTOK:
            # fail closed: an unpriced model can't be budgeted, so it can't be called
            raise ValueError(f"No price configured for model {model!r}")
        requested = kwargs.get("max_tokens", self.settings.max_output_tokens)
        kwargs["max_tokens"] = min(requested, self.settings.max_output_tokens)

        self.budget.check()
        started = time.perf_counter()
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            self._trace(purpose, document_id, model, started, error=repr(exc))
            raise
        usage = response.usage
        cost = self.budget.record(model, usage.input_tokens, usage.output_tokens)
        self._trace(
            purpose,
            document_id,
            model,
            started,
            usage.input_tokens,
            usage.output_tokens,
            cost,
            getattr(response, "stop_reason", None),
        )
        return response

    def _trace(
        self,
        purpose,
        document_id,
        model,
        started,
        input_tokens=0,
        output_tokens=0,
        cost=0.0,
        stop_reason=None,
        error=None,
    ) -> None:
        if self.recorder is None:
            return
        record = CallRecord(
            purpose,
            document_id,
            model,
            input_tokens,
            output_tokens,
            cost,
            int((time.perf_counter() - started) * 1000),
            stop_reason,
            error,
        )
        try:
            self.recorder(record)
        except Exception:  # noqa: BLE001, S110 - tracing must never break the real work
            pass
