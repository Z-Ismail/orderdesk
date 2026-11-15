"""First real API call, through our budget-guarded client."""

from orderdesk.config import get_settings
from orderdesk.llm import LLMClient

llm = LLMClient(get_settings())  # reads ANTHROPIC_API_KEY from .env

response = llm.create(
    max_tokens=200,
    messages=[{"role": "user", "content": "In one sentence, what is a purchase order?"}],
)

print("ANSWER:     ", response.content[0].text)
print("STOP REASON:", response.stop_reason)
print("TOKENS:     ", response.usage.input_tokens, "in /", response.usage.output_tokens, "out")
print(
    f"COST:        ${llm.budget.cost_usd:.6f}  (budget: {llm.budget.calls} of {llm.budget.max_calls} calls used)"
)
