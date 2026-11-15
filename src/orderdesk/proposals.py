"""The pipeline's output: a proposed order plus the reasons a human or the agent must look at it."""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from orderdesk.matching import Candidate
from orderdesk.models import ExtractedOrder

IssueCode = Literal[
    "unknown_customer",  # sender doesn't match any customer account
    "customer_on_hold",  # account exists but is not allowed to order
    "customer_matched_by_name",  # no email match; identified by company name only (weaker)
    "amendment",  # changes an earlier order instead of placing a new one
    "possible_duplicate",  # same customer + PO number already seen (retyped re-send)
    "no_match",  # nothing in the catalogue looks like this line
    "low_match_confidence",  # best product isn't clearly ahead of the others
    "unit_mismatch",  # customer's unit differs from how we sell it (box vs case)
    "non_integer_quantity",  # e.g. 2.5 cases
    "out_of_stock",  # not enough available across warehouses
    "erp_lookup_failed",  # couldn't get price or stock right now
]


class Issue(BaseModel):
    code: IssueCode
    message: str
    line_number: int | None = None  # None = applies to the whole order


class ProposedLine(BaseModel):
    line_number: int
    description: str
    quantity: Decimal
    customer_unit: str | None
    customer_sku: str | None
    candidates: list[Candidate]
    chosen_sku: str | None
    similarity: float | None = None
    margin: float | None = None
    unit_price: Decimal | None = None
    available_stock: int | None = None


class OrderProposal(BaseModel):
    document_id: str
    customer_id: str | None
    customer_name: str | None
    extracted: ExtractedOrder
    lines: list[ProposedLine]
    issues: list[Issue]

    @property
    def status(self) -> Literal["needs_investigation", "ready_for_approval"]:
        return "needs_investigation" if self.issues else "ready_for_approval"
