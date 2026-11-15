from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from datetime import date
from typing import Literal

from pydantic import EmailStr, model_validator


class OrderLine(BaseModel):
    """One line of a customer's purchase order, as the customer wrote it."""

    model_config = ConfigDict(extra="forbid")

    line_number: int = Field(ge=1)
    customer_sku: str | None = None
    description: str = Field(min_length=1)
    quantity: Decimal = Field(gt=0)
    unit_of_measure: str | None = None
    unit_price: Decimal | None = Field(default=None, ge=0)

class ExtractedOrder(BaseModel):
    """What the LLM must produce from a PO document. Nothing else gets through."""

    model_config = ConfigDict(extra="forbid")

    customer_name: str | None = None
    customer_email: EmailStr | None = None
    customer_po_number: str | None = None
    order_date: date | None = None
    requested_delivery_date: date | None = None
    delivery_address: str | None = None
    currency: Literal["GBP", "EUR", "USD"] = "GBP"
    notes: str | None = None
    is_amendment: bool = False
    lines: list[OrderLine] = Field(min_length=1)

    @model_validator(mode="after")
    def check_cross_field_rules(self) -> "ExtractedOrder":
        numbers = [line.line_number for line in self.lines]
        if len(numbers) != len(set(numbers)):
            raise ValueError("line numbers must be unique")
        if (self.order_date and self.requested_delivery_date
                and self.requested_delivery_date < self.order_date):
            raise ValueError("delivery date cannot be before order date")
        return self