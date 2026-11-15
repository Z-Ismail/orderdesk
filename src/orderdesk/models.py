from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class OrderLine(BaseModel):
    """One line of a customer's purchase order, as the customer wrote it."""

    model_config = ConfigDict(extra="forbid")

    line_number: int = Field(ge=1)
    customer_sku: str | None = None
    description: str = Field(min_length=1)
    quantity: Decimal = Field(gt=0)
    unit_of_measure: str | None = None
    unit_price: Decimal | None = Field(default=None, ge=0)