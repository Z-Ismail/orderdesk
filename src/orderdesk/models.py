from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, EmailStr, Field, model_validator

# --- Deterministic clean-up rules ------------------------------------------------------
# These run on every value before validation. They fix things an LLM gets "nearly right"
# with code (always correct) instead of prompt wording (usually correct).

NULL_LIKE = {"", "none", "null", "n/a", "na", "-", "unknown"}

UNIT_ALIASES = {
    "cases": "case",
    "cs": "case",
    "carton": "case",
    "cartons": "case",
    "ctn": "case",
    "boxes": "box",
    "bx": "box",
    "packs": "pack",
    "pk": "pack",
    "pkt": "pack",
    "rolls": "roll",
    "ea": "each",
    "unit": "each",
    "units": "each",
    "pcs": "each",
    "piece": "each",
    "pieces": "each",
}


def empty_to_none(value: object) -> object:
    """'None', 'null', '' and friends become a real None."""
    if isinstance(value, str) and value.strip().lower() in NULL_LIKE:
        return None
    return value


def normalise_unit(value: object) -> object:
    """'Cases', 'cs.', 'CTN' all become 'case'. Unknown units pass through lower-cased."""
    value = empty_to_none(value)
    if isinstance(value, str):
        cleaned = value.strip().lower().rstrip(".")
        return UNIT_ALIASES.get(cleaned, cleaned)
    return value


OptionalText = Annotated[str | None, BeforeValidator(empty_to_none)]
OptionalEmail = Annotated[EmailStr | None, BeforeValidator(empty_to_none)]
Unit = Annotated[str | None, BeforeValidator(normalise_unit)]


class OrderLine(BaseModel):
    """One line of a customer's purchase order, as the customer wrote it."""

    model_config = ConfigDict(extra="forbid")

    line_number: int = Field(ge=1)
    customer_sku: OptionalText = None
    description: str = Field(min_length=1)
    quantity: Decimal = Field(gt=0)
    unit_of_measure: Unit = None
    unit_price: Decimal | None = Field(default=None, ge=0)


class ExtractedOrder(BaseModel):
    """What the LLM must produce from a PO document. Nothing else gets through."""

    model_config = ConfigDict(extra="forbid")

    customer_name: OptionalText = None
    customer_email: OptionalEmail = None
    customer_po_number: OptionalText = None
    order_date: date | None = None
    requested_delivery_date: date | None = None
    delivery_address: OptionalText = None
    currency: Literal["GBP", "EUR", "USD"] = "GBP"
    notes: OptionalText = None
    is_amendment: bool = False
    lines: list[OrderLine] = Field(min_length=1)

    @model_validator(mode="after")
    def check_cross_field_rules(self) -> "ExtractedOrder":
        numbers = [line.line_number for line in self.lines]
        if len(numbers) != len(set(numbers)):
            raise ValueError("line numbers must be unique")
        if (
            self.order_date
            and self.requested_delivery_date
            and self.requested_delivery_date < self.order_date
        ):
            raise ValueError("delivery date cannot be before order date")
        return self


class RawDocument(BaseModel):
    """A document as it arrived: text plus metadata, before any AI touches it."""

    model_config = ConfigDict(extra="forbid")

    document_id: str  # first 16 chars of the hash; short and readable
    content_sha256: str  # fingerprint of the raw bytes
    source_type: Literal["pdf", "email"]
    filename: str
    text: str
    page_count: int | None = None  # PDFs only
    needs_ocr: bool = False
    sender: str | None = None  # emails only
    subject: str | None = None
    received_at: datetime | None = None
    message_id: str | None = None
    attachments: list["RawDocument"] = Field(default_factory=list)
    # Only for scans: the original PDF (base64) so a vision model can read the page images.
    ocr_source_b64: str | None = Field(default=None, repr=False)
