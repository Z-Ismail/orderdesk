from decimal import Decimal

import pytest
from pydantic import ValidationError

from orderdesk.models import ExtractedOrder, OrderLine

GOOD = {
    "line_number": 1,
    "customer_sku": None,
    "description": "8oz cups",
    "quantity": 2,
    "unit_of_measure": "case",
    "unit_price": "42.00",
}


def test_valid_line():
    line = OrderLine.model_validate(GOOD)
    assert line.unit_price == Decimal("42.00")


@pytest.mark.parametrize(
    "change",
    [
        {"quantity": 0},
        {"quantity": -3},
        {"description": ""},
        {"line_number": 0},
        {"unit_price": "-1"},
        {"discount_pct": 90},
    ],
)
def test_bad_lines_rejected(change):
    with pytest.raises(ValidationError):
        OrderLine.model_validate({**GOOD, **change})


ORDER = {
    "customer_name": "The Gull Cafe",
    "customer_email": "orders@thegullcafe.co.uk",
    "order_date": "2026-11-10",
    "requested_delivery_date": "2026-11-13",
    "lines": [GOOD],
}


def test_valid_order():
    order = ExtractedOrder.model_validate(ORDER)
    assert order.currency == "GBP"  # default applied
    assert order.order_date.year == 2026  # string became a date


@pytest.mark.parametrize(
    "change",
    [
        {"lines": []},
        {"lines": [GOOD, GOOD]},  # duplicate line numbers
        {"requested_delivery_date": "2026-11-01"},
        {"currency": "BTC"},
        {"customer_email": "not-an-email"},
        {"discount_pct": 90},
    ],
)
def test_bad_orders_rejected(change):
    with pytest.raises(ValidationError):
        ExtractedOrder.model_validate({**ORDER, **change})


def test_schema_for_llm():
    schema = ExtractedOrder.model_json_schema()
    assert "lines" in schema["properties"]


@pytest.mark.parametrize("raw", ["None", "null", "", "  N/A "])
def test_null_like_text_becomes_none(raw):
    order = ExtractedOrder.model_validate({**ORDER, "notes": raw, "customer_po_number": raw})
    assert order.notes is None
    assert order.customer_po_number is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("cases", "case"),
        ("Cases", "case"),
        ("CTN", "case"),
        ("boxes", "box"),
        ("packs", "pack"),
        ("ea.", "each"),
        ("pack of 6", "pack of 6"),
        (None, None),
    ],
)
def test_units_are_normalised(raw, expected):
    line = OrderLine.model_validate({**GOOD, "unit_of_measure": raw})
    assert line.unit_of_measure == expected
