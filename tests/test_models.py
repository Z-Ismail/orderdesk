import pytest
from decimal import Decimal
from pydantic import ValidationError
from orderdesk.models import OrderLine

GOOD = {"line_number": 1, "customer_sku": None, "description": "8oz cups",
        "quantity": 2, "unit_of_measure": "case", "unit_price": "42.00"}

def test_valid_line():
    line = OrderLine.model_validate(GOOD)
    assert line.unit_price == Decimal("42.00")

@pytest.mark.parametrize("change", [
    {"quantity": 0}, {"quantity": -3}, {"description": ""},
    {"line_number": 0}, {"unit_price": "-1"}, {"discount_pct": 90},
])
def test_bad_lines_rejected(change):
    with pytest.raises(ValidationError):
        OrderLine.model_validate({**GOOD, **change})