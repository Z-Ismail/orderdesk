"""Simulated ERP + client tests. Runs the real FastAPI app in-process (no server needed)."""

from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from orderdesk.erp_client import ERPError, HttpERPClient
from orderdesk.erp_sim.app import DraftOrderIn, create_app

ROOT = Path(__file__).resolve().parents[1]
ORDER = DraftOrderIn(
    customer_id="C001", customer_po_number="GC-24117", lines=[{"sku": "HH-CUP-08W", "quantity": 2}]
)


@pytest.fixture
def http(tmp_path) -> TestClient:
    return TestClient(create_app(db_path=tmp_path / "erp.db", data_dir=ROOT / "sample_data"))


def client(http: TestClient, key: str = "dev-write-key", **kwargs) -> HttpERPClient:
    return HttpERPClient("http://testserver", key, http=http, backoff_seconds=0, **kwargs)


def test_product_and_stock_lookup(http):
    erp = client(http, "dev-read-key")
    assert erp.get_product("HH-CUP-08W").pack_size == 1000
    stock = erp.get_stock("HH-GLV-NIT-L")
    assert stock.total_available == sum(w.available for w in stock.warehouses)
    avonmouth = next(w for w in stock.warehouses if w.warehouse == "AVONMOUTH")
    assert avonmouth.on_hand == 0  # seeded out of stock there, for the agent to discover


def test_tier_pricing(http):
    erp = client(http, "dev-read-key")
    assert erp.get_price("C001", "HH-ROLL-BLU-2").unit_price == Decimal("19.80")  # standard
    gold = erp.get_price("C002", "HH-ROLL-BLU-2")  # gold tier: 12% off
    assert (gold.discount_pct, gold.unit_price) == (Decimal(12), Decimal("17.42"))


def test_discontinued_sku_cannot_be_priced(http):
    with pytest.raises(ERPError) as err:
        client(http).get_price("C001", "HH-CUP-08K")
    assert err.value.status_code == 422


def test_create_draft_order_computes_totals(http):
    order = client(http).create_draft_order(ORDER, idempotency_key="po-doc-0001")
    assert order.status == "draft"
    assert order.order_total == Decimal("84.00")  # 2 x 42.00


def test_same_idempotency_key_never_creates_a_second_order(http):
    erp = client(http)
    first = erp.create_draft_order(ORDER, idempotency_key="po-doc-0002")
    again = erp.create_draft_order(ORDER, idempotency_key="po-doc-0002")
    assert first.order_id == again.order_id
    assert len(http.get("/draft-orders", headers={"X-API-Key": "dev-read-key"}).json()) == 1


def test_read_key_cannot_create_orders(http):
    with pytest.raises(ERPError) as err:
        client(http, "dev-read-key").create_draft_order(ORDER, idempotency_key="po-doc-0003")
    assert err.value.status_code == 403


def test_invalid_key_is_rejected(http):
    with pytest.raises(ERPError) as err:
        client(http, "not-a-key").get_product("HH-CUP-08W")
    assert err.value.status_code == 401


def test_on_hold_customer_cannot_order(http):
    on_hold = DraftOrderIn(customer_id="C005", lines=[{"sku": "HH-TP-JUMBO", "quantity": 1}])
    with pytest.raises(ERPError) as err:
        client(http).create_draft_order(on_hold, idempotency_key="po-doc-0004")
    assert err.value.status_code == 409


def test_retries_temporary_outages_then_gives_up(http, monkeypatch):
    monkeypatch.setenv("ERP_FAULT_RATE", "1.0")  # every request fails with 503
    erp = client(http, max_attempts=3)
    with pytest.raises(ERPError, match="after 3 attempts"):
        erp.get_product("HH-CUP-08W")


def test_does_not_retry_client_errors(http):
    calls = []
    original = http.request

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    http.request = counting
    with pytest.raises(ERPError):
        client(http).get_product("NOT-A-SKU")  # 404: retrying won't help
    assert len(calls) == 1
