"""Simulated ERP: products, stock, customer prices and draft sales orders.

    uv run uvicorn orderdesk.erp_sim.app:create_app --factory --port 8100
    # interactive API docs: http://localhost:8100/docs

Behaves like a real ERP API in the ways that matter to us:
  - API keys with scopes: a read key can look things up; only the write key can create orders.
  - Idempotent order creation: the same Idempotency-Key always returns the same order.
  - Optional fault injection (ERP_FAULT_RATE, ERP_LATENCY_MS) to practise retries and timeouts.
Data lives in a SQLite file seeded from sample_data/ on first start.
"""

import asyncio
import csv
import os
import random
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, Header, HTTPException, Request, Security
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# --- API models (this is the ERP's public contract) ---------------------------------------


class Product(BaseModel):
    sku: str
    name: str
    unit_of_measure: str
    pack_size: int
    list_price: Decimal
    active: bool


class WarehouseStock(BaseModel):
    warehouse: str
    on_hand: int
    allocated: int
    available: int


class StockLevel(BaseModel):
    sku: str
    total_available: int
    warehouses: list[WarehouseStock]


class Price(BaseModel):
    sku: str
    customer_id: str
    price_tier: str
    list_price: Decimal
    discount_pct: Decimal
    unit_price: Decimal
    currency: Literal["GBP"] = "GBP"


class DraftOrderLineIn(BaseModel):
    sku: str
    quantity: int = Field(gt=0)


class DraftOrderIn(BaseModel):
    customer_id: str
    customer_po_number: str | None = None
    requested_delivery_date: str | None = None
    delivery_address: str | None = None
    notes: str | None = None
    lines: list[DraftOrderLineIn] = Field(min_length=1)


class DraftOrderLine(BaseModel):
    sku: str
    quantity: int
    unit_price: Decimal
    line_total: Decimal


class DraftOrder(BaseModel):
    order_id: str
    status: Literal["draft"] = "draft"
    customer_id: str
    customer_po_number: str | None
    requested_delivery_date: str | None
    delivery_address: str | None
    notes: str | None
    lines: list[DraftOrderLine]
    order_total: Decimal
    created_at: str


# --- storage ---------------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (sku TEXT PRIMARY KEY, name TEXT, unit_of_measure TEXT,
    pack_size INTEGER, list_price TEXT, active INTEGER);
CREATE TABLE IF NOT EXISTS stock (sku TEXT, warehouse TEXT, on_hand INTEGER, allocated INTEGER,
    PRIMARY KEY (sku, warehouse));
CREATE TABLE IF NOT EXISTS customers (customer_id TEXT PRIMARY KEY, price_tier TEXT,
    account_status TEXT);
CREATE TABLE IF NOT EXISTS price_tiers (price_tier TEXT PRIMARY KEY, discount_pct TEXT);
CREATE TABLE IF NOT EXISTS draft_orders (order_id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE,
    customer_id TEXT, body TEXT, created_at TEXT);
"""


def _seed(db: sqlite3.Connection, data_dir: Path) -> None:
    def rows(rel: str) -> list[dict[str, str]]:
        with (data_dir / rel).open(newline="") as handle:
            return list(csv.DictReader(handle))

    for p in rows("catalogue/products.csv"):
        db.execute(
            "INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?, ?, ?)",
            (
                p["sku"],
                p["name"],
                p["unit_of_measure"],
                int(p["pack_size"]),
                p["list_price_gbp"],
                p["active"].lower() == "true",
            ),
        )
    for s in rows("erp/inventory.csv"):
        db.execute(
            "INSERT OR REPLACE INTO stock VALUES (?, ?, ?, ?)",
            (s["sku"], s["warehouse"], int(s["on_hand"]), int(s["allocated"])),
        )
    for c in rows("customers/customers.csv"):
        db.execute(
            "INSERT OR REPLACE INTO customers VALUES (?, ?, ?)",
            (c["customer_id"], c["price_tier"], c["account_status"]),
        )
    for t in rows("erp/price_tiers.csv"):
        db.execute(
            "INSERT OR REPLACE INTO price_tiers VALUES (?, ?)", (t["price_tier"], t["discount_pct"])
        )
    db.commit()


class Store:
    def __init__(self, db_path: Path, data_dir: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript(SCHEMA)
            if db.execute("SELECT count(*) FROM products").fetchone()[0] == 0:
                _seed(db, data_dir)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.db_path)
        try:
            yield db
        finally:
            db.close()


# --- security: API keys with scopes ------------------------------------------------------------

Scope = Literal["read", "write"]


def _keys() -> dict[str, set[Scope]]:
    return {
        os.environ.get("ERP_READ_KEY", "dev-read-key"): {"read"},
        os.environ.get("ERP_WRITE_KEY", "dev-write-key"): {"read", "write"},
    }


# Declared as a security scheme so /docs shows an "Authorize" button.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require(scope: Scope):
    def check(api_key: Annotated[str | None, Security(api_key_header)] = None) -> None:
        scopes = _keys().get(api_key or "")
        if scopes is None:
            raise HTTPException(401, "missing or invalid API key")
        if scope not in scopes:
            raise HTTPException(403, f"this key lacks the '{scope}' scope")

    return Security(check)


# --- app -------------------------------------------------------------------------------------------


def money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def create_app(db_path: Path | None = None, data_dir: Path | None = None) -> FastAPI:
    store = Store(
        db_path or Path(os.environ.get("ERP_DB_PATH", ".erp/erp.db")),
        data_dir or Path(os.environ.get("ERP_DATA_DIR", "sample_data")),
    )
    app = FastAPI(title="Simulated ERP", version="1.0")

    @app.middleware("http")
    async def inject_faults(request: Request, call_next):
        """Chaos switch: slow or failing ERP on demand. Off unless the env vars are set."""
        latency_ms = int(os.environ.get("ERP_LATENCY_MS", "0"))
        fault_rate = float(os.environ.get("ERP_FAULT_RATE", "0"))
        if latency_ms:
            await asyncio.sleep(latency_ms / 1000)  # async sleep: don't block other requests
        if request.url.path != "/health" and random.random() < fault_rate:
            return JSONResponse({"detail": "simulated ERP outage"}, status_code=503)
        return await call_next(request)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/products/{sku}", dependencies=[require("read")])
    def get_product(sku: str) -> Product:
        with store.connection() as db:
            row = db.execute("SELECT * FROM products WHERE sku = ?", (sku,)).fetchone()
        if row is None:
            raise HTTPException(404, f"unknown SKU {sku}")
        return Product(
            sku=row[0],
            name=row[1],
            unit_of_measure=row[2],
            pack_size=row[3],
            list_price=Decimal(row[4]),
            active=bool(row[5]),
        )

    @app.get("/stock/{sku}", dependencies=[require("read")])
    def get_stock(sku: str) -> StockLevel:
        with store.connection() as db:
            rows = db.execute(
                "SELECT warehouse, on_hand, allocated FROM stock WHERE sku = ? ORDER BY warehouse",
                (sku,),
            ).fetchall()
        if not rows:
            raise HTTPException(404, f"unknown SKU {sku}")
        warehouses = [
            WarehouseStock(warehouse=w, on_hand=o, allocated=a, available=max(o - a, 0))
            for w, o, a in rows
        ]
        return StockLevel(
            sku=sku, warehouses=warehouses, total_available=sum(w.available for w in warehouses)
        )

    def _price(db: sqlite3.Connection, customer_id: str, sku: str) -> Price:
        customer = db.execute(
            "SELECT c.price_tier, t.discount_pct FROM customers c "
            "JOIN price_tiers t USING (price_tier) WHERE c.customer_id = ?",
            (customer_id,),
        ).fetchone()
        if customer is None:
            raise HTTPException(404, f"unknown customer {customer_id}")
        product = db.execute(
            "SELECT list_price, active FROM products WHERE sku = ?", (sku,)
        ).fetchone()
        if product is None:
            raise HTTPException(404, f"unknown SKU {sku}")
        if not product[1]:
            raise HTTPException(422, f"SKU {sku} is discontinued")
        tier, discount = customer[0], Decimal(customer[1])
        list_price = Decimal(product[0])
        return Price(
            sku=sku,
            customer_id=customer_id,
            price_tier=tier,
            list_price=list_price,
            discount_pct=discount,
            unit_price=money(list_price * (1 - discount / 100)),
        )

    @app.get("/prices", dependencies=[require("read")])
    def get_price(customer_id: str, sku: str) -> Price:
        with store.connection() as db:
            return _price(db, customer_id, sku)

    @app.post("/draft-orders", status_code=201, dependencies=[require("write")])
    def create_draft_order(
        order: DraftOrderIn, idempotency_key: Annotated[str, Header(min_length=8)]
    ) -> DraftOrder:
        with store.connection() as db:
            existing = db.execute(
                "SELECT body FROM draft_orders WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            if existing:
                return DraftOrder.model_validate_json(existing[0])  # replay, not a new order
            status = db.execute(
                "SELECT account_status FROM customers WHERE customer_id = ?", (order.customer_id,)
            ).fetchone()
            if status is None:
                raise HTTPException(404, f"unknown customer {order.customer_id}")
            if status[0] != "active":
                raise HTTPException(409, f"customer account is {status[0]}")
            lines = []
            for line in order.lines:
                price = _price(db, order.customer_id, line.sku)
                lines.append(
                    DraftOrderLine(
                        sku=line.sku,
                        quantity=line.quantity,
                        unit_price=price.unit_price,
                        line_total=money(price.unit_price * line.quantity),
                    )
                )
            draft = DraftOrder(
                order_id=f"SO-{uuid.uuid4().hex[:8].upper()}",
                customer_id=order.customer_id,
                customer_po_number=order.customer_po_number,
                requested_delivery_date=order.requested_delivery_date,
                delivery_address=order.delivery_address,
                notes=order.notes,
                lines=lines,
                order_total=money(sum((ln.line_total for ln in lines), Decimal(0))),
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            db.execute(
                "INSERT INTO draft_orders VALUES (?, ?, ?, ?, ?)",
                (
                    draft.order_id,
                    idempotency_key,
                    draft.customer_id,
                    draft.model_dump_json(),
                    draft.created_at,
                ),
            )
            db.commit()
            return draft

    @app.get("/draft-orders/{order_id}", dependencies=[require("read")])
    def get_draft_order(order_id: str) -> DraftOrder:
        with store.connection() as db:
            row = db.execute(
                "SELECT body FROM draft_orders WHERE order_id = ?", (order_id,)
            ).fetchone()
        if row is None:
            raise HTTPException(404, f"unknown order {order_id}")
        return DraftOrder.model_validate_json(row[0])

    @app.get("/draft-orders", dependencies=[require("read")])
    def list_draft_orders(customer_id: str | None = None) -> list[DraftOrder]:
        with store.connection() as db:
            rows = db.execute(
                "SELECT body FROM draft_orders WHERE ?1 IS NULL OR customer_id = ?1 "
                "ORDER BY created_at",
                (customer_id,),
            ).fetchall()
        return [DraftOrder.model_validate_json(r[0]) for r in rows]

    return app
