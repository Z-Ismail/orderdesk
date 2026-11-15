"""Load the product catalogue, customers and aliases into Postgres, with embeddings.

uv run python -m orderdesk.catalogue          # migrate and load sample_data/
"""

import csv
import sys
from pathlib import Path

import psycopg

from orderdesk.embeddings import Embedder


def product_search_text(row: dict[str, str]) -> str:
    """The text we embed for each product. Name first: it carries the most meaning."""
    return f"{row['name']}. {row['description']}. Category: {row['category']}"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_catalogue(conn: psycopg.Connection, embedder: Embedder, data_dir: Path) -> dict[str, int]:
    """Upsert everything, so running it twice is safe. Returns row counts."""
    products = _read(data_dir / "catalogue" / "products.csv")
    vectors = embedder.embed_documents([product_search_text(p) for p in products])
    for product, vector in zip(products, vectors, strict=True):
        conn.execute(
            """
            INSERT INTO products (sku, name, description, category, unit_of_measure,
                                  pack_size, list_price, active, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (sku) DO UPDATE SET
                name = EXCLUDED.name, description = EXCLUDED.description,
                category = EXCLUDED.category, unit_of_measure = EXCLUDED.unit_of_measure,
                pack_size = EXCLUDED.pack_size, list_price = EXCLUDED.list_price,
                active = EXCLUDED.active, embedding = EXCLUDED.embedding
            """,
            (
                product["sku"],
                product["name"],
                product["description"],
                product["category"],
                product["unit_of_measure"],
                int(product["pack_size"]),
                product["list_price_gbp"],
                product["active"].lower() == "true",
                vector,
            ),
        )

    customers = _read(data_dir / "customers" / "customers.csv")
    for c in customers:
        conn.execute(
            """
            INSERT INTO customers (customer_id, name, email_domain, contact_email, price_tier,
                                   account_status, default_delivery_address)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (customer_id) DO UPDATE SET
                name = EXCLUDED.name, email_domain = EXCLUDED.email_domain,
                contact_email = EXCLUDED.contact_email, price_tier = EXCLUDED.price_tier,
                account_status = EXCLUDED.account_status,
                default_delivery_address = EXCLUDED.default_delivery_address
            """,
            (
                c["customer_id"],
                c["name"],
                c["email_domain"],
                c["contact_email"],
                c["price_tier"],
                c["account_status"],
                c["default_delivery_address"] or None,
            ),
        )

    # The file fully describes the catalogue aliases, so replace those wholesale. Aliases that
    # reviewers taught the system ('learned') survive a reload, unless the file now says otherwise.
    aliases = _read(data_dir / "customers" / "customer_aliases.csv")
    conn.execute("DELETE FROM customer_aliases WHERE source = 'catalogue'")
    for a in aliases:
        conn.execute(
            "DELETE FROM customer_aliases WHERE customer_id = %s AND "
            "(lower(customer_sku) = lower(%s) OR lower(customer_text) = lower(%s))",
            (a["customer_id"], a["customer_sku"] or "", a["customer_text"] or ""),
        )
        conn.execute(
            "INSERT INTO customer_aliases (customer_id, customer_sku, customer_text, sku) "
            "VALUES (%s, %s, %s, %s)",
            (a["customer_id"], a["customer_sku"] or None, a["customer_text"] or None, a["sku"]),
        )
    conn.commit()
    return {"products": len(products), "customers": len(customers), "aliases": len(aliases)}


def main() -> None:
    from orderdesk.config import get_settings
    from orderdesk.db import connect, migrate
    from orderdesk.embeddings import FastEmbedEmbedder

    settings = get_settings()
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample_data")
    with connect(settings.database_url) as conn:
        applied = migrate(conn)
        print(f"migrations applied: {applied or 'none (already up to date)'}")
        print("loading embedding model (first run downloads ~130 MB)...")
        counts = load_catalogue(conn, FastEmbedEmbedder(settings.embedding_model), data_dir)
    print(f"loaded: {counts}")


if __name__ == "__main__":
    main()
