"""Catalogue, customer and matching tests. Use the offline HashingEmbedder, so they check our
SQL and logic, not the quality of a real embedding model (that's evals/score_matching.py)."""

from pathlib import Path

import pytest

from orderdesk.catalogue import load_catalogue
from orderdesk.customers import identify_customer
from orderdesk.db import connect, migrate
from orderdesk.matching import hybrid_search, keyword_query, match_line
from orderdesk.models import OrderLine

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.db


def line(description: str, customer_sku: str | None = None) -> OrderLine:
    return OrderLine(line_number=1, description=description, quantity=1, customer_sku=customer_sku)


def test_migrations_are_idempotent(loaded_db):
    with connect(loaded_db) as conn:
        assert migrate(conn) == []  # nothing new the second time


def test_catalogue_load_counts_and_rerun_is_safe(loaded_db, embedder):
    with connect(loaded_db) as conn:
        counts = load_catalogue(conn, embedder, ROOT / "sample_data")
        assert counts == {"products": 40, "customers": 5, "aliases": 4}
        assert conn.execute("SELECT count(*) FROM products").fetchone()[0] == 40


@pytest.mark.parametrize(
    ("email", "expected"),
    [
        ("orders@thegullcafe.co.uk", "C001"),  # exact contact address
        ("someone.else@cliftongrand.co.uk", "C002"),  # known domain, different person
        ("PURCHASING@CliftonGrand.co.uk", "C002"),  # case doesn't matter
        ("procurement@quickbuy-trading.biz", None),  # unknown sender
        (None, None),
    ],
)
def test_identify_customer(conn, email, expected):
    customer = identify_customer(conn, email)
    assert (customer.customer_id if customer else None) == expected


def test_alias_by_customer_code(conn, embedder):
    result = match_line(conn, embedder, line("Blue roll 2ply", "CGH-PAP-11"), "C002")
    assert result.best.sku == "HH-ROLL-BLU-2"
    assert result.best.source == "alias"


def test_alias_by_customer_wording_is_case_insensitive(conn, embedder):
    result = match_line(conn, embedder, line("Blue Roll"), "C003")
    assert (result.best.sku, result.best.source) == ("HH-ROLL-BLU-2", "alias")


def test_alias_belongs_to_one_customer_only(conn, embedder):
    # 'blue roll' is C003's alias; for C001 it must go through search instead
    result = match_line(conn, embedder, line("blue roll"), "C001")
    assert result.best.source == "hybrid"


def test_exact_sku_match(conn, embedder):
    result = match_line(conn, embedder, line("8oz white paper hot cup", "HH-CUP-08W"), "C001")
    assert (result.best.sku, result.best.source) == ("HH-CUP-08W", "exact_sku")
    assert "HH-CUP-08W" not in [c.sku for c in result.candidates[1:]]  # no duplicate


def test_discontinued_products_are_never_returned(conn, embedder):
    skus = [c.sku for c in hybrid_search(conn, embedder, "8oz kraft paper hot cup", limit=10)]
    assert "HH-CUP-08K" not in skus


def test_hybrid_search_finds_the_obvious_product(conn, embedder):
    skus = [c.sku for c in hybrid_search(conn, embedder, "nitrile gloves large", limit=3)]
    assert "HH-GLV-NIT-L" in skus


def test_returns_at_most_limit_candidates(conn, embedder):
    assert len(match_line(conn, embedder, line("cups"), None, limit=5).candidates) <= 5


def test_keyword_query_is_safe_with_punctuation():
    assert keyword_query("Napkins 40cm (3-ply) & black!") == "napkins | 40cm | 3 | ply | black"
    assert keyword_query("!!!") == ""


def test_identify_customer_falls_back_to_name(conn):
    customer = identify_customer(conn, None, "exe valley care home")
    assert (customer.customer_id, customer.matched_by) == ("C005", "name")


def test_name_fallback_ignores_punctuation_and_company_suffixes(conn):
    customer = identify_customer(conn, None, "The Gull Café Ltd.")
    assert customer.customer_id == "C001"


def test_email_beats_name(conn):
    customer = identify_customer(conn, "orders@thegullcafe.co.uk", "Clifton Grand Hotel")
    assert (customer.customer_id, customer.matched_by) == ("C001", "email")
