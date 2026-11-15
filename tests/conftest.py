"""Shared fixtures. Database tests use a separate 'orderdesk_test' database, rebuilt each run,
so they never touch your development data. They skip cleanly if Postgres isn't running."""

import os
from pathlib import Path

import psycopg
import pytest

from orderdesk.catalogue import load_catalogue
from orderdesk.db import connect, migrate
from orderdesk.embeddings import HashingEmbedder

ROOT = Path(__file__).resolve().parents[1]
ADMIN_URL = os.environ.get(
    "TEST_DATABASE_ADMIN_URL", "postgresql://orderdesk:orderdesk@localhost:5433/orderdesk"
)
TEST_URL = ADMIN_URL.rsplit("/", 1)[0] + "/orderdesk_test"


@pytest.fixture(scope="session")
def db_url() -> str:
    try:
        with psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=3) as admin:
            admin.execute("DROP DATABASE IF EXISTS orderdesk_test WITH (FORCE)")
            admin.execute("CREATE DATABASE orderdesk_test")
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable: run `docker compose up -d` to include DB tests")
    return TEST_URL


@pytest.fixture(scope="session")
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture(scope="session")
def loaded_db(db_url, embedder) -> str:
    with connect(db_url) as conn:
        migrate(conn)
        load_catalogue(conn, embedder, ROOT / "sample_data")
    return db_url


@pytest.fixture
def conn(loaded_db):
    with connect(loaded_db) as connection:
        yield connection
        connection.rollback()
