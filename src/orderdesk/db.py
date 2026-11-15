"""Postgres connection and schema migrations.

Migrations are plain SQL files in orderdesk/sql/, applied in filename order and recorded in
schema_migrations, so running migrate() again only applies what is new.
"""

from importlib import resources

import psycopg
from pgvector.psycopg import register_vector


def connect(database_url: str) -> psycopg.Connection:
    conn = psycopg.connect(database_url)
    # The extension must exist before pgvector can register its type adapters.
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def migrate(conn: psycopg.Connection) -> list[str]:
    """Apply any SQL migrations not yet recorded. Returns the names applied this time."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
    )
    applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
    newly_applied = []
    sql_files = sorted(
        (f for f in resources.files("orderdesk.sql").iterdir() if f.name.endswith(".sql")),
        key=lambda f: f.name,
    )
    for sql_file in sql_files:
        if sql_file.name in applied:
            continue
        conn.execute(sql_file.read_text())  # each file runs inside this transaction
        conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (sql_file.name,))
        newly_applied.append(sql_file.name)
    conn.commit()
    return newly_applied
