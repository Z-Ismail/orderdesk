"""Public demo: realistic data, zero LLM calls.

    uv run python -m orderdesk.demo export   # on your machine: save your real agent results
    uv run python -m orderdesk.demo seed     # on the server: (re)build the demo from scratch

`seed` loads the catalogue, runs the sample inbox through the real pipeline (with the golden
extractions standing in for Claude), and attaches the agent investigations you recorded with
`export`. Visitors see genuine agent output without the demo ever spending money, and nobody
can use the public site to run prompts on your API key.
"""

import json
import sys
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from orderdesk.models import ExtractedOrder, RawDocument

DEMO_DIR = Path("sample_data") / "demo"
RECORDED = DEMO_DIR / "investigations.json"
GOLDEN = Path("sample_data") / "golden"


def golden_extractor(doc: RawDocument) -> ExtractedOrder:
    return ExtractedOrder.model_validate_json(
        (GOLDEN / f"{Path(doc.filename).stem}.json").read_text()
    )


def export_investigations(conn: psycopg.Connection, path: Path = RECORDED) -> int:
    """Latest successful investigation per document, keyed by filename (ids differ per DB)."""
    rows = conn.execute(
        """
        SELECT DISTINCT ON (d.filename) d.filename, i.findings, i.transcript, i.steps,
               i.cost_usd, i.model
        FROM investigations i JOIN documents d USING (document_id)
        WHERE i.findings IS NOT NULL
        ORDER BY d.filename, i.created_at DESC
        """
    ).fetchall()
    records = [
        {
            "filename": f,
            "findings": findings,
            "transcript": transcript,
            "steps": steps,
            "cost_usd": float(cost),
            "model": model,
        }
        for f, findings, transcript, steps, cost, model in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, default=str) + "\n")
    return len(records)


def reset(conn: psycopg.Connection) -> None:
    """Delete all order data (not the catalogue or users). Demo only."""
    conn.execute("TRUNCATE documents, llm_calls, login_failures CASCADE")
    conn.execute("DELETE FROM customer_aliases WHERE source = 'learned'")
    conn.commit()


def seed(conn: psycopg.Connection, embedder, erp, thresholds, path: Path = RECORDED) -> dict:
    from orderdesk.pipeline import Pipeline

    reset(conn)
    pipeline = Pipeline(conn, golden_extractor, embedder, erp, thresholds)
    outcomes = pipeline.run_inbox(Path("sample_data") / "inbox")

    attached = 0
    if path.exists():
        ids = {o.filename: o.document_id for o in outcomes}
        for rec in json.loads(path.read_text()):
            document_id = ids.get(rec["filename"])
            if document_id is None:
                continue
            conn.execute(
                "INSERT INTO investigations (document_id, findings, transcript, steps, cost_usd, "
                "model) VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    document_id,
                    Jsonb(rec["findings"]),
                    Jsonb(rec["transcript"]),
                    rec["steps"],
                    rec["cost_usd"],
                    rec["model"],
                ),
            )
            conn.execute(
                "UPDATE documents SET status = 'investigated' WHERE document_id = %s "
                "AND status = 'needs_investigation'",
                (document_id,),
            )
            conn.execute(
                "INSERT INTO audit_log (document_id, actor, event, detail) "
                "VALUES (%s, 'agent', 'investigated', %s)",
                (
                    document_id,
                    Jsonb({"action": rec["findings"]["recommended_action"], "recorded": True}),
                ),
            )
            attached += 1
        conn.commit()
    return {"documents": len(outcomes), "investigations": attached}


def main(argv: list[str] | None = None) -> int:
    from orderdesk.auth import create_user
    from orderdesk.catalogue import load_catalogue
    from orderdesk.config import get_settings
    from orderdesk.db import connect, migrate
    from orderdesk.embeddings import FastEmbedEmbedder
    from orderdesk.erp_client import HttpERPClient
    from orderdesk.pipeline import Thresholds

    command = (argv or sys.argv[1:] or ["seed"])[0]
    settings = get_settings()
    with connect(settings.database_url) as conn:
        migrate(conn)
        if command == "export":
            print(f"Saved {export_investigations(conn)} investigations to {RECORDED}")
            return 0
        if command != "seed":
            print("usage: python -m orderdesk.demo [export|seed]")
            return 2
        if not settings.demo_mode:
            print("Refusing: seed wipes all orders. Set DEMO_MODE=true if this is a demo site.")
            return 1
        embedder = FastEmbedEmbedder(settings.embedding_model)
        load_catalogue(conn, embedder, Path("sample_data"))
        erp = HttpERPClient(settings.erp_base_url, settings.erp_read_key.get_secret_value())
        counts = seed(
            conn,
            embedder,
            erp,
            Thresholds(settings.match_min_similarity, settings.match_min_margin),
        )
        password = settings.demo_password.get_secret_value()
        if password:
            create_user(conn, settings.demo_email, password)
        print(
            f"Demo ready: {counts}; login {settings.demo_email}"
            if password
            else f"Demo ready: {counts}; set DEMO_PASSWORD to create the demo login"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
