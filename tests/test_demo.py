"""Demo mode: seeding, recorded investigations, the reset button, and its safety guard."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from orderdesk import demo
from orderdesk.auth import create_user
from orderdesk.db import connect
from orderdesk.erp_client import HttpERPClient
from orderdesk.erp_sim.app import create_app as create_erp
from orderdesk.pipeline import Thresholds
from orderdesk.web.app import create_app

pytestmark = pytest.mark.db
ROOT = Path(__file__).resolve().parents[1]
FINDINGS = {
    "recommended_action": "reject",
    "customer_id": None,
    "summary": "Fraud attempt.",
    "risk_flags": ["possible_prompt_injection"],
    "lines": [],
}


@pytest.fixture
def erp(tmp_path):
    http = TestClient(create_erp(db_path=tmp_path / "erp.db", data_dir=ROOT / "sample_data"))
    return HttpERPClient("http://testserver", "dev-write-key", http=http, backoff_seconds=0)


@pytest.fixture
def recorded(tmp_path):
    path = tmp_path / "investigations.json"
    path.write_text(
        json.dumps(
            [
                {
                    "filename": "email_008_unknown_sender.eml",
                    "findings": FINDINGS,
                    "transcript": [],
                    "steps": 2,
                    "cost_usd": 0.03,
                    "model": "claude-sonnet-5-5",
                }
            ]
        )
    )
    return path


@pytest.fixture(autouse=True)
def run_from_repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)  # the demo reads sample_data/ relative to the working directory


def test_seed_runs_the_inbox_and_attaches_recorded_investigations(
    loaded_db, embedder, erp, recorded
):
    with connect(loaded_db) as conn:
        counts = demo.seed(conn, embedder, erp, Thresholds(0.0, 0.0), path=recorded)
        assert counts == {"documents": 7, "investigations": 1}
        status = conn.execute(
            "SELECT status FROM documents WHERE filename LIKE 'email_008%'"
        ).fetchone()[0]
        assert status == "investigated"


def test_seed_twice_gives_the_same_result(loaded_db, embedder, erp, recorded):
    with connect(loaded_db) as conn:
        demo.seed(conn, embedder, erp, Thresholds(0.0, 0.0), path=recorded)
        conn.execute("UPDATE documents SET status = 'approved'")
        conn.commit()
        demo.seed(conn, embedder, erp, Thresholds(0.0, 0.0), path=recorded)
        assert (
            conn.execute("SELECT count(*) FROM documents WHERE status = 'approved'").fetchone()[0]
            == 0
        )


def test_export_then_seed_round_trip(loaded_db, embedder, erp, recorded, tmp_path):
    with connect(loaded_db) as conn:
        demo.seed(conn, embedder, erp, Thresholds(0.0, 0.0), path=recorded)
        out = tmp_path / "exported.json"
        assert demo.export_investigations(conn, out) == 1
        assert json.loads(out.read_text())[0]["findings"]["summary"] == "Fraud attempt."


def test_seed_refuses_unless_demo_mode(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "false")
    assert demo.main(["seed"]) == 1


def _client(loaded_db, erp, demo_settings, reset):
    with connect(loaded_db) as conn:
        conn.execute("TRUNCATE users CASCADE")
        conn.execute("DELETE FROM login_failures")
        create_user(conn, "demo@orderdesk.app", "public demo password")
    app = create_app(
        connect=lambda: connect(loaded_db),
        erp=erp,
        secure_cookies=False,
        session_hours=1,
        demo=demo_settings,
        demo_reset=reset,
    )
    return TestClient(app)


def test_login_page_shows_demo_credentials_only_in_demo_mode(loaded_db, erp):
    on = _client(
        loaded_db,
        erp,
        {"email": "demo@orderdesk.app", "password": "public demo password"},
        lambda c: None,
    )
    assert "public demo password" in on.get("/login").text
    off = _client(loaded_db, erp, None, None)
    assert "Public demo" not in off.get("/login").text


def test_reset_button_reseeds(loaded_db, erp):
    calls = []
    client = _client(
        loaded_db,
        erp,
        {"email": "demo@orderdesk.app", "password": "public demo password"},
        calls.append,
    )
    client.post("/login", data={"email": "demo@orderdesk.app", "password": "public demo password"})
    import re

    csrf = re.search(r'name="csrf" value="([^"]+)"', client.get("/").text).group(1)
    assert client.post("/demo/reset", data={"csrf": "wrong"}).status_code == 403
    assert client.post("/demo/reset", data={"csrf": csrf}).status_code == 200
    assert len(calls) == 1


def test_reset_does_not_exist_outside_demo_mode(loaded_db, erp):
    client = _client(loaded_db, erp, None, None)
    client.post("/login", data={"email": "demo@orderdesk.app", "password": "public demo password"})
    assert client.post("/demo/reset", data={}).status_code == 404
