"""Password hashing and sessions, at the unit level."""

import pytest

from orderdesk import auth
from orderdesk.db import connect


def test_password_hash_verifies_and_is_salted():
    first, second = auth.hash_password("a long passphrase"), auth.hash_password("a long passphrase")
    assert first != second  # different salt each time
    assert auth.verify_password("a long passphrase", first)
    assert not auth.verify_password("a long passphrasE", first)
    assert "a long passphrase" not in first


def test_garbage_hash_never_verifies():
    assert not auth.verify_password("anything", "not-a-real-hash")


@pytest.mark.db
def test_short_passwords_are_refused(loaded_db):
    with connect(loaded_db) as conn, pytest.raises(auth.AuthError, match="12"):
        auth.create_user(conn, "a@example.com", "short")


@pytest.mark.db
def test_session_round_trip_and_token_not_stored(loaded_db):
    with connect(loaded_db) as conn:
        conn.execute("TRUNCATE users CASCADE")
        conn.execute("DELETE FROM login_failures")
        auth.create_user(conn, "Rev@Example.com", "twelve chars ok")
        token = auth.login(conn, "rev@example.com", "twelve chars ok")
        session = auth.session_for(conn, token)
        assert session.email == "rev@example.com" and session.role == "reviewer"
        stored = conn.execute("SELECT token_hash FROM sessions").fetchone()[0]
        assert stored != token  # only the hash is in the database
        auth.logout(conn, token)
        assert auth.session_for(conn, token) is None


@pytest.mark.db
def test_unknown_email_gives_the_same_error_as_wrong_password(loaded_db):
    with connect(loaded_db) as conn:
        conn.execute("DELETE FROM login_failures")
        with pytest.raises(auth.AuthError, match="wrong email or password"):
            auth.login(conn, "nobody@example.com", "whatever password")
