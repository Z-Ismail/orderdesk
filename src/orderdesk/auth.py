"""Reviewer accounts and login sessions, built on the standard library only.

    uv run python -m orderdesk.auth add-user you@example.com           # prompts for a password
    uv run python -m orderdesk.auth add-user boss@example.com --admin

Choices, and why:
- Passwords are hashed with scrypt (deliberately slow and memory-hard), with a random salt per
  user. We store the hash, never the password.
- Sessions live in Postgres. The browser holds a random token; the database holds only its
  SHA-256. Logging out deletes the row, so a stolen cookie stops working immediately (a
  signed-cookie session can't be revoked like that).
- Each session has a CSRF token that every form must send back, so another website can't make
  a logged-in reviewer's browser approve an order.
- Five wrong passwords in 15 minutes locks that email out for the rest of the window.
"""

import argparse
import base64
import getpass
import hashlib
import hmac
import secrets
import sys
from datetime import UTC, datetime, timedelta

import psycopg
from pydantic import BaseModel

SCRYPT = {"n": 2**14, "r": 8, "p": 1}
MIN_PASSWORD_LENGTH = 12
MAX_FAILURES, FAILURE_WINDOW = 5, timedelta(minutes=15)


class AuthError(RuntimeError):
    """Deliberately vague message: never reveal whether the email or the password was wrong."""


class Session(BaseModel):
    email: str
    role: str
    csrf_token: str


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **SCRYPT)
    return f"scrypt${SCRYPT['n']}${SCRYPT['r']}${SCRYPT['p']}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, expected = stored.split("$")
        digest = hashlib.scrypt(
            password.encode(), salt=base64.b64decode(salt), dklen=32, n=int(n), r=int(r), p=int(p)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, base64.b64decode(expected))  # constant-time compare


# Checked when the email doesn't exist, so a missing user takes as long as a wrong password
# (otherwise response times reveal which emails have accounts).
_DUMMY_HASH = hash_password(secrets.token_hex(16))


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_user(
    conn: psycopg.Connection, email: str, password: str, role: str = "reviewer"
) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
    conn.execute(
        "INSERT INTO users (email, password_hash, role) VALUES (lower(%s), %s, %s) "
        "ON CONFLICT (email) DO UPDATE SET password_hash = EXCLUDED.password_hash, "
        "role = EXCLUDED.role, active = true",
        (email.strip(), hash_password(password), role),
    )
    conn.commit()


def login(conn: psycopg.Connection, email: str, password: str, hours: int = 12) -> str:
    """Returns a new session token for the cookie, or raises AuthError."""
    email = email.strip().lower()
    recent = conn.execute(
        "SELECT count(*) FROM login_failures WHERE email = %s AND at > now() - %s",
        (email, FAILURE_WINDOW),
    ).fetchone()[0]
    if recent >= MAX_FAILURES:
        raise AuthError("too many failed attempts; try again in 15 minutes")

    row = conn.execute(
        "SELECT password_hash FROM users WHERE email = %s AND active", (email,)
    ).fetchone()
    if not verify_password(password, row[0] if row else _DUMMY_HASH) or row is None:
        conn.execute("INSERT INTO login_failures (email) VALUES (%s)", (email,))
        conn.commit()
        raise AuthError("wrong email or password")

    token = secrets.token_urlsafe(32)
    conn.execute(
        "INSERT INTO sessions (token_hash, email, csrf_token, expires_at) VALUES (%s, %s, %s, %s)",
        (
            _token_hash(token),
            email,
            secrets.token_urlsafe(32),
            datetime.now(UTC) + timedelta(hours=hours),
        ),
    )
    conn.execute("DELETE FROM login_failures WHERE email = %s", (email,))
    conn.execute("DELETE FROM sessions WHERE expires_at < now()")  # tidy up while we're here
    conn.commit()
    return token


def session_for(conn: psycopg.Connection, token: str | None) -> Session | None:
    if not token:
        return None
    row = conn.execute(
        """
        SELECT s.email, u.role, s.csrf_token FROM sessions s JOIN users u USING (email)
        WHERE s.token_hash = %s AND s.expires_at > now() AND u.active
        """,
        (_token_hash(token),),
    ).fetchone()
    return Session(email=row[0], role=row[1], csrf_token=row[2]) if row else None


def logout(conn: psycopg.Connection, token: str | None) -> None:
    if token:
        conn.execute("DELETE FROM sessions WHERE token_hash = %s", (_token_hash(token),))
        conn.commit()


def csrf_ok(session: Session, submitted: str | None) -> bool:
    return bool(submitted) and hmac.compare_digest(session.csrf_token, submitted)


def main(argv: list[str] | None = None) -> int:
    from orderdesk.config import get_settings
    from orderdesk.db import connect, migrate

    parser = argparse.ArgumentParser(prog="orderdesk.auth")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add-user", help="create a reviewer, or reset their password")
    add.add_argument("email")
    add.add_argument("--admin", action="store_true")
    args = parser.parse_args(argv)

    password = getpass.getpass(f"Password for {args.email} (min {MIN_PASSWORD_LENGTH} chars): ")
    if password != getpass.getpass("Again: "):
        print("Passwords don't match.")
        return 1
    with connect(get_settings().database_url) as conn:
        migrate(conn)
        try:
            create_user(conn, args.email, password, "admin" if args.admin else "reviewer")
        except AuthError as exc:
            print(exc)
            return 1
    print(f"Saved {args.email}. Log in at http://localhost:8000/login")
    return 0


if __name__ == "__main__":
    sys.exit(main())
