"""Work out which customer sent a document."""

import re
from typing import Literal

import psycopg
from pydantic import BaseModel


class Customer(BaseModel):
    customer_id: str
    name: str
    email_domain: str
    contact_email: str
    price_tier: str
    account_status: str
    default_delivery_address: str | None = None
    matched_by: Literal["email", "domain", "name"] | None = None


FIELDS = [
    "customer_id",
    "name",
    "email_domain",
    "contact_email",
    "price_tier",
    "account_status",
    "default_delivery_address",
]
COLUMNS = ", ".join(FIELDS)


def normalise_name(name: str) -> str:
    """'The Gull Café Ltd.' and 'gull cafe' compare equal."""
    name = name.lower().replace("&", " and ").replace("é", "e")
    words = re.findall(r"[a-z0-9]+", name)
    return " ".join(w for w in words if w not in {"the", "ltd", "limited", "plc", "llp"})


def identify_customer(
    conn: psycopg.Connection, email: str | None, name: str | None = None
) -> Customer | None:
    """Strongest evidence first: exact contact address, then email domain, then company name.

    A name match is the weakest (anyone can type a name), so the pipeline flags it for review.
    """
    if email and "@" in email:
        email = email.strip().lower()
        for matched_by, sql, value in (
            ("email", "lower(contact_email) = %s", email),
            ("domain", "lower(email_domain) = %s", email.rsplit("@", 1)[1]),
        ):
            row = conn.execute(f"SELECT {COLUMNS} FROM customers WHERE {sql}", (value,)).fetchone()
            if row:
                return Customer(**dict(zip(FIELDS, row, strict=True)), matched_by=matched_by)
    if name and normalise_name(name):
        wanted = normalise_name(name)
        for row in conn.execute(f"SELECT {COLUMNS} FROM customers"):
            if normalise_name(row[1]) == wanted:
                return Customer(**dict(zip(FIELDS, row, strict=True)), matched_by="name")
    return None
