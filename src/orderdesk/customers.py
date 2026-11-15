"""Work out which customer sent a document."""

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


COLUMNS = (
    "customer_id, name, email_domain, contact_email, price_tier, account_status, "
    "default_delivery_address"
)


def identify_customer(conn: psycopg.Connection, email: str | None) -> Customer | None:
    """Exact contact address first, then the email domain. Unknown senders return None."""
    if not email or "@" not in email:
        return None
    email = email.strip().lower()
    domain = email.rsplit("@", 1)[1]
    row = conn.execute(
        f"SELECT {COLUMNS} FROM customers WHERE lower(contact_email) = %s "
        f"UNION ALL SELECT {COLUMNS} FROM customers WHERE lower(email_domain) = %s LIMIT 1",
        (email, domain),
    ).fetchone()
    if row is None:
        return None
    return Customer(**dict(zip(COLUMNS.split(", "), row, strict=True)))
