"""Stage 3: match each order line to catalogue SKUs.

Order of attack, cheapest and most certain first:
  1. Customer alias  - this customer has told us before what their code or wording means.
  2. Exact SKU       - the customer quoted one of our own SKUs.
  3. Hybrid search   - keyword + vector search, combined with reciprocal rank fusion (RRF).
Discontinued products are never returned.
"""

import re
from typing import Literal

import psycopg
from pydantic import BaseModel

from orderdesk.embeddings import Embedder
from orderdesk.models import OrderLine

RRF_K = 60  # standard RRF constant: dampens the advantage of being ranked 1st vs 2nd
SEARCH_DEPTH = 20  # candidates taken from each of keyword and vector search before fusing


class Candidate(BaseModel):
    sku: str
    name: str
    unit_of_measure: str
    pack_size: int
    score: float  # alias/exact = 1.0; hybrid = RRF score (only meaningful as a ranking)
    similarity: float | None = None  # cosine similarity to the query, 0-1 (hybrid only)
    source: Literal["alias", "exact_sku", "hybrid"]


class MatchResult(BaseModel):
    line_number: int
    query: str
    candidates: list[Candidate]

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def margin(self) -> float | None:
        """How far ahead the best vector match is of the runner-up. Small = ambiguous."""
        sims = [c.similarity for c in self.candidates[:2] if c.similarity is not None]
        return sims[0] - sims[1] if len(sims) == 2 else None


PRODUCT_COLUMNS = "p.sku, p.name, p.unit_of_measure, p.pack_size"


def _product_candidate(row: tuple, source: str) -> Candidate:
    sku, name, unit, pack = row
    return Candidate(
        sku=sku, name=name, unit_of_measure=unit, pack_size=pack, score=1.0, source=source
    )


def _alias(conn: psycopg.Connection, customer_id: str, line: OrderLine) -> Candidate | None:
    row = conn.execute(
        f"""
        SELECT {PRODUCT_COLUMNS} FROM customer_aliases a JOIN products p ON p.sku = a.sku
        WHERE a.customer_id = %(cid)s AND p.active
          AND (lower(a.customer_sku) = lower(%(csku)s) OR lower(a.customer_text) = lower(%(text)s))
        ORDER BY (a.customer_sku IS NOT NULL) DESC  -- a code match beats a wording match
        LIMIT 1
        """,
        {"cid": customer_id, "csku": line.customer_sku or "", "text": line.description.strip()},
    ).fetchone()
    return _product_candidate(row, "alias") if row else None


def _exact_sku(conn: psycopg.Connection, customer_sku: str) -> Candidate | None:
    row = conn.execute(
        f"SELECT {PRODUCT_COLUMNS} FROM products p WHERE lower(p.sku) = lower(%s) AND p.active",
        (customer_sku,),
    ).fetchone()
    return _product_candidate(row, "exact_sku") if row else None


def keyword_query(text: str) -> str:
    """'Black sacks 90ltr heavy' -> 'black | sacks | 90ltr | heavy' (any word may match)."""
    return " | ".join(re.findall(r"[a-z0-9]+", text.lower()))


HYBRID_SQL = f"""
WITH q AS (
    SELECT %(qv)s::vector AS v, to_tsquery('english', %(tsq)s) AS ts
), semantic AS (
    SELECT sku, 1 - (embedding <=> q.v) AS similarity,
           row_number() OVER (ORDER BY embedding <=> q.v) AS rank
    FROM products, q WHERE active
    ORDER BY embedding <=> q.v LIMIT {SEARCH_DEPTH}
), keyword AS (
    SELECT sku, row_number() OVER (ORDER BY ts_rank_cd(search_tsv, q.ts) DESC) AS rank
    FROM products, q WHERE active AND search_tsv @@ q.ts
    ORDER BY ts_rank_cd(search_tsv, q.ts) DESC LIMIT {SEARCH_DEPTH}
)
SELECT {PRODUCT_COLUMNS}, s.similarity,
       coalesce(1.0 / ({RRF_K} + s.rank), 0) + coalesce(1.0 / ({RRF_K} + k.rank), 0) AS score
FROM semantic s
FULL OUTER JOIN keyword k USING (sku)
JOIN products p USING (sku)
ORDER BY score DESC, p.sku
LIMIT %(limit)s
"""


def hybrid_search(
    conn: psycopg.Connection, embedder: Embedder, text: str, limit: int = 5
) -> list[Candidate]:
    rows = conn.execute(
        HYBRID_SQL,
        {"qv": embedder.embed_query(text), "tsq": keyword_query(text), "limit": limit},
    ).fetchall()
    return [
        Candidate(
            sku=sku,
            name=name,
            unit_of_measure=unit,
            pack_size=pack,
            similarity=float(sim) if sim is not None else None,
            score=float(score),
            source="hybrid",
        )
        for sku, name, unit, pack, sim, score in rows
    ]


def match_line(
    conn: psycopg.Connection,
    embedder: Embedder,
    line: OrderLine,
    customer_id: str | None,
    limit: int = 5,
) -> MatchResult:
    certain = None
    if customer_id:
        certain = _alias(conn, customer_id, line)
    if certain is None and line.customer_sku:
        certain = _exact_sku(conn, line.customer_sku)

    searched = hybrid_search(conn, embedder, line.description, limit=limit)
    if certain:
        # Keep search results as alternatives for the reviewer, minus the duplicate.
        searched = [c for c in searched if c.sku != certain.sku][: limit - 1]
        candidates = [certain, *searched]
    else:
        candidates = searched
    return MatchResult(line_number=line.line_number, query=line.description, candidates=candidates)
