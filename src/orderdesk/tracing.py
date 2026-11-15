"""Store a record of every LLM call in Postgres.

Uses its own autocommit connection on purpose: if the pipeline rolls back after a failure, the
trace of the call that failed must survive, because that's exactly the one you'll want to see.
"""

import psycopg

from orderdesk.llm import CallRecord


class PostgresRecorder:
    def __init__(self, database_url: str) -> None:
        self.conn = psycopg.connect(database_url, autocommit=True)

    def __call__(self, r: CallRecord) -> None:
        self.conn.execute(
            "INSERT INTO llm_calls (purpose, document_id, model, input_tokens, output_tokens, "
            "cost_usd, latency_ms, stop_reason, error) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                r.purpose,
                r.document_id,
                r.model,
                r.input_tokens,
                r.output_tokens,
                r.cost_usd,
                r.latency_ms,
                r.stop_reason,
                r.error,
            ),
        )

    def close(self) -> None:
        self.conn.close()
