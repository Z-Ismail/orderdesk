-- Pipeline state: one row per unique document, the proposed order, and an audit trail.

CREATE TABLE documents (
    document_id    text PRIMARY KEY,           -- first 16 chars of the content hash
    content_sha256 text NOT NULL UNIQUE,       -- the same bytes can never be stored twice
    filename       text NOT NULL,
    source_type    text NOT NULL,
    sender         text,
    subject        text,
    received_at    timestamptz,
    raw            jsonb NOT NULL,             -- the full RawDocument, for re-processing
    status         text NOT NULL DEFAULT 'received' CHECK (status IN (
                       'received', 'needs_investigation', 'ready_for_approval',
                       'approved', 'rejected', 'failed')),
    error          text,
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX documents_status_idx ON documents (status);

-- What the system proposes to order, with everything a reviewer needs to decide.
CREATE TABLE order_proposals (
    document_id text PRIMARY KEY REFERENCES documents (document_id) ON DELETE CASCADE,
    customer_id text REFERENCES customers (customer_id),
    extracted   jsonb NOT NULL,                -- ExtractedOrder exactly as the LLM produced it
    lines       jsonb NOT NULL,                -- each line with match candidates, price, stock
    issues      jsonb NOT NULL,                -- why it can't go straight to approval
    updated_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX order_proposals_customer_idx ON order_proposals (customer_id);

-- Append-only record of everything that happened to a document, and who did it.
CREATE TABLE audit_log (
    id          bigserial PRIMARY KEY,
    document_id text NOT NULL REFERENCES documents (document_id) ON DELETE CASCADE,
    at          timestamptz NOT NULL DEFAULT now(),
    actor       text NOT NULL,                 -- 'pipeline', 'agent', or a user's email
    event       text NOT NULL,
    detail      jsonb NOT NULL DEFAULT '{}'
);
CREATE INDEX audit_log_document_idx ON audit_log (document_id, at);
