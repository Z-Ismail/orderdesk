-- The investigator agent's findings, and the full record of how it reached them.

ALTER TABLE documents DROP CONSTRAINT documents_status_check;
ALTER TABLE documents ADD CONSTRAINT documents_status_check CHECK (status IN (
    'received', 'needs_investigation', 'investigated', 'ready_for_approval',
    'approved', 'rejected', 'failed'));

CREATE TABLE investigations (
    id          bigserial PRIMARY KEY,
    document_id text NOT NULL REFERENCES documents (document_id) ON DELETE CASCADE,
    findings    jsonb,                 -- null if the investigation failed
    transcript  jsonb NOT NULL,        -- every tool call and result, for review and debugging
    steps       integer NOT NULL,
    cost_usd    numeric(10, 6) NOT NULL,
    model       text NOT NULL,
    error       text,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX investigations_document_idx ON investigations (document_id, created_at);
