-- Human decisions, and the corrections reviewers make (the most valuable data in the system).

CREATE TABLE decisions (
    document_id    text PRIMARY KEY REFERENCES documents (document_id) ON DELETE CASCADE,
    decision       text NOT NULL CHECK (decision IN ('approved', 'rejected')),
    decided_by     text NOT NULL,
    reason         text,
    suggested      jsonb,                -- what the system proposed
    final_lines    jsonb,                -- what the human actually approved
    corrections    jsonb NOT NULL DEFAULT '[]',  -- the difference between the two
    erp_order_id   text,
    order_total    numeric(12, 2),
    decided_at     timestamptz NOT NULL DEFAULT now()
);

-- Aliases now come from two places: the catalogue file, and reviewers' corrections.
ALTER TABLE customer_aliases
    ADD COLUMN source text NOT NULL DEFAULT 'catalogue' CHECK (source IN ('catalogue', 'learned'));
