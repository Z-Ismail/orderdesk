-- Reviewer accounts, login sessions, and a trace of every LLM call.

CREATE TABLE users (
    email         text PRIMARY KEY,
    password_hash text NOT NULL,           -- scrypt; never the password itself
    role          text NOT NULL DEFAULT 'reviewer' CHECK (role IN ('reviewer', 'admin')),
    active        boolean NOT NULL DEFAULT true,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE sessions (
    token_hash  text PRIMARY KEY,          -- sha256 of the cookie value: a DB leak can't log anyone in
    email       text NOT NULL REFERENCES users (email) ON DELETE CASCADE,
    csrf_token  text NOT NULL,             -- must accompany every form POST in this session
    created_at  timestamptz NOT NULL DEFAULT now(),
    expires_at  timestamptz NOT NULL
);
CREATE INDEX sessions_email_idx ON sessions (email);

-- Failed logins, for throttling password guessing.
CREATE TABLE login_failures (
    id     bigserial PRIMARY KEY,
    email  text NOT NULL,
    at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX login_failures_email_at_idx ON login_failures (email, at);

CREATE TABLE llm_calls (
    id            bigserial PRIMARY KEY,
    at            timestamptz NOT NULL DEFAULT now(),
    purpose       text NOT NULL,           -- extraction | vision | agent
    document_id   text,                    -- no FK: traces outlive deleted documents
    model         text NOT NULL,
    input_tokens  integer NOT NULL,
    output_tokens integer NOT NULL,
    cost_usd      numeric(12, 6) NOT NULL,
    latency_ms    integer NOT NULL,
    stop_reason   text,
    error         text
);
CREATE INDEX llm_calls_document_idx ON llm_calls (document_id, at);
CREATE INDEX llm_calls_at_idx ON llm_calls (at);
