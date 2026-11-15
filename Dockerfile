# Production image for orderdesk. One image, several roles (web app, ERP simulator, jobs):
# the command decides which runs. Build: docker build -t orderdesk .

# --- stage 1: build the virtual environment ---------------------------------------------------
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first, in their own layer: rebuilt only when the lockfile changes, not on every
# code edit. That turns a 2-minute rebuild into a few seconds.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY README.md ./
RUN uv sync --locked --no-dev

# Bake the embedding model into the image so containers don't download 130 MB on every start.
ENV FASTEMBED_CACHE_PATH=/app/models
RUN .venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# --- stage 2: the slim runtime image ------------------------------------------------------------
FROM python:3.12-slim
RUN useradd --create-home --uid 10001 orderdesk   # never run as root
WORKDIR /app
COPY --from=build --chown=orderdesk /app /app
COPY --chown=orderdesk sample_data ./sample_data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/models \
    ERP_DB_PATH=/app/.erp/erp.db \
    ERP_DATA_DIR=/app/sample_data
RUN mkdir -p /app/.erp && chown orderdesk /app/.erp
USER orderdesk

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=4)"

# Default role: the web app. PORT is set by most hosting platforms.
CMD ["sh", "-c", "uvicorn orderdesk.web.app:create_app --factory --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
