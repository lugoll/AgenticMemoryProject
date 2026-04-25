FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Layer 1: uv-managed dependencies (invalidated when pyproject.toml/uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra app --no-dev --no-install-project

# Layer 2: project source + final uv sync
COPY . .
RUN uv sync --frozen --extra app --no-dev

# Layer 3: torch CPU + sentence-transformers — installed AFTER uv sync so they
# are not removed by uv's lockfile reconciliation. BuildKit cache mount keeps
# downloaded wheels on the host so subsequent rebuilds skip the download.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install "transformers>=4.40.0" "sentence-transformers>=3.0"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN chmod +x /app/docker-entrypoint.sh
ENTRYPOINT ["/app/docker-entrypoint.sh"]
