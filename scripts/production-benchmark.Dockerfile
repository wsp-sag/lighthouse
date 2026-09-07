FROM ghcr.io/astral-sh/uv:0.12.1-python3.10-trixie-slim

ENV UV_PROJECT_ENVIRONMENT=/opt/lighthouse-venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MALLOC_ARENA_MAX=2 \
    MALLOC_TRIM_THRESHOLD_=65536

WORKDIR /build
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

WORKDIR /workspace
ENTRYPOINT ["/opt/lighthouse-venv/bin/python", "/workspace/scripts/production-benchmark.py"]
