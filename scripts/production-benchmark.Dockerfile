FROM ghcr.io/astral-sh/uv:0.12.1-python3.10-trixie-slim

ENV UV_PROJECT_ENVIRONMENT=/opt/lighthouse-venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MALLOC_ARENA_MAX=2 \
    MALLOC_TRIM_THRESHOLD_=65536

WORKDIR /build

# These named contexts are supplied by production-benchmark.py. Copy only the
# package sources and build metadata, not either repository's Git history or
# development artifacts.
COPY --from=activitysim pyproject.toml README.md LICENSE.txt ./activitysim/
COPY --from=activitysim activitysim ./activitysim/activitysim
COPY --from=sharrow pyproject.toml README.md LICENSE ./sharrow/
COPY --from=sharrow sharrow ./sharrow/sharrow

WORKDIR /build/lighthouse
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# uv records editable sources outside the project as absolute paths. Refresh
# those local path records inside the image, retaining the copied lockfile's
# version preferences, then install the resolved lock non-editably to match
# production behavior.
RUN uv lock && uv sync --locked --no-dev --no-editable

WORKDIR /workspace
ENTRYPOINT ["/opt/lighthouse-venv/bin/python", "/workspace/scripts/production-benchmark.py"]
