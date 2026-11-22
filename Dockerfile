FROM python:3.12-slim

WORKDIR /srv/repo

ENV UV_PROJECT_ENVIRONMENT="/venv"
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
ENV PYTHONPATH="/srv/repo"

COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /usr/local/bin/uv

# Dependencies only: the code is mounted at /srv/repo or run from the working directory.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --all-extras --all-groups
