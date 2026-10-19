# syntax=docker/dockerfile:1
#
# Multi-stage image for the `structured-agent` CLI.
#
#   docker build -t structured-output-agent .
#   docker run --rm structured-output-agent demo
#
# Optional extras (e.g. the Anthropic client) can be baked in at build time:
#
#   docker build --build-arg EXTRAS=anthropic -t structured-output-agent:anthropic .

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------
# Builder: produce a wheel from the source tree.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip wheel --no-deps --wheel-dir /dist .

# ---------------------------------------------------------------------------
# Runtime: install only the wheel (and requested extras) on a clean base.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

# Comma-separated list of optional extras to install, e.g. "anthropic".
ARG EXTRAS=""

LABEL org.opencontainers.image.title="structured-output-agent" \
      org.opencontainers.image.description="Pydantic-enforced structured output, validated tool calls and retry-on-parse-error for LLMs." \
      org.opencontainers.image.source="https://github.com/imranmali/structured-output-agent" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY --from=builder /dist /tmp/dist

# The glob is expanded by the shell first; the extras suffix is appended only
# when EXTRAS is non-empty, producing e.g. "/tmp/dist/<wheel>.whl[anthropic]".
RUN set -eu; \
    wheel="$(ls /tmp/dist/structured_output_agent-*.whl)"; \
    pip install "${wheel}${EXTRAS:+[${EXTRAS}]}"; \
    rm -rf /tmp/dist; \
    useradd --create-home --uid 10001 --shell /usr/sbin/nologin agent

USER agent
WORKDIR /home/agent

ENTRYPOINT ["structured-agent"]
CMD ["--help"]
