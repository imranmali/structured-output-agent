# syntax=docker/dockerfile:1
#
# Multi-stage image serving the structured-output HTTP API by default.
#
#   docker build -t structured-output-agent .
#   docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY=... \
#       -e SOA_MODELS=structured_output_agent.cli:WeatherReport structured-output-agent
#   docker run --rm structured-output-agent demo        # offline demo, no network
#
# EXTRAS selects the optional extras baked in (default: Claude client + server):
#
#   docker build --build-arg EXTRAS=anthropic -t structured-output-agent:cli .
#
# See docs/deployment.md for configuration (SOA_* variables) and operations.

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------
# Builder: install the package and its dependencies into a virtualenv.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

# Comma-separated list of optional extras to install; empty for the bare CLI.
ARG EXTRAS="anthropic,server"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Installing from the source directory builds exactly one wheel, so no
# filename globbing is needed; the extras suffix is added only when EXTRAS is
# non-empty, e.g. "/src[anthropic,server]".
RUN pip install "/src${EXTRAS:+[${EXTRAS}]}"

# ---------------------------------------------------------------------------
# Runtime: the virtualenv on a clean base, running as an unprivileged user.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="structured-output-agent" \
      org.opencontainers.image.description="Pydantic-enforced structured output, validated tool calls and retry-on-parse-error for LLMs, served over HTTP." \
      org.opencontainers.image.source="https://github.com/imranmali/structured-output-agent" \
      org.opencontainers.image.documentation="https://github.com/imranmali/structured-output-agent/blob/main/docs/deployment.md" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    SOA_LOG_FORMAT=json

RUN groupadd --gid 10001 agent \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin agent

COPY --from=builder /opt/venv /opt/venv

# Numeric so Kubernetes can verify runAsNonRoot without a runAsUser override.
USER 10001:10001
WORKDIR /home/agent

EXPOSE 8000

# Only meaningful for `serve`; one-shot commands such as `demo` exit before
# the first check runs.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"]

ENTRYPOINT ["structured-agent"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
