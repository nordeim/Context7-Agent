# ==============================================================================
# Context7 AI Agent - Multi-stage Production Dockerfile
# ==============================================================================

# Build stage - for compiling dependencies
FROM python:3.11-slim as builder

# Set build arguments
ARG POETRY_VERSION=1.8.0
ARG POETRY_HOME=/opt/poetry

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION

# Set environment variables
ENV PATH="${POETRY_HOME}/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

# Production stage - minimal runtime
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r context7 && useradd -r -g context7 context7

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @upstash/context7-mcp@latest

# Create directories
RUN mkdir -p /app/data /app/logs && \
    chown -R context7:context7 /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=context7:context7 src/ /app/src/

# Switch to non-root user
USER context7

# Set working directory
WORKDIR /app

# Create data directory
RUN mkdir -p /app/data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, 'src'); from config import config; print('OK')" || exit 1

# Expose port (for future web interface)
EXPOSE 8080

# Default command
CMD ["python", "-m", "src.cli"]
