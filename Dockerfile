# Multi-stage build for Linnaeus
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./

# Copy source code
COPY src/ ./src/

# Install dependencies and build wheel
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install build
RUN python -m build --wheel

# Production stage
FROM python:3.11-slim

# Install system dependencies for runtime
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r linnaeus && useradd -r -g linnaeus linnaeus

# Install UV
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy built wheel from builder stage
COPY --from=builder /app/dist/*.whl ./

# Create virtual environment and install package
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install *.whl

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs && \
    chown -R linnaeus:linnaeus /app

# Copy default configuration
COPY config.yaml /app/

# Switch to non-root user
USER linnaeus

# Set environment variables
ENV PYTHONPATH="/app"
ENV AMEGHINO_CONFIG="/app/config.yaml"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD linnaeus --version || exit 1

# Default command
CMD ["linnaeus", "--help"]

# Labels
LABEL maintainer="Esteban <linnaeus@example.com>"
LABEL version="0.1.0"
LABEL description="AI-powered classification of Internet Autonomous Systems"
LABEL org.opencontainers.image.source="https://github.com/NU-AquaLab/linnaeus"
LABEL org.opencontainers.image.title="Linnaeus"
LABEL org.opencontainers.image.description="AI-powered classification of Internet Autonomous Systems"
LABEL org.opencontainers.image.version="0.1.0"
