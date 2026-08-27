# Multi-stage build for Linneaus
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
RUN groupadd -r linneaus && useradd -r -g linneaus linneaus

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
    chown -R linneaus:linneaus /app

# Copy default configuration
COPY config.yaml /app/

# Switch to non-root user
USER linneaus

# Set environment variables
ENV PYTHONPATH="/app"
ENV AMEGHINO_CONFIG="/app/config.yaml"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD linneaus --version || exit 1

# Default command
CMD ["linneaus", "--help"]

# Labels
LABEL maintainer="Esteban <linneaus@example.com>"
LABEL version="0.1.0"
LABEL description="AI-powered classification of Internet Autonomous Systems"
LABEL org.opencontainers.image.source="https://github.com/linneaus-project/linneaus"
LABEL org.opencontainers.image.title="Linneaus"
LABEL org.opencontainers.image.description="AI-powered classification of Internet Autonomous Systems"
LABEL org.opencontainers.image.version="0.1.0"
