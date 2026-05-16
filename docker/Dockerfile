# ──────────────────────────────────────────────────────────────────────────────
# ODVS Dockerfile — Multi-stage build
#
# Stages:
#   1. base      — JDK 17 + Python 3.11 + system deps
#   2. builder   — pip install into a virtual environment
#   3. runtime   — minimal runtime image (no build tools)
#   4. streamlit — extends runtime, exposes Streamlit port
#
# Build targets:
#   docker build --target runtime   -t odvs:latest .       # pipeline runner
#   docker build --target streamlit -t odvs-ui:latest .    # Streamlit UI
# 

# Stage 1: base 
FROM python:3.11.9-slim-bookworm AS base

LABEL org.opencontainers.image.title="ODVS"
LABEL org.opencontainers.image.description="Open Dataset Versioning System — Apache Iceberg + Spark + S3"
LABEL org.opencontainers.image.source="https://github.com/odvs/odvs"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    # Pip configuration
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    # Java — required by PySpark
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PATH="/usr/lib/jvm/java-17-openjdk-amd64/bin:$PATH" \
    # Spark tuning
    SPARK_LOCAL_DIRS=/tmp/spark-local \
    SPARK_LOG_DIR=/var/log/spark \
    # ODVS defaults (override via docker run -e or docker-compose env_file)
    ODVS_ENV=production \
    ODVS_LOG_LEVEL=INFO \
    ODVS_STRUCTURED_LOGS=true

# System packages:
#   - openjdk-17-jre-headless: PySpark JVM runtime
#   - procps: spark process management
#   - curl: health checks
#   - ca-certificates: S3/HTTPS TLS
#   - libgomp1: PyArrow SIMD optimisation
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        procps \
        curl \
        ca-certificates \
        libgomp1 \
        tini \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # Verify Java
    && java -version \
    # Create Spark temp directories
    && groupadd --gid 1001 odvs \
    && useradd --uid 1001 --gid odvs --shell /bin/bash --create-home odvs \
    && mkdir -p /tmp/spark-local /var/log/spark \
    && chown -R odvs:odvs /tmp/spark-local /var/log/spark

WORKDIR /app


# Stage 2: builder
FROM base AS builder

# Install build toolchain (not carried into runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a venv so we can COPY it cleanly into the runtime stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Layer-cache pip: copy requirements first so changes to source don't bust this
COPY requirements.txt ./

# Install production deps only (no dev extras)
# Exclude dev/test packages that are not needed at runtime
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && pip check


# Stage 3: runtime
FROM base AS runtime

# Copy the pre-built venv from builder — no compiler toolchain in this layer
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY --chown=odvs:odvs . /app/

# Pre-download Iceberg + AWS bundle JARs into the PySpark jars directory so
# the first pipeline run doesn't require internet access (important in air-gapped envs)
# Jars are fetched from Maven Central during build, not at runtime.
RUN python - <<'EOF' || echo "JAR pre-fetch skipped (no network). JARs will be fetched at first run."
from pyspark.sql import SparkSession
import os
os.environ.setdefault('SPARK_LOCAL_DIRS', '/tmp/spark-local')
spark = SparkSession.builder \
    .master('local[1]') \
    .config('spark.jars.packages',
        'org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.10.1,'
        'org.apache.iceberg:iceberg-aws-bundle:1.10.1,'
        'org.apache.hadoop:hadoop-aws:3.4.2') \
    .config('spark.driver.memory', '512m') \
    .getOrCreate()
spark.stop()
print('Iceberg JARs pre-downloaded.')
EOF

# Registry directory
RUN mkdir -p /home/odvs/.odvs/registry /home/odvs/.odvs/lineage \
    && chown -R odvs:odvs /home/odvs/.odvs

# Drop to non-root
USER odvs

# Expose metrics / health port (no direct HTTP from runtime image, just a placeholder)
EXPOSE 8080

# Use tini as PID 1 to handle SIGTERM correctly for Spark child processes
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default: show help (override with docker run odvs:latest python scripts/run_pipeline.py ...)
CMD ["python", "src/odvs/scripts/run_pipeline.py", "--help"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import odvs.config; print('OK')" || exit 1


# Stage 4: streamlit 
FROM runtime AS streamlit

# Streamlit configuration via environment (no config file needed)
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_THEME_BASE=dark \
    STREAMLIT_THEME_PRIMARY_COLOR=#2a7aff \
    STREAMLIT_THEME_BACKGROUND_COLOR=#0d0f14 \
    STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR=#131720 \
    STREAMLIT_THEME_TEXT_COLOR=#e8ecf4 \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500 \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true

EXPOSE 8501

CMD ["streamlit", "run", "src/odvs/streamlit_app/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--logger.level=warning"]

HEALTHCHECK --interval=20s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:8501/_stcore/health || exit 1