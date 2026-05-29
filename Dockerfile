FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    HF_HOME=/data/models/.cache \
    HF_HUB_CACHE=/data/models/.cache/hub \
    TRANSFORMERS_CACHE=/data/models/.cache/hub \
    PATH=/opt/venv/bin:/root/.local/bin:${PATH}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip uv

COPY pyproject.toml setup.py MANIFEST.in uv.lock .python-version readme.md ./

RUN uv sync --frozen --no-dev --no-install-project

COPY . .

RUN mkdir -p /data/models /data/models/.cache /app/results /app/results_latent \
 && uv sync --frozen --no-dev \
 && rm -rf /root/.cache/pip /root/.cache/uv

VOLUME ["/data"]

# For GPU workloads, run the container with `--gpus all`.
CMD ["bash"]
