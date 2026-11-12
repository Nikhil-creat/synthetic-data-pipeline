# syntax=docker/dockerfile:1

# ---------- Stage 1: build the virtualenv (CPU-only torch keeps the image small) ----------
FROM python:3.15-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/opt/torch

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Pre-download ResNet18 weights so the runtime container works offline.
# If the build has no internet, the app falls back to random init and logs a warning.
RUN mkdir -p /opt/torch && \
    (python -c "from torchvision import models; models.resnet18(weights=models.ResNet18_Weights.DEFAULT)" \
     || echo "WARN: could not pre-download ResNet18 weights")

# ---------- Stage 2: minimal runtime ----------
FROM python:3.15-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    TORCH_HOME=/opt/torch \
    QDRANT_URL=http://qdrant:6333

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/torch /opt/torch
COPY pipeline.py main.py ./

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
