# Sunnyview segmenter (FastAPI). Build from this directory:
#   docker build -t sunnyview-segmenter .
#   docker run --rm -p 127.0.0.1:8000:8000 sunnyview-segmenter
# Add --build-arg WITH_ONNX=1 to include onnxruntime for the ONNX U-Net stage.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

ARG WITH_ONNX=0
COPY requirements.txt requirements-onnx.txt ./
RUN pip install -r requirements.txt \
    && if [ "$WITH_ONNX" = "1" ]; then pip install -r requirements-onnx.txt; fi

COPY *.py ./
COPY models/ ./models/

RUN useradd --create-home --uid 10001 segmenter
USER segmenter

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"

# 0.0.0.0 is required inside the container; publish it on localhost only (see README).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
