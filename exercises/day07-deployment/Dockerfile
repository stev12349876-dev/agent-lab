FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd --create-home appuser

WORKDIR /app

COPY exercises/day06-rag-api/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements.txt

COPY --chown=appuser:appuser exercises/day06-rag-api /app/exercises/day06-rag-api
COPY --chown=appuser:appuser exercises/day03-rag-loading/data/sample.md /app/exercises/day03-rag-loading/data/sample.md

RUN mkdir -p /app/exercises/day06-rag-api/data \
    && chown -R appuser:appuser /app

USER appuser
WORKDIR /app/exercises/day06-rag-api

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
