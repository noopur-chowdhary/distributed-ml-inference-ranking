FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-docker.txt .

RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements-docker.txt

COPY src ./src
COPY artifacts ./artifacts
COPY data/processed ./data/processed
COPY serve_config.yaml .

EXPOSE 8000

CMD ["serve", "run", "serve_config.yaml"]
