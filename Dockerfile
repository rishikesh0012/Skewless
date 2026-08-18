# Skewless FastAPI backend.
#
# Build (from repo root): docker build -t skewless-backend .
# Run:                    docker run -p 8000:8000 skewless-backend
FROM python:3.13-slim

WORKDIR /app

# libgomp1 is required at runtime by LightGBM's compiled OpenMP extension.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# src/ and models/ mirror the local layout so the existing parents[2]-relative
# paths in model/predictor.py and model/train.py resolve unchanged.
COPY src ./src
COPY models ./models

ENV PYTHONPATH=/app/src

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
