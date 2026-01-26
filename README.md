# Housing Price Prediction — ML API (Production Style)

[![Deployed on Railway](https://img.shields.io/badge/Deployed%20on-Railway-7B3FE4?logo=railway&logoColor=white)](https://housing-price-ml-api-production.up.railway.app)
[![Swagger](https://img.shields.io/badge/API-Docs-green)](https://housing-price-ml-api-production.up.railway.app/docs)
[![CI - FastAPI Tests](https://github.com/KaushalyaDasanayake/housing-price-ml-api/actions/workflows/ci.yml/badge.svg)](https://github.com/KaushalyaDasanayake/housing-price-ml-api/actions/workflows/ci.yml)
[![Weekly Retraining](https://github.com/KaushalyaDasanayake/housing-price-ml-api/actions/workflows/retrain.yml/badge.svg)](https://github.com/KaushalyaDasanayake/housing-price-ml-api/actions/workflows/retrain.yml)


Production-style Machine Learning API built with FastAPI, featuring Redis caching, request logging, data drift detection, and automated weekly retraining via GitHub Actions.
Deployed using Docker and Railway.

This project demonstrates an end-to-end ML engineering workflow, not just model training.

**Live Demo**

API Base URL: https://housing-price-ml-api-production.up.railway.app

Swagger UI: https://housing-price-ml-api-production.up.railway.app/docs

Use /v1/predict to test predictions directly in the browser

**What This Project Demonstrates**

| Area             | Implemented                     |
| ---------------- | ------------------------------- |
| Model Training   | scikit-learn Linear Regression  |
| API Serving      | FastAPI                         |
| Input Validation | Pydantic                        |
| Feature Safety   | Fixed feature order enforcement |
| Caching          | Redis                           |
| Logging          | CSV prediction logs             |
| Monitoring       | Stats & drift endpoints         |
| CI Testing       | GitHub Actions + pytest         |
| Retraining       | Weekly GitHub Actions pipeline  |
| Deployment       | Docker + Railway                |


**System Architecture**

Client (Swagger / curl / frontend)
            |
            v
     FastAPI Application
            |
            +--> Redis Cache (prediction cache)
            |
            +--> ML Model (scikit-learn)
            |
            +--> CSV Logs (prediction history)
                        |
                        v
                 Drift Detection (/drift)
                        |
                        v
            GitHub Actions (Weekly Retraining)


**Features**

✅ Prediction API
- Endpoint: POST /v1/predict
- Input validation with Pydantic
- Feature order safety to prevent silent bugs

✅ Redis Caching
- Same input → same cache key (SHA256)
- Reduces latency for repeated predictions
- Cache TTL = 10 minutes

✅ Logging & Dataset Building

Logs every prediction to CSV:

- input features
- predicted price
- latency
- cache hit
- model version

This creates a production-style dataset for monitoring and retraining.

✅ Monitoring Endpoints
Health
- GET /health

Checks if server is running.

Readiness
- GET /ready

Checks if model & scaler are loaded.

Stats
- GET /stats

Returns:
- total predictions
- cache hit ratio
- average latency
- last 5 predictions

Drift Detection
- GET /drift

Compares recent prediction feature means with training baseline using z-score:

**Flags drift when z-score > 3**

**Redis Caching Strategy**

- Input JSON is hashed using SHA-256
- Same input → same cache key
- Cached predictions expire after 10 minutes
- Improves latency and reduces model compute

**Prediction Logging**

Every request is logged to CSV:

- timestamp
- request id
- model version
- input features
- predicted price
- latency
- cache hit flag
- error (if any)

This creates a real production-style dataset for:
- monitoring
- drift analysis
- retraining

**Automated Weekly Retraining**

A GitHub Actions workflow runs weekly:

Steps:
1. Load production prediction logs
2. Combine with original training data
3. Retrain model & scaler
4. Update:
- model.joblib
- scaler.joblib
- training_stats.json

This simulates a continuous ML lifecycle.

- Workflow file:
.github/workflows/retrain.yml

[![Weekly Retraining](https://github.com/KaushalyaDasanayake/housing-price-ml-api/actions/workflows/retrain.yml/badge.svg)](https://github.com/KaushalyaDasanayake/housing-price-ml-api/actions/workflows/retrain.yml)

**Testing & CI**

**Local Tests**
- pytest

**CI Pipeline**
- Runs on every push to main
- Executes FastAPI endpoint tests

Workflow:

.github/workflows/ci.yml

[![CI - FastAPI Tests](https://github.com/KaushalyaDasanayake/housing-price-ml-api/actions/workflows/ci.yml/badge.svg)](https://github.com/KaushalyaDasanayake/housing-price-ml-api/actions/workflows/ci.yml)


**Run Locally with Docker**

docker compose up --build

Then open:

http://localhost:8000/docs

Redis and API run as separate containers.

**Environment Variables**

| Variable        | Purpose                       |
| --------------- | ----------------------------- |
| `REDIS_URL`     | Redis connection string       |
| `PRED_LOG_PATH` | CSV file path for predictions |
| `MODEL_VERSION` | Model version label           |


Example:

PRED_LOG_PATH=/app/data/predictions.csv
MODEL_VERSION=v1

**Tech Stack**

- FastAPI
- scikit-learn
- Redis
- Docker & Docker Compose
- GitHub Actions
- Pandas, NumPy

**Limitations & Future Improvements**

- No real ground-truth labels in production → accuracy cannot be measured live
- Drift detection is statistical only (mean/std), not model performance based
- Prometheus/Grafana not included (intentionally kept simple)
- No A/B model versioning yet

Future upgrades:
- Performance monitoring
- Shadow deployments
- Feature store integration

**Author**

Kaushalya Rathnayake | ML Engineering Portfolio Project


```md
## Why This Project Matters

- Simulates production ML system
- Includes monitoring and retraining
- Uses CI/CD and cloud deployment