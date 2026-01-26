# 🏠 Housing Price Prediction --- Production ML API

[![Deployed on
Railway](https://img.shields.io/badge/Deployed%20on-Railway-7B3FE4?logo=railway&logoColor=white)](https://housing-price-ml-api-production.up.railway.app)
[![Swagger](https://img.shields.io/badge/API-Docs-green)](https://housing-price-ml-api-production.up.railway.app/docs)
[![CI - FastAPI
Tests](https://github.com/KaushalyaDasanayake/housing-price-ml-api/actions/workflows/ci.yml/badge.svg)](https://github.com/KaushalyaDasanayake/housing-price-ml-api/actions/workflows/ci.yml)
[![Weekly
Retraining](https://github.com/KaushalyaDasanayake/housing-price-ml-api/actions/workflows/retrain.yml/badge.svg)](https://github.com/KaushalyaDasanayake/housing-price-ml-api/actions/workflows/retrain.yml)

Production-style **Machine Learning API** built with **FastAPI**,
featuring **Redis caching, request logging, data drift detection, and
automated weekly retraining via GitHub Actions**.\
Deployed using **Docker** and **Railway**.

This project demonstrates an **end-to-end ML engineering workflow**, not
just model training.

------------------------------------------------------------------------

## 🚀 Live Demo

-   **API Base URL:**
    https://housing-price-ml-api-production.up.railway.app\
-   **Swagger UI:**
    https://housing-price-ml-api-production.up.railway.app/docs\
-   Test predictions using: `POST /v1/predict`

------------------------------------------------------------------------

## 🎯 What This Project Demonstrates

  Area               Implementation
  ------------------ ---------------------------------
  Model Training     scikit-learn Linear Regression
  API Serving        FastAPI
  Input Validation   Pydantic
  Feature Safety     Fixed feature order enforcement
  Caching            Redis
  Logging            CSV prediction logs
  Monitoring         Stats & drift endpoints
  CI Testing         GitHub Actions + pytest
  Retraining         Weekly GitHub Actions pipeline
  Deployment         Docker + Railway

------------------------------------------------------------------------

## 🧱 System Architecture

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

------------------------------------------------------------------------

## ✨ Features

### ✅ Prediction API

-   **Endpoint:** `POST /v1/predict`
-   Input validation with **Pydantic**
-   Fixed feature order to prevent silent bugs

------------------------------------------------------------------------

### ✅ Redis Caching

-   Same input → same cache key (SHA-256)
-   Reduces latency for repeated predictions
-   Cache TTL = **10 minutes**

------------------------------------------------------------------------

### ✅ Logging & Dataset Building

Every prediction is logged to CSV:

-   input features
-   predicted price
-   latency
-   cache hit flag
-   model version

Creates a production-style dataset for: - monitoring - drift detection -
retraining

------------------------------------------------------------------------

### ✅ Monitoring Endpoints

#### 🔍 Health

`GET /health` --- checks server status

#### ⚙️ Readiness

`GET /ready` --- checks model & scaler availability

#### 📊 Stats

`GET /stats` --- returns: - total predictions - cache hit ratio -
average latency - last 5 predictions

#### 📈 Drift Detection

`GET /drift`

-   Compares recent feature means with training baseline
-   Uses **z-score**
-   Flags drift when:

```{=html}
<!-- -->
```
    |z| > 3

------------------------------------------------------------------------

## ⚡ Redis Caching Strategy

-   Input JSON → SHA-256 hash → cache key
-   Same input = same prediction result
-   Prevents unnecessary model computation
-   Improves response latency

------------------------------------------------------------------------

## 🧾 Prediction Logging

Each request logs:

-   timestamp
-   request id
-   model version
-   input features
-   predicted price
-   latency
-   cache hit flag
-   error (if any)

Enables: - monitoring - drift analysis - continuous retraining

------------------------------------------------------------------------

## 🔁 Automated Weekly Retraining

GitHub Actions workflow runs **weekly**:

### Steps

1.  Load production logs
2.  Merge with training dataset
3.  Retrain model & scaler
4.  Update:
    -   `model.joblib`
    -   `scaler.joblib`
    -   `training_stats.json`

Simulates real-world ML lifecycle automation.

📄 Workflow file:

    .github/workflows/retrain.yml

------------------------------------------------------------------------

## 🧪 Testing & CI

### ✅ Local Testing

    pytest

### ✅ CI Pipeline

-   Runs on every push to `main`
-   Executes FastAPI endpoint tests

📄 Workflow:

    .github/workflows/ci.yml

------------------------------------------------------------------------

## 🐳 Run Locally with Docker

    docker compose up --build

Then open:

    http://localhost:8000/docs

Redis and API run as separate containers.

------------------------------------------------------------------------

## 🔐 Environment Variables

  Variable        Purpose
  --------------- -------------------------------
  REDIS_URL       Redis connection string
  PRED_LOG_PATH   CSV file path for predictions
  MODEL_VERSION   Model version label

Example:

    PRED_LOG_PATH=/app/data/predictions.csv
    MODEL_VERSION=v1

------------------------------------------------------------------------

## 🛠 Tech Stack

-   FastAPI
-   scikit-learn
-   Redis
-   Docker & Docker Compose
-   GitHub Actions
-   Pandas, NumPy

------------------------------------------------------------------------

## ⚠️ Limitations & Future Improvements

### Current Limitations

-   No real production labels → cannot measure live accuracy
-   Drift detection is statistical only (mean/std)
-   No Prometheus/Grafana monitoring
-   No A/B testing or model version routing

### Future Enhancements

-   Performance-based monitoring
-   Shadow deployments
-   Feature store integration
-   Model registry support

------------------------------------------------------------------------

## 🌟 Why This Project Matters

-   Simulates real production ML systems
-   Includes monitoring and retraining
-   Demonstrates CI/CD and cloud deployment
-   Shows ML engineering, not just modeling


------------------------------------------------------------------------



---

##  Example API Usage

###  Example Prediction Request

Using **curl**:

```bash
curl -X POST "https://housing-price-ml-api-production.up.railway.app/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252,
    "HouseAge": 41,
    "AveRooms": 6.984127,
    "AveBedrms": 1.02381,
    "Population": 322,
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23,
    "RoomsPerHousehold": 2.0,
    "BedroomsPerHouse": 0.3,
    "PopulationPerHousehold": 3.0
  }'
```

✅ **Successful Response**

```json
{
  "status": "success",
  "model_version": "v1",
  "predicted_price": 4.425209713467011,
  "error": null
}
```

---

###  Example Drift Check

Using **curl**:

```bash
curl -X GET "https://housing-price-ml-api-production.up.railway.app/drift"
```

✅ **Drift Report Response**

```json
{
  "status": "ok",
  "checked_rows": 50,
  "features": {
    "MedInc": {
      "train_mean": 3.881,
      "current_mean": 8.325,
      "z_score": 2.33,
      "drift": false
    },
    "HouseAge": {
      "train_mean": 28.608,
      "current_mean": 41.0,
      "z_score": 0.98,
      "drift": false
    }
  }
}
```

---



---

## 🐳 Run Locally with Docker

You can run the full system (**API + Redis**) locally using **Docker Compose**.

### ✅ Prerequisites
- Docker installed
- Docker Compose installed

### ▶️ Steps

```bash
git clone https://github.com/KaushalyaDasanayake/housing-price-ml-api.git
cd housing-price-ml-api
docker-compose up --build
```

### Access the API

- **API Base URL:** http://localhost:8000  
- **Swagger UI:** http://localhost:8000/docs

Redis runs inside Docker and is automatically connected to the API container.

### ⏹ Stop Services

```bash
docker-compose down
```

---

## Run Tests Locally (Optional)

If you want to run tests without Docker:

```bash
pip install -r requirements.txt
pytest
```

---

## 👩‍💻 Author

**Kaushalya Rathnayake**\
ML Engineering Portfolio Project