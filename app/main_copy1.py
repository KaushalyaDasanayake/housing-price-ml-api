import redis
import os
import json
import hashlib
from fastapi import FastAPI
import pandas as pd
from fastapi import Body
from typing import Optional, Any, List, Dict
from pydantic import ConfigDict
import joblib
import numpy as np
import logging
import time
from uuid import uuid4
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import Request
from pathlib import Path
from pydantic import BaseModel, Field

# logger 
from datetime import datetime
import csv
from threading import Lock

# log stats
from collections import Counter

# export csv
from fastapi.responses import FileResponse
import csv

# Logs
PRED_LOG_PATH = os.getenv("PRED_LOG_PATH", "data/predictions.csv")
_csv_lock = Lock()

def append_prediction_log(row: dict) -> None:
    """
    Append one prediction log row to a CSV file.
    Uses a lock to avoid race conditions when multiple requests happen.
    """
    fieldnames = list(row.keys())

    os.makedirs(os.path.dirname(PRED_LOG_PATH), exist_ok=True)

    with _csv_lock:
        file_exists = os.path.exists(PRED_LOG_PATH) and os.path.getsize(PRED_LOG_PATH) > 0
        with open(PRED_LOG_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


redis_url = os.getenv("REDIS_URL") or os.getenv("REDIS_PUBLIC_URL")

app = FastAPI() #creates a fastapi app object

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("house-api")

# testing
logger.info("server starting...")

BASE_DIR = Path(__file__).resolve().parent.parent

# will be loaded during startup (avoid heady loading during startup)
model = None
scaler = None

redis_client = None

# model = None # for testing purpose for rediness

@app.on_event("startup")
def load_artifacts():
    global model, scaler, redis_client

    # 1) Load model + scaler
    try:
        logger.info("Loading model and scaler...")
        model = joblib.load(BASE_DIR / "model" / "model.joblib")
        scaler = joblib.load(BASE_DIR / "model" / "scaler.joblib")
        logger.info("✅ Model and scaler loaded successfully")
    except Exception as e:
        model = None
        scaler = None
        logger.exception(f"❌ Failed to load artifacts: {e}")

    # 2) Connect to Redis
    try:
        logger.info("Connecting to Redis...")

        # Railway gives REDIS_PUBLIC_URL (you already have it)
        redis_url = os.getenv("REDIS_URL") or os.getenv("REDIS_PUBLIC_URL")

        if redis_url:
            redis_client = redis.from_url(redis_url, decode_responses=True)
        else:
            # Local dev (optional)
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", 6379))
            redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

        redis_client.ping()
        logger.info("✅ Connected to Redis")

    except Exception as e:
        redis_client = None
        logger.exception(f"❌ Failed to connect to Redis: {e}")


# startup runs once per server process
# global allows you to assign to module-level variables
# if loading fails, we keep model/scaler=None → readiness fails safely

# Feature order
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
# MODEL_VERSION = "v1"

FEATURE_ORDER = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
    "RoomsPerHousehold",
    "BedroomsPerHouse",
    "PopulationPerHousehold",
]

class HouseFeatures(BaseModel):

    # Forbid extra fields → protects model from wrong feature inputs
    model_config = ConfigDict(extra="forbid")

    # ge = greater than or equal
    # examples show sample values in Swagger UI
    MedInc: float = Field(..., ge=0, examples=[8.3252])   #  Field(...)  -  required
    HouseAge: float = Field(..., ge=0, examples=[41])
    AveRooms: float = Field(..., ge=0, examples=[6.984127])
    AveBedrms: float = Field(..., ge=0, examples=[1.02381])
    Population: float = Field(..., ge=0, examples=[322])
    AveOccup: float = Field(..., ge=0, examples=[2.555556])

    # Latitude must be between -90 and 90
    Latitude: float = Field(..., ge=-90, le=90, examples=[37.88])

    # Longitude must be between -180 and 180
    Longitude: float = Field(..., ge=-180, le=180, examples=[-122.23])

    RoomsPerHousehold: float = Field(..., ge=0, examples=[2.0])
    BedroomsPerHouse: float = Field(..., ge=0, examples=[0.3])
    PopulationPerHousehold: float = Field(..., ge=0, examples=[3.0])


class ApiError(BaseModel):
    code: str                # machine-readable error code
    message: str             # human-readable message
    details: Optional[Any] = None   # optional debug info

class PredictResponse(BaseModel):
    status: str                     # "success" or "error"
    model_version: str              # API/model version
    predicted_price: Optional[float] = None
    error: Optional[ApiError] = None


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    rid = getattr(request.state, "request_id", "no-request-id")
    logger.warning(f"[{rid}] VALIDATION_ERROR | details={exc.errors()}")

    return JSONResponse(
        status_code=422,
        content=PredictResponse(
            status="error",
            model_version=MODEL_VERSION,
            predicted_price=None,
            error=ApiError(
                code="VALIDATION_ERROR",
                message="Input validation failed",
                details=exc.errors()   # detailed field errors
            )
        ).model_dump()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", "no-request-id")
    logger.exception(f"[{rid}] INTERNAL_SERVER_ERROR: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "model_version": MODEL_VERSION,
            "predicted_price": None,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "Something went wrong during prediction."
            }
        }
    )


# Logging Middleware
@app.middleware("http")
async def log_requests(request:Request, call_next):
    request_id = str(uuid4())
    request.state.request_id = request_id  # store for later use
    start = time.time()

    # Log request starts
    logger.info(f"[{request_id}] START {request.method} {request.url.path}")

    try:
        response = await call_next(request)
    except Exception as e:
        # Log unexpected crash (500)
        logger.exception(f"[{request_id}] CRASH {request.method} {request.url.path}: {e}")
        # send error again and fastapi can send 500 response
        raise

    # How many milliseconds the request took
    duration_ms = (time.time() - start) * 1000
    logger.info(f"[{request_id}] END {request.method} {request.url.path} -> {response.status_code} ({duration_ms:.2f}ms)")

    # Attach request_id for the response headers so client can report it
    response.headers["X-Request-ID"] = request_id
    return response


# Health check (liveness)
@app.get("/health")
def health(request:Request):
    
    # add logs
    rid = getattr(request.state, "request_id", "no-request-id")
    logger.info(f"[{rid}] Health check")

    # server is up and responding
    return {"status":"ok"}

# check service is ready
@app.get("/ready", response_model=Dict[str, Any])
def ready():
    # readiness: model/scaler loaded and usable
    if model is None or scaler is None:
        return JSONResponse(
            status_code=503,
            content={
                "status":"not_ready",
                "model_version":MODEL_VERSION
            }
        )
    return{
        "status":"ready",
        "model_version":MODEL_VERSION
    }


@app.get("/")
def home():
    return {"message": "Api is working"}

# get logs
@app.get("/logs/stats")
def log_stats():
    if not os.path.exists(PRED_LOG_PATH) or os.path.getsize(PRED_LOG_PATH) == 0:
        return {"status": "no_logs_yet"}

    rows = []
    with open(PRED_LOG_PATH, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    total = len(rows)
    hits = sum(1 for r in rows if str(r.get("cache_hit")).lower() == "true")

    # latency might be empty sometimes, handle safely
    latencies = []
    for r in rows:
        try:
            latencies.append(float(r.get("latency_ms", 0)))
        except:
            pass

    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else None

    return {
        "total_predictions": total,
        "cache_hit_ratio": round((hits / total) * 100, 2),
        "avg_latency_ms": avg_latency,
        "last_5": rows[-5:]
    }

# export a clean dataset file from logs
@app.get("/export/predictions.csv")
def export_predictions_csv():
    if not os.path.exists(PRED_LOG_PATH):
        return JSONResponse(status_code=404, content={"error": "predictions log not found"})

    # Return the file directly (browser downloads it)
    return FileResponse(
        PRED_LOG_PATH,
        media_type="text/csv",
        filename="predictions.csv"
    )

# clean dataset export
@app.get("/export/dataset.csv")
def export_dataset_csv():
    if not os.path.exists(PRED_LOG_PATH):
        return JSONResponse(status_code=404, content={"error": "predictions log not found"})

    out_path = os.path.join(os.path.dirname(PRED_LOG_PATH), "dataset.csv")

    keep_cols = [
        "timestamp",
        "model_version",
        *FEATURE_ORDER,
        "predicted_price",
        "cache_hit",
        "latency_ms",
    ]

    with open(PRED_LOG_PATH, "r") as fin, open(out_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=keep_cols)
        writer.writeheader()

        for r in reader:
            if r.get("error"):          # skip errored rows
                continue
            if not r.get("predicted_price"):
                continue

            clean_row = {k: r.get(k, "") for k in keep_cols}
            writer.writerow(clean_row)

    return FileResponse(out_path, media_type="text/csv", filename="dataset.csv")


@app.post("/v1/predict", response_model=PredictResponse)
def predict(data: HouseFeatures, request:Request):

    rid = request.state.request_id
    start_time = time.time()

    # ✅ FIRST: check if model and scaler are ready (A guard)
    if model is None or scaler is None:
        return JSONResponse(
            status_code=503,
            content=PredictResponse(
                status="error",
                model_version=MODEL_VERSION,
                predicted_price=None,
                error=ApiError(
                    code="NOT_READY",
                    message="Model is not loaded yet. Try again later."
                )
            ).model_dump()
        )
    

    # ✅ # ---- Cache key (same input -> same key) ----
    input_dict = data.model_dump()
    input_str = json.dumps(input_dict, sort_keys=True)
    cache_key = hashlib.sha256(input_str.encode()).hexdigest()

    cached_value = None
    if redis_client is not None:
        cached_value = redis_client.get(cache_key)

    # ✅ Cache hit path
    if cached_value is not None:
        latency_ms = (time.time() - start_time) * 1000

        append_prediction_log({
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": rid,
            "model_version": MODEL_VERSION,
            **input_dict,
            "predicted_price": float(cached_value),
            "cache_hit": True,
            "latency_ms": round(latency_ms, 2),
            "error": ""
        })

        logger.info(f"[{rid}] Cache hit")
        return PredictResponse(
            status="success",
            model_version=MODEL_VERSION,
            predicted_price=float(cached_value),
            error=None
        )

    # ✅ THEN: build features

    # Build feature vector in a fixed, declared order (prevents silent bugs)
    row = {feature: getattr(data, feature) for feature in FEATURE_ORDER}
    X = pd.DataFrame([row], columns=FEATURE_ORDER)

    logger.info(
    f" [{rid}] Predict input | Lat={data.Latitude} Lon={data.Longitude} Pop={data.Population}"
    )

    logger.info(f"[{rid}] Feature order used: {FEATURE_ORDER}")

    # ✅ scale + predict
    X_scaled = scaler.transform(X)
    # Predict using trained model
    pred = model.predict(X_scaled)

    pred_value = float(pred[0])

    # store in cache (expire in 10 minutes) 
    # API still works
    # just no caching
    if redis_client is not None:
        redis_client.setex(cache_key, 600, pred_value)
        logger.info(f"[{rid}] Cache stored")

    
    latency_ms = (time.time() - start_time) * 1000

    append_prediction_log({
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": rid,
        "model_version": MODEL_VERSION,
        **input_dict,
        "predicted_price": pred_value,
        "cache_hit": False,
        "latency_ms": round(latency_ms, 2),
        "error": ""
    })

    logger.info(f"[{rid}] Prediction result | price={pred_value}")

    # Return structured response
    return PredictResponse(
    status="success",
    model_version=MODEL_VERSION,
    predicted_price=float(pred[0]),
    error=None
)
