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

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

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
    """
    Runs once when the FastAPI server starts.
    Loads ML artifacts into memory so requests are fast.
    """
    global model, scaler, redis_client

    try:
        logger.info("Loading model and scaler...")
        model = joblib.load(BASE_DIR / "model" / "model.joblib")
        scaler = joblib.load(BASE_DIR / "model" / "scaler.joblib")
        logger.info("✅ Model and scaler loaded successfully")
    except Exception as e:
        # Keep them as None so /ready returns 503
        model = None
        scaler = None
        logger.exception(f"❌ Failed to load artifacts: {e}")

    # ---- Connect to Redis ----
    try:
        logger.info("Connecting to Redis...")
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True
        )

        redis_client.ping()   # test connection
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


@app.post("/v1/predict", response_model=PredictResponse)
def predict(data: HouseFeatures, request:Request):

    rid = request.state.request_id

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

    if cached_value is not None:
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

    logger.info(f"[{rid}] Prediction result | price={float(pred[0])}")

    # Return structured response
    return PredictResponse(
    status="success",
    model_version=MODEL_VERSION,
    predicted_price=float(pred[0]),
    error=None
)
