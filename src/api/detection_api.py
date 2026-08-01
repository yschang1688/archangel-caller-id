"""
detection_api.py — Archangel Anti-Fraud Detection API
======================================================
FastAPI-based REST API for real-time fraud detection and reporting.

Endpoints:
    POST /api/v1/predict        — Heuristic fraud scoring (fallback)
    POST /api/v1/predict_model  — ML model-based fraud prediction
    POST /api/v1/report         — Submit a fraud report (Guardian Score weighted)
    GET  /api/v1/blacklist      — Get current blacklist candidates
    GET  /api/v1/health         — Service health check
    GET  /api/v1/stats          — Pipeline statistics

Role Target: Data Research Engineer @ Gogolook ISL
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import time
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

app = FastAPI(
    title="🛡️ Archangel Anti-Fraud API",
    description="Real-time fraud detection and Guardian Score reporting system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────

class FraudPredictionRequest(BaseModel):
    phone_number: str = Field(..., example="+886-800-1234")
    call_duration_sec: int = Field(ge=0, example=45)
    is_voip: bool = Field(default=False, example=True)
    caller_country: str = Field(default="TW", example="TW")
    report_count: int = Field(ge=0, example=12)

class FraudPredictionResponse(BaseModel):
    phone_number: str
    fraud_probability: float
    risk_level: str  # HIGH / MEDIUM / LOW / SAFE
    recommendation: str
    latency_ms: float

class ReportRequest(BaseModel):
    user_id: str = Field(..., example="user_001")
    phone_number: str = Field(..., example="+886-800-SCAM-01")
    scam_category: str = Field(..., example="investment")
    description: Optional[str] = Field(None, example="假冒銀行來電要求轉帳")

class ReportResponse(BaseModel):
    report_id: str
    phone_number: str
    reporter_weight: float
    weighted_scam_score: float
    decision: str
    total_reports: int

class ModelPredictionRequest(BaseModel):
    """Feature vector for ML model-based prediction."""
    features: List[float] = Field(
        ...,
        description="Numeric feature vector matching training schema",
        example=[0.5, -0.2, 1.3, 0.0, 0.8, -1.1, 0.3, 0.0, 1.0, 0.0],
    )

class ModelPredictionResponse(BaseModel):
    fraud_probability: float
    predicted_label: int
    risk_level: str
    model_name: str
    feature_count: int
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_sec: float
    timestamp: str
    model_loaded: bool

class StatsResponse(BaseModel):
    total_predictions: int
    total_model_predictions: int
    total_reports: int
    blacklisted_numbers: int
    avg_latency_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# In-memory State (production: Redis + DB)
# ─────────────────────────────────────────────────────────────────────────────

_start_time = time.time()
_prediction_count = 0
_model_prediction_count = 0
_total_latency_ms = 0.0

# Lazy-import Guardian Score engine
_guardian_engine = None

# ML model artifacts (lazy-loaded)
_ml_model = None
_ml_scaler = None
_ml_feature_names = None
_model_load_attempted = False

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')


def _get_guardian_engine():
    global _guardian_engine
    if _guardian_engine is None:
        from src.feature_engineering.guardian_score import GuardianScoreEngine
        _guardian_engine = GuardianScoreEngine()
        # Register a demo user
        _guardian_engine.register_user("demo_user", "fp_demo", "TW")
    return _guardian_engine


def _load_ml_model():
    """Lazy-load serialized ML model, scaler, and feature names."""
    global _ml_model, _ml_scaler, _ml_feature_names, _model_load_attempted
    if _model_load_attempted:
        return _ml_model is not None
    _model_load_attempted = True

    try:
        import joblib
        model_path = os.path.join(MODEL_DIR, 'xgboost_spam_model.pkl')
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        features_path = os.path.join(MODEL_DIR, 'feature_names.pkl')

        if os.path.exists(model_path):
            _ml_model = joblib.load(model_path)
            logger.info(f"[API] ML model loaded from {model_path}")
        else:
            logger.warning(f"[API] Model not found at {model_path}")
            return False

        if os.path.exists(scaler_path):
            _ml_scaler = joblib.load(scaler_path)

        if os.path.exists(features_path):
            _ml_feature_names = joblib.load(features_path)

        return True
    except Exception as e:
        logger.error(f"[API] Failed to load model: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Risk Scoring Logic
# ─────────────────────────────────────────────────────────────────────────────

def _compute_risk(req: FraudPredictionRequest) -> tuple[float, str]:
    """
    Heuristic risk scoring (production: ML model inference).

    Combines:
      - VoIP flag (scam centers heavily use VoIP)
      - Report count (crowd-sourced signal)
      - Call duration pattern (very short = robocall, medium = social engineering)
    """
    score = 0.0

    # VoIP is a strong scam indicator
    if req.is_voip:
        score += 0.35

    # Report count signal (log-scaled)
    import math
    if req.report_count > 0:
        score += min(0.40, 0.1 * math.log1p(req.report_count))

    # Call duration pattern
    if req.call_duration_sec < 5:
        score += 0.10  # Robocall pattern
    elif 30 < req.call_duration_sec < 180:
        score += 0.05  # Social engineering window

    # Foreign caller boost
    if req.caller_country not in ("TW", "HK", "SG", "JP"):
        score += 0.15

    score = min(1.0, score)

    if score >= 0.7:
        level = "HIGH"
    elif score >= 0.4:
        level = "MEDIUM"
    elif score >= 0.15:
        level = "LOW"
    else:
        level = "SAFE"

    return score, level


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Service health check with uptime and model status."""
    model_ok = _load_ml_model()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_sec=round(time.time() - _start_time, 2),
        timestamp=datetime.now().isoformat(),
        model_loaded=model_ok,
    )


@app.post("/api/v1/predict", response_model=FraudPredictionResponse)
async def predict_fraud(req: FraudPredictionRequest):
    """
    Predict fraud probability for a phone number.

    Uses heuristic scoring in demo mode.
    Production: XGBoost model inference with <50ms p99 latency.
    """
    global _prediction_count, _total_latency_ms

    t0 = time.perf_counter()

    score, level = _compute_risk(req)

    recommendations = {
        "HIGH": "🔴 Immediate block recommended. Add to Redis blacklist.",
        "MEDIUM": "🟡 Flag for human review. Monitor call patterns.",
        "LOW": "🟠 Display warning label to user.",
        "SAFE": "🟢 No action needed.",
    }

    latency_ms = (time.perf_counter() - t0) * 1000
    _prediction_count += 1
    _total_latency_ms += latency_ms

    return FraudPredictionResponse(
        phone_number=req.phone_number,
        fraud_probability=round(score, 4),
        risk_level=level,
        recommendation=recommendations[level],
        latency_ms=round(latency_ms, 3),
    )


@app.post("/api/v1/predict_model", response_model=ModelPredictionResponse)
async def predict_model(req: ModelPredictionRequest):
    """
    ML model-based fraud prediction.

    Accepts a numeric feature vector and returns real model inference.
    Requires `models/xgboost_spam_model.pkl` (generated by Notebook 02).

    Falls back to error if model is not serialized yet.
    """
    global _model_prediction_count, _total_latency_ms

    if not _load_ml_model():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run notebook 02 or run_ml_dev.py first to serialize the model.",
        )

    t0 = time.perf_counter()

    features = np.array(req.features).reshape(1, -1)

    # Validate feature count
    if _ml_feature_names and features.shape[1] != len(_ml_feature_names):
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(_ml_feature_names)} features, got {features.shape[1]}. "
                   f"Feature names: {_ml_feature_names[:5]}...",
        )

    proba = _ml_model.predict_proba(features)[0, 1]
    label = int(proba >= 0.5)

    if proba >= 0.7:
        level = "HIGH"
    elif proba >= 0.4:
        level = "MEDIUM"
    elif proba >= 0.15:
        level = "LOW"
    else:
        level = "SAFE"

    latency_ms = (time.perf_counter() - t0) * 1000
    _model_prediction_count += 1
    _total_latency_ms += latency_ms

    return ModelPredictionResponse(
        fraud_probability=round(float(proba), 4),
        predicted_label=label,
        risk_level=level,
        model_name=type(_ml_model).__name__,
        feature_count=features.shape[1],
        latency_ms=round(latency_ms, 3),
    )


@app.post("/api/v1/report", response_model=ReportResponse)
async def submit_report(req: ReportRequest):
    """
    Submit a fraud report with Guardian Score weighting.

    The reporter's reputation affects the weight of their report
    in the consensus-based blacklist decision.
    """
    engine = _get_guardian_engine()

    # Ensure user exists
    if req.user_id not in engine.users:
        engine.register_user(req.user_id, f"fp_{req.user_id}", "TW")

    result = engine.submit_report(
        user_id=req.user_id,
        phone_number=req.phone_number,
        scam_category=req.scam_category,
        description=req.description,
    )

    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result["reason"])

    return ReportResponse(
        report_id=result["report_id"],
        phone_number=result["phone_number"],
        reporter_weight=result["reporter_weight"],
        weighted_scam_score=result["weighted_scam_score"],
        decision=result["decision"],
        total_reports=result["total_reports"],
    )


@app.get("/api/v1/blacklist")
async def get_blacklist():
    """Get current blacklist candidates based on Guardian Score consensus."""
    engine = _get_guardian_engine()
    candidates = engine.get_blacklist_candidates()
    return {
        "count": len(candidates),
        "candidates": candidates,
    }


@app.get("/api/v1/stats", response_model=StatsResponse)
async def get_stats():
    """Pipeline statistics."""
    engine = _get_guardian_engine()
    total = _prediction_count + _model_prediction_count
    avg_latency = _total_latency_ms / total if total > 0 else 0

    return StatsResponse(
        total_predictions=_prediction_count,
        total_model_predictions=_model_prediction_count,
        total_reports=sum(len(r.reports) for r in engine.phone_risk.values()),
        blacklisted_numbers=len(engine.get_blacklist_candidates()),
        avg_latency_ms=round(avg_latency, 3),
    )


@app.get("/")
async def root():
    """API root — redirect hint."""
    return {
        "message": "🛡️ Archangel Anti-Fraud API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
