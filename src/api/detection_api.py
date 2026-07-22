"""
detection_api.py — Archangel Anti-Fraud Detection API
======================================================
FastAPI-based REST API for real-time fraud detection and reporting.

Endpoints:
    POST /api/v1/predict   — Predict fraud probability for a phone number
    POST /api/v1/report    — Submit a fraud report (Guardian Score weighted)
    GET  /api/v1/blacklist — Get current blacklist candidates
    GET  /api/v1/health    — Service health check
    GET  /api/v1/stats     — Pipeline statistics

Portfolio: Caller-ID & Anti-Fraud Data Platform
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import time

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

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_sec: float
    timestamp: str

class StatsResponse(BaseModel):
    total_predictions: int
    total_reports: int
    blacklisted_numbers: int
    avg_latency_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# In-memory State (production: Redis + DB)
# ─────────────────────────────────────────────────────────────────────────────

_start_time = time.time()
_prediction_count = 0
_total_latency_ms = 0.0

# Lazy-import Guardian Score engine
_guardian_engine = None

def _get_guardian_engine():
    global _guardian_engine
    if _guardian_engine is None:
        from src.feature_engineering.guardian_score import GuardianScoreEngine
        _guardian_engine = GuardianScoreEngine()
        # Register a demo user
        _guardian_engine.register_user("demo_user", "fp_demo", "TW")
    return _guardian_engine


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
    """Service health check with uptime."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_sec=round(time.time() - _start_time, 2),
        timestamp=datetime.now().isoformat(),
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
    avg_latency = _total_latency_ms / _prediction_count if _prediction_count > 0 else 0

    return StatsResponse(
        total_predictions=_prediction_count,
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
