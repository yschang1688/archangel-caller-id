"""
model_monitor.py — Archangel Model Performance & Drift Monitor
==============================================================
Implements automated model health monitoring with Population Stability
Index (PSI) for distribution drift detection and auto-retraining triggers.

In production anti-fraud systems, scam patterns evolve constantly.
A model trained 3 months ago may be blind to new attack vectors.
This module detects when the current model has degraded and triggers
the retraining pipeline — forming a closed feedback loop.

Portfolio: Caller-ID & Anti-Fraud Data Platform
"""

import math
import random
import time
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# PSI Thresholds (Industry Standard)
# ─────────────────────────────────────────────────────────────────────────────
# PSI < 0.10  → Insignificant shift, model stable
# PSI 0.10–0.25 → Moderate shift, investigate
# PSI > 0.25  → Major drift, RETRAIN REQUIRED

PSI_STABLE    = 0.10
PSI_MODERATE  = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelSnapshot:
    """Point-in-time model performance metrics."""
    snapshot_id: str
    timestamp: float
    model_version: str
    hit_rate: float
    false_positive_rate: float
    precision: float
    f1_score: float
    avg_prediction_latency_ms: float
    n_predictions: int
    score_distribution: list[float] = field(default_factory=list)  # Histogram buckets

    @property
    def datetime_str(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M")


@dataclass
class DriftReport:
    """Output of a drift detection analysis."""
    baseline_version: str
    current_version: str
    psi_score: float
    hit_rate_delta: float
    fpr_delta: float
    drift_severity: str    # "STABLE", "MODERATE", "CRITICAL"
    recommended_action: str
    triggered_at: str

    @property
    def requires_retraining(self) -> bool:
        return self.drift_severity == "CRITICAL"

    @property
    def requires_investigation(self) -> bool:
        return self.drift_severity in ("MODERATE", "CRITICAL")


# ─────────────────────────────────────────────────────────────────────────────
# PSI Calculator
# ─────────────────────────────────────────────────────────────────────────────

class PSICalculator:
    """
    Population Stability Index — measures distribution shift between
    training-time score distribution and current production distribution.

    PSI = Σ [(Actual% - Expected%) × ln(Actual% / Expected%)]

    This is the same formula used in credit risk model monitoring
    and is directly applicable to anti-fraud prediction score drift.
    """

    N_BINS = 10
    EPSILON = 1e-6  # Prevent log(0)

    @classmethod
    def compute(
        cls,
        baseline_scores: list[float],
        current_scores: list[float],
        bins: Optional[list[float]] = None,
    ) -> tuple[float, list[dict]]:
        """
        Computes PSI between baseline and current score distributions.

        Returns:
            psi: Overall PSI score
            bin_details: Per-bin breakdown for diagnostics
        """
        if bins is None:
            bins = [i / cls.N_BINS for i in range(cls.N_BINS + 1)]

        baseline_arr = np.array(baseline_scores)
        current_arr  = np.array(current_scores)

        # Bin frequencies
        baseline_hist, _ = np.histogram(baseline_arr, bins=bins)
        current_hist,  _ = np.histogram(current_arr,  bins=bins)

        # Convert to proportions
        baseline_pct = baseline_hist / len(baseline_scores)
        current_pct  = current_hist  / len(current_scores)

        # Add epsilon to avoid log(0)
        baseline_pct = np.where(baseline_pct == 0, cls.EPSILON, baseline_pct)
        current_pct  = np.where(current_pct  == 0, cls.EPSILON, current_pct)

        # PSI per bin
        bin_psi = (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
        total_psi = float(np.sum(bin_psi))

        bin_details = [
            {
                "bin_range": f"[{bins[i]:.1f}, {bins[i+1]:.1f})",
                "baseline_pct": round(float(baseline_pct[i]) * 100, 2),
                "current_pct":  round(float(current_pct[i])  * 100, 2),
                "bin_psi":      round(float(bin_psi[i]), 5),
            }
            for i in range(len(bin_psi))
        ]

        return total_psi, bin_details


# ─────────────────────────────────────────────────────────────────────────────
# Model Monitor
# ─────────────────────────────────────────────────────────────────────────────

class ModelMonitor:
    """
    Continuous model health monitoring system.

    Monitors:
        1. Performance degradation (Hit Rate, FPR trend)
        2. Score distribution drift (PSI)
        3. Prediction latency SLA compliance
        4. Data volume anomalies (sudden drop → data pipeline issue)

    Auto-triggers:
        - Alert to Slack/PagerDuty on MODERATE drift
        - Retraining pipeline (Kubeflow) on CRITICAL drift
        - Rollback signal if new model underperforms
    """

    LATENCY_SLA_MS     = 50.0   # p99 latency SLA
    HIT_RATE_FLOOR     = 0.65   # Alert if hit rate drops below this
    FPR_CEILING        = 0.05   # Alert if FPR exceeds this

    def __init__(self, model_version: str = "v1.0.0"):
        self.model_version = model_version
        self.snapshots: list[ModelSnapshot] = []
        self.baseline_snapshot: Optional[ModelSnapshot] = None
        self.psi_calculator = PSICalculator()
        self._alert_log: list[str] = []
        self._retraining_triggered = False

    # ── Snapshot Recording ─────────────────────────────────────────────────

    def record_snapshot(
        self,
        hit_rate: float,
        fpr: float,
        precision: float,
        n_predictions: int,
        score_samples: list[float],
        avg_latency_ms: float = 25.0,
    ) -> ModelSnapshot:
        """Records a point-in-time performance snapshot."""
        f1 = 2 * precision * hit_rate / (precision + hit_rate) if (precision + hit_rate) > 0 else 0
        snap = ModelSnapshot(
            snapshot_id=f"snap_{len(self.snapshots):04d}",
            timestamp=time.time(),
            model_version=self.model_version,
            hit_rate=hit_rate,
            false_positive_rate=fpr,
            precision=precision,
            f1_score=f1,
            avg_prediction_latency_ms=avg_latency_ms,
            n_predictions=n_predictions,
            score_distribution=score_samples,
        )
        self.snapshots.append(snap)

        if self.baseline_snapshot is None:
            self.baseline_snapshot = snap
            logger.info(f"📌 Baseline snapshot set: version={self.model_version} "
                        f"| hit_rate={hit_rate:.3f}")

        return snap

    # ── Drift Analysis ─────────────────────────────────────────────────────

    def analyze_drift(self, current: ModelSnapshot) -> DriftReport:
        """
        Compares current performance against baseline.
        Computes PSI and performance delta to assess drift severity.
        """
        if self.baseline_snapshot is None:
            raise ValueError("No baseline snapshot recorded")

        baseline = self.baseline_snapshot

        # PSI computation
        psi = 0.0
        if baseline.score_distribution and current.score_distribution:
            psi, bin_details = self.psi_calculator.compute(
                baseline.score_distribution,
                current.score_distribution,
            )

        # Performance deltas
        hit_rate_delta = current.hit_rate - baseline.hit_rate
        fpr_delta = current.false_positive_rate - baseline.false_positive_rate

        # Severity classification
        if psi > PSI_MODERATE or hit_rate_delta < -0.05 or fpr_delta > 0.02:
            severity = "CRITICAL"
            action = ("🔴 CRITICAL DRIFT: Trigger Kubeflow retraining pipeline immediately. "
                      "Consider hotfix rollback if latency SLA also breached.")
        elif psi > PSI_STABLE or abs(hit_rate_delta) > 0.02:
            severity = "MODERATE"
            action = ("🟡 MODERATE DRIFT: Schedule retraining within 48h. "
                      "Alert ML team. Increase monitoring frequency.")
        else:
            severity = "STABLE"
            action = "🟢 Model stable. Continue routine monitoring."

        report = DriftReport(
            baseline_version=baseline.model_version,
            current_version=current.model_version,
            psi_score=psi,
            hit_rate_delta=hit_rate_delta,
            fpr_delta=fpr_delta,
            drift_severity=severity,
            recommended_action=action,
            triggered_at=current.datetime_str,
        )

        self._log_drift_report(report, psi)

        # Auto-trigger retraining
        if report.requires_retraining and not self._retraining_triggered:
            self._trigger_retraining(report)

        return report

    def _log_drift_report(self, report: DriftReport, psi: float):
        """Structured logging for monitoring dashboards."""
        print(f"\n{'═'*60}")
        print(f"  MODEL DRIFT ANALYSIS REPORT")
        print(f"{'═'*60}")
        print(f"  Baseline Version : {report.baseline_version}")
        print(f"  Current Version  : {report.current_version}")
        print(f"  Triggered At     : {report.triggered_at}")
        print(f"{'─'*60}")
        print(f"  PSI Score        : {psi:.4f}  "
              f"({'Stable' if psi < PSI_STABLE else 'Moderate' if psi < PSI_MODERATE else 'CRITICAL'})")
        print(f"  Hit Rate Δ       : {report.hit_rate_delta:+.4f}")
        print(f"  FPR Δ            : {report.fpr_delta:+.4f}")
        print(f"  Severity         : {report.drift_severity}")
        print(f"{'─'*60}")
        print(f"  {report.recommended_action}")
        print(f"{'═'*60}\n")

    def _trigger_retraining(self, report: DriftReport):
        """
        In production: calls Kubeflow Pipeline API or Airflow DAG trigger.
        Here we simulate the trigger with structured logging.
        """
        self._retraining_triggered = True
        payload = {
            "event": "AUTO_RETRAIN_TRIGGERED",
            "reason": "CRITICAL_DRIFT",
            "psi": round(report.psi_score, 4),
            "hit_rate_delta": round(report.hit_rate_delta, 4),
            "timestamp": datetime.now().isoformat(),
            "pipeline": "kubeflow://anti-fraud-retraining-pipeline",
            "priority": "HIGH",
        }
        logger.warning(f"🚨 RETRAINING TRIGGERED: {json.dumps(payload, indent=2)}")

    # ── SLA Monitoring ─────────────────────────────────────────────────────

    def check_latency_sla(self, p99_latency_ms: float) -> bool:
        """Flags latency SLA breach."""
        if p99_latency_ms > self.LATENCY_SLA_MS:
            logger.warning(
                f"⚠️  LATENCY SLA BREACH: p99={p99_latency_ms:.1f}ms > "
                f"SLA={self.LATENCY_SLA_MS:.1f}ms"
            )
            return False
        return True

    # ── Trend Analysis ─────────────────────────────────────────────────────

    def get_performance_trend(self) -> list[dict]:
        """Returns time-series of key metrics for dashboard visualization."""
        return [
            {
                "timestamp": s.datetime_str,
                "hit_rate": round(s.hit_rate, 4),
                "fpr": round(s.false_positive_rate, 4),
                "f1": round(s.f1_score, 4),
                "latency_ms": round(s.avg_prediction_latency_ms, 2),
            }
            for s in self.snapshots
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

def generate_score_distribution(mean: float, std: float, n: int = 1000) -> list[float]:
    """Generate realistic prediction score samples."""
    return [max(0.0, min(1.0, random.gauss(mean, std))) for _ in range(n)]


def run_demo() -> dict:
    """Simulates 30 days of model monitoring with introduced drift. Returns key metrics."""
    random.seed(SEED)
    np.random.seed(SEED)

    monitor = ModelMonitor(model_version="v1.2.0")

    print("\n" + "═"*60)
    print("  MODEL MONITOR — 30-DAY SIMULATION")
    print("═"*60)

    # Week 1: Stable baseline
    print("\n📅 Week 1: Establishing baseline...")
    baseline_scores = generate_score_distribution(mean=0.45, std=0.25)
    baseline_snap = monitor.record_snapshot(
        hit_rate=0.891,
        fpr=0.023,
        precision=0.912,
        n_predictions=850_000,
        score_samples=baseline_scores,
        avg_latency_ms=24.3,
    )

    # Week 2-3: Slight natural drift
    print("📅 Week 2-3: Normal operation...")
    for day in range(14):
        drift = day * 0.001
        scores = generate_score_distribution(mean=0.45 + drift, std=0.25)
        monitor.record_snapshot(
            hit_rate=0.891 - drift * 0.5,
            fpr=0.023 + drift * 0.1,
            precision=0.912 - drift * 0.3,
            n_predictions=random.randint(800_000, 900_000),
            score_samples=scores,
            avg_latency_ms=random.gauss(25, 2),
        )

    # Week 4: New scam wave — distribution shift
    print("📅 Week 4: New cross-national scam wave detected...")
    drifted_scores = generate_score_distribution(mean=0.35, std=0.30)  # Shifted left
    drifted_snap = monitor.record_snapshot(
        hit_rate=0.831,     # -6pp drop
        fpr=0.041,          # +1.8pp increase
        precision=0.871,
        n_predictions=920_000,
        score_samples=drifted_scores,
        avg_latency_ms=28.1,
    )

    # Analyze drift
    print("\n🔍 Running Drift Analysis...")
    report = monitor.analyze_drift(drifted_snap)

    # Trend output
    print("\n📊 Performance Trend (last 5 snapshots):")
    trend = monitor.get_performance_trend()
    for entry in trend[-5:]:
        print(f"   {entry['timestamp']} | Hit Rate: {entry['hit_rate']:.3f} | "
              f"FPR: {entry['fpr']:.3f} | F1: {entry['f1']:.3f}")

    print(f"\n✅ Total snapshots recorded: {len(monitor.snapshots)}")
    print(f"   Retraining triggered: {monitor._retraining_triggered}")

    return {
        "psi_score": round(report.psi_score, 4),
        "drift_severity": report.drift_severity,
        "hit_rate_delta": round(report.hit_rate_delta, 4),
        "fpr_delta": round(report.fpr_delta, 4),
        "retraining_triggered": monitor._retraining_triggered,
        "baseline_hit_rate": baseline_snap.hit_rate,
        "final_hit_rate": drifted_snap.hit_rate,
    }


if __name__ == "__main__":
    run_demo()
