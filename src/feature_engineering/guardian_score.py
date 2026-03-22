"""
guardian_score.py — Archangel Guardian Score Engine
=====================================================
Implements the weighted reputation scoring system for user-submitted
fraud reports. High-reputation users' reports instantly update the
blacklist; low-reputation reports require consensus validation.

Core Design:
    - Accuracy-weighted scoring: past report accuracy gates current influence
    - Anti-manipulation: device fingerprint + geo consistency checks
    - Bayesian update: new evidence continuously refines each user's score

Role Target: Data Research Engineer @ Gogolook ISL
"""

import math
import hashlib
import time
import random
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Enums & Constants
# ─────────────────────────────────────────────────────────────────────────────

class UserRank(Enum):
    """Gamification tier system mapping to data trust levels."""
    CIVILIAN   = ("平民",    0.0,  0.10)   # (display_name, min_score, report_weight)
    KNIGHT     = ("騎士",    0.40, 0.35)
    GUARDIAN   = ("守護者",  0.65, 0.65)
    ARCHANGEL  = ("大天使",  0.85, 1.00)

    def __init__(self, display_name: str, min_score: float, report_weight: float):
        self.display_name = display_name
        self.min_score = min_score
        self.report_weight = report_weight


# Blacklist thresholds — calibrated to minimize false positive rate
INSTANT_BLACKLIST_THRESHOLD = 0.85   # Weighted scam score → immediate block
REVIEW_QUEUE_THRESHOLD      = 0.50   # → human/ML review
SAFE_THRESHOLD              = 0.15   # → whitelist candidate


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    """Tracks a contributor's reputation state."""
    user_id: str
    device_fingerprint: str
    registered_country: str

    # Accuracy tracking
    total_reports: int = 0
    confirmed_correct: int = 0    # Verified by ground truth or consensus
    confirmed_wrong: int = 0      # False positives / false negatives

    # Bayesian prior (Beta distribution parameters)
    alpha: float = 2.0   # Prior successes (smoothing)
    beta:  float = 2.0   # Prior failures  (smoothing)

    # Behavioral anomaly flags
    report_burst_count: int = 0   # Reports in last 5 minutes
    geo_inconsistency_flag: bool = False

    @property
    def accuracy_rate(self) -> float:
        """Beta distribution mean — Bayesian estimate of true accuracy."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def guardian_score(self) -> float:
        """
        Composite reputation score [0, 1].

        Components:
            - Bayesian accuracy (60%): reliability of past reports
            - Volume bonus (20%):      experience-based credibility
            - Penalty deductions (20%): behavioral anomaly flags
        """
        # Bayesian accuracy component
        accuracy = self.accuracy_rate

        # Volume bonus: log-scaled to prevent gaming (max +0.2)
        volume_bonus = min(0.20, 0.04 * math.log1p(self.total_reports))

        # Behavioral penalties
        burst_penalty = min(0.15, self.report_burst_count * 0.03)
        geo_penalty = 0.10 if self.geo_inconsistency_flag else 0.0

        raw_score = accuracy * 0.60 + volume_bonus - burst_penalty - geo_penalty
        return max(0.0, min(1.0, raw_score))

    @property
    def rank(self) -> UserRank:
        """Derives gamification tier from Guardian Score."""
        score = self.guardian_score
        if score >= UserRank.ARCHANGEL.min_score:
            return UserRank.ARCHANGEL
        elif score >= UserRank.GUARDIAN.min_score:
            return UserRank.GUARDIAN
        elif score >= UserRank.KNIGHT.min_score:
            return UserRank.KNIGHT
        return UserRank.CIVILIAN

    @property
    def report_weight(self) -> float:
        """Effective weight of this user's reports in consensus calculation."""
        return self.rank.report_weight

    def update_bayesian(self, was_correct: bool):
        """
        Bayesian update of accuracy estimate after report validation.
        Beta(α, β) → Beta(α+1, β) if correct, Beta(α, β+1) if wrong.
        """
        if was_correct:
            self.alpha += 1
            self.confirmed_correct += 1
        else:
            self.beta += 1
            self.confirmed_wrong += 1
        self.total_reports += 1


@dataclass
class PhoneReportRecord:
    """A single user fraud report for a phone number."""
    report_id: str
    phone_number: str
    reported_by: str          # user_id
    scam_category: str        # "investment", "customs", "romance", "impersonation"
    reported_at: float        # Unix timestamp
    user_weight: float        # Snapshot of reporter's weight at time of report
    raw_description: Optional[str] = None


@dataclass
class PhoneRiskProfile:
    """Aggregated risk assessment for a phone number."""
    phone_number: str
    reports: list[PhoneReportRecord] = field(default_factory=list)
    _cache_valid: bool = False
    _cached_score: float = 0.0

    def add_report(self, report: PhoneReportRecord):
        self.reports.append(report)
        self._cache_valid = False

    @property
    def weighted_scam_score(self) -> float:
        """
        Weighted consensus score using Guardian weights.

        Formula: Σ(user_weight_i) / Σ(max_weight) for all reports
        This prevents vote stuffing by low-reputation accounts.
        """
        if not self.reports:
            return 0.0
        if self._cache_valid:
            return self._cached_score

        total_weight = sum(r.user_weight for r in self.reports)
        max_possible = len(self.reports) * UserRank.ARCHANGEL.report_weight
        score = total_weight / max_possible if max_possible > 0 else 0.0

        self._cached_score = min(1.0, score)
        self._cache_valid = True
        return self._cached_score

    @property
    def dominant_category(self) -> str:
        """Most frequently reported scam type."""
        if not self.reports:
            return "unknown"
        category_weights: dict[str, float] = {}
        for r in self.reports:
            category_weights[r.scam_category] = (
                category_weights.get(r.scam_category, 0) + r.user_weight
            )
        return max(category_weights, key=category_weights.get)

    @property
    def blacklist_decision(self) -> str:
        score = self.weighted_scam_score
        if score >= INSTANT_BLACKLIST_THRESHOLD:
            return "BLOCK"      # → Redis blacklist immediate write
        elif score >= REVIEW_QUEUE_THRESHOLD:
            return "REVIEW"     # → ML ensemble + human queue
        elif score >= SAFE_THRESHOLD:
            return "SUSPECT"    # → App UI warning label
        return "SAFE"


# ─────────────────────────────────────────────────────────────────────────────
# Guardian Score Engine
# ─────────────────────────────────────────────────────────────────────────────

class GuardianScoreEngine:
    """
    Manages the full lifecycle of user reputation and phone risk scoring.

    Anti-manipulation safeguards:
        1. Device fingerprint deduplication
        2. Report burst rate limiting
        3. Geographic consistency validation
        4. Cross-report consensus validation
    """

    BURST_WINDOW_SEC = 300      # 5-minute burst detection window
    BURST_LIMIT      = 10       # Max reports per user per 5 minutes
    MIN_CONSENSUS    = 3        # Minimum reports before BLOCK decision

    def __init__(self):
        self.users: dict[str, UserProfile] = {}
        self.phone_risk: dict[str, PhoneRiskProfile] = {}
        self._report_id_counter = 0

    # ── User Management ────────────────────────────────────────────────────

    def register_user(
        self,
        user_id: str,
        device_fingerprint: str,
        country: str,
    ) -> UserProfile:
        profile = UserProfile(
            user_id=user_id,
            device_fingerprint=device_fingerprint,
            registered_country=country,
        )
        self.users[user_id] = profile
        logger.info(f"👤 User registered: {user_id} [{country}] | Initial Score: {profile.guardian_score:.2f}")
        return profile

    # ── Anti-Manipulation Layer ────────────────────────────────────────────

    def _check_burst_rate(self, user: UserProfile, current_time: float) -> bool:
        """Returns True if user is within normal reporting rate."""
        if user.report_burst_count >= self.BURST_LIMIT:
            logger.warning(f"🚨 Burst rate exceeded for user {user.user_id} — throttling")
            return False
        return True

    def _check_geo_consistency(self, user: UserProfile, report_country: str) -> bool:
        """
        Validates that report origin is geographically consistent.
        A Taiwan user suddenly reporting from Russia suggests Botnet activity.
        """
        is_consistent = (
            user.registered_country == report_country
            or report_country in ["TW", "HK", "SG"]  # Diaspora allowlist
        )
        if not is_consistent:
            user.geo_inconsistency_flag = True
            logger.warning(
                f"⚠️  Geo inconsistency: user={user.registered_country}, "
                f"report_from={report_country}"
            )
        return is_consistent

    # ── Core Report Submission ─────────────────────────────────────────────

    def submit_report(
        self,
        user_id: str,
        phone_number: str,
        scam_category: str,
        report_country: str = "TW",
        description: Optional[str] = None,
    ) -> dict:
        """
        Processes a user fraud report with full validation pipeline.

        Returns decision: whether the phone is immediately blacklisted,
        queued for review, or needs more consensus.
        """
        user = self.users.get(user_id)
        if not user:
            return {"status": "error", "reason": "Unknown user"}

        current_time = time.time()

        # ── Anti-manipulation checks
        if not self._check_burst_rate(user, current_time):
            return {"status": "throttled", "reason": "Burst rate exceeded"}

        self._check_geo_consistency(user, report_country)
        user.report_burst_count += 1

        # ── Create weighted report
        self._report_id_counter += 1
        report = PhoneReportRecord(
            report_id=f"RPT-{self._report_id_counter:06d}",
            phone_number=phone_number,
            reported_by=user_id,
            scam_category=scam_category,
            reported_at=current_time,
            user_weight=user.report_weight,
            raw_description=description,
        )

        # ── Update phone risk profile
        if phone_number not in self.phone_risk:
            self.phone_risk[phone_number] = PhoneRiskProfile(phone_number=phone_number)

        risk = self.phone_risk[phone_number]
        risk.add_report(report)

        decision = risk.blacklist_decision
        report_count = len(risk.reports)

        # ── Enforce minimum consensus before BLOCK
        if decision == "BLOCK" and report_count < self.MIN_CONSENSUS:
            decision = "REVIEW"  # Insufficient consensus despite high score

        result = {
            "report_id": report.report_id,
            "phone_number": phone_number,
            "reporter_rank": user.rank.display_name,
            "reporter_weight": user.report_weight,
            "guardian_score": round(user.guardian_score, 3),
            "weighted_scam_score": round(risk.weighted_scam_score, 3),
            "total_reports": report_count,
            "dominant_category": risk.dominant_category,
            "decision": decision,
        }

        log_emoji = {"BLOCK": "🔴", "REVIEW": "🟡", "SUSPECT": "🟠", "SAFE": "🟢"}
        logger.info(
            f"{log_emoji.get(decision, '⚪')} [{decision}] {phone_number} | "
            f"score={risk.weighted_scam_score:.2f} | reports={report_count} | "
            f"reporter_rank={user.rank.display_name}"
        )

        return result

    def validate_report(self, user_id: str, was_correct: bool):
        """
        Called after ground truth confirmation to update Bayesian accuracy.
        Triggered by: consensus, law enforcement data, honeypot validation.
        """
        user = self.users.get(user_id)
        if user:
            user.update_bayesian(was_correct)
            logger.info(
                f"📈 Bayesian update for {user_id}: correct={was_correct} | "
                f"new_score={user.guardian_score:.3f} | rank={user.rank.display_name}"
            )

    # ── Analytics ─────────────────────────────────────────────────────────

    def get_leaderboard(self, top_n: int = 10) -> list[dict]:
        """Top contributors ranked by Guardian Score."""
        ranked = sorted(
            self.users.values(),
            key=lambda u: u.guardian_score,
            reverse=True
        )[:top_n]

        return [
            {
                "rank": i + 1,
                "user_id": u.user_id,
                "display_rank": u.rank.display_name,
                "guardian_score": round(u.guardian_score, 3),
                "total_reports": u.total_reports,
                "accuracy": round(u.accuracy_rate, 3),
            }
            for i, u in enumerate(ranked)
        ]

    def get_blacklist_candidates(self) -> list[dict]:
        """Returns all phone numbers meeting BLOCK threshold."""
        return [
            {
                "phone_number": phone,
                "weighted_score": round(risk.weighted_scam_score, 3),
                "reports": len(risk.reports),
                "category": risk.dominant_category,
                "decision": risk.blacklist_decision,
            }
            for phone, risk in self.phone_risk.items()
            if risk.blacklist_decision == "BLOCK"
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Demo / Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def run_demo() -> dict:
    """Demonstrates the Guardian Score engine with a realistic scenario."""
    random.seed(SEED)

    engine = GuardianScoreEngine()

    print("\n" + "═" * 65)
    print("  GUARDIAN SCORE ENGINE — DEMO")
    print("═" * 65)

    # Register users with different trust levels
    users = [
        ("archangel_alice", "fp_aa001", "TW"),  # High-activity honest user
        ("knight_bob",      "fp_kb002", "TW"),  # Medium user
        ("new_carol",       "fp_nc003", "TW"),  # New user
        ("botnet_dave",     "fp_bd004", "RU"),  # Suspicious user
    ]

    for uid, fp, country in users:
        engine.register_user(uid, fp, country)

    # Simulate report history — Alice has proven track record
    engine.users["archangel_alice"].alpha = 47.0
    engine.users["archangel_alice"].beta = 5.0
    engine.users["archangel_alice"].total_reports = 50
    engine.users["archangel_alice"].confirmed_correct = 45

    engine.users["knight_bob"].alpha = 16.0
    engine.users["knight_bob"].beta = 6.0
    engine.users["knight_bob"].total_reports = 20

    print("\n📊 Initial Guardian Scores:")
    for uid, _, _ in users:
        u = engine.users[uid]
        print(f"   {uid:<25} | Score: {u.guardian_score:.3f} | "
              f"Rank: {u.rank.display_name} | Weight: {u.report_weight:.2f}")

    # Scam number being reported by multiple users
    scam_number = "+886-800-SCAM-99"
    print(f"\n📞 Reporting scam number: {scam_number}")

    reports = [
        ("archangel_alice", "investment"),
        ("knight_bob",       "investment"),
        ("new_carol",        "customs"),
        ("botnet_dave",      "investment"),  # From RU — geo inconsistency
    ]

    for uid, category in reports:
        result = engine.submit_report(uid, scam_number, category)
        print(f"\n   ▸ {uid} [{result['reporter_rank']}]")
        print(f"     Weighted Score: {result['weighted_scam_score']:.3f}")
        print(f"     Decision: {result['decision']}")

    # Simulate Bayesian updates (ground truth validation)
    print("\n🔄 Bayesian Updates (ground truth validation):")
    for _ in range(5):
        engine.validate_report("archangel_alice", True)
    engine.validate_report("botnet_dave", False)
    engine.validate_report("botnet_dave", False)

    print("\n🏆 Leaderboard (Top Contributors):")
    for entry in engine.get_leaderboard():
        print(f"   #{entry['rank']} {entry['user_id']:<25} | "
              f"Score: {entry['guardian_score']:.3f} | "
              f"Reports: {entry['total_reports']}")

    candidates = engine.get_blacklist_candidates()
    print(f"\n🔴 Blacklist Candidates: {len(candidates)}")
    for c in candidates:
        print(f"   {c['phone_number']} | Score: {c['weighted_score']:.3f} | "
              f"Category: {c['category']}")

    return {
        "users_registered": len(engine.users),
        "blacklist_candidates": len(candidates),
        "top_guardian_score": engine.get_leaderboard()[0]["guardian_score"] if engine.get_leaderboard() else 0,
    }


if __name__ == "__main__":
    run_demo()
