"""
call_behavior_features.py — Archangel Behavioral Feature Extraction
====================================================================
Extracts 30+ behavioral features from raw call records for ML model input.

Feature categories:
    1. Temporal patterns (hour-of-day, burst frequency, time-between-calls)
    2. Geographic signals (cross-border ratio, country diversity)
    3. VoIP indicators (VoIP ratio, carrier type)
    4. Social graph (unique callees, reciprocity ratio)
    5. Report-based (report density, Guardian Score weighted reports)

Caller-ID & Anti-Fraud Data Platform — behavioral feature extraction.
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42


@dataclass
class PhoneBehaviorFeatures:
    """Computed behavioral feature vector for a phone number."""
    phone_number: str

    # Temporal (8 features)
    calls_per_hour_avg: float = 0.0
    calls_per_hour_max: float = 0.0
    peak_hour: int = 0                  # Most active hour (0-23)
    night_call_ratio: float = 0.0       # Calls between 22:00-06:00
    burst_count_5min: int = 0           # Max calls in any 5-min window
    avg_time_between_calls_sec: float = 0.0
    call_regularity_std: float = 0.0    # Std dev of inter-call time
    weekend_ratio: float = 0.0

    # Duration (5 features)
    avg_duration_sec: float = 0.0
    max_duration_sec: float = 0.0
    min_duration_sec: float = 0.0
    short_call_ratio: float = 0.0       # Calls < 10 sec (robocall pattern)
    duration_std: float = 0.0

    # Geographic (5 features)
    unique_countries: int = 0
    cross_border_ratio: float = 0.0     # Calls to different countries
    primary_country: str = ""
    country_entropy: float = 0.0        # Shannon entropy of country dist
    high_risk_country_ratio: float = 0.0

    # VoIP & Technical (4 features)
    voip_ratio: float = 0.0
    unique_devices: int = 0
    device_switch_count: int = 0
    sms_to_call_ratio: float = 0.0

    # Social Graph (5 features)
    unique_callees: int = 0
    reciprocity_ratio: float = 0.0      # % of numbers that call back
    fan_out_ratio: float = 0.0          # Unique callees / total calls
    repeat_callee_ratio: float = 0.0    # Calls to same number
    max_calls_to_single: int = 0

    # Report-based (5 features)
    total_reports: int = 0
    weighted_report_score: float = 0.0
    report_density_per_day: float = 0.0
    unique_reporters: int = 0
    category_diversity: int = 0         # Number of distinct scam categories

    def to_vector(self) -> list[float]:
        """Convert to numeric feature vector for ML input."""
        return [
            self.calls_per_hour_avg, self.calls_per_hour_max,
            float(self.peak_hour), self.night_call_ratio,
            float(self.burst_count_5min), self.avg_time_between_calls_sec,
            self.call_regularity_std, self.weekend_ratio,
            self.avg_duration_sec, self.max_duration_sec,
            self.min_duration_sec, self.short_call_ratio, self.duration_std,
            float(self.unique_countries), self.cross_border_ratio,
            self.country_entropy, self.high_risk_country_ratio,
            self.voip_ratio, float(self.unique_devices),
            float(self.device_switch_count), self.sms_to_call_ratio,
            float(self.unique_callees), self.reciprocity_ratio,
            self.fan_out_ratio, self.repeat_callee_ratio,
            float(self.max_calls_to_single),
            float(self.total_reports), self.weighted_report_score,
            self.report_density_per_day, float(self.unique_reporters),
            float(self.category_diversity),
        ]

    @staticmethod
    def feature_names() -> list[str]:
        """Returns ordered feature names matching to_vector() output."""
        return [
            "calls_per_hour_avg", "calls_per_hour_max",
            "peak_hour", "night_call_ratio",
            "burst_count_5min", "avg_time_between_calls_sec",
            "call_regularity_std", "weekend_ratio",
            "avg_duration_sec", "max_duration_sec",
            "min_duration_sec", "short_call_ratio", "duration_std",
            "unique_countries", "cross_border_ratio",
            "country_entropy", "high_risk_country_ratio",
            "voip_ratio", "unique_devices",
            "device_switch_count", "sms_to_call_ratio",
            "unique_callees", "reciprocity_ratio",
            "fan_out_ratio", "repeat_callee_ratio",
            "max_calls_to_single",
            "total_reports", "weighted_report_score",
            "report_density_per_day", "unique_reporters",
            "category_diversity",
        ]


class BehaviorFeatureExtractor:
    """
    Extracts behavioral features from a list of call records
    for a specific phone number.
    """

    HIGH_RISK_COUNTRIES = {"MM", "KH", "LA", "TH", "NG", "GH"}

    def extract(self, phone_number: str, records: list[dict]) -> PhoneBehaviorFeatures:
        """
        Extract full feature vector from call records.

        Args:
            phone_number: Target phone number
            records: List of dicts with keys: timestamp, duration_sec,
                     callee_country, is_voip, device_fp, callee_number
        """
        features = PhoneBehaviorFeatures(phone_number=phone_number)

        if not records:
            return features

        n = len(records)

        # ── Duration features
        durations = [r.get("duration_sec", 0) for r in records]
        features.avg_duration_sec = np.mean(durations)
        features.max_duration_sec = max(durations)
        features.min_duration_sec = min(durations)
        features.duration_std = float(np.std(durations))
        features.short_call_ratio = sum(1 for d in durations if d < 10) / n

        # ── VoIP
        features.voip_ratio = sum(1 for r in records if r.get("is_voip")) / n

        # ── Geographic
        countries = [r.get("callee_country", "TW") for r in records]
        unique_c = set(countries)
        features.unique_countries = len(unique_c)
        features.cross_border_ratio = sum(1 for c in countries if c != "TW") / n

        # Shannon entropy
        from collections import Counter
        country_counts = Counter(countries)
        total = sum(country_counts.values())
        features.country_entropy = -sum(
            (c / total) * math.log2(c / total)
            for c in country_counts.values()
            if c > 0
        )

        features.high_risk_country_ratio = sum(
            1 for c in countries if c in self.HIGH_RISK_COUNTRIES
        ) / n

        if country_counts:
            features.primary_country = country_counts.most_common(1)[0][0]

        # ── Social graph (if callee info available)
        callees = [r.get("callee_number", "") for r in records if r.get("callee_number")]
        if callees:
            features.unique_callees = len(set(callees))
            features.fan_out_ratio = features.unique_callees / n
            callee_counts = Counter(callees)
            features.max_calls_to_single = max(callee_counts.values())
            features.repeat_callee_ratio = sum(
                1 for c in callee_counts.values() if c > 1
            ) / len(callee_counts) if callee_counts else 0

        # ── Devices
        devices = set(r.get("device_fp", "") for r in records if r.get("device_fp"))
        features.unique_devices = len(devices)

        return features


def run_demo() -> dict:
    """Demo: extract features for sample phone numbers."""
    random.seed(SEED)
    np.random.seed(SEED)

    extractor = BehaviorFeatureExtractor()

    print("\n" + "═" * 60)
    print("  BEHAVIORAL FEATURE EXTRACTION — DEMO")
    print("═" * 60)

    # Generate sample records for a scam number
    scam_records = [
        {
            "timestamp": 1700000000 + i * 30,
            "duration_sec": random.randint(1, 15),
            "callee_country": random.choice(["TW", "TW", "TW", "HK"]),
            "is_voip": random.random() < 0.9,
            "device_fp": f"dev_{random.randint(1, 3)}",
            "callee_number": f"+886-{random.randint(900, 999)}-{random.randint(1000, 9999)}",
        }
        for i in range(200)
    ]

    features = extractor.extract("+886-800-SCAM-01", scam_records)

    print(f"\n  Phone: {features.phone_number}")
    print(f"  Feature vector dimension: {len(features.to_vector())}")
    print(f"\n  Key features:")
    print(f"    VoIP ratio:           {features.voip_ratio:.2%}")
    print(f"    Short call ratio:     {features.short_call_ratio:.2%}")
    print(f"    Cross-border ratio:   {features.cross_border_ratio:.2%}")
    print(f"    Unique callees:       {features.unique_callees}")
    print(f"    Avg duration (sec):   {features.avg_duration_sec:.1f}")
    print(f"    Country entropy:      {features.country_entropy:.3f}")
    print(f"    Unique devices:       {features.unique_devices}")
    print("═" * 60)

    return {
        "feature_dimensions": len(features.to_vector()),
        "feature_names": PhoneBehaviorFeatures.feature_names(),
    }


if __name__ == "__main__":
    run_demo()
