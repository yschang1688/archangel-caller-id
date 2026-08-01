"""
spark_etl.py — Archangel Anti-Fraud Batch ETL Pipeline
=======================================================
CORE DEMONSTRATION: Data Skew handling via Salting Technique

In global anti-fraud scenarios, certain phone numbers (e.g., scam call centers)
generate extreme query hotspots — a classic Data Skew problem. This module
demonstrates production-grade solutions.

Role Target: Data Research Engineer @ Gogolook ISL
"""

import hashlib
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import logging

# In production: from pyspark.sql import SparkSession, functions as F
# This module uses native Python to demonstrate the algorithm logic
# identically to how it would operate in a Spark distributed environment.

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CallRecord:
    """Represents a single call event ingested from Kafka."""
    call_id: str
    phone_number: str       # May be a hotkey (scam center) → causes skew
    caller_country: str
    callee_country: str
    call_duration_sec: int
    timestamp: float
    is_voip: bool
    reported_as_scam: bool = False
    guardian_score_weight: float = 1.0


@dataclass
class PartitionStats:
    """Tracks partition load for skew detection."""
    partition_id: int
    record_count: int = 0
    processing_time_ms: float = 0.0

    @property
    def load_ratio(self) -> float:
        """Ratio vs. ideal even distribution."""
        return self.record_count  # Will be normalized by caller


# ─────────────────────────────────────────────────────────────────────────────
# ⭐ CORE: Salting Technique for Data Skew Mitigation
# ─────────────────────────────────────────────────────────────────────────────

class SparkSkewHandler:
    """
    Implements the Salting Technique to resolve Data Skew in Spark.

    Problem:
        In anti-fraud pipelines, certain phone numbers (hot keys) generate
        millions of queries — e.g., a scam call center in Myanmar floods
        one Spark partition while others sit idle. This causes:
        - Stage stragglers → entire job blocked by 1 slow task
        - OOM errors on overloaded executor nodes
        - 10-100x slower batch completion

    Solution:
        1. Pre-analysis: identify hot keys (call_count > threshold)
        2. Salting: append random suffix [0, SALT_FACTOR) to hot key values
        3. Distribute: one logical group now spans SALT_FACTOR partitions
        4. Partial aggregate: reduce within each salted partition
        5. Final aggregate: strip salt suffix, merge partial results

    Spark SQL equivalent:
        -- Step 1: Salt hot keys
        SELECT CONCAT(phone_number, '_', FLOOR(RAND() * 32)) AS salted_key, ...
        FROM call_records WHERE phone_number IN (SELECT phone FROM hot_keys)

        -- Step 2: Partial aggregation
        GROUP BY salted_key

        -- Step 3: Final merge
        SELECT REGEXP_REPLACE(salted_key, '_[0-9]+$', '') AS phone_number,
               SUM(partial_count) AS total_count
        FROM partial_results GROUP BY phone_number
    """

    SALT_FACTOR: int = 32          # Number of virtual partitions per hot key
    HOT_KEY_THRESHOLD: int = 1000  # Min records to classify as hot key
    SKEW_RATIO_ALERT: float = 5.0  # Alert if max/avg partition ratio > 5x

    def __init__(self):
        self.hot_keys: set[str] = set()
        self.partition_stats: dict[int, PartitionStats] = {}
        self._detected_skew = False
        self._pre_salt_skew_ratio: float = 0.0

    # ── Step 1: Hot Key Detection ──────────────────────────────────────────

    def detect_hot_keys(self, records: list[CallRecord]) -> dict[str, int]:
        """
        Identifies phone numbers generating extreme data skew.

        In Spark: spark.sql("SELECT phone_number, COUNT(*) as cnt
                             FROM records GROUP BY phone_number
                             ORDER BY cnt DESC LIMIT 100")
        """
        frequency_map: dict[str, int] = defaultdict(int)

        for record in records:
            frequency_map[record.phone_number] += 1

        hot_keys = {
            phone: count
            for phone, count in frequency_map.items()
            if count >= self.HOT_KEY_THRESHOLD
        }

        self.hot_keys = set(hot_keys.keys())

        # Skew severity analysis
        if frequency_map:
            counts = list(frequency_map.values())
            avg_count = sum(counts) / len(counts)
            max_count = max(counts)
            skew_ratio = max_count / avg_count if avg_count > 0 else 1
            self._pre_salt_skew_ratio = skew_ratio

            if skew_ratio > self.SKEW_RATIO_ALERT:
                self._detected_skew = True
                logger.warning(
                    f"⚠️  DATA SKEW DETECTED: max/avg ratio = {skew_ratio:.1f}x | "
                    f"Hot keys found: {len(hot_keys)} | "
                    f"Activating Salting Technique..."
                )
            else:
                logger.info(f"✅ Distribution healthy: skew ratio = {skew_ratio:.1f}x")

        return hot_keys

    # ── Step 2: Apply Salt to Hot Key Records ─────────────────────────────

    def apply_salt(self, record: CallRecord) -> str:
        """
        Appends random salt suffix to hot key phone numbers.

        Before salting: "+886-800-000-001" → maps to 1 partition
        After salting:  "+886-800-000-001_17" → maps to partition 17 of 32
                        "+886-800-000-001_03" → maps to partition 3 of 32
        """
        if record.phone_number in self.hot_keys:
            salt = random.randint(0, self.SALT_FACTOR - 1)
            return f"{record.phone_number}_{salt:02d}"
        return record.phone_number

    # ── Step 3: Simulate Distributed Partition Processing ─────────────────

    def partition_and_process(
        self,
        records: list[CallRecord],
        n_partitions: int = 32
    ) -> dict[str, dict]:
        """
        Simulates Spark's partition-based parallel processing.

        Returns partial aggregation results per salted key,
        ready for final reduce step.
        """
        # Initialize partition tracking
        partitions: dict[int, list] = defaultdict(list)

        for record in records:
            salted_key = self.apply_salt(record)
            # Deterministic hash partitioning (mirrors Spark's HashPartitioner)
            partition_id = int(hashlib.md5(salted_key.encode()).hexdigest(), 16) % n_partitions
            partitions[partition_id].append((salted_key, record))

        # Partial aggregation within each partition
        partial_results: dict[str, dict] = {}

        for partition_id, partition_data in partitions.items():
            start = time.perf_counter()

            for salted_key, record in partition_data:
                if salted_key not in partial_results:
                    partial_results[salted_key] = {
                        "count": 0,
                        "scam_reports": 0,
                        "total_duration": 0,
                        "voip_count": 0,
                        "weighted_scam_score": 0.0,
                    }
                r = partial_results[salted_key]
                r["count"] += 1
                r["total_duration"] += record.call_duration_sec
                if record.reported_as_scam:
                    r["scam_reports"] += 1
                    r["weighted_scam_score"] += record.guardian_score_weight
                if record.is_voip:
                    r["voip_count"] += 1

            elapsed_ms = (time.perf_counter() - start) * 1000
            self.partition_stats[partition_id] = PartitionStats(
                partition_id=partition_id,
                record_count=len(partition_data),
                processing_time_ms=elapsed_ms,
            )

        return partial_results

    # ── Step 4: Final Aggregation — Strip Salt & Merge ────────────────────

    def final_aggregate(self, partial_results: dict[str, dict]) -> dict[str, dict]:
        """
        Strips salt suffix and merges partial results into final phone-level stats.

        This is the equivalent of a second GROUP BY in Spark after salting.
        The extra shuffle is a worthwhile tradeoff vs. straggler-blocked stages.
        """
        final: dict[str, dict] = {}

        for salted_key, stats in partial_results.items():
            # Strip salt: "+886-800-000-001_17" → "+886-800-000-001"
            parts = salted_key.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                original_key = parts[0]
            else:
                original_key = salted_key

            if original_key not in final:
                final[original_key] = {
                    "count": 0, "scam_reports": 0,
                    "total_duration": 0, "voip_count": 0,
                    "weighted_scam_score": 0.0,
                }
            for key in ("count", "scam_reports", "total_duration",
                        "voip_count", "weighted_scam_score"):
                final[original_key][key] += stats[key]

        # Compute derived metrics
        for phone, stats in final.items():
            n = stats["count"]
            stats["scam_report_rate"] = stats["scam_reports"] / n if n > 0 else 0
            stats["avg_duration_sec"] = stats["total_duration"] / n if n > 0 else 0
            stats["voip_ratio"] = stats["voip_count"] / n if n > 0 else 0
            # Risk classification based on weighted score + report rate
            stats["risk_level"] = self._classify_risk(stats)

        return final

    def _classify_risk(self, stats: dict) -> str:
        """Business rule: risk classification for blacklist decision."""
        rate = stats["scam_report_rate"]
        weighted = stats["weighted_scam_score"]
        is_voip_dominant = stats["voip_ratio"] > 0.8

        if rate >= 0.7 or (weighted >= 50 and is_voip_dominant):
            return "HIGH"      # → Immediate Redis blacklist
        elif rate >= 0.3 or weighted >= 20:
            return "MEDIUM"    # → Flagged for human review
        elif rate >= 0.1:
            return "LOW"       # → Warn user but allow call
        return "SAFE"

    # ── Performance Report ─────────────────────────────────────────────────

    def get_skew_report(self, n_partitions: int = 32) -> dict:
        """Returns before/after skew metrics."""
        if not self.partition_stats:
            return {}

        counts = [s.record_count for s in self.partition_stats.values()]
        if not counts:
            return {}

        avg = sum(counts) / len(counts)
        max_count = max(counts)
        min_count = min(counts)
        post_salt_ratio = max_count / avg if avg > 0 else 0

        return {
            "active_partitions": len(self.partition_stats),
            "total_partitions": n_partitions,
            "avg_records_per_partition": round(avg, 0),
            "max_records_per_partition": max_count,
            "min_records_per_partition": min_count,
            "pre_salt_skew_ratio": round(self._pre_salt_skew_ratio, 2),
            "post_salt_skew_ratio": round(post_salt_ratio, 2),
            "hot_keys_salted": len(self.hot_keys),
        }

    def print_skew_report(self, n_partitions: int = 32):
        """Prints before/after comparison of partition load distribution."""
        report = self.get_skew_report(n_partitions)
        if not report:
            return

        print("\n" + "═" * 60)
        print("  PARTITION LOAD DISTRIBUTION REPORT (After Salting)")
        print("═" * 60)
        print(f"  Active partitions : {report['active_partitions']} / {report['total_partitions']}")
        print(f"  Avg records/part  : {report['avg_records_per_partition']:.0f}")
        print(f"  Max records/part  : {report['max_records_per_partition']}")
        print(f"  Min records/part  : {report['min_records_per_partition']}")
        print(f"  Pre-salt skew     : {report['pre_salt_skew_ratio']:.2f}x")
        print(f"  Post-salt skew    : {report['post_salt_skew_ratio']:.2f}x  (target: <5x)")
        print(f"  Hot keys salted   : {report['hot_keys_salted']}")
        print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ETL Pipeline Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class AntiFraudETL:
    """
    Orchestrates the complete batch ETL pipeline:
    Raw Call Records → Feature Store → Risk Classification → Blacklist Update
    """

    def __init__(self):
        self.skew_handler = SparkSkewHandler()

    def generate_synthetic_data(self, n_records: int = 50_000) -> list[CallRecord]:
        """
        Generates realistic synthetic call data with intentional skew.
        Simulates 3 scam call centers generating 60% of all traffic.
        """
        countries = ["TW", "US", "UK", "JP", "SG", "AU", "HK", "DE"]
        scam_centers = ["+886-800-SCAM-01", "+886-800-SCAM-02", "+886-800-SCAM-03"]
        normal_pool = [f"+{random.randint(1,99)}-{random.randint(100,999)}-{random.randint(1000,9999)}"
                       for _ in range(500)]

        records = []
        for i in range(n_records):
            # 60% of calls come from 3 scam centers (extreme skew simulation)
            if random.random() < 0.60:
                phone = random.choice(scam_centers)
                is_scam = random.random() < 0.85   # 85% scam rate for centers
                is_voip = random.random() < 0.95
            else:
                phone = random.choice(normal_pool)
                is_scam = random.random() < 0.02   # 2% scam rate for normal
                is_voip = random.random() < 0.1

            records.append(CallRecord(
                call_id=f"call_{i:08d}",
                phone_number=phone,
                caller_country=random.choice(countries),
                callee_country="TW",
                call_duration_sec=random.randint(1, 300),
                timestamp=time.time() - random.randint(0, 86400),
                is_voip=is_voip,
                reported_as_scam=is_scam,
                guardian_score_weight=round(random.uniform(0.1, 3.0), 2),
            ))

        return records

    def load_from_fcc_csv(self, data_path: str, sample_n: int = 50_000) -> list:
        """
        載入 FCC 原始 CSV，轉換為 CallRecord 列表。

        FCC 資料所有記錄本身即為「投訴案件」，因此 reported_as_scam=True。
        VoIP 判定：Method 欄位含 robocall / prerecorded 的視為 VoIP。

        參數：
            data_path:  FCC CSV 路徑（raw_fcc.csv 或 staging CSV 均可）
            sample_n:   最多取樣筆數（預設 50,000，避免記憶體壓力）

        回傳：
            list[CallRecord]
        """
        import os
        import pandas as pd
        import numpy as np

        logger.info(f"📂 Loading FCC CSV: {os.path.basename(data_path)}")
        df = pd.read_csv(data_path, on_bad_lines="skip", low_memory=False)
        logger.info(f"   原始列數: {len(df):,}")

        # 保留有 Caller ID 的記錄
        df = df[df["Caller ID Number"].notna()].copy()

        # 移除明顯假號碼
        fake = {"000-000-0000", "555-555-5555", "111-111-1111", "999-999-9999"}
        df = df[~df["Caller ID Number"].isin(fake)].copy()
        logger.info(f"   清洗後有效列數: {len(df):,}")

        # 取樣（避免資料過大拖慢 ETL demo）
        if len(df) > sample_n:
            df = df.sample(n=sample_n, random_state=SEED).reset_index(drop=True)
            logger.info(f"   已取樣至 {sample_n:,} 筆")

        # VoIP 判定（向量化）
        method_series = df.get("Method", None)
        if method_series is not None:
            is_voip_arr = method_series.str.lower().str.contains(
                "robocall|prerecorded|auto-dialer", na=False
            ).tolist()
        else:
            is_voip_arr = [False] * len(df)

        # 產生隨機輔助欄位（確定性）
        rng = random.Random(SEED)
        phone_arr = df["Caller ID Number"].astype(str).tolist()

        records = []
        import time as _time
        ts_base = _time.time()
        for i, (phone, is_voip) in enumerate(zip(phone_arr, is_voip_arr)):
            records.append(CallRecord(
                call_id=f"fcc_{i:08d}",
                phone_number=phone.strip(),
                caller_country="US",
                callee_country="US",
                call_duration_sec=rng.randint(1, 60),
                timestamp=ts_base - rng.randint(0, 365 * 24 * 3600),
                is_voip=bool(is_voip),
                reported_as_scam=True,   # FCC 投訴本身即為詐騙回報
                guardian_score_weight=round(rng.uniform(0.5, 2.0), 2),
            ))

        logger.info(f"   轉換完成: {len(records):,} 筆 CallRecord")
        return records

    def run_from_csv(self, data_path: str, sample_n: int = 50_000) -> dict:
        """
        使用真實 FCC CSV 資料執行完整 ETL pipeline。
        流程與 run() 相同，差別在於資料來源為外部 CSV 而非合成資料。
        """
        import os
        random.seed(SEED)

        logger.info(
            f"🚀 Starting Archangel ETL Pipeline | Source: {os.path.basename(data_path)}"
        )

        t0 = time.perf_counter()
        records = self.load_from_fcc_csv(data_path, sample_n=sample_n)
        logger.info(f"✅ Phase 1 — FCC data loaded: {len(records):,} records")

        hot_keys = self.skew_handler.detect_hot_keys(records)
        logger.info(f"✅ Phase 2 — Hot keys detected: {len(hot_keys)}")

        partial = self.skew_handler.partition_and_process(records)
        logger.info(f"✅ Phase 3 — Partial results: {len(partial)} salted groups")

        final_results = self.skew_handler.final_aggregate(partial)
        t_total = (time.perf_counter() - t0) * 1000

        risk_counts = defaultdict(int)
        for stats in final_results.values():
            risk_counts[stats["risk_level"]] += 1

        self.skew_handler.print_skew_report()

        logger.info(f"✅ Pipeline complete in {t_total:.0f}ms")
        logger.info(f"   📊 Risk Distribution: {dict(risk_counts)}")
        logger.info(f"   🔴 HIGH risk numbers → Redis blacklist: {risk_counts['HIGH']}")

        skew_report = self.skew_handler.get_skew_report()

        return {
            "total_records": len(records),
            "unique_numbers": len(final_results),
            "hot_keys_handled": len(hot_keys),
            "risk_distribution": dict(risk_counts),
            "pipeline_time_ms": round(t_total, 2),
            "pre_salt_skew_ratio": skew_report.get("pre_salt_skew_ratio", 0),
            "post_salt_skew_ratio": skew_report.get("post_salt_skew_ratio", 0),
        }

    def run(self, n_records: int = 50_000) -> dict:
        """Execute the full ETL pipeline with skew handling."""
        random.seed(SEED)

        logger.info(f"🚀 Starting Archangel ETL Pipeline | Records: {n_records:,}")

        # ── Phase 1: Data Generation / Ingestion
        t0 = time.perf_counter()
        records = self.generate_synthetic_data(n_records)
        logger.info(f"✅ Phase 1 — Data ingested: {len(records):,} records")

        # ── Phase 2: Hot Key Detection
        hot_keys = self.skew_handler.detect_hot_keys(records)
        logger.info(f"✅ Phase 2 — Hot keys detected: {len(hot_keys)}")

        # ── Phase 3: Salted Partition & Partial Aggregation
        partial = self.skew_handler.partition_and_process(records)
        logger.info(f"✅ Phase 3 — Partial results: {len(partial)} salted groups")

        # ── Phase 4: Final Aggregation
        final_results = self.skew_handler.final_aggregate(partial)
        t_total = (time.perf_counter() - t0) * 1000

        # ── Phase 5: Risk Summary
        risk_counts = defaultdict(int)
        for stats in final_results.values():
            risk_counts[stats["risk_level"]] += 1

        self.skew_handler.print_skew_report()

        logger.info(f"✅ Pipeline complete in {t_total:.0f}ms")
        logger.info(f"   📊 Risk Distribution: {dict(risk_counts)}")
        logger.info(f"   🔴 HIGH risk numbers → Redis blacklist: {risk_counts['HIGH']}")

        skew_report = self.skew_handler.get_skew_report()

        return {
            "total_records": len(records),
            "unique_numbers": len(final_results),
            "hot_keys_handled": len(hot_keys),
            "risk_distribution": dict(risk_counts),
            "pipeline_time_ms": round(t_total, 2),
            "pre_salt_skew_ratio": skew_report.get("pre_salt_skew_ratio", 0),
            "post_salt_skew_ratio": skew_report.get("post_salt_skew_ratio", 0),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    etl = AntiFraudETL()
    results = etl.run(n_records=50_000)

    print("\n" + "═" * 60)
    print("  FINAL ETL PIPELINE RESULTS")
    print("═" * 60)
    for k, v in results.items():
        print(f"  {k:<30}: {v}")
    print("═" * 60)
