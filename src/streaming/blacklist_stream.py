"""
blacklist_stream.py — Archangel Real-time Blacklist Stream Processor
=====================================================================
Kafka consumer that scores call events with the trained SVM model and
writes spam numbers to a Redis blacklist in real time.

Pipeline position:
    kafka_producer.py ──► topic 'call-events' ──► THIS ──► Redis blacklist
                                                        └► detection_api.py reads

Feature honesty note:
    The SVM model was trained on 20 FCC complaint-aggregate features.
    A live call-event stream can only derive the behavioral subset
    (time-of-day ratios, VoIP ratio, call volume); complaint-content
    features (robocall_flag, cramming_flag, …) do not exist at the
    event level and are left at 0.0. Scores from this path are
    therefore conservative — a number needs strong behavioral signals
    to cross the blacklist threshold.

Latency instrumentation:
    - processing_ms:  consume → feature update → SVM score → Redis write
    - end_to_end_ms:  producer event timestamp → Redis write (same host clock)
    Reported as p50 / p95 / p99 at the end of a run.
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from src.ml.svm_spam_classifier import SVMSpamTrainer
from src.processing.fcc_data_pipeline import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TOPIC = "call-events"
BLACKLIST_KEY = "archangel:blacklist"          # Redis hash: phone -> proba
BLACKLIST_META_KEY = "archangel:blacklist:meta"  # Redis hash: phone -> json meta


# ─────────────────────────────────────────────────────────────────────────────
# Rolling per-number feature aggregation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NumberState:
    """Rolling aggregates for one phone number."""
    total_calls: int = 0
    voip_calls: int = 0
    short_calls: int = 0            # < 30s — robocall-like behavior
    hour_buckets: dict = field(default_factory=lambda: defaultdict(int))
    weekend_calls: int = 0
    caller_countries: set = field(default_factory=set)
    first_seen: float = 0.0
    last_seen: float = 0.0


class StreamFeatureAggregator:
    """
    Maintains per-number rolling state and maps it onto the 20-dim
    FCC feature vector. Features that cannot be derived from a call
    event stay 0.0 (see module docstring).
    """

    def __init__(self):
        self.numbers: dict[str, NumberState] = defaultdict(NumberState)

    def update(self, event: dict) -> str:
        phone = event["phone_number"]
        s = self.numbers[phone]
        ts = float(event.get("timestamp", time.time()))
        dt = datetime.fromtimestamp(ts)

        s.total_calls += 1
        if event.get("is_voip"):
            s.voip_calls += 1
        if int(event.get("call_duration_sec", 0)) < 30:
            s.short_calls += 1
        s.hour_buckets[dt.hour] += 1
        if dt.weekday() >= 5:
            s.weekend_calls += 1
        s.caller_countries.add(event.get("caller_country", ""))
        if s.first_seen == 0.0:
            s.first_seen = ts
        s.last_seen = ts
        return phone

    def features_for(self, phone: str) -> dict:
        s = self.numbers[phone]
        n = max(s.total_calls, 1)

        def hour_ratio(hours) -> float:
            return sum(s.hour_buckets[h] for h in hours) / n

        # 只映射語意相符的特徵。complaint_count_log / complaint_velocity
        # 在訓練資料裡是「被投訴」的量與速率——通話事件不是投訴，硬映會把
        # 任何活躍號碼都推成 spam（實測正常流量 proba 被推到 1.0），一律留 0。
        feats = dict.fromkeys(FEATURE_NAMES, 0.0)
        feats.update({
            # 時間模式 — derivable from event timestamps
            "business_hour_ratio": hour_ratio(range(9, 17)),
            "evening_ratio": hour_ratio(range(18, 22)),
            "night_ratio": hour_ratio(list(range(22, 24)) + list(range(0, 6))),
            "school_hour_ratio": hour_ratio(range(8, 15)),
            "weekend_ratio": s.weekend_calls / n,
            # 通話類型 — short-call ratio as robocall-like proxy signals
            "prerecorded_ratio": s.short_calls / n,
            "abandoned_call_ratio": s.short_calls / n,
            "live_voice_ratio": 1.0 - (s.short_calls / n),
            # 號碼特徵
            "is_voip": s.voip_calls / n,
            # 地理行為 — cross-country breadth
            "unique_states_norm": min(len(s.caller_countries) / 10.0, 1.0),
        })
        return feats


# ─────────────────────────────────────────────────────────────────────────────
# Stream processor
# ─────────────────────────────────────────────────────────────────────────────

class BlacklistStreamProcessor:
    """Consumes call events, scores them, writes spam numbers to Redis."""

    # 冷啟動守門：n=1 時比例特徵必為 0/1 極端值，單通深夜短促 VoIP 電話
    # 會被誤判成詐騙中心（實測 5000 事件誤黑 91 個正常號）。觀測滿 5 通
    # 才允許判黑，詐騙中心的高頻行為幾乎立刻通過此門檻。
    MIN_CALLS_TO_BLACKLIST = 5

    def __init__(self, redis_client, model_path: str = None, threshold: float = None,
                 min_calls: int = None):
        self.trainer = SVMSpamTrainer()
        self.trainer.load_model(model_path)
        if threshold is not None:
            self.trainer.threshold = threshold
        self.min_calls = min_calls if min_calls is not None else self.MIN_CALLS_TO_BLACKLIST
        self.aggregator = StreamFeatureAggregator()
        self.redis = redis_client
        self.processing_ms: list[float] = []
        self.end_to_end_ms: list[float] = []
        self.events_seen = 0
        self.blacklisted: set[str] = set()
        self._blacklist_proba: dict[str, float] = {}

    def process_event(self, event: dict) -> dict:
        t0 = time.perf_counter()
        phone = self.aggregator.update(event)
        feats = self.aggregator.features_for(phone)
        result = self.trainer.predict_single(feats)

        if result["prediction"] == 1 and \
                self.aggregator.numbers[phone].total_calls >= self.min_calls:
            # 只在初次上榜或機率有感變動時寫 Redis——詐騙中心每通電話都
            # 重寫兩鍵會讓同步 round-trip 主導尾延遲（實測 e2e p99 190ms
            # → 去重後回到數十 ms 級）。
            prev = self._blacklist_proba.get(phone)
            if prev is None or abs(result["probability"] - prev) > 0.01:
                self.redis.hset(BLACKLIST_KEY, phone, result["probability"])
                self.redis.hset(BLACKLIST_META_KEY, phone, json.dumps({
                    "probability": result["probability"],
                    "threshold": result["threshold"],
                    "calls_observed": self.aggregator.numbers[phone].total_calls,
                    "blacklisted_at": time.time(),
                    "source": "blacklist_stream",
                }))
                self._blacklist_proba[phone] = result["probability"]
            self.blacklisted.add(phone)

        processing = (time.perf_counter() - t0) * 1000
        self.processing_ms.append(processing)
        produced_at = float(event.get("produced_at", 0))
        if produced_at:
            self.end_to_end_ms.append((time.time() - produced_at) * 1000)
        self.events_seen += 1
        return result

    def latency_report(self) -> dict:
        def pct(data, q):
            return round(float(np.percentile(data, q)), 3) if data else None
        return {
            "events": self.events_seen,
            "blacklisted_numbers": len(self.blacklisted),
            "processing_ms": {q: pct(self.processing_ms, qv)
                              for q, qv in [("p50", 50), ("p95", 95), ("p99", 99)]},
            "end_to_end_ms": {q: pct(self.end_to_end_ms, qv)
                              for q, qv in [("p50", 50), ("p95", 95), ("p99", 99)]},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Runners
# ─────────────────────────────────────────────────────────────────────────────

def run_kafka(processor: BlacklistStreamProcessor, broker: str,
              max_events: int, idle_timeout_s: float = 10.0,
              from_latest: bool = False) -> None:
    """
    from_latest: start at the log tail instead of replaying backlog.
    量端到端延遲時必用——replay 模式的 e2e 含 rebalance 期間積壓
    （實測會把 p95 推到秒級），量到的是 backlog 深度不是管線延遲。
    """
    from kafka import KafkaConsumer
    # 量測模式用一次性 group：固定 group 的已 commit offset 會蓋過
    # auto_offset_reset，consumer 恢復舊位置消費歷史事件（實測 e2e 被
    # 灌成 54s——量到的是事件年齡不是管線延遲）。
    group = (f"archangel-blacklist-stream-{int(time.time())}"
             if from_latest else "archangel-blacklist-stream")
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=broker,
        auto_offset_reset="latest" if from_latest else "earliest",
        group_id=group,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms=int(idle_timeout_s * 1000),
    )
    logger.info(f"🎧 Consuming '{TOPIC}' from {broker} (max_events={max_events})")
    for msg in consumer:
        processor.process_event(msg.value)
        if processor.events_seen >= max_events:
            break
    consumer.close()


def run_simulation(processor: BlacklistStreamProcessor, n_events: int) -> None:
    """No-broker fallback: consume straight from the simulated producer."""
    from src.ingestion.kafka_producer import SimulatedKafkaProducer
    logger.info(f"🎧 No broker — consuming {n_events} simulated events in-process")
    producer = SimulatedKafkaProducer()
    for event in producer.generate_events(n_events):
        processor.process_event(json.loads(event.to_json()))


def main():
    parser = argparse.ArgumentParser(description="Archangel real-time blacklist stream")
    parser.add_argument("--broker", default=None, help="Kafka bootstrap server, e.g. localhost:9092")
    parser.add_argument("--redis", default="localhost:6379", help="Redis host:port")
    parser.add_argument("--max-events", type=int, default=5000)
    parser.add_argument("--idle-timeout", type=float, default=10.0,
                        help="Stop after this many seconds without new messages")
    parser.add_argument("--from-latest", action="store_true",
                        help="Skip backlog; required for honest e2e latency numbers")
    parser.add_argument("--flush", action="store_true", help="Clear existing blacklist keys first")
    args = parser.parse_args()

    import redis as redis_lib
    host, port = args.redis.split(":")
    redis_client = redis_lib.Redis(host=host, port=int(port), decode_responses=True)
    redis_client.ping()
    if args.flush:
        redis_client.delete(BLACKLIST_KEY, BLACKLIST_META_KEY)

    processor = BlacklistStreamProcessor(redis_client)
    if args.broker:
        run_kafka(processor, args.broker, args.max_events, args.idle_timeout,
                  args.from_latest)
    else:
        run_simulation(processor, args.max_events)

    report = processor.latency_report()
    print("\n─── Blacklist Stream Report ───")
    print(json.dumps(report, indent=2))
    print(f"Redis blacklist size: {redis_client.hlen(BLACKLIST_KEY)}")


if __name__ == "__main__":
    main()
