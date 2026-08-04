"""
kafka_producer.py — Archangel Simulated Kafka Event Producer
=============================================================
Generates realistic call event streams for pipeline ingestion.

Two modes:
    default          — in-process simulation (no broker needed)
    --broker <addr>  — real publishing to Kafka topic 'call-events'
                       via confluent_kafka (pairs with
                       src/streaming/blacklist_stream.py downstream)
"""

import json
import random
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Generator
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Event Schema
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CallEvent:
    """Mirrors the Kafka message schema for call events."""
    event_id: str
    phone_number: str
    caller_country: str
    callee_country: str
    call_duration_sec: int
    is_voip: bool
    timestamp: float            # Simulated call time (drives time-of-day features)
    device_fingerprint: str
    sms_content_hash: str = ""  # Hash of SMS body (privacy-preserving)
    produced_at: float = 0.0    # Publish wall-clock time (drives e2e latency measurement)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Producer
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedKafkaProducer:
    """
    Simulates a Kafka producer generating call events.

    Production equivalent:
        from confluent_kafka import Producer
        producer = Producer({'bootstrap.servers': 'kafka:29092'})
        producer.produce('call-events', value=event.to_json())
    """

    TOPIC = "call-events"

    def __init__(self):
        random.seed(SEED)
        self.countries = ["TW", "US", "UK", "JP", "SG", "AU", "HK", "DE", "MY", "MM"]
        self.scam_centers = [
            "+886-800-SCAM-01", "+886-800-SCAM-02", "+886-800-SCAM-03",
            "+95-800-FRAUD-01", "+44-900-SCAM-01",
        ]
        self.normal_pool = [
            f"+{random.randint(1, 99)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
            for _ in range(1000)
        ]
        self._event_counter = 0

    def generate_events(self, n_events: int = 1000) -> Generator[CallEvent, None, None]:
        """Yields realistic call events with configurable scam ratio."""
        for _ in range(n_events):
            self._event_counter += 1

            # 15% scam center traffic (realistic hotspot)
            if random.random() < 0.15:
                phone = random.choice(self.scam_centers)
                is_voip = random.random() < 0.90
                caller_country = random.choice(["MM", "MY", "TH", "KH"])
                duration = random.randint(1, 30)  # Short robocalls
                # Scam centers work the night shift (22-06)
                call_hour = random.choice([22, 23, 0, 1, 2, 3, 4, 5])
            else:
                phone = random.choice(self.normal_pool)
                is_voip = random.random() < 0.08
                caller_country = random.choice(self.countries)
                duration = random.randint(5, 600)
                # Normal traffic concentrates in business hours + evening
                call_hour = random.choices(
                    population=list(range(24)),
                    weights=[1, 1, 1, 1, 1, 1, 2, 4, 8, 10, 10, 10,
                             8, 10, 10, 10, 10, 8, 9, 9, 6, 4, 2, 1],
                )[0]

            # Simulated call time within the past 24h at the profile's hour.
            # Wall-clock must NOT drive this — a demo run at 2am would mark
            # every caller as a night caller (night_ratio=1.0 across the board).
            # Day start is computed in LOCAL time: the consumer buckets hours
            # via datetime.fromtimestamp(); a UTC midnight base (now % 86400)
            # shifts every profile by the UTC offset (+8h here 把深夜平移成上班時段).
            now = time.time()
            lt = time.localtime(now)
            day_start = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
            call_ts = day_start + call_hour * 3600 + random.randint(0, 3599)
            if call_ts > now:
                call_ts -= 86400

            event = CallEvent(
                event_id=f"evt_{self._event_counter:08d}",
                phone_number=phone,
                caller_country=caller_country,
                callee_country="TW",
                call_duration_sec=duration,
                is_voip=is_voip,
                timestamp=call_ts,
                device_fingerprint=hashlib.md5(phone.encode()).hexdigest()[:12],
            )
            yield event

    def produce_batch(self, n_events: int = 1000) -> list[dict]:
        """Produce a batch and return summary stats."""
        events = list(self.generate_events(n_events))

        voip_count = sum(1 for e in events if e.is_voip)
        scam_center_count = sum(1 for e in events if e.phone_number in self.scam_centers)
        unique_phones = len(set(e.phone_number for e in events))

        summary = {
            "topic": self.TOPIC,
            "events_produced": len(events),
            "unique_phone_numbers": unique_phones,
            "voip_ratio": round(voip_count / len(events), 3),
            "scam_center_ratio": round(scam_center_count / len(events), 3),
        }

        logger.info(f"📤 Produced {len(events)} events to topic '{self.TOPIC}'")
        logger.info(f"   VoIP: {voip_count} ({summary['voip_ratio']:.1%}) | "
                     f"Scam centers: {scam_center_count} ({summary['scam_center_ratio']:.1%})")

        return summary


def run_demo() -> dict:
    """Demo: produce 5000 simulated events."""
    random.seed(SEED)
    producer = SimulatedKafkaProducer()

    print("\n" + "═" * 60)
    print("  KAFKA PRODUCER — SIMULATED EVENT STREAM")
    print("═" * 60)

    summary = producer.produce_batch(n_events=5000)

    print(f"\n  Events produced:    {summary['events_produced']:,}")
    print(f"  Unique phones:      {summary['unique_phone_numbers']}")
    print(f"  VoIP ratio:         {summary['voip_ratio']:.1%}")
    print(f"  Scam center ratio:  {summary['scam_center_ratio']:.1%}")
    print("═" * 60)

    return summary


def run_publish(broker: str, n_events: int = 5000, rate: float = None) -> dict:
    """
    Publish simulated events to a real Kafka broker.

    rate: events/sec pacing. Without it the producer bursts everything at
    once and downstream end-to-end latency measures queue backlog, not the
    pipeline (實測 burst 模式 e2e p50 會被推到秒級).
    """
    from confluent_kafka import Producer

    random.seed(SEED)
    sim = SimulatedKafkaProducer()
    producer = Producer({"bootstrap.servers": broker})

    interval = 1.0 / rate if rate else 0.0
    published = 0
    t0 = time.perf_counter()
    for event in sim.generate_events(n_events):
        # Stamp at publish time so downstream end-to-end latency is real
        # (call-time `timestamp` stays simulated — it drives features)
        event.produced_at = time.time()
        producer.produce(sim.TOPIC, value=event.to_json())
        published += 1
        producer.poll(0)
        if interval:
            next_at = t0 + published * interval
            sleep_for = next_at - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
    producer.flush(timeout=30)
    elapsed = time.perf_counter() - t0

    logger.info(f"📤 Published {published} events to '{sim.TOPIC}' @ {broker} "
                f"in {elapsed:.2f}s ({published / elapsed:,.0f} ev/s)")
    return {"topic": sim.TOPIC, "events_published": published,
            "elapsed_s": round(elapsed, 2)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Archangel call-event producer")
    parser.add_argument("--broker", default=None,
                        help="Kafka bootstrap server (e.g. localhost:9092); omit to simulate")
    parser.add_argument("--n-events", type=int, default=5000)
    parser.add_argument("--rate", type=float, default=None,
                        help="events/sec pacing; omit to burst")
    args = parser.parse_args()

    if args.broker:
        run_publish(args.broker, args.n_events, args.rate)
    else:
        run_demo()
