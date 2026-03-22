"""
kafka_producer.py — Archangel Simulated Kafka Event Producer
=============================================================
Generates realistic call event streams for pipeline ingestion.

In production: publishes to Kafka topic 'call-events' via confluent_kafka.
Here: simulates the event stream with configurable throughput.

Role Target: Data Research Engineer @ Gogolook ISL
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
    timestamp: float
    device_fingerprint: str
    sms_content_hash: str = ""  # Hash of SMS body (privacy-preserving)

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
            else:
                phone = random.choice(self.normal_pool)
                is_voip = random.random() < 0.08
                caller_country = random.choice(self.countries)
                duration = random.randint(5, 600)

            event = CallEvent(
                event_id=f"evt_{self._event_counter:08d}",
                phone_number=phone,
                caller_country=caller_country,
                callee_country="TW",
                call_duration_sec=duration,
                is_voip=is_voip,
                timestamp=time.time(),
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


if __name__ == "__main__":
    run_demo()
