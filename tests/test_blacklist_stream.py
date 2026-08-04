"""
test_blacklist_stream.py — 串流黑名單處理器單元測試
====================================================
含商業邏輯反向測試：詐騙行為模式必須進黑名單、正常模式必須不進——
把 threshold 邏輯反向（>= 改 <）時這兩條會同時翻紅。
"""

import time

import pytest

from src.processing.fcc_data_pipeline import FEATURE_NAMES
from src.streaming.blacklist_stream import (
    BLACKLIST_KEY,
    BlacklistStreamProcessor,
    StreamFeatureAggregator,
)


class FakeRedis:
    """In-memory stand-in for redis.Redis (hset/hgetall/hlen only)."""

    def __init__(self):
        self.store = {}

    def hset(self, key, field, value):
        self.store.setdefault(key, {})[field] = value

    def hgetall(self, key):
        return self.store.get(key, {})

    def hlen(self, key):
        return len(self.store.get(key, {}))


def make_event(phone, *, is_voip, duration, hour, country="MM"):
    # 固定在 2026-08-03（週一）的指定時段
    base = time.mktime((2026, 8, 3, hour, 0, 0, 0, 0, -1))
    return {
        "event_id": "evt_test",
        "phone_number": phone,
        "caller_country": country,
        "callee_country": "TW",
        "call_duration_sec": duration,
        "is_voip": is_voip,
        "timestamp": base,
        "device_fingerprint": "test",
    }


# ─── Aggregator ──────────────────────────────────────────────────────────────

def test_aggregator_voip_ratio():
    agg = StreamFeatureAggregator()
    for voip in [True, True, True, False]:
        agg.update(make_event("+886-1", is_voip=voip, duration=10, hour=23))
    feats = agg.features_for("+886-1")
    assert feats["is_voip"] == pytest.approx(0.75)


def test_aggregator_night_ratio_and_zero_fill():
    agg = StreamFeatureAggregator()
    agg.update(make_event("+886-2", is_voip=True, duration=5, hour=23))
    agg.update(make_event("+886-2", is_voip=True, duration=5, hour=3))
    agg.update(make_event("+886-2", is_voip=True, duration=5, hour=10))
    feats = agg.features_for("+886-2")
    assert feats["night_ratio"] == pytest.approx(2 / 3)
    # 事件流推不出的投訴類特徵必須維持 0（不得憑空捏造）
    assert feats["robocall_flag"] == 0.0
    assert feats["cramming_flag"] == 0.0
    assert set(feats) == set(FEATURE_NAMES)


# ─── Processor（真模型 + FakeRedis）──────────────────────────────────────────
# 需要已訓練的 models/svm_spam_model.pkl（repo 不含模型檔時整組跳過，
# 先跑 run_ml_dev.py 訓練即可解鎖）。

import os

from src.ml.svm_spam_classifier import _PROJECT_ROOT

_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "svm_spam_model.pkl")


@pytest.fixture(scope="module")
def processor():
    if not os.path.exists(_MODEL_PATH):
        pytest.skip("trained model not present — run run_ml_dev.py first")
    return BlacklistStreamProcessor(FakeRedis())


def _scam_burst(processor, phone, n=50):
    """高頻深夜短促 VoIP 呼出、來源國輪替——比照模擬 producer 的詐騙中心行為。"""
    countries = ["MM", "MY", "TH", "KH"]
    for i in range(n):
        processor.process_event(
            make_event(phone, is_voip=True, duration=3, hour=23,
                       country=countries[i % len(countries)])
        )


def _normal_traffic(processor, phone, n=5):
    """低頻日間長通話——正常號碼行為模式。"""
    for i in range(n):
        processor.process_event(
            make_event(phone, is_voip=False, duration=300, hour=10, country="TW")
        )


def test_scam_pattern_gets_blacklisted(processor):
    _scam_burst(processor, "+886-800-SCAM-T1")
    assert "+886-800-SCAM-T1" in processor.blacklisted
    assert "+886-800-SCAM-T1" in processor.redis.hgetall(BLACKLIST_KEY)


def test_normal_pattern_not_blacklisted(processor):
    _normal_traffic(processor, "+886-2-2345-6789")
    assert "+886-2-2345-6789" not in processor.blacklisted


def test_cold_start_single_call_not_blacklisted(processor):
    """單通深夜短促 VoIP 電話（n=1 比例特徵全極端）不得直接判黑。"""
    processor.process_event(
        make_event("+886-9-1111-2222", is_voip=True, duration=3, hour=23)
    )
    assert "+886-9-1111-2222" not in processor.blacklisted


def test_latency_report_shape(processor):
    report = processor.latency_report()
    assert report["events"] > 0
    assert report["processing_ms"]["p99"] is not None
    assert report["processing_ms"]["p50"] <= report["processing_ms"]["p99"]
