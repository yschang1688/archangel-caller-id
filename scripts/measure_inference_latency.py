"""
measure_inference_latency.py — 推論延遲量測條件查證
====================================================
README 的 <0.057ms/record 出自 run_ml_ops.py 的批次量法：
    predict_proba(X_test_scaled[:100]) 計時 ÷ 100（批次攤提）
本腳本同時量測：
    A. 批次攤提  — 與 run_ml_ops.py 相同條件（100 筆批次）
    B. 真單筆    — predict_single()（含 DataFrame 建構 + scaler.transform）
    C. 單筆核心  — 預先 scale 後單筆 predict_proba（不含前處理）
輸出各自的 p50/p95/p99，供 README 措辭對齊實測。
"""

import time

import numpy as np
import pandas as pd

from src.ml.svm_spam_classifier import SVMSpamTrainer
from src.processing.fcc_data_pipeline import FEATURE_NAMES

N_ROUNDS = 200
BATCH = 100
rng = np.random.default_rng(42)


def pct(xs):
    return {q: round(float(np.percentile(xs, v)), 4)
            for q, v in [("p50", 50), ("p95", 95), ("p99", 99)]}


def main():
    trainer = SVMSpamTrainer()
    trainer.load_model()

    X = pd.DataFrame(rng.random((BATCH, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    X_scaled = trainer.scaler.transform(X)

    # A. 批次攤提（run_ml_ops.py 的量法）
    amortized = []
    for _ in range(N_ROUNDS):
        t0 = time.perf_counter()
        trainer.model.predict_proba(X_scaled)
        amortized.append((time.perf_counter() - t0) * 1000 / BATCH)

    # B. 真單筆（predict_single 完整路徑）
    single_full = []
    feat_dict = dict(zip(FEATURE_NAMES, X.iloc[0]))
    for _ in range(N_ROUNDS):
        t0 = time.perf_counter()
        trainer.predict_single(feat_dict)
        single_full.append((time.perf_counter() - t0) * 1000)

    # C. 單筆核心（僅 predict_proba，一列）
    one_row = X_scaled[:1]
    single_core = []
    for _ in range(N_ROUNDS):
        t0 = time.perf_counter()
        trainer.model.predict_proba(one_row)
        single_core.append((time.perf_counter() - t0) * 1000)

    print(f"rounds={N_ROUNDS}, batch={BATCH}, model threshold={trainer.threshold:.4f}")
    print(f"A. batch-amortized per-record (run_ml_ops 條件): {pct(amortized)}")
    print(f"B. true single-record via predict_single():      {pct(single_full)}")
    print(f"C. single-record predict_proba only:             {pct(single_core)}")


if __name__ == "__main__":
    main()
