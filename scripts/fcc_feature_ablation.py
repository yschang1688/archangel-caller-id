#!/usr/bin/env python3
"""
FCC 特徵消融與洩漏檢查（complaint_count_log）
==========================================

目的：
    驗證是否主要依賴 `complaint_count_log` 一維就能決定勝負。
    - 單特徵消融：移除 / 置零 / 打亂該特徵
    - 評估流程修正：train/val/test，門檻只在 val 上選，test 固定門檻
    - permutation 重要度：打亂該特徵對測試 F1 的影響（ΔF1）

輸入：
    - raw_fcc.csv（或指定 CSV 路徑）

輸出：
    - outputs/experiments/fcc_feature_ablation.csv
    - outputs/experiments/fcc_feature_ablation.md

用法：
    python scripts/fcc_feature_ablation.py --data-path raw_fcc.csv --hard-negative-ratio 0.03
"""
from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, roc_auc_score

SEED = 42
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "raw_fcc.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "experiments")

# 允許用 `python scripts/xxx.py` 直接執行時，也能 import `src.*`
import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_fcc_dataset(data_path: str, negative_ratio: float, hard_negative_ratio: float) -> Tuple[pd.DataFrame, pd.Series]:
    from src.processing.fcc_data_pipeline import fcc_clean_and_prepare
    X, y, _, _ = fcc_clean_and_prepare(
        data_path,
        negative_ratio=negative_ratio,
        hard_negative_ratio=hard_negative_ratio,
    )
    return X, y


def train_eval_with_val_threshold(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> Dict[str, float]:
    """
    三分法：train/val/test；門檻只在 val 選，test 固定門檻評估。
    回傳：precision/recall/f1/roc_auc/threshold
    """
    # 先切 test，再從 train 切 val
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=SEED, stratify=y_train_full
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=SEED,
        cache_size=500,
    )
    model.fit(X_train_s, y_train)

    # 選門檻（只用 val）
    y_val_proba = model.predict_proba(X_val_s)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1_scores))
    thr = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    # 測試評估（固定 thr）
    y_test_proba = model.predict_proba(X_test_s)[:, 1]
    y_test_pred = (y_test_proba >= thr).astype(int)

    precision = float(precision_score(y_test, y_test_pred, zero_division=0))
    recall = float(recall_score(y_test, y_test_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_test_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_test_proba))

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "threshold": round(thr, 4),
    }


def permutation_delta_f1(
    X: pd.DataFrame, y: pd.Series, feature: str
) -> float:
    """
    在固定 val 選門檻、test 評估的流程下，打亂單一特徵，量測 ΔF1。
    回傳：打亂後 F1 - 原始 F1（負值代表該特徵重要）
    """
    base = train_eval_with_val_threshold(X, y)
    X_perm = X.copy()
    X_perm[feature] = np.random.RandomState(SEED).permutation(X_perm[feature].values)
    after = train_eval_with_val_threshold(X_perm, y)
    return round(after["f1"] - base["f1"], 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="FCC 特徵消融與洩漏檢查（complaint_count_log）")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--hard-negative-ratio", type=float, default=0.03)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "fcc_feature_ablation.csv")
    md_path = os.path.join(OUTPUT_DIR, "fcc_feature_ablation.md")

    X, y = load_fcc_dataset(args.data_path, args.negative_ratio, args.hard_negative_ratio)
    assert "complaint_count_log" in X.columns, "缺少 complaint_count_log 特徵"

    experiments = []

    # 0) 原始（baseline）
    res_base = train_eval_with_val_threshold(X, y)
    experiments.append({"scenario": "baseline", **res_base})

    # 1) remove（直接移除欄位）
    X_remove = X.drop(columns=["complaint_count_log"])
    res_remove = train_eval_with_val_threshold(X_remove, y)
    experiments.append({"scenario": "remove_feature", **res_remove})

    # 2) zero（將該欄位置零）
    X_zero = X.copy()
    X_zero["complaint_count_log"] = 0.0
    res_zero = train_eval_with_val_threshold(X_zero, y)
    experiments.append({"scenario": "zero_feature", **res_zero})

    # 3) shuffle（隨機打亂該欄位）
    X_shuffle = X.copy()
    X_shuffle["complaint_count_log"] = np.random.RandomState(SEED).permutation(
        X_shuffle["complaint_count_log"].values
    )
    res_shuffle = train_eval_with_val_threshold(X_shuffle, y)
    experiments.append({"scenario": "shuffle_feature", **res_shuffle})

    # 4) permutation 重要度（ΔF1）
    delta = permutation_delta_f1(X, y, "complaint_count_log")

    df = pd.DataFrame(experiments)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # Markdown
    md = []
    md.append("# FCC 特徵消融與洩漏檢查（complaint_count_log）")
    md.append("")
    md.append(f"- Dataset: `{os.path.basename(args.data_path)}`")
    md.append(f"- Seed: `{SEED}`，negative_ratio=`{args.negative_ratio}`，hard_negative_ratio=`{args.hard_negative_ratio}`")
    md.append(f"- Evaluation: train/val/test；threshold 僅在 val 選，test 固定門檻")
    md.append("")
    md.append("## 結果總表")
    md.append("")
    md.append(df.to_markdown(index=False))
    md.append("")
    md.append("## 洩漏訊號檢驗（permutation importance on F1）")
    md.append("")
    md.append(f"- ΔF1 after shuffling `complaint_count_log`: **{delta:+.4f}**（負值代表該特徵重要）")
    md.append("")
    md.append("## 解讀建議")
    md.append("")
    md.append("- 若 remove/zero/shuffle 任一情境導致 F1 顯著下降，代表該欄位對模型決策影響極大（可能形成隱性洩漏）。")
    md.append("- 若 ΔF1 顯著為負，代表該欄位是主要決策依據之一；建議做分箱/截斷/降權，或以更嚴格規則定義 hard pool。")
    md.append("- 建議將本流程納入每次資料與特徵更新的 regression 檢查。")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("\n✅ Ablation 完成")
    print(f"  CSV: {csv_path}")
    print(f"  MD:  {md_path}")


if __name__ == "__main__":
    main()

