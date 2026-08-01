#!/usr/bin/env python3
"""
FCC hard negatives ratio 敏感度分析（0.00 → 0.10）
=================================================

目的：
    掃描 FCC 負樣本策略中的 hard negatives 比例，觀察 precision / recall / F1 的變化趨勢，
    用於評估「更貼近真實世界」的負樣本混合策略對模型表現的影響。

輸入：
    - raw_fcc.csv（或指定 CSV 路徑）

輸出：
    - outputs/experiments/fcc_hard_negative_sensitivity.csv
    - outputs/experiments/fcc_hard_negative_sensitivity.md

依賴：
    - pandas / numpy
    - src.processing.fcc_data_pipeline.fcc_clean_and_prepare
    - src.ml.svm_spam_classifier.SVMSpamTrainer
    - sklearn.metrics (precision_score, recall_score, f1_score)

用法：
    python scripts/fcc_hard_negative_sensitivity.py --data-path raw_fcc.csv
    python scripts/fcc_hard_negative_sensitivity.py --start 0.0 --end 0.1 --step 0.01
"""

from __future__ import annotations

import os
import sys
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score

SEED = 42

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "raw_fcc.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "experiments")

# 允許用 `python scripts/xxx.py` 直接執行時，也能 import `src.*`
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@dataclass(frozen=True)
class SweepPoint:
    """單一 sweep 點位的輸出紀錄。"""

    hard_negative_ratio: float
    n_samples: int
    precision: float
    recall: float
    f1: float
    threshold: float
    roc_auc: float
    support_vectors: int


def _frange(start: float, end: float, step: float) -> list[float]:
    """
    產生包含 end 的等差數列（浮點）。

    參數：
        start: 起點
        end: 終點
        step: 間距

    回傳：
        list[float]
    """
    if step <= 0:
        raise ValueError("step 必須 > 0")
    n = int(round((end - start) / step))
    values = [round(start + i * step, 4) for i in range(n + 1)]
    # 修正浮點誤差：確保最後一點 <= end
    return [v for v in values if v <= end + 1e-9]


def run_sweep(
    data_path: str,
    negative_ratio: float,
    start: float,
    end: float,
    step: float,
) -> tuple[pd.DataFrame, str, str]:
    """
    執行 hard negatives 比例 sweep，輸出結果 DataFrame + 檔案路徑。

    參數：
        data_path: raw_fcc.csv 路徑
        negative_ratio: 負/正樣本比例（1.0 = 1:1 平衡）
        start: hard_negative_ratio 起點
        end: hard_negative_ratio 終點
        step: hard_negative_ratio 間距

    回傳：
        (df, csv_path, md_path)
    """
    from src.processing.fcc_data_pipeline import fcc_clean_and_prepare
    from src.ml.svm_spam_classifier import SVMSpamTrainer

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "fcc_hard_negative_sensitivity.csv")
    md_path = os.path.join(OUTPUT_DIR, "fcc_hard_negative_sensitivity.md")

    ratios = _frange(start, end, step)
    results: list[SweepPoint] = []

    for ratio in ratios:
        print("\n" + "─" * 72)
        print(f"[Sweep] hard_negative_ratio={ratio:.2f}  negative_ratio={negative_ratio:.2f}")
        print("─" * 72)

        X, y, _, _ = fcc_clean_and_prepare(
            data_path,
            negative_ratio=negative_ratio,
            hard_negative_ratio=ratio,
        )

        trainer = SVMSpamTrainer()
        report = trainer.train(X, y, test_size=0.2, do_grid_search=False)

        # 使用最佳門檻值計算 precision/recall/f1（符合「可調 threshold」的產品邏輯）
        y_test = np.array(report["y_test"])
        y_proba = np.array(report["y_proba"])
        y_pred_opt = (y_proba >= float(report["optimal_threshold"])).astype(int)

        precision = float(precision_score(y_test, y_pred_opt, zero_division=0))
        recall = float(recall_score(y_test, y_pred_opt, zero_division=0))
        f1 = float(f1_score(y_test, y_pred_opt, zero_division=0))

        support_vectors = report.get("support_vectors", 0)
        if isinstance(support_vectors, list):
            n_support = int(np.sum(np.array(support_vectors, dtype=int)))
        else:
            n_support = int(support_vectors)

        results.append(
            SweepPoint(
                hard_negative_ratio=float(ratio),
                n_samples=int(len(X)),
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                threshold=float(report["optimal_threshold"]),
                roc_auc=float(report["roc_auc"]),
                support_vectors=n_support,
            )
        )

    df = pd.DataFrame([r.__dict__ for r in results]).sort_values("hard_negative_ratio").reset_index(drop=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # Markdown 報表
    md_lines: list[str] = []
    md_lines.append("# FCC hard negatives ratio 敏感度分析")
    md_lines.append("")
    md_lines.append(f"- Dataset: `{os.path.basename(data_path)}`")
    md_lines.append(f"- Seed: `{SEED}`")
    md_lines.append(f"- negative_ratio (neg/pos): `{negative_ratio}`")
    md_lines.append(f"- Sweep: hard_negative_ratio `{start}` → `{end}` (step `{step}`)")
    md_lines.append("")
    md_lines.append("## 結果總表（以最佳門檻值計算 precision/recall/F1）")
    md_lines.append("")
    md_lines.append(df.to_markdown(index=False))
    md_lines.append("")
    md_lines.append("## 觀察重點")
    md_lines.append("")
    md_lines.append("- hard_negative_ratio 越高 → 負樣本越接近 spam 邊界 → 通常 precision/recall/F1 會下降（更貼近真實）。")
    md_lines.append("- support_vectors 若顯著增加，代表邊界變得更複雜、重疊/噪音增大。")
    md_lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return df, csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="FCC hard negatives ratio sensitivity sweep")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH, help="FCC dataset path (raw_fcc.csv).")
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="Negative/positive ratio (1.0 = balanced).")
    parser.add_argument("--start", type=float, default=0.0, help="Sweep start hard_negative_ratio.")
    parser.add_argument("--end", type=float, default=0.10, help="Sweep end hard_negative_ratio.")
    parser.add_argument("--step", type=float, default=0.01, help="Sweep step hard_negative_ratio.")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"找不到資料集: {args.data_path}")

    df, csv_path, md_path = run_sweep(
        data_path=args.data_path,
        negative_ratio=float(args.negative_ratio),
        start=float(args.start),
        end=float(args.end),
        step=float(args.step),
    )

    print("\n✅ Sweep 完成")
    print(f"  CSV: {csv_path}")
    print(f"  MD:  {md_path}")
    print(f"  Rows: {len(df)}")


if __name__ == "__main__":
    main()

