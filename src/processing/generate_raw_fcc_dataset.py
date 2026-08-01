#!/usr/bin/env python3
"""
generate_raw_fcc_dataset.py — 產生「極髒」原始層資料集 (raw_fcc.csv)
====================================================================

四層資料契約（Data Layer Contract）：
  ┌──────────────────┬──────────────────────────┬──────────────────────────────────┐
  │ 層級              │ 檔案名稱                  │ 品質保證                          │
  ├──────────────────┼──────────────────────────┼──────────────────────────────────┤
  │ raw_*（原始層）    │ raw_fcc.csv              │ 無保證；含缺失/錯誤/重複/亂碼      │
  │ staging_*（初清層）│ FCC_Consumer_..._2017.csv│ 已移除假號碼/極端日期/基礎格式      │
  │ clean_*（可訓練層）│ （pipeline 中間產物）      │ 完整清洗 + 特徵工程、無缺失值      │
  │ label_*（已標註層）│ label_100000_dataset.csv │ clean + Label 標記，可直接訓練     │
  └──────────────────┴──────────────────────────┴──────────────────────────────────┘

本腳本以 FCC staging 資料為基底，注入中度污染 (medium profile)：
  - 額外缺失值：非空值的 10-20% → NaN
  - 錯誤值：     非空值的 8-12%  → 格式污染/非法值/錯型態
  - 重複列：     原資料的 ~3%    → 完全重複
  - 亂碼列：     ~1%            → 完全隨機垃圾列

用途：壓力測試清洗管線的韌性 (robustness)。

Usage:
    python -m src.processing.generate_raw_fcc_dataset                      # 預設 medium
    python -m src.processing.generate_raw_fcc_dataset --profile heavy      # 重度污染
    python -m src.processing.generate_raw_fcc_dataset --profile light      # 輕度污染
    python -m src.processing.generate_raw_fcc_dataset --sample 200000      # 只取 20 萬列加速
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FCC_STAGING_PATH = os.path.join(_PROJECT_ROOT, "FCC_Consumer_Complaints_RAW_Data_2017.csv")
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs", "quality")
DEFAULT_OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "raw_fcc.csv")

# ── 污染設定檔 (profile) ──────────────────────────────────────────────────────
CORRUPTION_PROFILES = {
    "light": {
        "name": "light",
        "description": "輕度污染 — 缺失 5-10%、錯誤值 3-5%",
        "missing_rate_range": (0.05, 0.10),
        "error_rate_range": (0.03, 0.05),
        "duplicate_rate": 0.01,
        "garbage_row_rate": 0.005,
    },
    "medium": {
        "name": "medium",
        "description": "中度污染 — 缺失 10-20%、錯誤值 8-12%",
        "missing_rate_range": (0.10, 0.20),
        "error_rate_range": (0.08, 0.12),
        "duplicate_rate": 0.03,
        "garbage_row_rate": 0.01,
    },
    "heavy": {
        "name": "heavy",
        "description": "重度污染 — 缺失 20-35%、錯誤值 15-25%",
        "missing_rate_range": (0.20, 0.35),
        "error_rate_range": (0.15, 0.25),
        "duplicate_rate": 0.05,
        "garbage_row_rate": 0.02,
    },
}

# ── 各欄位的污染策略 ──────────────────────────────────────────────────────────

# 缺失值注入權重（越高 = 該欄位被注入越多缺失）
MISSING_WEIGHTS = {
    "Ticket ID": 0.0,                     # 保留 ID
    "Ticket Created": 0.6,
    "Date of Issue": 0.8,
    "Time of Issue": 0.9,
    "Form": 0.4,
    "Method": 0.8,
    "Issue": 0.9,
    "Caller ID Number": 0.7,
    "Type of Call or Messge": 0.8,
    "Advertiser Business Number": 0.5,    # 已經很高，少加
    "City": 0.6,
    "State": 0.6,
    "Zip": 0.6,
    "Location (Center point of the Zip Code)": 0.5,
}

# ── 錯誤值生成器 ─────────────────────────────────────────────────────────────

def _corrupt_caller_id(rng: np.random.RandomState, n: int) -> list:
    """產生格式錯誤的電話號碼。"""
    templates = [
        lambda: "".join(rng.choice(list("abcdefghij"), 10)),    # 全字母
        lambda: f"{rng.randint(0, 9999)}",                     # 位數不足
        lambda: "000-000-0000",                                # 已知假號碼
        lambda: "555-555-5555",
        lambda: f"+1-{rng.randint(100,999)}-{rng.randint(0,9999999):07d}",  # 國際格式 (超長)
        lambda: f"({rng.randint(100,999)}) {rng.randint(100,999)}-{rng.randint(1000,9999)}",  # 括號格式
        lambda: "N/A",
        lambda: "UNKNOWN",
        lambda: "111-111-1111",
        lambda: "999-999-9999",
        lambda: f"{rng.randint(-999, -100)}-{rng.randint(100,999)}-{rng.randint(1000,9999)}",  # 負數區號
    ]
    return [templates[rng.randint(0, len(templates))]() for _ in range(n)]


def _corrupt_time(rng: np.random.RandomState, n: int) -> list:
    """產生格式錯誤的時間字串。"""
    templates = [
        lambda: f"{rng.randint(25, 99)}:{rng.randint(0, 59):02d} am",  # 無效小時
        lambda: f"{rng.randint(1, 12)}:{rng.randint(60, 99)} pm",      # 無效分鐘
        lambda: "abc",
        lambda: "N/A",
        lambda: "midnight",
        lambda: "noon-ish",
        lambda: f"{rng.randint(0, 23):02d}{rng.randint(0, 59):02d}",   # 缺冒號 "1430"
        lambda: "",
        lambda: f"{rng.uniform(0, 24):.6f}",                            # 浮點數
        lambda: "25:00 pm",
    ]
    return [templates[rng.randint(0, len(templates))]() for _ in range(n)]


def _corrupt_date(rng: np.random.RandomState, n: int) -> list:
    """產生格式錯誤的日期字串。"""
    templates = [
        lambda: f"{rng.randint(13, 20)}/{rng.randint(32, 50)}/{rng.randint(2000, 2030)}",  # 無效月/日
        lambda: "not_a_date",
        lambda: "N/A",
        lambda: f"{rng.randint(1990, 2000)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",  # 錯誤年份格式 ISO
        lambda: f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}",   # 缺年份
        lambda: "01/01/1900",                                            # 極端過早
        lambda: "12/31/2099",                                            # 極端未來
        lambda: f"2017-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T00:00:00Z",  # ISO 8601
        lambda: str(rng.randint(1000000000, 2000000000)),                # Unix timestamp 字串
    ]
    return [templates[rng.randint(0, len(templates))]() for _ in range(n)]


def _corrupt_state(rng: np.random.RandomState, n: int) -> list:
    """產生無效的州別代碼。"""
    templates = [
        lambda: "XX",
        lambda: "ZZ",
        lambda: str(rng.randint(10, 99)),  # 數字
        lambda: "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 3)),  # 三碼
        lambda: "n/a",
        lambda: "Unknown",
        lambda: "",
    ]
    return [templates[rng.randint(0, len(templates))]() for _ in range(n)]


def _corrupt_zip(rng: np.random.RandomState, n: int) -> list:
    """產生無效的郵遞區號。"""
    templates = [
        lambda: float("nan"),
        lambda: float(rng.randint(-99999, -1)),        # 負數
        lambda: float(rng.randint(100000, 999999)),     # 六位數
        lambda: 0.0,
        lambda: 99999.0,
        lambda: float(rng.uniform(0, 1)),               # 小數
    ]
    return [templates[rng.randint(0, len(templates))]() for _ in range(n)]


def _corrupt_method(rng: np.random.RandomState, n: int) -> list:
    """產生垃圾 Method 值。"""
    templates = [
        lambda: "Pigeon Post",
        lambda: "Telepathy",
        lambda: "N/A",
        lambda: "",
        lambda: "unknowN",
        lambda: str(rng.randint(0, 100)),
        lambda: "Fax Machine (legacy)",
    ]
    return [templates[rng.randint(0, len(templates))]() for _ in range(n)]


def _corrupt_issue(rng: np.random.RandomState, n: int) -> list:
    """產生垃圾 Issue 值。"""
    templates = [
        lambda: "asdfghjkl",
        lambda: "COMPLAINT!!!",
        lambda: "N/A",
        lambda: "",
        lambda: "Robocals",       # typo
        lambda: "Telemarkting",   # typo
        lambda: "其他問題",        # 中文（格式不一致）
        lambda: "TEST_ENTRY",
        lambda: str(rng.randint(0, 100)),
    ]
    return [templates[rng.randint(0, len(templates))]() for _ in range(n)]


# 各欄位對應的錯誤值生成器
ERROR_GENERATORS = {
    "Caller ID Number": _corrupt_caller_id,
    "Time of Issue": _corrupt_time,
    "Date of Issue": _corrupt_date,
    "State": _corrupt_state,
    "Zip": _corrupt_zip,
    "Method": _corrupt_method,
    "Issue": _corrupt_issue,
}

# 錯誤值注入權重
ERROR_WEIGHTS = {
    "Caller ID Number": 1.0,
    "Time of Issue": 1.0,
    "Date of Issue": 1.0,
    "State": 0.8,
    "Zip": 0.8,
    "Method": 0.8,
    "Issue": 0.8,
}


# ── 主生成器 ──────────────────────────────────────────────────────────────────

def generate_dirty_dataset(
    source_path: str = None,
    output_path: str = None,
    profile: str = "medium",
    sample_rows: int = None,
    seed: int = 42,
) -> dict:
    """
    從 FCC staging 資料生成極髒原始層資料集。

    參數：
        source_path:  FCC staging CSV 路徑
        output_path:  輸出 CSV 路徑
        profile:      污染等級 ("light" / "medium" / "heavy")
        sample_rows:  只取前 N 列（加速測試用）
        seed:         隨機種子

    回傳：
        dict — 污染統計報告
    """
    if source_path is None:
        source_path = FCC_STAGING_PATH
    if output_path is None:
        output_path = DEFAULT_OUTPUT_PATH

    config = CORRUPTION_PROFILES[profile]
    rng = np.random.RandomState(seed)

    logger.info(f"=== 開始生成髒資料集 (profile={profile}) ===")
    logger.info(f"來源: {source_path}")

    # ── 載入 ──
    t0 = time.time()
    df = pd.read_csv(source_path, on_bad_lines="skip")
    if sample_rows is not None and sample_rows < len(df):
        df = df.head(sample_rows).copy()
        logger.info(f"取樣前 {sample_rows:,} 列")
    original_rows = len(df)
    original_cols = list(df.columns)
    logger.info(f"載入完成: {original_rows:,} × {len(original_cols)} ({time.time()-t0:.1f}s)")

    # 記錄原始缺失率（做 before/after 比較）
    original_missing = {c: float(df[c].isna().mean()) for c in df.columns}

    # ── 統計收集器 ──
    report = {
        "generated_at": datetime.now().isoformat(),
        "source_path": source_path,
        "output_path": output_path,
        "profile": config["name"],
        "profile_description": config["description"],
        "seed": seed,
        "original_rows": original_rows,
        "original_columns": original_cols,
        "original_missing_pct": {k: round(v * 100, 2) for k, v in original_missing.items()},
        "injected": {
            "missing": {},
            "errors": {},
            "duplicates": 0,
            "garbage_rows": 0,
        },
    }

    # ══════════════════════════════════════════════════════════════
    # Phase 1: 注入額外缺失值
    # ══════════════════════════════════════════════════════════════
    logger.info("Phase 1/4: 注入缺失值...")
    miss_lo, miss_hi = config["missing_rate_range"]

    for col in df.columns:
        weight = MISSING_WEIGHTS.get(col, 0.0)
        if weight == 0:
            continue

        # 只對目前非空的值下手
        non_null_mask = df[col].notna()
        n_non_null = non_null_mask.sum()
        if n_non_null == 0:
            continue

        # 該欄位的實際缺失注入率 = range 內按 weight 線性插值
        rate = miss_lo + (miss_hi - miss_lo) * weight
        n_to_nullify = int(n_non_null * rate)

        if n_to_nullify == 0:
            continue

        non_null_indices = df.index[non_null_mask].to_numpy()
        chosen = rng.choice(non_null_indices, size=min(n_to_nullify, len(non_null_indices)), replace=False)
        df.loc[chosen, col] = np.nan

        report["injected"]["missing"][col] = {
            "n_injected": int(len(chosen)),
            "rate_of_nonnull": round(len(chosen) / n_non_null * 100, 2),
        }
        logger.info(f"  {col}: +{len(chosen):,} NaN ({len(chosen)/n_non_null*100:.1f}% of non-null)")

    # ══════════════════════════════════════════════════════════════
    # Phase 2: 注入錯誤值
    # ══════════════════════════════════════════════════════════════
    logger.info("Phase 2/4: 注入錯誤值...")
    err_lo, err_hi = config["error_rate_range"]

    for col, gen_fn in ERROR_GENERATORS.items():
        if col not in df.columns:
            continue

        weight = ERROR_WEIGHTS.get(col, 0.5)

        # 只對目前非空的值下手
        non_null_mask = df[col].notna()
        n_non_null = int(non_null_mask.sum())
        if n_non_null == 0:
            continue

        rate = err_lo + (err_hi - err_lo) * weight
        n_to_corrupt = int(n_non_null * rate)
        if n_to_corrupt == 0:
            continue

        non_null_indices = df.index[non_null_mask].to_numpy()
        chosen = rng.choice(non_null_indices, size=min(n_to_corrupt, len(non_null_indices)), replace=False)

        # 生成錯誤值
        bad_values = gen_fn(rng, len(chosen))
        df.loc[chosen, col] = bad_values

        report["injected"]["errors"][col] = {
            "n_injected": int(len(chosen)),
            "rate_of_nonnull": round(len(chosen) / n_non_null * 100, 2),
            "sample_errors": bad_values[:5],
        }
        logger.info(f"  {col}: +{len(chosen):,} errors ({len(chosen)/n_non_null*100:.1f}% of non-null)")

    # ══════════════════════════════════════════════════════════════
    # Phase 3: 注入重複列
    # ══════════════════════════════════════════════════════════════
    logger.info("Phase 3/4: 注入重複列...")
    dup_rate = config["duplicate_rate"]
    n_dups = int(len(df) * dup_rate)

    if n_dups > 0:
        dup_indices = rng.choice(df.index, size=n_dups, replace=True)
        dup_rows = df.loc[dup_indices].copy().reset_index(drop=True)
        df = pd.concat([df, dup_rows], ignore_index=True)
        report["injected"]["duplicates"] = int(n_dups)
        logger.info(f"  +{n_dups:,} 重複列")

    # ══════════════════════════════════════════════════════════════
    # Phase 4: 注入垃圾列（完全隨機）
    # ══════════════════════════════════════════════════════════════
    logger.info("Phase 4/4: 注入垃圾列...")
    garbage_rate = config["garbage_row_rate"]
    n_garbage = int(original_rows * garbage_rate)

    if n_garbage > 0:
        garbage_data = {}
        for col in original_cols:
            if col == "Ticket ID":
                garbage_data[col] = rng.randint(-999999, -1, size=n_garbage)
            elif col == "Zip":
                garbage_data[col] = rng.uniform(-999, 999999, size=n_garbage)
            else:
                # 隨機 ASCII 字串
                garbage_data[col] = [
                    "".join(chr(rng.randint(32, 126)) for _ in range(rng.randint(1, 30)))
                    for _ in range(n_garbage)
                ]
        garbage_df = pd.DataFrame(garbage_data)
        df = pd.concat([df, garbage_df], ignore_index=True)
        report["injected"]["garbage_rows"] = int(n_garbage)
        logger.info(f"  +{n_garbage:,} 垃圾列")

    # ── 打亂行序 ──
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # ══════════════════════════════════════════════════════════════
    # 輸出
    # ══════════════════════════════════════════════════════════════
    final_rows = len(df)
    report["final_rows"] = final_rows
    report["final_missing_pct"] = {
        c: round(float(df[c].isna().mean()) * 100, 2) for c in original_cols
    }

    # 計算品質對比
    report["quality_delta"] = {}
    for c in original_cols:
        before = report["original_missing_pct"].get(c, 0)
        after = report["final_missing_pct"].get(c, 0)
        report["quality_delta"][c] = {
            "missing_before_pct": before,
            "missing_after_pct": after,
            "delta_pct": round(after - before, 2),
        }

    # 寫 CSV
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    logger.info(f"寫入 CSV: {output_path}")
    t1 = time.time()
    df.to_csv(output_path, index=False)
    logger.info(f"寫入完成 ({time.time()-t1:.1f}s)")
    logger.info(f"最終: {final_rows:,} 列 (原始 {original_rows:,} + "
                f"重複 {report['injected']['duplicates']:,} + "
                f"垃圾 {report['injected']['garbage_rows']:,})")

    return report


def write_corruption_report(report: dict, output_dir: str = None):
    """
    將污染統計報告寫成 JSON + Markdown。

    參數：
        report:     generate_dirty_dataset() 的回傳值
        output_dir: 輸出目錄
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # ── JSON ──
    json_path = os.path.join(output_dir, "raw_fcc_corruption_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"JSON 報告: {json_path}")

    # ── Markdown ──
    md_path = os.path.join(output_dir, "raw_fcc_corruption_report.md")
    md = []
    md.append("# FCC Raw 資料污染報告")
    md.append("")
    md.append(f"- **生成時間**: `{report['generated_at']}`")
    md.append(f"- **來源**: `{report['source_path']}`")
    md.append(f"- **輸出**: `{report['output_path']}`")
    md.append(f"- **污染等級**: `{report['profile']}` — {report['profile_description']}")
    md.append(f"- **隨機種子**: `{report['seed']}`")
    md.append(f"- **原始列數**: `{report['original_rows']:,}`")
    md.append(f"- **最終列數**: `{report['final_rows']:,}`")
    md.append("")

    md.append("## 一、注入缺失值")
    md.append("")
    md.append("| 欄位 | 注入數 | 佔非空% |")
    md.append("|------|--------|---------|")
    for col, info in report["injected"]["missing"].items():
        md.append(f"| `{col}` | {info['n_injected']:,} | {info['rate_of_nonnull']}% |")
    md.append("")

    md.append("## 二、注入錯誤值")
    md.append("")
    md.append("| 欄位 | 注入數 | 佔非空% | 範例 |")
    md.append("|------|--------|---------|------|")
    for col, info in report["injected"]["errors"].items():
        samples = ", ".join(str(s)[:25] for s in info.get("sample_errors", [])[:3])
        md.append(f"| `{col}` | {info['n_injected']:,} | {info['rate_of_nonnull']}% | {samples} |")
    md.append("")

    md.append("## 三、重複與垃圾列")
    md.append("")
    md.append(f"- 重複列: `{report['injected']['duplicates']:,}`")
    md.append(f"- 垃圾列: `{report['injected']['garbage_rows']:,}`")
    md.append("")

    md.append("## 四、欄位缺失率對比 (Before → After)")
    md.append("")
    md.append("| 欄位 | Before% | After% | Δ% |")
    md.append("|------|---------|--------|-----|")
    for col, delta in report["quality_delta"].items():
        md.append(f"| `{col}` | {delta['missing_before_pct']} | {delta['missing_after_pct']} | +{delta['delta_pct']} |")
    md.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info(f"Markdown 報告: {md_path}")

    return json_path, md_path


# ── CLI 入口 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="產生 FCC 極髒原始層資料集 (raw_fcc.csv)",
    )
    parser.add_argument(
        "--profile", type=str, default="medium",
        choices=list(CORRUPTION_PROFILES.keys()),
        help="污染等級 (default: medium)",
    )
    parser.add_argument(
        "--source", type=str, default=FCC_STAGING_PATH,
        help="來源 CSV 路徑",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_PATH,
        help="輸出 CSV 路徑",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="只取前 N 列（加速測試）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="隨機種子",
    )
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║       🧪 FCC Raw Dataset Generator — 髒資料產生器           ║
╠══════════════════════════════════════════════════════════════╣
║  Profile: {profile:<49}║
║  Source:  {source:<49}║
║  Output:  {output:<49}║
║  Seed:    {seed:<49}║
╚══════════════════════════════════════════════════════════════╝
""".format(
        profile=args.profile,
        source=os.path.basename(args.source)[:49],
        output=os.path.basename(args.output)[:49],
        seed=args.seed,
    ))

    t_start = time.time()

    report = generate_dirty_dataset(
        source_path=args.source,
        output_path=args.output,
        profile=args.profile,
        sample_rows=args.sample,
        seed=args.seed,
    )

    json_path, md_path = write_corruption_report(report)

    elapsed = time.time() - t_start
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                   ✅ 生成完成                                ║
╠══════════════════════════════════════════════════════════════╣
║  原始列數:   {report['original_rows']:<46,}║
║  最終列數:   {report['final_rows']:<46,}║
║  重複列:     {report['injected']['duplicates']:<46,}║
║  垃圾列:     {report['injected']['garbage_rows']:<46,}║
║  耗時:       {elapsed:.1f}s{' '*(44-len(f'{elapsed:.1f}s'))}║
╠══════════════════════════════════════════════════════════════╣
║  Outputs:                                                    ║
║    📁 {os.path.basename(args.output):<55}║
║    📄 {os.path.basename(json_path):<55}║
║    📝 {os.path.basename(md_path):<55}║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
