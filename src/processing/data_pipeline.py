"""
data_pipeline.py — Archangel Data Cleaning & Feature Engineering Pipeline
=========================================================================
Loads raw fraud dataset, applies cleaning rules, feature engineering,
and prepares train-ready feature matrices.

Supports TWO dataset schemas automatically:
  • fraud_1000_dataset.csv  — legacy schema (Tags, Transaction_Amount, Is_Fraud…)
  • fraud_100000_dataset.csv — production schema (Financial_Loss, Age_Group, Label…)

Role Target: Data Research Engineer @ Gogolook ISL
"""

import os
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42

# Resolve default dataset path relative to project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATASET    = os.path.join(_PROJECT_ROOT, "fraud_1000_dataset.csv")
LARGE_DATASET      = os.path.join(_PROJECT_ROOT, "fraud_100000_dataset.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Schema Detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_schema(df: pd.DataFrame) -> str:
    """
    Auto-detect which dataset schema we're working with.

    Returns:
        'legacy'      → fraud_1000 schema  (Tags / Transaction_Amount / Is_Fraud)
        'production'  → fraud_100000 schema (Financial_Loss / Age_Group / Label)
    """
    if "Is_Fraud" in df.columns:
        return "legacy"
    elif "Label" in df.columns:
        return "production"
    raise ValueError(f"Unknown schema — columns: {df.columns.tolist()}")


# ─────────────────────────────────────────────────────────────────────────────
# Shared Utilities
# ─────────────────────────────────────────────────────────────────────────────

def remove_outliers_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0) -> pd.DataFrame:
    """
    Z-score outlier removal (guides §1.1).

    Keeps rows where |z-score| < threshold.  More principled than raw
    value caps — adapts to dataset scale automatically.

    Args:
        df:         Input DataFrame
        col:        Numeric column to apply Z-score filter on
        threshold:  Standard deviation cutoff (default: 3σ)

    Returns:
        Filtered DataFrame
    """
    z = np.abs(scipy_stats.zscore(df[col].dropna()))
    valid_idx = df[col].dropna().index[z < threshold]
    removed = len(df) - len(valid_idx)
    logger.info(f"   Z-score [{col}]: 移除 {removed} 筆離群值 (|z|>{threshold}σ)")
    return df.loc[valid_idx].copy()


def clip_extremes(df: pd.DataFrame, col: str, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    """
    Percentile-based clipping to smooth extreme values (guides §1.1).

    Preferred over hard deletion when extreme values may carry signal
    (e.g., unusually high Report_Count could indicate a scam call center).
    """
    lo = df[col].quantile(lower_q)
    hi = df[col].quantile(upper_q)
    df[col] = df[col].clip(lower=lo, upper=hi)
    logger.info(f"   Clipping [{col}]: [{lo:.2f}, {hi:.2f}]")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Production Schema Pipeline  (fraud_100000_dataset.csv)
# ─────────────────────────────────────────────────────────────────────────────

def _pipeline_production(df: pd.DataFrame) -> tuple:
    """
    Full pipeline for the 100k production dataset.

    Schema: Report_Time, Phone_Number, Report_Count, Financial_Loss,
            Age_Group, Education, Cluster_ID, Label

    Design notes:
      - Cluster_ID is ground-truth cluster info; drop before training to
        avoid data leakage, but keep for unsupervised analysis (§3).
      - Financial_Loss has heavy right-tail → RobustScaler preferred.
      - Age_Group / Education use One-Hot Encoding (low cardinality,
        no ordinal relationship to assume).
      - C_POISON_BOT records are filtered out (bot-injected noise).
    """
    print("\n  📋 Schema: production (fraud_100000_dataset.csv)")

    # ── Step 1: Filter bot / poison records ──────────────────────────────
    n_before = len(df)
    df = df[df["Cluster_ID"] != "C_POISON_BOT"].copy()
    logger.info(f"   移除 C_POISON_BOT 記錄: {n_before - len(df)} 筆")

    # ── Step 2: Z-score outlier removal on Report_Count ─────────────────
    df = remove_outliers_zscore(df, "Report_Count", threshold=3.0)

    # ── Step 3: Clip Financial_Loss (heavy right-tail) ────────────────
    df = clip_extremes(df, "Financial_Loss", lower_q=0.01, upper_q=0.99)

    # ── Step 4: Drop non-predictive / leaky columns ──────────────────────
    # Keep Cluster_ID in a separate column for unsupervised analysis
    cluster_ids = df["Cluster_ID"].copy()
    drop_cols = [c for c in ["Phone_Number", "Report_Time", "Cluster_ID"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # ── Step 5: One-Hot Encoding for categoricals ─────────────────────
    cat_cols = [c for c in ["Age_Group", "Education"] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    logger.info(f"   One-Hot Encoding 完成 — 新增 {df.shape[1] - 3} 個 dummy 特徵")

    # ── Step 6: RobustScaler on numerics (outlier-resistant) ──────────
    # RobustScaler uses median and IQR → handles financial data skew better
    # than StandardScaler which is distorted by extreme fraud amounts.
    scaler = RobustScaler()
    numeric_cols = [c for c in ["Report_Count", "Financial_Loss"] if c in df.columns]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    logger.info(f"   RobustScaler 已套用: {numeric_cols}")

    # ── Step 7: Separate X / y ──────────────────────────────────────────
    y = df["Label"]
    X = df.drop(columns=["Label"])

    return X, y, scaler, cluster_ids


# ─────────────────────────────────────────────────────────────────────────────
# Legacy Schema Pipeline  (fraud_1000_dataset.csv)
# ─────────────────────────────────────────────────────────────────────────────

def _pipeline_legacy(df: pd.DataFrame) -> tuple:
    """
    Pipeline for the 1k legacy dataset.

    Schema: Incident_ID, Report_Time, Phone_Number, Tags,
            Report_Count, Transaction_Amount, Victim_Demographic, Is_Fraud
    """
    print("\n  📋 Schema: legacy (fraud_1000_dataset.csv)")

    # ── Step 1: Remove bot / noise records ───────────────────────────────
    n_before = len(df)
    if "Tags" in df.columns:
        df = df[~df["Tags"].str.contains("機器人測試", na=False)].copy()
    df = df[df["Report_Count"] < 5000]
    logger.info(f"   清洗後: {len(df)} 筆 (移除 {n_before - len(df)} 筆雜訊)")

    # ── Step 2: Z-score on Report_Count ──────────────────────────────────
    df = remove_outliers_zscore(df, "Report_Count", threshold=3.0)

    # ── Step 3: Drop non-predictive columns ──────────────────────────────
    drop_cols = [c for c in ["Incident_ID", "Phone_Number", "Report_Time"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # ── Step 4: One-Hot Encoding ──────────────────────────────────────────
    cat_cols = [c for c in ["Victim_Demographic", "Tags"] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # ── Step 5: RobustScaler ─────────────────────────────────────────────
    scaler = RobustScaler()
    numeric_cols = [c for c in ["Report_Count", "Transaction_Amount"] if c in df.columns]
    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    logger.info(f"   RobustScaler 已套用: {numeric_cols}")

    # ── Step 6: Separate X / y ──────────────────────────────────────────
    y = df["Is_Fraud"]
    X = df.drop(columns=["Is_Fraud"])

    return X, y, scaler, pd.Series(dtype=str)  # no cluster_ids in legacy


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def clean_and_prepare_data(file_path: str = None) -> tuple:
    """
    Full data cleaning + feature engineering pipeline.

    Auto-detects dataset schema and routes to the appropriate sub-pipeline.

    Returns:
        (X, y, scaler, cluster_ids)
        cluster_ids is a Series with ground-truth cluster labels (production schema)
        or empty Series (legacy schema).
    """
    if file_path is None:
        file_path = DEFAULT_DATASET

    np.random.seed(SEED)
    print("啟動防詐數據集清洗與特徵工程管線...")

    df = pd.read_csv(file_path)
    print(f"原始資料筆數: {len(df)}")

    schema = _detect_schema(df)

    if schema == "production":
        X, y, scaler, cluster_ids = _pipeline_production(df)
    else:
        X, y, scaler, cluster_ids = _pipeline_legacy(df)

    print(f"\n✅ 特徵工程完成！共萃取 {X.shape[1]} 個特徵維度。")
    print(f"   正常案例: {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"   詐騙案例: {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")

    return X, y, scaler, cluster_ids


if __name__ == "__main__":
    # Quick smoke test — run both datasets
    print("=== Legacy 1k dataset ===")
    X, y, scaler, cids = clean_and_prepare_data(DEFAULT_DATASET)
    print(f"X shape: {X.shape}")

    print("\n=== Production 100k dataset ===")
    X, y, scaler, cids = clean_and_prepare_data(LARGE_DATASET)
    print(f"X shape: {X.shape}, cluster_ids: {cids.value_counts().to_dict()}")
