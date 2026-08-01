"""
eda.py — Archangel Exploratory Data Analysis Toolkit
=====================================================
Reusable EDA visualization functions aligned with poc-ml.mdc Stage 1 (§2.2).

Produces 6 standard EDA outputs:
  1. plot_distributions()          — Histograms + KDE (§2.2.5, §2.2.6)
  2. plot_correlation_heatmap()    — Feature correlation matrix (§2.2.7)
  3. plot_boxplots()               — Outlier detection (§2.2.9)
  4. plot_target_distribution()    — Class imbalance visualization
  5. plot_scatter_matrix()         — Feature-target relationships (§2.2.8)
  6. generate_data_quality_report() — Checklist verification

Scope: Exploratory data analysis for the anti-fraud detection pipeline
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from typing import Optional
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Matplotlib 中文支援
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Distributions — Histogram + KDE  (§2.2.5, §2.2.6)
# ─────────────────────────────────────────────────────────────────────────────

def plot_distributions(
    df: pd.DataFrame,
    numeric_cols: list[str] = None,
    target_col: str = None,
    save_path: str = "eda_distributions.png",
) -> None:
    """
    Plot histograms with overlaid KDE for each numeric feature.

    If target_col is provided, distributions are split by class
    to reveal feature separation between fraud / non-fraud.

    Args:
        df:           DataFrame with features
        numeric_cols: Columns to plot (auto-detected if None)
        target_col:   Optional target column for class-split view
        save_path:    Output file path
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)

    n = len(numeric_cols)
    if n == 0:
        logger.warning("No numeric columns to plot.")
        return

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        if target_col and target_col in df.columns:
            for label, color, name in [(0, '#2196F3', '正常(0)'), (1, '#F44336', '詐騙(1)')]:
                subset = df[df[target_col] == label][col].dropna()
                ax.hist(subset, bins=50, alpha=0.5, color=color, label=name, density=True)
                if len(subset) > 1:
                    try:
                        subset.plot.kde(ax=ax, color=color, linewidth=1.5)
                    except Exception:
                        pass  # KDE 在常數值特徵上會失敗，跳過
            ax.legend(fontsize=8)
        else:
            ax.hist(df[col].dropna(), bins=50, alpha=0.7, color='steelblue', density=True)
            if len(df[col].dropna()) > 1:
                try:
                    df[col].dropna().plot.kde(ax=ax, color='darkblue', linewidth=1.5)
                except Exception:
                    pass  # KDE 在常數值特徵上會失敗，跳過

        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.set_xlabel('')

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("數值特徵分佈 — Histogram + KDE", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"[EDA] 分佈圖已儲存: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Correlation Heatmap  (§2.2.7)
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    df: pd.DataFrame,
    numeric_cols: list[str] = None,
    save_path: str = "eda_correlation_heatmap.png",
) -> pd.DataFrame:
    """
    Plot feature correlation matrix as a heatmap.

    Flags feature pairs with |correlation| > 0.9 as potential
    multicollinearity issues (poc-ml.mdc §2.2.7).

    Returns:
        Correlation matrix DataFrame
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    corr = df[numeric_cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols) * 0.8),
                                     max(6, len(numeric_cols) * 0.6)))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax,
                vmin=-1, vmax=1, annot_kws={"size": 8})
    ax.set_title("特徵相關性矩陣", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Flag high correlations
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.9:
                high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
    if high_corr:
        logger.warning(f"[EDA] 高相關性特徵對 (|r| > 0.9): {high_corr}")
    else:
        logger.info("[EDA] 無高度共線性特徵 (|r| < 0.9)")

    logger.info(f"[EDA] 相關性熱圖已儲存: {save_path}")
    return corr


# ─────────────────────────────────────────────────────────────────────────────
# 3. Box Plots — Outlier Detection  (§2.2.9, §2.3)
# ─────────────────────────────────────────────────────────────────────────────

def plot_boxplots(
    df: pd.DataFrame,
    numeric_cols: list[str] = None,
    target_col: str = None,
    save_path: str = "eda_boxplots.png",
) -> dict:
    """
    Box plots for outlier detection.

    Returns dict with IQR-based outlier counts per column.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)

    n = len(numeric_cols)
    if n == 0:
        return {}

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    outlier_counts = {}

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        data = df[col].dropna()

        if target_col and target_col in df.columns:
            plot_df = df[[col, target_col]].dropna()
            sns.boxplot(x=target_col, y=col, data=plot_df, ax=ax)
        else:
            sns.boxplot(y=data, ax=ax, color='steelblue')

        # IQR outlier count
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        n_outliers = ((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).sum()
        outlier_counts[col] = int(n_outliers)

        ax.set_title(f"{col}\n(離群值: {n_outliers})", fontsize=10, fontweight='bold')

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("盒鬚圖 — 離群值偵測", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"[EDA] 盒鬚圖已儲存: {save_path}  |  離群值: {outlier_counts}")
    return outlier_counts


# ─────────────────────────────────────────────────────────────────────────────
# 4. Target Distribution — Class Imbalance
# ─────────────────────────────────────────────────────────────────────────────

def plot_target_distribution(
    y: pd.Series,
    label_names: dict = None,
    save_path: str = "eda_target_distribution.png",
) -> dict:
    """
    Visualize class distribution with count + percentage.

    Critical for fraud detection: highlights the imbalance ratio
    that drives model selection (scale_pos_weight, SMOTE, etc.).

    Returns:
        dict with class counts and ratio
    """
    if label_names is None:
        label_names = {0: "正常 (0)", 1: "詐騙 (1)"}

    counts = y.value_counts().sort_index()
    total = len(y)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    colors = ['#2196F3', '#F44336']
    bars = axes[0].bar(
        [label_names.get(k, str(k)) for k in counts.index],
        counts.values,
        color=colors[:len(counts)],
        edgecolor='white',
        linewidth=1.5,
    )
    for bar, count in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.005,
                     f"{count:,}\n({count/total*100:.1f}%)",
                     ha='center', fontsize=11, fontweight='bold')
    axes[0].set_title("類別分佈", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("樣本數", fontsize=11)

    # Pie chart
    axes[1].pie(
        counts.values,
        labels=[label_names.get(k, str(k)) for k in counts.index],
        colors=colors[:len(counts)],
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 11},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    )
    axes[1].set_title("類別比例", fontsize=13, fontweight='bold')

    ratio = counts.iloc[0] / counts.iloc[-1] if len(counts) > 1 and counts.iloc[-1] > 0 else 0
    fig.suptitle(f"目標變數分佈 — 不均衡比例 {ratio:.0f}:1",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    result = {
        "counts": counts.to_dict(),
        "total": total,
        "imbalance_ratio": round(ratio, 1),
    }
    logger.info(f"[EDA] 目標分佈已儲存: {save_path}  |  不均衡比例: {ratio:.0f}:1")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Scatter Matrix — Feature-Target Relationships  (§2.2.8)
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter_matrix(
    df: pd.DataFrame,
    target_col: str,
    numeric_cols: list[str] = None,
    max_features: int = 6,
    sample_size: int = 5000,
    save_path: str = "eda_scatter_matrix.png",
) -> None:
    """
    Pair-wise scatter plot matrix colored by target class.

    Subsamples for performance when dataset is large.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

    # Limit features for readability
    numeric_cols = numeric_cols[:max_features]
    if not numeric_cols:
        logger.warning("[EDA] 無可用數值欄位，略過散佈圖矩陣")
        return

    if target_col not in df.columns:
        logger.warning(f"[EDA] 找不到目標欄位 {target_col}，改用無 hue 的散佈圖矩陣")
        plot_df = df[numeric_cols].dropna()
        if plot_df.empty:
            logger.warning("[EDA] 散佈圖矩陣資料為空，略過")
            return
        if len(plot_df) > sample_size:
            plot_df = plot_df.sample(sample_size, random_state=42)

        g = sns.pairplot(
            plot_df,
            diag_kind='kde',
            plot_kws={'alpha': 0.3, 's': 10},
            height=2.2,
            aspect=1,
        )
        g.figure.suptitle("特徵散佈圖矩陣 (無標籤)", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        logger.info(f"[EDA] 散佈圖矩陣已儲存: {save_path}")
        return

    plot_df = df[numeric_cols + [target_col]].dropna()
    if plot_df.empty:
        logger.warning("[EDA] 散佈圖矩陣資料為空，略過")
        return
    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(sample_size, random_state=42)

    g = sns.pairplot(
        plot_df, hue=target_col,
        diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10},
        height=2.2, aspect=1,
    )
    g.figure.suptitle("特徵散佈圖矩陣 (藍=正常 / 紅=詐騙)",
                       fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    logger.info(f"[EDA] 散佈圖矩陣已儲存: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Data Quality Report  (poc-ml.mdc Checklist)
# ─────────────────────────────────────────────────────────────────────────────

def generate_data_quality_report(
    df: pd.DataFrame,
    target_col: str = "Label",
) -> dict:
    """
    Automated data quality report aligned with poc-ml.mdc checklist.

    Checks:
      - Missing value ratio per column (< 20% threshold)
      - Outlier counts (IQR method)
      - Class distribution / imbalance ratio
      - Feature types summary
      - Z-score outlier detection
      - Duplicate rows

    Returns:
        dict with all quality metrics and pass/fail flags
    """
    report = {}
    total = len(df)

    print("\n" + "═" * 60)
    print("  📋 資料品質檢查報告 (poc-ml.mdc Checklist)")
    print("═" * 60)

    # ── 1. Shape & Types ────────────────────────────────────────────
    report["shape"] = df.shape
    report["dtypes"] = df.dtypes.value_counts().to_dict()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"\n  形狀: {df.shape[0]:,} 列 × {df.shape[1]} 欄")
    print(f"  數值欄位: {len(numeric_cols)}  |  類別欄位: {len(cat_cols)}")

    # ── 2. Missing Values ──────────────────────────────────────────
    missing = df.isnull().sum()
    missing_pct = (missing / total * 100).round(2)
    report["missing"] = missing_pct.to_dict()
    max_missing = missing_pct.max()
    missing_pass = max_missing < 20

    print(f"\n  ▸ 缺失值檢查:")
    if missing.sum() == 0:
        print(f"    ✅ 無缺失值")
    else:
        for col in missing[missing > 0].index:
            flag = "⚠️" if missing_pct[col] >= 20 else "✅"
            print(f"    {flag} {col}: {missing[col]} ({missing_pct[col]:.1f}%)")
    report["missing_check_pass"] = missing_pass

    # ── 3. Duplicates ──────────────────────────────────────────────
    n_dupes = df.duplicated().sum()
    report["duplicates"] = int(n_dupes)
    print(f"\n  ▸ 重複列: {n_dupes} ({n_dupes/total*100:.2f}%)")

    # ── 4. Outliers (IQR) ──────────────────────────────────────────
    outlier_summary = {}
    target_in_num = target_col in numeric_cols
    check_cols = [c for c in numeric_cols if c != target_col]

    print(f"\n  ▸ 離群值偵測 (IQR 方法):")
    for col in check_cols:
        data = df[col].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        n_out = int(((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).sum())
        outlier_summary[col] = {"count": n_out, "pct": round(n_out / len(data) * 100, 2)}
        flag = "⚠️" if n_out / len(data) > 0.05 else "✅"
        print(f"    {flag} {col}: {n_out} ({n_out/len(data)*100:.1f}%)")
    report["outliers"] = outlier_summary

    # ── 5. Z-score Extreme Values ──────────────────────────────────
    print(f"\n  ▸ Z-score 極端值 (|z| > 3):")
    zscore_summary = {}
    for col in check_cols:
        data = df[col].dropna()
        if data.std() == 0:
            continue
        z = np.abs(scipy_stats.zscore(data))
        n_extreme = int((z > 3).sum())
        zscore_summary[col] = n_extreme
        flag = "⚠️" if n_extreme > 0 else "✅"
        print(f"    {flag} {col}: {n_extreme} 筆")
    report["zscore_extremes"] = zscore_summary

    # ── 6. Target Distribution ─────────────────────────────────────
    if target_col in df.columns:
        target_dist = df[target_col].value_counts()
        ratio = target_dist.iloc[0] / target_dist.iloc[-1] if len(target_dist) > 1 else 0
        report["target_distribution"] = target_dist.to_dict()
        report["imbalance_ratio"] = round(ratio, 1)
        print(f"\n  ▸ 目標變數分佈:")
        for label, count in target_dist.items():
            print(f"    Label {label}: {count:>7,} ({count/total*100:.1f}%)")
        print(f"    不均衡比例: {ratio:.0f}:1")
        if ratio > 5:
            print(f"    ⚠️ 建議使用 SMOTE 或 scale_pos_weight 處理不均衡")

    # ── 7. Categorical Feature Summary ─────────────────────────────
    if cat_cols:
        print(f"\n  ▸ 類別特徵摘要:")
        cat_summary = {}
        for col in cat_cols:
            n_unique = df[col].nunique()
            cat_summary[col] = n_unique
            encoding_hint = "One-Hot" if n_unique < 10 else "Target Encoding"
            print(f"    {col}: {n_unique} 類別 → 建議 {encoding_hint}")
        report["categorical_summary"] = cat_summary

    # ── 8. Basic Statistics ────────────────────────────────────────
    report["describe"] = df.describe().to_dict()

    # ── Checklist Summary ──────────────────────────────────────────
    print(f"\n  {'─' * 56}")
    print(f"  poc-ml.mdc 資料準備 Checklist:")
    checks = [
        ("缺失值比例 < 20%", missing_pass),
        ("離群值已識別", len(outlier_summary) > 0),
        ("類別特徵已檢視", len(cat_cols) >= 0),
        ("數值特徵已檢視", len(numeric_cols) > 0),
        ("目標變數分佈已確認", target_col in df.columns),
    ]
    for desc, passed in checks:
        mark = "✅" if passed else "❌"
        print(f"    {mark} {desc}")
    report["checklist"] = {desc: passed for desc, passed in checks}

    print("═" * 60 + "\n")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: Run Full EDA Suite
# ─────────────────────────────────────────────────────────────────────────────

def run_full_eda(
    df: pd.DataFrame,
    target_col: str = "Label",
    save_dir: str = ".",
) -> dict:
    """
    Execute complete EDA suite: all 6 functions in sequence.

    Args:
        df:         Raw DataFrame (before cleaning)
        target_col: Target variable column name
        save_dir:   Directory for output images

    Returns:
        dict with quality report and outlier counts
    """
    os.makedirs(save_dir, exist_ok=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_no_target = [c for c in numeric_cols if c != target_col]

    print("\n🔍 執行完整 EDA 分析...\n")

    # 1. Distributions
    plot_distributions(df, num_no_target, target_col,
                       save_path=os.path.join(save_dir, "eda_distributions.png"))

    # 2. Correlation
    corr = plot_correlation_heatmap(df, numeric_cols,
                                     save_path=os.path.join(save_dir, "eda_correlation_heatmap.png"))

    # 3. Boxplots
    outliers = plot_boxplots(df, num_no_target, target_col,
                              save_path=os.path.join(save_dir, "eda_boxplots.png"))

    # 4. Target distribution
    target_info = plot_target_distribution(
        df[target_col] if target_col in df.columns else pd.Series(dtype=int),
        save_path=os.path.join(save_dir, "eda_target_distribution.png"),
    )

    # 5. Scatter matrix
    plot_scatter_matrix(df, target_col, num_no_target,
                        save_path=os.path.join(save_dir, "eda_scatter_matrix.png"))

    # 6. Quality report
    report = generate_data_quality_report(df, target_col)

    print("✅ EDA 完成！所有圖表已儲存至:", save_dir)
    return {"quality_report": report, "outlier_counts": outliers, "target_info": target_info}


if __name__ == "__main__":
    print("請透過 notebooks/01_data_exploration.ipynb 執行 EDA。")
