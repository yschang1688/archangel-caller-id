"""
unsupervised.py — Archangel Unsupervised Fraud Pattern Discovery
================================================================
Identifies unknown scam syndicates and attack vectors without labels.

Modules:
  1. DBSCAN — density-based clustering to find scam call center "rings"
  2. t-SNE  — 2D projection to visually confirm cluster separation

Key insight: In production, ~30% of scam calls come from NEW patterns
not yet in the training labels. Unsupervised discovery feeds those signals
back into the labeling pipeline (Data-centric AI closed loop).

Dataset:  label_100000_dataset.csv
          Ground-truth clusters in Cluster_ID column:
            C_NORMAL       → 90k  legitimate calls
            C_88X_RING     →  4k  coordinated scam ring (hot-key pattern)
            C_DEMO_TARGET  →  4k  targeted demo victims
            C_POISON_BOT   →  2k  bot accounts (filtered upstream)

Role Target: Data Research Engineer @ Gogolook ISL
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42

# Matplotlib 中文支援
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────────────────────────────────────
# DBSCAN Clustering  (guides §3.1)
# ─────────────────────────────────────────────────────────────────────────────

def run_dbscan(
    X: pd.DataFrame,
    eps: float = 0.6,
    min_samples: int = 10,
    ground_truth: pd.Series = None,
) -> dict:
    """
    DBSCAN clustering to discover scam syndicates.

    Why DBSCAN over K-means?
      K-means requires specifying k and assumes spherical clusters.
      Scam call rings form irregular, high-density pockets in feature space
      (same call center → similar Report_Count, Financial_Loss patterns).
      DBSCAN finds these automatically and marks sparse regions as noise (-1),
      which is often legitimate traffic.

    Args:
        X:             Feature matrix (preprocessed, scaled)
        eps:           Neighborhood radius (tune based on silhouette score)
        min_samples:   Min points to form a core point
        ground_truth:  Optional Series of true cluster labels for ARI evaluation

    Returns:
        dict with cluster labels, stats, and ARI score
    """
    np.random.seed(SEED)
    print(f"\n[DBSCAN] eps={eps}, min_samples={min_samples}")

    # DBSCAN benefits from feature scaling (uses distance metric)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X_scaled)

    # ── Cluster statistics ───────────────────────────────────────────────
    unique_labels = sorted(set(labels))
    n_clusters    = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise       = (labels == -1).sum()

    print(f"   發現群集數: {n_clusters}")
    print(f"   雜訊點(正常通話): {n_noise} ({n_noise/len(labels)*100:.1f}%)")

    # Per-cluster fraud analysis
    y = pd.Series(index=X.index, dtype=int)
    if hasattr(X, 'index'):
        pass  # Keep index alignment

    cluster_stats = []
    for cid in unique_labels:
        mask = labels == cid
        name = "noise" if cid == -1 else f"cluster_{cid}"
        count = mask.sum()
        cluster_stats.append({"cluster": name, "count": int(count),
                               "pct": round(count / len(labels) * 100, 2)})
        print(f"   {name:>12}: {count:>6} 筆 ({count/len(labels)*100:.1f}%)")

    # ── Silhouette Score (exclude noise) ──────────────────────────────────
    mask_valid = labels != -1
    sil_score = None
    if mask_valid.sum() > 1 and n_clusters > 1:
        sil_score = silhouette_score(X_scaled[mask_valid], labels[mask_valid],
                                     sample_size=min(5000, mask_valid.sum()),
                                     random_state=SEED)
        print(f"\n   Silhouette Score: {sil_score:.4f}")
        print(f"   (0.7+ excellent | 0.5+ reasonable | <0.25 overlapping)")

    # ── Adjusted Rand Index vs ground truth ─────────────────────────────
    ari = None
    if ground_truth is not None and len(ground_truth) == len(labels):
        # Map ground truth strings to ints
        gt_int = pd.Categorical(ground_truth).codes
        ari = adjusted_rand_score(gt_int, labels)
        print(f"   Adjusted Rand Index vs Cluster_ID: {ari:.4f}")
        print(f"   (1.0 = perfect | 0.0 = random | negative = worse than random)")

    return {
        "labels":         labels,
        "n_clusters":     n_clusters,
        "n_noise":        int(n_noise),
        "cluster_stats":  cluster_stats,
        "silhouette":     sil_score,
        "ari":            ari,
    }


# ─────────────────────────────────────────────────────────────────────────────
# t-SNE Visualization  (guides §3.2)
# ─────────────────────────────────────────────────────────────────────────────

def run_tsne_and_plot(
    X: pd.DataFrame,
    y: pd.Series,
    dbscan_labels: np.ndarray = None,
    ground_truth_clusters: pd.Series = None,
    n_samples: int = 5000,
    save_dir: str = ".",
) -> np.ndarray:
    """
    t-SNE dimensionality reduction + dual visualization.

    Produces two side-by-side plots:
      Left:  Colored by fraud label (0/1)
      Right: Colored by DBSCAN cluster OR ground-truth Cluster_ID

    Why t-SNE?
      High-dimensional fraud features are hard to visualize.
      t-SNE preserves LOCAL structure — similar calls cluster together.
      If a scam ring exists, it should form a tight island separate from
      the mass of normal calls. This is the visual "proof" for stakeholders.

    Note: t-SNE is computationally expensive — subsample for large datasets.

    Args:
        X:                     Feature matrix
        y:                     Fraud labels (0/1)
        dbscan_labels:         Optional DBSCAN cluster assignments
        ground_truth_clusters: Optional Cluster_ID column for comparison
        n_samples:             Max samples for t-SNE (performance cap)
        save_dir:              Directory to save plots

    Returns:
        X_tsne: 2D embedding array
    """
    np.random.seed(SEED)

    # Subsample for performance
    if len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        X_sub = X.iloc[idx]
        y_sub = y.iloc[idx]
        dbscan_sub = dbscan_labels[idx] if dbscan_labels is not None else None
        gt_sub     = ground_truth_clusters.iloc[idx] if ground_truth_clusters is not None else None
        print(f"\n[t-SNE] 降採樣 {len(X)} → {n_samples} 筆以加速計算...")
    else:
        X_sub, y_sub = X, y
        dbscan_sub, gt_sub = dbscan_labels, ground_truth_clusters

    # Scale before t-SNE
    X_scaled = StandardScaler().fit_transform(X_sub)

    print(f"[t-SNE] 計算 2D 嵌入 (perplexity=30, iter=1000)...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                random_state=SEED, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_scaled)
    print(f"   KL-divergence: {tsne.kl_divergence_:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: fraud labels
    colors_label = ['#2196F3' if v == 0 else '#F44336' for v in y_sub]
    axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1],
                    c=colors_label, alpha=0.4, s=8, linewidths=0)
    axes[0].set_title("t-SNE — 標記分佈\n(藍=正常 / 紅=詐騙)", fontsize=13)
    axes[0].set_xlabel("t-SNE 維度 1")
    axes[0].set_ylabel("t-SNE 維度 2")

    # Right: clusters (DBSCAN or ground truth)
    if gt_sub is not None:
        cluster_col   = gt_sub
        right_title   = "t-SNE — Ground-Truth Cluster_ID"
        palette_name  = "tab10"
        unique_vals   = cluster_col.unique()
        cmap          = plt.cm.get_cmap(palette_name, len(unique_vals))
        color_map     = {v: cmap(i) for i, v in enumerate(sorted(unique_vals))}
        colors_right  = [color_map[v] for v in cluster_col]

        for val in sorted(unique_vals):
            mask = cluster_col == val
            axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                            c=[color_map[val]], alpha=0.4, s=8, label=val, linewidths=0)
        axes[1].legend(markerscale=3, fontsize=9)

    elif dbscan_sub is not None:
        unique_vals   = sorted(set(dbscan_sub))
        cmap          = plt.cm.get_cmap("tab10", len(unique_vals))
        for cid in unique_vals:
            mask  = dbscan_sub == cid
            label = "noise" if cid == -1 else f"cluster_{cid}"
            axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                            c=[cmap(cid + 1 if cid != -1 else 0)],
                            alpha=0.4, s=8, label=label, linewidths=0)
        axes[1].legend(markerscale=3, fontsize=9)
        right_title = "t-SNE — DBSCAN 群集結果"
    else:
        axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c='gray', alpha=0.3, s=8)
        right_title = "t-SNE — 特徵空間分佈"

    axes[1].set_title(right_title, fontsize=13)
    axes[1].set_xlabel("t-SNE 維度 1")
    axes[1].set_ylabel("t-SNE 維度 2")

    plt.suptitle("Archangel 防詐系統 — 高維特徵視覺化", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = f"{save_dir}/tsne_visualization.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   t-SNE 圖表已儲存: {save_path}")

    return X_tsne


# ─────────────────────────────────────────────────────────────────────────────
# Cluster-level Fraud Rate Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_cluster_fraud_rates(
    y: pd.Series,
    dbscan_labels: np.ndarray,
    cluster_ids: pd.Series = None,
) -> pd.DataFrame:
    """
    Compute fraud rate per DBSCAN cluster.

    This is the actionable output: clusters with high fraud rates are
    candidate scam syndicates to escalate for investigation / blacklisting.
    """
    df = pd.DataFrame({
        "label":      y.values,
        "dbscan":     dbscan_labels,
    })
    if cluster_ids is not None and len(cluster_ids) == len(y):
        df["ground_truth"] = cluster_ids.values

    stats = df.groupby("dbscan").agg(
        count=("label", "count"),
        fraud_count=("label", "sum"),
        fraud_rate=("label", "mean"),
    ).reset_index()
    stats["cluster_name"] = stats["dbscan"].apply(
        lambda x: "noise" if x == -1 else f"cluster_{x}"
    )
    stats = stats.sort_values("fraud_rate", ascending=False)

    print("\n[Cluster Fraud Rates]")
    print(f"  {'Cluster':<15} {'Count':>7} {'Fraud%':>8}")
    print("  " + "─" * 35)
    for _, row in stats.iterrows():
        flag = " ⚠️" if row["fraud_rate"] > 0.5 else ""
        print(f"  {row['cluster_name']:<15} {row['count']:>7} {row['fraud_rate']*100:>7.1f}%{flag}")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_unsupervised_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    cluster_ids: pd.Series = None,
    eps: float = 0.6,
    min_samples: int = 10,
    n_tsne_samples: int = 5000,
    save_dir: str = ".",
) -> dict:
    """
    Full unsupervised pipeline: DBSCAN → fraud rate analysis → t-SNE.

    Returns:
        dict with dbscan_result, fraud_stats, tsne_embedding
    """
    print("\n" + "=" * 60)
    print("  🔍 Unsupervised Fraud Pattern Discovery")
    print("=" * 60)

    # 1. DBSCAN
    dbscan_result = run_dbscan(X, eps=eps, min_samples=min_samples,
                                ground_truth=cluster_ids)

    # 2. Per-cluster fraud rate
    fraud_stats = analyze_cluster_fraud_rates(
        y, dbscan_result["labels"], cluster_ids
    )

    # 3. t-SNE visualization
    X_tsne = run_tsne_and_plot(
        X, y,
        dbscan_labels=dbscan_result["labels"],
        ground_truth_clusters=cluster_ids,
        n_samples=n_tsne_samples,
        save_dir=save_dir,
    )

    print("\n✅ 非監督式分析完成")
    return {
        "dbscan_result": dbscan_result,
        "fraud_stats":   fraud_stats,
        "tsne_embedding": X_tsne,
    }


if __name__ == "__main__":
    print("請透過 notebooks/ 或 run_ml_ops.py 執行非監督式分析。")
