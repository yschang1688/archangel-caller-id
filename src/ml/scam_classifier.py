"""
scam_classifier.py — Archangel XGBoost Anti-Fraud Classifier
=============================================================
Trains and evaluates an XGBoost ensemble model for fraud detection.
Handles class imbalance via scale_pos_weight parameter.

Key additions vs. basic classifier:
  • SHAP TreeExplainer — model-agnostic feature attribution
  • Precision-Recall Curve (AUPRC) — correct metric for imbalanced data
  • joblib Pipeline serialization — production-ready model packaging

Role Target: Data Research Engineer @ Gogolook ISL
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_curve, average_precision_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import xgboost as xgb
import logging

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logging.getLogger(__name__).warning("shap not installed — SHAP plots disabled")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
MODEL_SAVE_PATH = "models/xgboost_fraud_pipeline.pkl"

# Matplotlib 中文支援
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────────────────────────────────────
# Core Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost_and_evaluate(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Train XGBoost classifier with class imbalance handling.

    Returns:
        (model, feature_names, metrics_dict)
    """
    np.random.seed(SEED)

    print("啟動 XGBoost 防詐預測模型訓練管線...\n")

    # 1. Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"訓練集維度: {X_train.shape}, 測試集維度: {X_test.shape}")

    # 2. Baseline: Logistic Regression
    print("\n[Baseline — Logistic Regression]")
    lr_model = LogisticRegression(random_state=SEED, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_f1 = f1_score(y_test, lr_pred)
    print(f"  Baseline F1: {lr_f1:.4f}")

    # 3. XGBoost with class imbalance handling
    # scale_pos_weight = neg_count / pos_count → tells XGBoost how rare fraud is
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\n[XGBoost] scale_pos_weight = {pos_weight:.2f}x (class imbalance correction)")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=pos_weight,
        eval_metric='logloss',
        random_state=SEED,
    )
    xgb_model.fit(X_train, y_train)

    # 4. Evaluation
    y_pred   = xgb_model.predict(X_test)
    y_proba  = xgb_model.predict_proba(X_test)[:, 1]
    xgb_f1   = f1_score(y_test, y_pred)
    roc_auc  = roc_auc_score(y_test, y_proba)
    auprc    = average_precision_score(y_test, y_proba)

    print("\n[模型評估報告 — Classification Report]:")
    print("─" * 50)
    print(classification_report(y_test, y_pred, target_names=['正常(0)', '詐騙(1)']))
    print("─" * 50)
    print(f"ROC-AUC: {roc_auc:.4f}  |  AUPRC: {auprc:.4f}")

    # 5. Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(xgb_model, X, y, cv=skf, scoring='f1')
    print(f"\n5-Fold CV F1: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")

    improvement = (xgb_f1 - lr_f1) / lr_f1 * 100
    print(f"XGBoost vs Baseline 提升: {improvement:+.2f}%")

    metrics = {
        "baseline_f1":    round(lr_f1, 4),
        "xgboost_f1":     round(xgb_f1, 4),
        "roc_auc":        round(roc_auc, 4),
        "auprc":          round(auprc, 4),
        "cv_f1_mean":     round(cv_scores.mean(), 4),
        "cv_f1_std":      round(cv_scores.std(), 4),
        "improvement_pct": round(improvement, 2),
        # Pass along test split so callers can run SHAP / PR Curve
        "_X_test":        X_test,
        "_y_test":        y_test,
        "_y_proba":       y_proba,
    }

    return xgb_model, X_train.columns.tolist(), metrics


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Explainability  (guides §2.2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_summary(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    save_dir: str = ".",
    max_display: int = 15,
) -> None:
    """
    Generate SHAP Summary Plot using TreeExplainer.

    Why SHAP over feature_importance?
      Feature importance shows "which features are used most" globally.
      SHAP shows "how much each feature pushed this specific prediction
      toward fraud vs. not-fraud" — required for regulatory explainability
      and root-cause debugging in a production anti-fraud system.

    Saves:
      • shap_summary_bar.png  — global mean |SHAP| per feature
      • shap_summary_dot.png  — full distribution (direction + magnitude)
    """
    if not _SHAP_AVAILABLE:
        print("⚠️  shap package not installed. Run: pip install shap")
        return

    print("\n[SHAP] 計算 TreeExplainer SHAP 值...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Bar plot — global feature importance
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, plot_type="bar",
                      max_display=max_display, show=False)
    plt.title("SHAP 特徵重要性 (Mean |SHAP|)", fontsize=14)
    plt.tight_layout()
    bar_path = os.path.join(save_dir, "shap_summary_bar.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   SHAP bar plot 已儲存: {bar_path}")

    # Dot plot — direction + magnitude
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test,
                      max_display=max_display, show=False)
    plt.title("SHAP 特徵影響分佈 (方向 + 強度)", fontsize=14)
    plt.tight_layout()
    dot_path = os.path.join(save_dir, "shap_summary_dot.png")
    plt.savefig(dot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   SHAP dot plot 已儲存: {dot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Precision-Recall Curve  (guides §4.2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_precision_recall_curve(
    y_test: pd.Series,
    y_proba: np.ndarray,
    model_label: str = "XGBoost",
    save_path: str = "pr_curve.png",
) -> float:
    """
    Plot Precision-Recall Curve and return AUPRC.

    Why PR Curve > ROC in fraud detection?
      ROC-AUC is optimistic when negatives vastly outnumber positives
      (e.g., 90:10 ratio) because it counts true negatives.
      AUPRC focuses only on the minority fraud class — directly measures
      how well the model ranks actual fraud cases above non-fraud.

    Returns:
        auprc (float)
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    auprc = average_precision_score(y_test, y_proba)

    # Baseline: random classifier AUPRC ≈ fraud rate
    fraud_rate = y_test.mean()

    plt.figure(figsize=(9, 6))
    plt.plot(recall, precision, linewidth=2,
             label=f"{model_label} (AUPRC = {auprc:.4f})")
    plt.axhline(y=fraud_rate, color="gray", linestyle="--", linewidth=1,
                label=f"Random Baseline ({fraud_rate:.3f})")

    # Mark operating point at threshold=0.5
    idx_05 = np.argmin(np.abs(thresholds - 0.5)) if len(thresholds) else 0
    plt.scatter(recall[idx_05], precision[idx_05], s=100, color="red",
                zorder=5, label=f"Threshold=0.5 (P={precision[idx_05]:.2f}, R={recall[idx_05]:.2f})")

    plt.xlabel("Recall (詐騙案例召回率)", fontsize=12)
    plt.ylabel("Precision (命中精準率)", fontsize=12)
    plt.title("Precision-Recall Curve — 防詐模型評估", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"\n[PR Curve] AUPRC = {auprc:.4f}  (baseline: {fraud_rate:.4f})")
    print(f"   PR Curve 已儲存: {save_path}")
    return auprc


# ─────────────────────────────────────────────────────────────────────────────
# Feature Importance Plot  (legacy / quick view)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: list,
    save_path: str = "xgboost_feature_importance.png",
) -> None:
    """Plot and save top-10 XGBoost built-in feature importance."""
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature':    feature_names,
        'Importance': importances,
    }).sort_values(by='Importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('XGBoost 防詐模型 — Top 10 Feature Importance', fontsize=16, fontweight='bold')
    plt.xlabel('Importance (gain)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n特徵重要性圖表已儲存: '{save_path}'")


# ─────────────────────────────────────────────────────────────────────────────
# Model Serialization  (guides §5.2)
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model: xgb.XGBClassifier, save_path: str = MODEL_SAVE_PATH) -> str:
    """
    Serialize trained model with joblib.

    Why joblib over pickle?
      joblib is optimized for NumPy arrays and sklearn/XGBoost objects —
      faster serialization + memory-mapped loading for large models.

    Returns:
        Path where model was saved.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    joblib.dump(model, save_path)
    size_kb = os.path.getsize(save_path) / 1024
    print(f"\n[joblib] 模型已序列化 → {save_path}  ({size_kb:.1f} KB)")
    return save_path


def load_model(load_path: str = MODEL_SAVE_PATH) -> xgb.XGBClassifier:
    """Load a previously serialized model."""
    model = joblib.load(load_path)
    print(f"[joblib] 模型已載入: {load_path}")
    return model


if __name__ == "__main__":
    print("請透過 run_demo.py 或 notebooks/ 執行完整管線。")
