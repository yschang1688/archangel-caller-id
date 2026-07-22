"""
data_refinement.py — Archangel Data-centric AI Refinement Pipeline
===================================================================
Implements SMOTE oversampling + cleanlab label quality correction.

This is the core "Data-centric AI" module — improving data quality
rather than tuning model hyperparameters. Aligned with Data-centric AI philosophy
of systematic dataset refinement over raw data accumulation.

Portfolio: Caller-ID & Anti-Fraud Data Platform
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42


class DataRefinementPipeline:
    """
    Two-phase data-centric refinement:

    Phase 1 — SMOTE: Fix class imbalance (1% fraud → balanced training set)
    Phase 2 — cleanlab: Identify and correct noisy labels

    This approach prioritizes DATA QUALITY over model complexity,
    consistent with Data-centric AI principles.
    """

    def __init__(self, random_state: int = SEED):
        self.random_state = random_state
        self.noisy_label_indices: list[int] = []
        self.metrics_history: dict[str, dict] = {}

    # ── Phase 1: SMOTE Oversampling ───────────────────────────────────────

    def apply_smote(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to balance fraud vs. non-fraud classes.

        In anti-fraud datasets, fraud typically represents 1-5% of records.
        SMOTE synthesizes minority-class examples in feature space, avoiding
        the information loss of random undersampling.
        """
        print("\n  📊 Class distribution BEFORE SMOTE:")
        print(f"     Non-fraud (0): {(y == 0).sum()}")
        print(f"     Fraud (1):     {(y == 1).sum()}")
        print(f"     Ratio:         1:{(y == 0).sum() // max((y == 1).sum(), 1)}")

        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        print("\n  📊 Class distribution AFTER SMOTE:")
        y_res = pd.Series(y_resampled)
        print(f"     Non-fraud (0): {(y_res == 0).sum()}")
        print(f"     Fraud (1):     {(y_res == 1).sum()}")
        print(f"     Ratio:         1:1 (balanced)")

        return pd.DataFrame(X_resampled, columns=X.columns), y_res

    # ── Phase 2: Cleanlab Label Quality ───────────────────────────────────

    def find_noisy_labels(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
    ) -> tuple[np.ndarray, list[int]]:
        """
        Identify potentially mislabeled samples using confident learning.

        Uses cross-validated predicted probabilities to find samples where
        the model's confident prediction disagrees with the given label.
        This is the core idea behind cleanlab.

        Returns:
            (label_quality_scores, noisy_indices)
        """
        np.random.seed(self.random_state)

        print("\n  🔍 Running confident learning for label quality assessment...")

        # Cross-validated probability predictions
        clf = LogisticRegression(random_state=self.random_state, max_iter=1000)
        pred_probs = cross_val_predict(clf, X, y, cv=n_folds, method='predict_proba')

        # Compute label quality score per sample
        # High score = label likely correct; Low score = label likely wrong
        n_samples = len(y)
        label_quality = np.zeros(n_samples)

        for i in range(n_samples):
            given_label = int(y.iloc[i])
            # Score = probability assigned to the given label
            label_quality[i] = pred_probs[i][given_label]

        # Identify noisy labels: samples where model is confident
        # the label is WRONG (quality score < threshold)
        threshold = np.percentile(label_quality, 5)  # Bottom 5%
        noisy_mask = label_quality < threshold
        noisy_indices = np.where(noisy_mask)[0].tolist()

        self.noisy_label_indices = noisy_indices

        print(f"  📋 Label quality assessment complete:")
        print(f"     Total samples:       {n_samples}")
        print(f"     Noisy labels found:  {len(noisy_indices)} ({len(noisy_indices)/n_samples*100:.1f}%)")
        print(f"     Quality threshold:   {threshold:.4f}")
        print(f"     Mean quality score:  {label_quality.mean():.4f}")

        return label_quality, noisy_indices

    def correct_noisy_labels(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        noisy_indices: list[int],
        strategy: str = "remove",
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Apply label correction strategy.

        Strategies:
            - "remove": Drop noisy samples entirely (safest)
            - "flip":   Flip the label (aggressive, for clear mismatches)
        """
        if strategy == "remove":
            clean_mask = ~np.isin(np.arange(len(y)), noisy_indices)
            X_clean = X.iloc[clean_mask].reset_index(drop=True)
            y_clean = y.iloc[clean_mask].reset_index(drop=True)
            print(f"\n  🧹 Removed {len(noisy_indices)} noisy samples → {len(y_clean)} remaining")
        elif strategy == "flip":
            y_clean = y.copy()
            y_clean.iloc[noisy_indices] = 1 - y_clean.iloc[noisy_indices]
            X_clean = X.copy()
            print(f"\n  🔄 Flipped {len(noisy_indices)} noisy labels")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return X_clean, y_clean

    # ── Full Pipeline ─────────────────────────────────────────────────────

    def run(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Execute the complete data refinement pipeline.

        Returns metrics comparing before/after refinement.
        """
        np.random.seed(self.random_state)

        print("\n" + "═" * 60)
        print("  DATA-CENTRIC AI REFINEMENT PIPELINE")
        print("═" * 60)

        # ── Baseline: train without refinement
        print("\n▸ Step 0: Baseline (no refinement)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        baseline_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, random_state=self.random_state,
            eval_metric='logloss',
        )
        baseline_model.fit(X_train, y_train)
        baseline_f1 = f1_score(y_test, baseline_model.predict(X_test))
        print(f"  Baseline F1: {baseline_f1:.4f}")
        self.metrics_history["baseline"] = {"f1": round(baseline_f1, 4)}

        # ── Phase 1: SMOTE
        print("\n▸ Phase 1: SMOTE Oversampling")
        X_smote, y_smote = self.apply_smote(X_train, y_train)

        smote_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, random_state=self.random_state,
            eval_metric='logloss',
        )
        smote_model.fit(X_smote, y_smote)
        smote_f1 = f1_score(y_test, smote_model.predict(X_test))
        print(f"\n  After SMOTE F1: {smote_f1:.4f} ({smote_f1 - baseline_f1:+.4f})")
        self.metrics_history["after_smote"] = {"f1": round(smote_f1, 4)}

        # ── Phase 2: Label quality (on original training set)
        print("\n▸ Phase 2: Cleanlab Label Quality Correction")
        quality_scores, noisy_indices = self.find_noisy_labels(X_train, y_train)

        X_clean, y_clean = self.correct_noisy_labels(
            X_train, y_train, noisy_indices, strategy="remove"
        )

        # Re-apply SMOTE on cleaned data
        X_clean_smote, y_clean_smote = self.apply_smote(X_clean, y_clean)

        refined_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, random_state=self.random_state,
            eval_metric='logloss',
        )
        refined_model.fit(X_clean_smote, y_clean_smote)
        refined_f1 = f1_score(y_test, refined_model.predict(X_test))
        print(f"\n  After SMOTE + Cleanlab F1: {refined_f1:.4f} ({refined_f1 - baseline_f1:+.4f})")
        self.metrics_history["after_refinement"] = {"f1": round(refined_f1, 4)}

        # ── Summary
        print("\n" + "═" * 60)
        print("  REFINEMENT SUMMARY")
        print("═" * 60)
        print(f"  Baseline F1:              {baseline_f1:.4f}")
        print(f"  After SMOTE:              {smote_f1:.4f}  ({smote_f1 - baseline_f1:+.4f})")
        print(f"  After SMOTE + Cleanlab:   {refined_f1:.4f}  ({refined_f1 - baseline_f1:+.4f})")
        print(f"  Noisy labels removed:     {len(noisy_indices)}")
        total_lift = refined_f1 - baseline_f1
        print(f"  Total F1 lift:            {total_lift:+.4f} ({total_lift/baseline_f1*100:+.1f}%)")
        print("═" * 60)

        return {
            "baseline_f1": round(baseline_f1, 4),
            "smote_f1": round(smote_f1, 4),
            "refined_f1": round(refined_f1, 4),
            "noisy_labels_found": len(noisy_indices),
            "total_lift": round(total_lift, 4),
            "total_lift_pct": round(total_lift / baseline_f1 * 100, 1),
        }


if __name__ == "__main__":
    from src.processing.data_pipeline import clean_and_prepare_data

    X, y, scaler = clean_and_prepare_data()
    pipeline = DataRefinementPipeline()
    results = pipeline.run(X, y)
