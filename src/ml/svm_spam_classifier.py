"""
svm_spam_classifier.py — SVM 垃圾電話二元分類器
=================================================
使用 Support Vector Machine 對電話號碼進行即時 spam/not-spam 分類。

設計決策：
  - 使用 RBF kernel SVM：適合非線性邊界，能捕捉特徵間的交互作用
  - StandardScaler 前處理：SVM 對特徵尺度敏感，必須標準化
  - GridSearchCV 超參數搜尋：C / gamma 自動調優
  - class_weight='balanced'：自動處理類別不均衡
  - 推論延遲 < 0.5 秒：20 維特徵向量 → SVM predict 為 O(n_sv × d)，n_sv ~ 數千級

效能與精確度權衡：
  - 20 維特徵（而非全部 30+ 維）→ 犧牲少量資訊換取推論速度
  - RBF kernel（而非多項式核）→ 較好泛化能力 + 可接受的推論時間
  - 門檻值可調：predict_proba + 自訂 threshold 控制 precision/recall 平衡

使用方式：
  trainer = SVMSpamTrainer()
  trainer.train(X_train, y_train)
  predictions = trainer.predict(X_test)
  trainer.save_model("models/svm_spam_model.pkl")
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Optional

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    accuracy_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SVMSpamTrainer:
    """
    SVM 垃圾電話分類器的訓練與推論引擎。

    包含完整的 ML 生命週期：
      - 資料預處理（StandardScaler）
      - 超參數搜尋（GridSearchCV with StratifiedKFold）
      - 模型訓練與評估
      - 門檻值最佳化（Precision-Recall 曲線）
      - 模型序列化（joblib）
      - 即時推論（< 0.5 秒）
    """

    def __init__(self, threshold: float = 0.5):
        """
        初始化 SVM 訓練器。

        參數：
            threshold: 分類門檻值，predict_proba >= threshold → spam (1)
        """
        self.scaler = StandardScaler()
        self.model: Optional[SVC] = None
        self.threshold = threshold
        self.best_params: Optional[dict] = None
        self.training_report: Optional[dict] = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        do_grid_search: bool = True,
    ) -> dict:
        """
        完整訓練流程：分割 → 標準化 → Grid Search → 訓練 → 評估。

        參數：
            X:              特徵矩陣 (n_samples × 20)
            y:              二元標籤 (0/1)
            test_size:      測試集比例
            do_grid_search: 是否執行超參數搜尋（False 時使用預設值加速）

        回傳：
            dict 包含完整評估指標

        依賴：
            sklearn.svm.SVC, sklearn.preprocessing.StandardScaler
        """
        print("\n" + "═" * 60)
        print("  🤖 SVM Spam Classifier — 訓練開始")
        print("═" * 60)

        # ── Step 1: Train/Test Split ──────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=SEED, stratify=y
        )
        print(f"\n  訓練集: {X_train.shape[0]:,} | 測試集: {X_test.shape[0]:,}")
        print(f"  訓練集 spam 比例: {y_train.mean()*100:.1f}%")

        # ── Step 2: StandardScaler ────────────────────────────────────
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print(f"  StandardScaler 已套用")

        # ── Step 3: Grid Search / 直接訓練 ────────────────────────────
        if do_grid_search:
            print(f"\n  🔍 GridSearchCV 超參數搜尋...")
            param_grid = {
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 0.01, 0.1],
                "kernel": ["rbf"],
            }

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            grid_search = GridSearchCV(
                SVC(
                    class_weight="balanced",
                    probability=True,
                    random_state=SEED,
                    cache_size=500,
                ),
                param_grid,
                cv=cv,
                scoring="f1",
                n_jobs=-1,
                verbose=0,
            )

            t0 = time.time()
            grid_search.fit(X_train_scaled, y_train)
            search_time = time.time() - t0

            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            print(f"  最佳參數: {self.best_params}")
            print(f"  最佳 CV F1: {grid_search.best_score_:.4f}")
            print(f"  搜尋耗時: {search_time:.1f}s")
        else:
            print(f"\n  使用預設參數訓練...")
            self.model = SVC(
                C=10,
                gamma="scale",
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=SEED,
                cache_size=500,
            )
            t0 = time.time()
            self.model.fit(X_train_scaled, y_train)
            train_time = time.time() - t0
            self.best_params = {"C": 10, "gamma": "scale", "kernel": "rbf"}
            print(f"  訓練耗時: {train_time:.1f}s")

        # ── Step 4: 評估 ─────────────────────────────────────────────
        report = self._evaluate(X_test_scaled, y_test, X_train_scaled, y_train)
        self.training_report = report
        return report

    def _evaluate(
        self,
        X_test_scaled: np.ndarray,
        y_test: pd.Series,
        X_train_scaled: np.ndarray,
        y_train: pd.Series,
    ) -> dict:
        """
        完整模型評估，包含分類報告、混淆矩陣、ROC-AUC、最佳門檻值。

        參數：
            X_test_scaled:  標準化後的測試特徵
            y_test:         測試標籤
            X_train_scaled: 標準化後的訓練特徵
            y_train:        訓練標籤

        回傳：
            dict 包含所有評估指標
        """
        # 預測
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # 基本指標
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        cls_report = classification_report(y_test, y_pred, target_names=["Not Spam", "Spam"])

        # 最佳門檻值 (F1-optimal)
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_threshold_idx = np.argmax(f1_scores)
        optimal_threshold = float(thresholds[best_threshold_idx]) if best_threshold_idx < len(thresholds) else 0.5
        self.threshold = optimal_threshold

        # 用最佳門檻值重新預測
        y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
        f1_optimal = f1_score(y_test, y_pred_optimal)
        acc_optimal = accuracy_score(y_test, y_pred_optimal)

        # 訓練集表現 (偵測過擬合)
        y_train_pred = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, y_train_pred)

        # Support vectors 資訊
        n_support = self.model.n_support_

        # 推論速度測試
        inference_time = self._benchmark_inference(X_test_scaled[:100])

        print(f"\n{'─' * 56}")
        print(f"  📊 SVM 模型評估報告")
        print(f"{'─' * 56}")
        print(f"  訓練集準確率: {train_acc*100:.2f}%")
        print(f"  測試集準確率: {acc*100:.2f}%")
        print(f"  F1 Score:     {f1:.4f}")
        print(f"  ROC-AUC:      {auc:.4f}")
        print(f"  最佳門檻值:   {optimal_threshold:.4f}")
        print(f"  F1 (最佳門檻): {f1_optimal:.4f}")
        print(f"  Accuracy (最佳門檻): {acc_optimal*100:.2f}%")
        print(f"  Support Vectors: {n_support}")
        print(f"  推論延遲 (100筆): {inference_time*1000:.2f}ms")
        print(f"  單筆推論延遲:     {inference_time*10:.4f}ms")
        print(f"\n  混淆矩陣:")
        print(f"            Pred=0  Pred=1")
        print(f"  Actual=0  {cm[0][0]:>6}  {cm[0][1]:>6}")
        print(f"  Actual=1  {cm[1][0]:>6}  {cm[1][1]:>6}")
        print(f"\n{cls_report}")

        report = {
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "optimal_threshold": round(optimal_threshold, 4),
            "f1_optimal": round(f1_optimal, 4),
            "accuracy_optimal": round(acc_optimal, 4),
            "confusion_matrix": cm.tolist(),
            "support_vectors": n_support.tolist(),
            "best_params": self.best_params,
            "inference_time_100_samples_ms": round(inference_time * 1000, 2),
            "single_inference_ms": round(inference_time * 10, 4),
            # 供敏感度分析/實驗紀錄使用（避免外部重算 split）
            "y_test": y_test.astype(int).tolist(),
            "y_proba": y_proba.astype(float).tolist(),
        }
        return report

    def _benchmark_inference(self, X_sample: np.ndarray) -> float:
        """
        測量推論延遲，驗證 < 0.5 秒目標。

        參數：
            X_sample: 標準化後的測試資料 (subset)

        回傳：
            float — 100 筆的總推論時間（秒）
        """
        times = []
        for _ in range(10):  # 跑 10 次取平均
            t0 = time.perf_counter()
            self.model.predict_proba(X_sample)
            times.append(time.perf_counter() - t0)
        return float(np.median(times))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        即時推論：給定特徵矩陣，回傳 spam/not-spam 預測。

        使用 optimal threshold 進行二元分類。

        參數：
            X: 特徵矩陣 (n_samples × 20)

        回傳：
            np.ndarray — 0 (正常) 或 1 (垃圾電話)
        """
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        回傳垃圾電話機率值 [0, 1]。

        參數：
            X: 特徵矩陣 (n_samples × 20)

        回傳：
            np.ndarray — 每筆的 spam 機率
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict_single(self, features: dict) -> dict:
        """
        單筆即時推論（用於 API 即時查詢）。

        參數：
            features: dict，key 為 20 個特徵名稱

        回傳：
            dict 包含 prediction, probability, latency_ms

        依賴：
            需先呼叫 train() 或 load_model()
        """
        from src.processing.fcc_data_pipeline import FEATURE_NAMES

        feature_vector = np.array([[features.get(name, 0.0) for name in FEATURE_NAMES]])
        feature_df = pd.DataFrame(feature_vector, columns=FEATURE_NAMES)

        t0 = time.perf_counter()
        X_scaled = self.scaler.transform(feature_df)
        proba = self.model.predict_proba(X_scaled)[0, 1]
        prediction = int(proba >= self.threshold)
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "prediction": prediction,
            "label": "SPAM" if prediction == 1 else "NOT_SPAM",
            "probability": round(float(proba), 4),
            "threshold": self.threshold,
            "latency_ms": round(latency_ms, 4),
        }

    # ── 模型序列化 ────────────────────────────────────────────────────

    def save_model(self, path: str = None) -> str:
        """
        儲存模型（SVM + Scaler + 門檻值）至 pickle 檔。

        參數：
            path: 儲存路徑，預設為 models/svm_spam_model.pkl

        回傳：
            str — 實際儲存路徑
        """
        if path is None:
            path = os.path.join(_PROJECT_ROOT, "models", "svm_spam_model.pkl")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "threshold": self.threshold,
            "best_params": self.best_params,
            "training_report": self.training_report,
        }
        joblib.dump(payload, path)
        logger.info(f"模型已儲存: {path}")
        return path

    def load_model(self, path: str = None) -> None:
        """
        載入已訓練的模型。

        參數：
            path: pickle 檔路徑，預設為 models/svm_spam_model.pkl
        """
        if path is None:
            path = os.path.join(_PROJECT_ROOT, "models", "svm_spam_model.pkl")

        payload = joblib.load(path)
        self.model = payload["model"]
        self.scaler = payload["scaler"]
        self.threshold = payload["threshold"]
        self.best_params = payload.get("best_params")
        self.training_report = payload.get("training_report")
        logger.info(f"模型已載入: {path} | 門檻值: {self.threshold:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 視覺化
# ─────────────────────────────────────────────────────────────────────────────

def plot_svm_results(
    trainer: SVMSpamTrainer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_dir: str = None,
) -> None:
    """
    繪製 SVM 模型評估圖表：混淆矩陣、ROC 曲線、特徵重要性。

    參數：
        trainer:  已訓練的 SVMSpamTrainer
        X_test:   測試特徵
        y_test:   測試標籤
        save_dir: 圖表儲存目錄

    依賴：
        matplotlib, seaborn
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, auc

    if save_dir is None:
        save_dir = os.path.join(_PROJECT_ROOT, "outputs", "models")
    os.makedirs(save_dir, exist_ok=True)

    # 中文支援
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Microsoft JhengHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    X_test_scaled = trainer.scaler.transform(X_test)
    y_proba = trainer.model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= trainer.threshold).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 1. 混淆矩陣 ──
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
        xticklabels=["Not Spam", "Spam"],
        yticklabels=["Not Spam", "Spam"],
    )
    axes[0].set_title("SVM 混淆矩陣", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("預測")
    axes[0].set_ylabel("實際")

    # ── 2. ROC 曲線 ──
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color="#F44336", lw=2, label=f"AUC = {roc_auc:.4f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].fill_between(fpr, tpr, alpha=0.1, color="#F44336")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC 曲線", fontsize=13, fontweight="bold")
    axes[1].legend(loc="lower right", fontsize=11)

    # ── 3. 門檻值 vs F1 ──
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    axes[2].plot(thresholds, f1_scores[:-1], color="#2196F3", lw=2, label="F1 Score")
    axes[2].plot(thresholds, precisions[:-1], "--", color="#4CAF50", lw=1.5, label="Precision")
    axes[2].plot(thresholds, recalls[:-1], "--", color="#FF9800", lw=1.5, label="Recall")
    axes[2].axvline(x=trainer.threshold, color="red", linestyle=":", lw=1.5,
                    label=f"Threshold={trainer.threshold:.3f}")
    axes[2].set_xlabel("Threshold")
    axes[2].set_ylabel("Score")
    axes[2].set_title("門檻值調優", fontsize=13, fontweight="bold")
    axes[2].legend(fontsize=9)

    fig.suptitle("SVM Spam Classifier — 模型評估", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "svm_spam_evaluation.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"評估圖表已儲存: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.processing.fcc_data_pipeline import fcc_clean_and_prepare

    # 載入 FCC 資料 + 特徵工程
    X, y, fcc_features, raw_df = fcc_clean_and_prepare()

    # 訓練 SVM
    trainer = SVMSpamTrainer()
    report = trainer.train(X, y, do_grid_search=True)

    # 儲存模型
    trainer.save_model()

    # 單筆推論 demo
    sample_features = X.iloc[0].to_dict()
    result = trainer.predict_single(sample_features)
    print(f"\n🔍 單筆推論 Demo:")
    print(f"   預測: {result['label']}")
    print(f"   機率: {result['probability']}")
    print(f"   延遲: {result['latency_ms']:.4f} ms")
