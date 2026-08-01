#!/usr/bin/env python3
"""
run_ml_ops.py — Archangel Intelligence System: One-Click MLOps Pipeline
=====================================================================
Executes all core modules sequentially with deterministic output.
Designed for terminal presentation and reproducible demonstration.

Usage:
    python run_ml_ops.py                          # Full demo（合成資料）
    python run_ml_ops.py --quick                  # Quick mode（跳過資料精煉）
    python run_ml_ops.py --data-path raw_fcc.csv  # 使用真實 FCC CSV 資料 + 已訓練模型

Modules:
    1. Spark ETL — Data Skew Salting Technique
    2. A/B Testing — Statistical Rigor with Cohen's d
    3. Model Monitor — PSI Drift Detection & Auto-Retrain
    4. Guardian Score — Bayesian Reputation System

當指定 --data-path 且 models/svm_spam_model.pkl 存在時，
Module 2-4 將使用 run_ml_dev.py 訓練出的真實 SVM 模型指標，
而非硬編碼的模擬數值。
"""

import sys
import os
import time
import random
import argparse
import logging
import numpy as np

# Ensure project root is in Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# Global deterministic seed
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MODEL_PKL_PATH = os.path.join(PROJECT_ROOT, "models", "svm_spam_model.pkl")


def print_banner(data_path: str = None, model_loaded: bool = False):
    """
    列印 Demo 標題橫幅。

    參數：
        data_path: 資料集路徑（若有指定）
        model_loaded: 是否成功載入訓練模型
    """
    print("\n")
    print("═" * 68)
    print("  🛡️  ARCHANGEL INTELLIGENCE SYSTEM — Full Pipeline Demo")
    print("  ─────────────────────────────────────────────────────────")
    print("  Data-centric AI  |  Anti-Fraud  |  Closed-Loop Pipeline")
    print("  Deterministic seed: 42  |  All results are reproducible")
    if data_path:
        print(f"  📂 ETL Source: {os.path.basename(data_path)} (real FCC data)")
    else:
        print("  📂 ETL Source: synthetic data (seed=42)")
    if model_loaded:
        print("  🤖 Model: SVM from models/svm_spam_model.pkl（真實模型驅動 Module 2-4）")
    else:
        print("  🤖 Model: simulated data（Module 2-4 使用合成模擬數值）")
    print("═" * 68)


def print_module_header(num: int, total: int, title: str, subtitle: str):
    """
    列印模組區段標題。

    參數：
        num: 當前模組序號
        total: 總模組數
        title: 模組名稱
        subtitle: 模組副標題說明
    """
    print(f"\n\n{'▓' * 68}")
    print(f"  ▶ Module {num}/{total}: {title}")
    print(f"    {subtitle}")
    print(f"{'▓' * 68}")


def print_summary(results: dict):
    """
    列印最終摘要表格。

    參數：
        results: 各模組回傳結果的字典
    """
    print("\n\n")
    print("═" * 68)
    print("  📊 KEY RESULTS SUMMARY — Archangel Intelligence System")
    print("═" * 68)

    etl = results.get("etl", {})
    ab = results.get("ab_testing", {})
    monitor = results.get("model_monitor", {})
    guardian = results.get("guardian_score", {})

    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  📦 DATA SKEW HANDLING (spark_etl.py)                       │
  │    Pre-salt skew ratio:   {etl.get('pre_salt_skew_ratio', 'N/A'):>10}x                     │
  │    Post-salt skew ratio:  {etl.get('post_salt_skew_ratio', 'N/A'):>10}x                     │
  │    Hot keys handled:      {etl.get('hot_keys_handled', 'N/A'):>10}                      │
  │    Pipeline time:         {etl.get('pipeline_time_ms', 'N/A'):>10} ms                   │
  ├─────────────────────────────────────────────────────────────┤
  │  🔬 A/B TESTING (ab_testing.py)                             │
  │    P-value:               {ab.get('p_value', 'N/A'):>10}                      │
  │    Cohen's d:             {ab.get('cohen_d', 'N/A'):>10}                      │
  │    95% CI:                [{ab.get('ci_lower', 'N/A')}, {ab.get('ci_upper', 'N/A')}]          │
  │    Significant:           {str(ab.get('is_significant', 'N/A')):>10}                      │
  ├─────────────────────────────────────────────────────────────┤
  │  📈 MODEL MONITOR (model_monitor.py)                        │
  │    PSI Score:             {monitor.get('psi_score', 'N/A'):>10}                      │
  │    Drift Severity:        {monitor.get('drift_severity', 'N/A'):>10}                      │
  │    Hit Rate Δ:            {monitor.get('hit_rate_delta', 'N/A'):>10}                      │
  │    Retrain Triggered:     {str(monitor.get('retraining_triggered', 'N/A')):>10}                      │
  ├─────────────────────────────────────────────────────────────┤
  │  🏆 GUARDIAN SCORE (guardian_score.py)                       │
  │    Users Registered:      {guardian.get('users_registered', 'N/A'):>10}                      │
  │    Blacklist Candidates:  {guardian.get('blacklist_candidates', 'N/A'):>10}                      │
  │    Top Guardian Score:    {guardian.get('top_guardian_score', 'N/A'):>10}                      │
  └─────────────────────────────────────────────────────────────┘""")

    print("\n" + "═" * 68)
    print("  ✅ All modules executed successfully with seed=42")
    print("  📝 Results are deterministic and reproducible")
    print("═" * 68)

    # Quick reference for README alignment
    print(f"""
  ── README Key Results Reference ──────────────────────────────
  Data Skew:  {etl.get('pre_salt_skew_ratio', '?')}x → {etl.get('post_salt_skew_ratio', '?')}x (after salting)
  A/B Test:   p={ab.get('p_value', '?')}, Cohen's d={ab.get('cohen_d', '?')}, CI=[{ab.get('ci_lower', '?')}, {ab.get('ci_upper', '?')}]
  PSI Drift:  {monitor.get('psi_score', '?')} → {monitor.get('drift_severity', '?')} → retrain={monitor.get('retraining_triggered', '?')}
  Hit Rate:   {monitor.get('baseline_hit_rate', '?')} → {monitor.get('final_hit_rate', '?')} (after drift)
  ──────────────────────────────────────────────────────────────""")


# ─────────────────────────────────────────────────────────────────────────────
# 模型載入與指標計算
# ─────────────────────────────────────────────────────────────────────────────

def load_trained_model_metrics(data_path: str) -> dict | None:
    """
    載入 run_ml_dev.py 訓練出的 SVM 模型，並在真實 FCC 資料上計算指標。

    流程：
        1. 載入 models/svm_spam_model.pkl（含 model, scaler, threshold）
        2. 載入 FCC CSV → 特徵工程 → train/test split
        3. 用 default threshold (0.5) 和 optimal threshold 分別計算 hit_rate / FPR / precision
        4. 產出模型預測分佈（用於 PSI 監控）
        5. 產出逐號碼 spam 機率（用於 Guardian Score）

    參數：
        data_path: FCC CSV 檔案路徑

    回傳：
        dict 包含 default/optimal 指標、分佈、phone_predictions；若模型不存在回傳 None

    依賴：
        src.ml.svm_spam_classifier.SVMSpamTrainer,
        src.processing.fcc_data_pipeline.fcc_clean_and_prepare,
        sklearn.model_selection.train_test_split,
        sklearn.metrics (recall_score, precision_score)
    """
    if not os.path.exists(MODEL_PKL_PATH):
        print(f"  ⚠️  找不到已訓練模型: {MODEL_PKL_PATH}")
        print("     請先執行: python run_ml_dev.py --data-path <csv>")
        print("     Module 2-4 將使用合成模擬數值")
        return None

    print("\n" + "─" * 60)
    print("  🔧 Step 0: 載入已訓練模型 + 計算真實指標")
    print("─" * 60)

    from src.ml.svm_spam_classifier import SVMSpamTrainer
    from src.processing.fcc_data_pipeline import fcc_clean_and_prepare
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import recall_score, precision_score

    # 載入模型
    trainer = SVMSpamTrainer()
    trainer.load_model(MODEL_PKL_PATH)
    print(f"  ✅ 模型已載入: {MODEL_PKL_PATH}")
    print(f"     最佳門檻值: {trainer.threshold:.4f}")
    print(f"     參數: {trainer.best_params}")

    # 準備資料
    print(f"  📂 準備 FCC 資料: {os.path.basename(data_path)}")
    X, y, fcc_features, raw_df = fcc_clean_and_prepare(data_path, negative_ratio=1.0)

    # 分割（與訓練時一致的 split）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y,
    )

    # 預測
    X_train_scaled = trainer.scaler.transform(X_train)
    X_test_scaled = trainer.scaler.transform(X_test)
    y_proba_train = trainer.model.predict_proba(X_train_scaled)[:, 1]
    y_proba_test = trainer.model.predict_proba(X_test_scaled)[:, 1]

    # Default threshold (0.5) 指標
    y_pred_default = (y_proba_test >= 0.5).astype(int)
    default_hr = float(recall_score(y_test, y_pred_default))
    default_prec = float(precision_score(y_test, y_pred_default, zero_division=0))
    default_fpr = float(((y_pred_default == 1) & (y_test == 0)).sum() / max((y_test == 0).sum(), 1))

    # Optimal threshold 指標
    y_pred_optimal = (y_proba_test >= trainer.threshold).astype(int)
    optimal_hr = float(recall_score(y_test, y_pred_optimal))
    optimal_prec = float(precision_score(y_test, y_pred_optimal, zero_division=0))
    optimal_fpr = float(((y_pred_optimal == 1) & (y_test == 0)).sum() / max((y_test == 0).sum(), 1))

    # 推論延遲
    t0 = time.perf_counter()
    trainer.model.predict_proba(X_test_scaled[:100])
    inference_ms = (time.perf_counter() - t0) * 1000

    # Phone-level predictions（用於 Guardian Score）
    phone_predictions = []
    # fcc_features 的 index 是電話號碼
    fcc_scaled = trainer.scaler.transform(fcc_features)
    fcc_proba = trainer.model.predict_proba(fcc_scaled)[:, 1]
    for phone, proba in zip(fcc_features.index, fcc_proba):
        phone_predictions.append({
            "phone_number": str(phone),
            "spam_proba": float(proba),
        })

    print(f"\n  📊 真實模型指標:")
    print(f"     Default (threshold=0.5):  hit_rate={default_hr:.4f}  precision={default_prec:.4f}  FPR={default_fpr:.4f}")
    print(f"     Optimal (threshold={trainer.threshold:.4f}): hit_rate={optimal_hr:.4f}  precision={optimal_prec:.4f}  FPR={optimal_fpr:.4f}")
    print(f"     推論延遲 (100 筆): {inference_ms:.2f}ms")
    print(f"     Phone predictions: {len(phone_predictions)} 筆")

    return {
        "default": {
            "hit_rate": default_hr,
            "precision": default_prec,
            "fpr": default_fpr,
            "threshold": 0.5,
        },
        "optimal": {
            "hit_rate": optimal_hr,
            "precision": optimal_prec,
            "fpr": optimal_fpr,
            "threshold": float(trainer.threshold),
        },
        "y_proba_train": y_proba_train.tolist(),
        "y_proba_test": y_proba_test.tolist(),
        "inference_latency_ms": inference_ms,
        "phone_predictions": phone_predictions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_full_demo(quick: bool = False, data_path: str = None):
    """
    執行完整 Demo 管線。

    流程：
        Step 0: 若有 data_path，載入已訓練模型並計算真實指標
        Module 1: Spark ETL（真實/合成資料）
        Module 2: A/B Testing（真實/合成指標）
        Module 3: Model Monitor（真實/合成指標）← 關鍵：hit_rate 來自真實模型
        Module 4: Guardian Score（真實/合成指標）

    參數：
        quick: 是否跳過資料精煉
        data_path: FCC CSV 路徑（可選）

    回傳：
        dict 包含所有模組結果
    """
    # Configure logging to stdout
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    # Suppress logging during import
    logging.disable(logging.CRITICAL)

    # Pre-import all modules
    from src.processing.spark_etl import AntiFraudETL
    from src.ml.ab_testing import run_demo as ab_demo, run_demo_with_model as ab_demo_model
    from src.monitoring.model_monitor import run_demo as monitor_demo, run_demo_with_model as monitor_demo_model
    from src.feature_engineering.guardian_score import run_demo as guardian_demo, run_demo_with_model as guardian_demo_model

    # Re-enable logging
    logging.disable(logging.NOTSET)

    # ── Step 0: 載入模型指標（若有 data_path）─────────────────────
    model_metrics = None
    if data_path:
        model_metrics = load_trained_model_metrics(data_path)

    print_banner(data_path=data_path, model_loaded=model_metrics is not None)

    all_results = {}
    t_start = time.perf_counter()
    total_modules = 4

    # ═══════════════════════════════════════════════════════════════════════
    # Module 1: Spark ETL — Data Skew Salting
    # ═══════════════════════════════════════════════════════════════════════
    if data_path:
        subtitle = f"Loading real FCC data from {os.path.basename(data_path)} (sample 50k)"
    else:
        subtitle = "Demonstrates salting technique to resolve partition skew"

    print_module_header(1, total_modules,
                        "Spark ETL — Data Skew Salting",
                        subtitle)
    etl = AntiFraudETL()
    if data_path:
        etl_results = etl.run_from_csv(data_path, sample_n=50_000)
    else:
        etl_results = etl.run(n_records=50_000)
    all_results["etl"] = etl_results

    # ═══════════════════════════════════════════════════════════════════════
    # Module 2: A/B Testing — Statistical Rigor
    # ═══════════════════════════════════════════════════════════════════════
    if model_metrics:
        print_module_header(2, total_modules,
                            "A/B Testing — Statistical Rigor（真實模型驅動）",
                            "Control=threshold 0.5 vs Treatment=optimal threshold → 真實 hit_rate")
        ab_results = ab_demo_model(
            default_metrics=model_metrics["default"],
            optimal_metrics=model_metrics["optimal"],
        )
    else:
        print_module_header(2, total_modules,
                            "A/B Testing — Statistical Rigor",
                            "Power analysis → z-test → Cohen's d → Business decision")
        ab_results = ab_demo()
    all_results["ab_testing"] = ab_results

    # ═══════════════════════════════════════════════════════════════════════
    # Module 3: Model Monitor — Drift Detection  ← 關鍵改動
    # ═══════════════════════════════════════════════════════════════════════
    if model_metrics:
        print_module_header(3, total_modules,
                            "Model Monitor — PSI Drift Detection（真實模型驅動）",
                            f"Baseline hit_rate={model_metrics['optimal']['hit_rate']:.4f}（來自 SVM 模型，非硬編碼）")
        monitor_results = monitor_demo_model({
            "hit_rate": model_metrics["optimal"]["hit_rate"],
            "precision": model_metrics["optimal"]["precision"],
            "fpr": model_metrics["optimal"]["fpr"],
            "y_proba_train": model_metrics["y_proba_train"],
            "y_proba_test": model_metrics["y_proba_test"],
            "inference_latency_ms": model_metrics["inference_latency_ms"],
        })
    else:
        print_module_header(3, total_modules,
                            "Model Monitor — PSI Drift Detection",
                            "30-day simulation with auto-retraining trigger")
        monitor_results = monitor_demo()
    all_results["model_monitor"] = monitor_results

    # ═══════════════════════════════════════════════════════════════════════
    # Module 4: Guardian Score — Bayesian Reputation
    # ═══════════════════════════════════════════════════════════════════════
    if model_metrics:
        print_module_header(4, total_modules,
                            "Guardian Score — Bayesian Reputation（真實模型驅動）",
                            "使用模型預測的高信心 spam 號碼作為回報對象")
        guardian_results = guardian_demo_model(
            phone_predictions=model_metrics["phone_predictions"],
        )
    else:
        print_module_header(4, total_modules,
                            "Guardian Score — Bayesian Reputation",
                            "Beta distribution update + anti-manipulation + weighted consensus")
        guardian_results = guardian_demo()
    all_results["guardian_score"] = guardian_results

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    t_total = time.perf_counter() - t_start
    print_summary(all_results)

    print(f"\n  ⏱️  Total execution time: {t_total:.2f}s")
    if model_metrics:
        print(f"\n  🤖 本次 Demo 使用真實 SVM 模型驅動 Module 2-4")
        print(f"     模型來源: {MODEL_PKL_PATH}")
        print(f"     資料來源: {os.path.basename(data_path)}")
    print(f"\n  💡 Next steps:")
    print(f"     • Jupyter Notebook:  jupyter notebook notebooks/")
    print(f"     • FastAPI Swagger:   uvicorn src.api.detection_api:app --reload")
    print(f"     • Docker full stack: docker-compose up -d")
    print(f"     • Run tests:        pytest tests/ -v")
    print()

    return all_results


def main():
    """
    CLI 入口：解析引數並啟動 Demo。

    參數：
        --quick: 跳過資料精煉
        --data-path: 指定 FCC CSV 資料集路徑

    回傳：
        dict 包含所有模組結果
    """
    parser = argparse.ArgumentParser(
        description="Archangel Intelligence System — Pipeline Demo"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: skip data refinement pipeline"
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="指定真實資料集路徑（e.g. raw_fcc.csv）。"
             "若未指定，Module 1 使用合成資料（seed=42）。"
             "若指定且 models/svm_spam_model.pkl 存在，Module 2-4 使用真實模型。"
    )
    args = parser.parse_args()

    # 驗證資料集路徑
    if args.data_path and not os.path.exists(args.data_path):
        print(f"❌ 找不到資料集: {args.data_path}")
        sys.exit(1)

    results = run_full_demo(quick=args.quick, data_path=args.data_path)
    return results


if __name__ == "__main__":
    main()
