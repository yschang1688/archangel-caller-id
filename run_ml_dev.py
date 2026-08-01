#!/usr/bin/env python3
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOperatorIssue=false, reportCallIssue=false
"""
run_ml_dev.py — Archangel MLDev 全生命週期 CLI Pipeline
=============================================================
與 run_ml_ops.py 互補：
  - run_ml_ops.py  跑 MLOps 系統模組 (Spark ETL / A/B Testing / Guardian Score / PSI Monitor)
  - run_ml_dev.py 跑 MLDev 數據管線 (EDA → Train → Evaluate → Serialize → Monitor)

Usage:
    python run_ml_dev.py                     # 完整管線
    python run_ml_dev.py --skip-eda          # 跳過 EDA 視覺化
    python run_ml_dev.py --grid-search       # 啟用 GridSearchCV
    python run_ml_dev.py --skip-unsupervised # 跳過 DBSCAN + t-SNE
"""

import sys
import os
import time
import random
import argparse
import logging
import json

import numpy as np
import yaml

# ─── Deterministic seeds ─────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, 'FCC_Consumer_Complaints_RAW_Data_2017.csv')
EDA_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'eda')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'models')
UNSUP_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'unsupervised')
MONITOR_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'monitoring')
QUALITY_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'quality')
SERIALIZE_DIR = os.path.join(PROJECT_ROOT, 'models')

for d in [EDA_DIR, MODEL_DIR, UNSUP_DIR, MONITOR_DIR, QUALITY_DIR, SERIALIZE_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Centralized config ──────────────────────────────────────────────────────
PIPELINE_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "pipeline_config.yaml")


def load_pipeline_config(config_path: str = PIPELINE_CONFIG_PATH) -> dict:
    """
    載入集中化設定檔（pipeline_config.yaml）。

    參數：
        config_path: 設定檔路徑

    依賴：
        pyyaml (yaml.safe_load)

    回傳：
        dict：設定內容；若檔案不存在或解析失敗則回傳空 dict
    """
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"讀取設定檔失敗: {config_path} ({exc})")
        return {}


def _cfg_get(cfg: dict, dotted_key: str, default):
    """
    以 dotted path 取設定值（例：fcc.hard_negative_ratio）。

    參數：
        cfg: 設定 dict
        dotted_key: 以 '.' 分隔的 key
        default: 找不到時回傳值

    回傳：
        任意型別：對應設定值或 default
    """
    current = cfg
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def print_banner(data_path: str):
    print("""
╔══════════════════════════════════════════════════════════════╗
║            🛡️  Archangel ML Pipeline                         ║
║            MLOps 全生命週期 — 防詐模型訓練與部署                  ║
╠══════════════════════════════════════════════════════════════╣
║  Dataset:  {dataset:<50}                                     ║
║  Steps:    EDA → Clean → Train → Evaluate → Serialize → PSI  ║
║  Seed:     42 (deterministic)                                ║
╚══════════════════════════════════════════════════════════════╝
""".format(dataset=os.path.basename(data_path)[:50]))


def step_header(step_num: int, title: str):
    print(f"\n{'═'*60}")
    print(f"  Step {step_num}: {title}")
    print(f"{'═'*60}")


def _count_iqr_outliers(series):
    if series is None:
        return 0
    if not hasattr(series, "dropna"):
        return 0
    s = series.dropna()
    if len(s) < 5:
        return 0
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


def build_raw_clean_quality_report(raw_df, X, y, data_path):
    """Build a compact raw->clean quality comparison for demo output."""
    import pandas as pd

    raw_label_num = None
    if "Label" in raw_df.columns:
        raw_label_num = pd.to_numeric(raw_df["Label"], errors="coerce")

    raw_rows = int(len(raw_df))
    clean_rows = int(len(X))
    dropped_rows = raw_rows - clean_rows

    raw_missing_ratio = {
        c: round(float(raw_df[c].isna().mean() * 100), 3) for c in raw_df.columns
    }
    high_missing_cols = {k: v for k, v in raw_missing_ratio.items() if v >= 5}

    report_count_raw = pd.to_numeric(raw_df.get("Report_Count"), errors="coerce")
    fin_loss_raw = pd.to_numeric(raw_df.get("Financial_Loss"), errors="coerce")

    raw_summary = {
        "dataset_path": data_path,
        "rows": raw_rows,
        "columns": int(raw_df.shape[1]),
        "missing_ratio_pct": raw_missing_ratio,
        "high_missing_cols_pct_ge_5": high_missing_cols,
        "invalid_label_rows": int(raw_label_num.isna().sum()) if raw_label_num is not None else None,
        "label_0_rows": int((raw_label_num == 0).sum()) if raw_label_num is not None else None,
        "label_1_rows": int((raw_label_num == 1).sum()) if raw_label_num is not None else None,
        "negative_report_count_rows": int((report_count_raw < 0).sum()) if report_count_raw is not None else None,
        "negative_financial_loss_rows": int((fin_loss_raw < 0).sum()) if fin_loss_raw is not None else None,
        "report_count_iqr_outliers": _count_iqr_outliers(report_count_raw) if report_count_raw is not None else None,
        "financial_loss_iqr_outliers": _count_iqr_outliers(fin_loss_raw) if fin_loss_raw is not None else None,
    }

    clean_summary = {
        "rows": clean_rows,
        "columns": int(X.shape[1]),
        "class_0_rows": int((y == 0).sum()),
        "class_1_rows": int((y == 1).sum()),
        "class_imbalance_ratio_0_to_1": round(float((y == 0).sum() / max((y == 1).sum(), 1)), 4),
        "any_missing_in_features": bool(X.isna().any().any()),
        "missing_cells_in_features": int(X.isna().sum().sum()),
    }

    diff = {
        "rows_dropped_total": dropped_rows,
        "rows_retained_pct": round((clean_rows / max(raw_rows, 1)) * 100, 3),
        "feature_columns_after_engineering": int(X.shape[1]),
    }

    return {"raw": raw_summary, "clean": clean_summary, "diff": diff}


def write_quality_reports(report, output_dir):
    json_path = os.path.join(output_dir, "raw_to_clean_quality_report.json")
    md_path = os.path.join(output_dir, "raw_to_clean_quality_report.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md = []
    md.append("# Raw to Clean 資料品質報告")
    md.append("")
    md.append("## 一、資料集")
    md.append(f"- Dataset: `{report['raw']['dataset_path']}`")
    md.append(f"- Raw rows: `{report['raw']['rows']:,}`")
    md.append(f"- Clean rows: `{report['clean']['rows']:,}`")
    md.append(f"- Rows dropped: `{report['diff']['rows_dropped_total']:,}`")
    md.append(f"- Rows retained: `{report['diff']['rows_retained_pct']}%`")
    md.append("")
    md.append("## 二、Raw 品質概況")
    md.append(f"- Invalid label rows: `{report['raw']['invalid_label_rows']}`")
    md.append(f"- Negative Report_Count rows: `{report['raw']['negative_report_count_rows']}`")
    md.append(f"- Negative Financial_Loss rows: `{report['raw']['negative_financial_loss_rows']}`")
    md.append(f"- Report_Count IQR outliers: `{report['raw']['report_count_iqr_outliers']}`")
    md.append(f"- Financial_Loss IQR outliers: `{report['raw']['financial_loss_iqr_outliers']}`")
    md.append("")
    md.append("### 高缺失欄位（>=5%）")
    if report["raw"]["high_missing_cols_pct_ge_5"]:
        for k, v in report["raw"]["high_missing_cols_pct_ge_5"].items():
            md.append(f"- `{k}`: `{v}%`")
    else:
        md.append("- 無")
    md.append("")
    md.append("## 三、Clean 品質概況")
    md.append(f"- Feature columns: `{report['clean']['columns']}`")
    md.append(f"- Class 0 rows: `{report['clean']['class_0_rows']:,}`")
    md.append(f"- Class 1 rows: `{report['clean']['class_1_rows']:,}`")
    md.append(f"- Imbalance ratio (0:1): `{report['clean']['class_imbalance_ratio_0_to_1']}`")
    md.append(f"- Missing cells in features: `{report['clean']['missing_cells_in_features']}`")
    md.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    return json_path, md_path


def run_pipeline(args):
    t_start = time.perf_counter()
    results = {}
    data_path = args.data_path

    # ──────────────────────────────────────────────────────────────
    # Step 1: Load Dataset
    # ──────────────────────────────────────────────────────────────
    step_header(1, "載入資料集")
    import pandas as pd
    df = pd.read_csv(data_path)
    print(f"  資料集: {os.path.basename(data_path)}")
    print(f"  形狀: {df.shape[0]:,} × {df.shape[1]}")
    if "Label" in df.columns:
        y_num = pd.to_numeric(df["Label"], errors="coerce")
        fraud_ratio = (y_num == 1).mean()
        n0 = int((y_num == 0).sum())
        n1 = int((y_num == 1).sum())
        print(f"  詐騙率(可解析標籤): {fraud_ratio:.1%}")
        if n1 > 0:
            print(f"  不均衡比例(可解析標籤): {n0 / n1:.0f}:1")
    results['dataset_shape'] = df.shape

    # ──────────────────────────────────────────────────────────────
    # Step 2: EDA
    # ──────────────────────────────────────────────────────────────
    step_header(2, "EDA 分析")
    if args.skip_eda:
        print("  ⏭️ 已跳過 (--skip-eda)")
    else:
        from src.processing.eda import generate_data_quality_report, run_full_eda
        eda_results = run_full_eda(df, target_col='Label', save_dir=EDA_DIR)
        results['eda'] = eda_results

    # ──────────────────────────────────────────────────────────────
    # Step 3: Data Cleaning & Feature Engineering
    # ──────────────────────────────────────────────────────────────
    step_header(3, "資料清洗 & 特徵工程")
    is_fcc_schema = "Caller ID Number" in df.columns and "Issue" in df.columns and "Label" not in df.columns
    if is_fcc_schema:
        from src.processing.fcc_data_pipeline import fcc_clean_and_prepare
        X, y, _, _ = fcc_clean_and_prepare(
            data_path,
            negative_ratio=args.fcc_negative_ratio,
            hard_negative_ratio=args.fcc_hard_negative_ratio,
        )
        scaler = None
        cluster_ids = None
        print("  使用 FCC 專用管線: src.processing.fcc_data_pipeline")
    else:
        from src.processing.data_pipeline import clean_and_prepare_data
        X, y, scaler, cluster_ids = clean_and_prepare_data(data_path)
        print("  使用既有管線: src.processing.data_pipeline")
    print(f"  清洗後: {X.shape[0]:,} × {X.shape[1]}")
    print(f"  Scaler: {type(scaler).__name__}")
    print(f"  特徵: {list(X.columns)}")
    results['features_shape'] = X.shape
    quality_report = build_raw_clean_quality_report(df, X, y, data_path)
    quality_json_path, quality_md_path = write_quality_reports(quality_report, QUALITY_DIR)
    results["quality_report_json"] = quality_json_path
    results["quality_report_md"] = quality_md_path
    print(f"  📄 Raw→Clean 品質報告(JSON): {quality_json_path}")
    print(f"  📝 Raw→Clean 品質報告(MD):   {quality_md_path}")

    # ══════════════════════════════════════════════════════════════
    # FCC 分支  vs  通用分支（Step 4 ~ 7）
    # ══════════════════════════════════════════════════════════════
    if is_fcc_schema:
        # ── Step 4-FCC: SVM 訓練 ─────────────────────────────────
        step_header(4, "SVM Spam Classifier — 訓練")
        from src.ml.svm_spam_classifier import SVMSpamTrainer, plot_svm_results

        trainer = SVMSpamTrainer()
        svm_report = trainer.train(X, y, test_size=0.2, do_grid_search=args.grid_search)
        results['best_model'] = 'SVM (RBF)'
        results['best_f1'] = svm_report['f1_score']
        results['best_auprc'] = svm_report['roc_auc']  # 用 ROC-AUC 替代 AUPRC
        results['svm_report'] = svm_report

        # ── Step 5-FCC: SVM 評估圖表 ─────────────────────────────
        step_header(5, "SVM 模型評估")
        from sklearn.model_selection import train_test_split as tts, StratifiedKFold, cross_val_score
        X_tr, X_te, y_tr, y_te = tts(X, y, test_size=0.2, random_state=SEED, stratify=y)
        plot_svm_results(trainer, X_te, y_te, save_dir=MODEL_DIR)
        print(f"  評估圖表: {os.path.join(MODEL_DIR, 'svm_spam_evaluation.png')}")

        # 過擬合檢查
        X_tr_s = trainer.scaler.transform(X_tr)
        X_te_s = trainer.scaler.transform(X_te)
        train_acc = float((trainer.model.predict(X_tr_s) == y_tr).mean())
        test_acc = float((trainer.model.predict(X_te_s) == y_te).mean())
        gap = abs(train_acc - test_acc)
        overfit_status = "OVERFIT ⚠️" if gap > 0.05 else "OK ✅"
        overfit = {"train_acc": round(train_acc, 4), "test_acc": round(test_acc, 4),
                   "gap": round(gap, 4), "status": overfit_status}
        results['overfitting'] = overfit
        print(f"  Train Acc: {train_acc:.4f}  Test Acc: {test_acc:.4f}  Gap: {gap:.4f}  → {overfit_status}")

        # 5-Fold CV (使用 pipeline 方式包裝 scaler+svm)
        from sklearn.pipeline import Pipeline as SKPipeline
        from sklearn.preprocessing import StandardScaler as SS
        from sklearn.svm import SVC
        svm_cv = SKPipeline([
            ("scaler", SS()),
            ("svm", SVC(**trainer.best_params, class_weight="balanced",
                        probability=True, random_state=SEED, cache_size=500)),
        ])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        try:
            cv_f1 = cross_val_score(svm_cv, X, y, cv=skf, scoring='f1', n_jobs=-1)
        except PermissionError:
            # 某些受限執行環境（sandbox/CI）不允許 loky 建立平行 worker，改為單執行緒重試。
            logger.warning("CV 平行執行受限，改用 n_jobs=1 重新計算 5-Fold CV")
            cv_f1 = cross_val_score(svm_cv, X, y, cv=skf, scoring='f1', n_jobs=1)
        print(f"  5-Fold CV F1: {cv_f1.mean():.4f} (±{cv_f1.std()*2:.4f})")
        results['cv_f1_mean'] = round(cv_f1.mean(), 4)

        # ── Step 6-FCC: 序列化 ────────────────────────────────────
        step_header(6, "模型序列化 (joblib)")
        model_path = trainer.save_model(os.path.join(SERIALIZE_DIR, 'svm_spam_model.pkl'))
        print(f"  SVM 模型: {model_path}")

        # ── Step 7-FCC: PSI 監控 ──────────────────────────────────
        step_header(7, "PSI 監控 — SVM 模型輸出")
        from src.monitoring.model_monitor import PSICalculator
        psi_calc = PSICalculator()
        baseline_proba = trainer.model.predict_proba(X_tr_s)[:, 1]
        current_proba = trainer.model.predict_proba(X_te_s)[:, 1]
        psi, _ = psi_calc.compute(baseline_proba.tolist(), current_proba.tolist())
        print(f"  PSI (Train vs Test): {psi:.4f}  {'⚠️ DRIFT' if psi >= 0.15 else '✅ STABLE'}")
        results['psi'] = round(psi, 4)
        drifted_proba = np.clip(current_proba + np.random.normal(0.2, 0.1, len(current_proba)), 0, 1)
        psi_drift, _ = psi_calc.compute(baseline_proba.tolist(), drifted_proba.tolist())
        print(f"  PSI (Drifted):       {psi_drift:.4f}  {'⚠️ DRIFT → Retrain' if psi_drift >= 0.15 else '✅ STABLE'}")
        results['psi_drifted'] = round(psi_drift, 4)

        best_name = 'SVM (RBF)'

    else:
        # ──────────────────────────────────────────────────────────
        # Step 4: Multi-Model Training (通用)
        # ──────────────────────────────────────────────────────────
        step_header(4, "多模型訓練 — LR / XGBoost / RandomForest / SVM")
        from src.ml.scam_classifier import (
            train_multiple_models, plot_confusion_matrix,
            check_overfitting, plot_precision_recall_curve,
            plot_shap_summary, save_model,
        )

        model_results = train_multiple_models(
            X, y,
            run_grid_search=args.grid_search,
            run_grid_search_xgb=getattr(args, "grid_search_xgb", False),
            run_grid_search_svm=getattr(args, "grid_search_svm", False),
        )

        best_name = model_results['_best_model_name']
        best_model = model_results['_best_model']
        best_res = model_results[best_name]

        results['best_model'] = best_name
        results['best_f1'] = best_res['test_f1']
        results['best_auprc'] = best_res['auprc']

        # ── SMOTE + cleanlab ──────────────────────────────────────
        print("\n  📐 SMOTE + cleanlab 數據精煉...")
        from src.ml.data_refinement import DataRefinementPipeline
        refinement = DataRefinementPipeline()
        refine_res = refinement.run(X, y)
        results['refinement'] = {
            'baseline_f1': refine_res.get('baseline_f1', 'N/A'),
            'refined_f1': refine_res.get('refined_f1', 'N/A'),
            'noisy_labels': refine_res.get('noisy_labels_found', 0),
        }

        # ──────────────────────────────────────────────────────────
        # Step 5: Evaluation (通用)
        # ──────────────────────────────────────────────────────────
        step_header(5, "模型評估")
        plot_confusion_matrix(
            best_res['_y_test'], best_res['_y_pred'],
            save_path=os.path.join(MODEL_DIR, "confusion_matrix.png"),
        )
        overfit = check_overfitting(
            best_model,
            best_res['_X_train'], best_res['_y_train'],
            best_res['_X_test'], best_res['_y_test'],
        )
        results['overfitting'] = overfit
        plot_precision_recall_curve(
            best_res['_y_test'], best_res['_y_proba'],
            save_path=os.path.join(MODEL_DIR, "pr_curve.png"),
        )
        if hasattr(best_model, 'feature_importances_'):
            plot_shap_summary(best_model, best_res['_X_test'], save_dir=MODEL_DIR)

        from sklearn.model_selection import StratifiedKFold, cross_val_score
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_f1 = cross_val_score(best_model, X, y, cv=skf, scoring='f1')
        print(f"\n  5-Fold CV F1: {cv_f1.mean():.4f} (±{cv_f1.std()*2:.4f})")
        results['cv_f1_mean'] = round(cv_f1.mean(), 4)

        # ── DBSCAN + t-SNE (optional) ────────────────────────────
        if not args.skip_unsupervised:
            print("\n  🔬 DBSCAN + t-SNE 非監督分析...")
            from src.ml.unsupervised import run_unsupervised_analysis
            sample_idx = np.random.choice(len(X), min(5000, len(X)), replace=False)
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
            c_sample = cluster_ids.iloc[sample_idx] if cluster_ids is not None else None
            unsup = run_unsupervised_analysis(X_sample, y_sample, cluster_ids=c_sample, save_dir=UNSUP_DIR)
            results['dbscan_clusters'] = unsup.get('dbscan', {}).get('n_clusters', 0)
        else:
            print("\n  ⏭️ 非監督分析已跳過 (--skip-unsupervised)")

        # ──────────────────────────────────────────────────────────
        # Step 6: Serialize (通用)
        # ──────────────────────────────────────────────────────────
        step_header(6, "模型序列化 (joblib)")
        import joblib
        model_path = os.path.join(SERIALIZE_DIR, 'xgboost_spam_model.pkl')
        save_model(best_model, model_path)
        joblib.dump(scaler, os.path.join(SERIALIZE_DIR, 'scaler.pkl'))
        joblib.dump(list(X.columns), os.path.join(SERIALIZE_DIR, 'feature_names.pkl'))
        print(f"  模型: {model_path}")
        print(f"  Scaler: {os.path.join(SERIALIZE_DIR, 'scaler.pkl')}")
        print(f"  Features: {os.path.join(SERIALIZE_DIR, 'feature_names.pkl')}")

        # ──────────────────────────────────────────────────────────
        # Step 7: PSI Monitor (通用)
        # ──────────────────────────────────────────────────────────
        step_header(7, "PSI 監控 — 真實模型輸出")
        from src.monitoring.model_monitor import PSICalculator
        from sklearn.model_selection import train_test_split
        psi_calc = PSICalculator()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y,
        )
        baseline_proba = best_model.predict_proba(X_train)[:, 1]
        current_proba = best_model.predict_proba(X_test)[:, 1]
        psi, _ = psi_calc.compute(baseline_proba.tolist(), current_proba.tolist())
        print(f"  PSI (Train vs Test): {psi:.4f}  {'⚠️ DRIFT' if psi >= 0.15 else '✅ STABLE'}")
        results['psi'] = round(psi, 4)
        drifted_proba = np.clip(current_proba + np.random.normal(0.2, 0.1, len(current_proba)), 0, 1)
        psi_drift, _ = psi_calc.compute(baseline_proba.tolist(), drifted_proba.tolist())
        print(f"  PSI (Drifted):       {psi_drift:.4f}  {'⚠️ DRIFT → Retrain' if psi_drift >= 0.15 else '✅ STABLE'}")
        results['psi_drifted'] = round(psi_drift, 4)

    # ──────────────────────────────────────────────────────────────
    # Summary （共用）
    # ──────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                   📊 Pipeline 結果摘要                         ║
╠══════════════════════════════════════════════════════════════╣
║  Dataset:            {str(results['dataset_shape']):<39}     ║
║  Features:           {str(results['features_shape']):<39}    ║
║  Best Model:         {best_name:<39}                         ║
║  Test F1:            {results['best_f1']:<39}                ║
║  ROC-AUC / AUPRC:    {results['best_auprc']:<39}             ║
║  5-Fold CV F1:       {results['cv_f1_mean']:<39}             ║
║  Overfitting:        {results['overfitting']['status']:<39}  ║
║  PSI (normal):       {results['psi']:<39}                    ║
║  PSI (drifted):      {results['psi_drifted']:<39}            ║
║  Total Time:         {elapsed:.1f}s{' '*(36-len(f'{elapsed:.1f}s'))}║
╠══════════════════════════════════════════════════════════════╣
║  Outputs:                                                    ║
║    📁 outputs/eda/          — EDA 圖表                        ║
║    📁 outputs/models/       — CM / PR / SHAP / SVM eval      ║
║    📁 outputs/quality/      — raw→clean JSON / Markdown      ║
║    📁 models/               — 序列化 artifacts                ║
╠══════════════════════════════════════════════════════════════╣
║  Next Steps:                                                 ║
║    1. jupyter notebook notebooks/                            ║
║    2. uvicorn src.api.detection_api:app --port 8000          ║
║    3. docker-compose up -d                                   ║
╚══════════════════════════════════════════════════════════════╝
""")

    return results


def main():
    cfg = load_pipeline_config()

    parser = argparse.ArgumentParser(
        description="Archangel MLOps Pipeline — 防詐模型訓練與部署",
    )
    parser.add_argument('--skip-eda', action='store_true',
                        help='跳過 EDA 視覺化 (加速)')
    parser.add_argument('--grid-search', action='store_true',
                        help='啟用 GridSearchCV 超參數調整（通用分支：同時啟用 XGBoost 與 SVM；FCC 分支：啟用 SVM）')
    parser.add_argument('--grid-search-xgb', action='store_true',
                        help='僅啟用 XGBoost GridSearchCV（通用分支）')
    parser.add_argument('--grid-search-svm', action='store_true',
                        help='僅啟用 SVM GridSearchCV（通用分支）')
    parser.add_argument('--skip-unsupervised', action='store_true',
                        help='跳過 DBSCAN + t-SNE')
    parser.add_argument(
        '--data-path',
        type=str,
        default=DEFAULT_DATA_PATH,
        help='指定資料集路徑（可傳 FCC_Consumer_Complaints_RAW_Data_2017.csv）',
    )

    # FCC-only knobs（可用 pipeline_config.yaml 設定預設值，CLI 會覆蓋）
    parser.add_argument(
        "--fcc-negative-ratio",
        type=float,
        default=float(_cfg_get(cfg, "fcc.negative_ratio", 1.0)),
        help="FCC 負/正樣本比例 (1.0 = 1:1 平衡)。",
    )
    parser.add_argument(
        "--fcc-hard-negative-ratio",
        type=float,
        default=float(_cfg_get(cfg, "fcc.hard_negative_ratio", 0.03)),
        help="FCC hard negatives 比例（預設 0.03 = 3/100；來自 complaint_count=1 pool）。",
    )
    args = parser.parse_args()

    print_banner(args.data_path)

    if not os.path.exists(args.data_path):
        print(f"❌ 找不到資料集: {args.data_path}")
        print("   請確認資料集路徑正確")
        sys.exit(1)

    run_pipeline(args)


if __name__ == "__main__":
    main()
