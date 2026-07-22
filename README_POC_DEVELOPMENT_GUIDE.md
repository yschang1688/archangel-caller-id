# 防詐來電辨識：Python 實戰 POC 開發藍圖

本指南對應《全民瘋 AI 系列》教材，展現處理不平衡數據（Imbalanced Data）與模型解釋化（XAI）的專業即戰力。  
已同步更新，完整涵蓋 **Archangel Intelligence System** 的所有已實作模組。

---

## ⚡ Gap 對照表（指南 vs 實作）

| # | 指南原始描述 | 目前實作狀態 | 對應檔案 |
|---|---|---|---|
| 1 | Z-score 離群值移除 | ✅ **已實作**（`remove_outliers_zscore`） | `src/processing/data_pipeline.py` |
| 2 | Clipping 極端值平滑 | ✅ **已實作**（`clip_extremes`） | `src/processing/data_pipeline.py` |
| 3 | Target Encoding | ⚠️ 改用 One-Hot（低基數類別更適合） | `src/processing/data_pipeline.py` |
| 4 | RobustScaler | ✅ **已換用**（取代 StandardScaler） | `src/processing/data_pipeline.py` |
| 5 | XGBoost + scale_pos_weight | ✅ 已實作 | `src/ml/scam_classifier.py` |
| 6 | SHAP Summary Plot | ✅ **已新增** | `src/ml/scam_classifier.py` |
| 7 | Baseline Logistic Regression | ✅ 已實作 | `src/ml/scam_classifier.py` |
| 8 | DBSCAN 群集探索 | ✅ **已新增** | `src/ml/unsupervised.py` |
| 9 | t-SNE 視覺化 | ✅ **已新增** | `src/ml/unsupervised.py` |
| 10 | StratifiedKFold | ✅ 已實作 | `src/ml/scam_classifier.py` |
| 11 | Precision-Recall Curve / AUPRC | ✅ **已新增** | `src/ml/scam_classifier.py` |
| 12 | SMOTE 過採樣 | ✅ 已實作 | `src/ml/data_refinement.py` |
| 13 | FastAPI /predict | ✅ 已實作（含更多端點） | `src/api/detection_api.py` |
| 14 | joblib 模型序列化 | ✅ **已新增** | `src/ml/scam_classifier.py` |
| — | Spark ETL / Data Skew Salting | ✅ **指南未涵蓋，已補充** | `src/processing/spark_etl.py` |
| — | A/B Testing（Cohen's d） | ✅ **指南未涵蓋，已補充** | `src/ml/ab_testing.py` |
| — | PSI 模型漂移監控 | ✅ **指南未涵蓋，已補充** | `src/monitoring/model_monitor.py` |
| — | Guardian Score（貝葉斯） | ✅ **指南未涵蓋，已補充** | `src/feature_engineering/guardian_score.py` |
| — | cleanlab 標籤品質修正 | ✅ **指南未涵蓋，已補充** | `src/ml/data_refinement.py` |
| — | `fraud_100000_dataset.csv` 支援 | ✅ **已新增 schema 自動偵測** | `src/processing/data_pipeline.py` |

---

## 階段一：資料清洗與特徵萃取（數據工程力）

**核心套件：** `pandas`, `numpy`, `scipy`, `sklearn`  
**實作檔案：** `src/processing/data_pipeline.py`

### 1.1 雙資料集支援（fraud_1000 vs fraud_100000）

系統自動偵測 CSV schema 並路由至對應管線：

```python
from src.processing.data_pipeline import clean_and_prepare_data, LARGE_DATASET

# 使用 100k 生產資料集（schema 自動偵測）
X, y, scaler, cluster_ids = clean_and_prepare_data(LARGE_DATASET)
# cluster_ids 保留 Cluster_ID 欄位供非監督式分析使用

# 使用 1k 舊版資料集
X, y, scaler, _ = clean_and_prepare_data()
```

`fraud_100000_dataset.csv` 欄位結構：

| 欄位 | 類型 | 說明 |
|---|---|---|
| Report_Count | int | 同號被回報次數 |
| Financial_Loss | float | 財損金額（右偏分佈，用 RobustScaler） |
| Age_Group | cat | 18-25 / 26-40 / 41-60 / 70+ / UNKNOWN |
| Education | cat | High School / Bachelor / Master/PhD / UNKNOWN |
| Cluster_ID | cat | 地面真值群集（訓練時移除，保留供評估） |
| Label | 0/1 | 目標欄位（90% 正常 / 10% 詐騙） |

### 1.2 Z-score 離群值移除

```python
from scipy import stats
import numpy as np

# Z-score 方法 — 適應資料尺度，比硬門檻更科學
z_scores = np.abs(stats.zscore(df['Report_Count']))
df_clean = df[z_scores < 3.0]  # 保留 3σ 內

# 或使用 Clipping（保留攻擊訊號，避免直接刪除）
df['Financial_Loss'] = df['Financial_Loss'].clip(
    lower=df['Financial_Loss'].quantile(0.01),
    upper=df['Financial_Loss'].quantile(0.99)
)
```

### 1.3 RobustScaler（取代 StandardScaler）

```python
from sklearn.preprocessing import RobustScaler

# 為何用 RobustScaler？
# StandardScaler 用 mean/std → 被極端詐騙金額拉偏
# RobustScaler 用 median/IQR → 對離群值有穩健性
scaler = RobustScaler()
df[['Report_Count', 'Financial_Loss']] = scaler.fit_transform(
    df[['Report_Count', 'Financial_Loss']]
)
```

---

## 階段二：建立防詐預測核心模型（算法深度與解釋力）

**核心套件：** `xgboost`, `shap`, `sklearn`, `joblib`  
**實作檔案：** `src/ml/scam_classifier.py`

### 2.1 XGBoost + 類別不均衡處理

```python
import xgboost as xgb

# scale_pos_weight = 負樣本數 / 正樣本數
# 100k 資料集：90000 / 10000 = 9.0x
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=pos_weight,   # 關鍵：強制模型重視少數詐騙樣本
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)
```

### 2.2 SHAP 值解釋化

```python
import shap
from src.ml.scam_classifier import plot_shap_summary

# TreeExplainer 對 XGBoost 精確計算 SHAP（非近似）
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Summary Plot（Bar）— 全域特徵重要性（Mean |SHAP|）
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=15)

# Summary Plot（Dot）— 特徵方向 + 強度分佈
shap.summary_plot(shap_values, X_test, max_display=15)

# 便捷函數（自動儲存兩張圖）
plot_shap_summary(xgb_model, X_test, save_dir=".")
```

**面試關鍵論述：**
> 「我不僅知道模型預測它是詐騙，還能透過 SHAP 說明是因為該門號 Report_Count 異常高（特徵 A）且 Financial_Loss 偏高（特徵 B），這才是防詐業務真正需要的可解釋 AI，可以直接提交給法務部門做調查依據。」

### 2.3 Baseline 比較

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_f1  = f1_score(y_test, lr_model.predict(X_test))
xgb_f1 = f1_score(y_test, xgb_model.predict(X_test))

print(f"Baseline (LR) F1:  {lr_f1:.4f}")
print(f"XGBoost F1:         {xgb_f1:.4f}")
print(f"提升幅度: {(xgb_f1 - lr_f1) / lr_f1 * 100:+.2f}%")
```

### 2.4 模型序列化（joblib）

```python
import joblib
from src.ml.scam_classifier import save_model, load_model

# 儲存（含完整 XGBoost state）
save_model(xgb_model, "models/xgboost_fraud_pipeline.pkl")

# 載入（推論服務使用）
loaded_model = load_model("models/xgboost_fraud_pipeline.pkl")
pred = loaded_model.predict(X_new)
```

---

## 階段三：探索未知詐騙犯罪集團（非監督式洞察）

**核心套件：** `scikit-learn`, `matplotlib`  
**實作檔案：** `src/ml/unsupervised.py`

### 3.1 DBSCAN 詐騙集群發現

```python
from src.ml.unsupervised import run_dbscan, analyze_cluster_fraud_rates

# DBSCAN — 發現高密度詐騙集群（不需指定 k）
result = run_dbscan(X, eps=0.6, min_samples=10, ground_truth=cluster_ids)
# result["n_clusters"]    → 發現群集數
# result["silhouette"]    → 輪廓係數（0.7+ excellent）
# result["ari"]           → 與 Cluster_ID 的 Adjusted Rand Index

# 分析各群集的詐騙率 → 識別高風險詐騙集群
stats = analyze_cluster_fraud_rates(y, result["labels"], cluster_ids)
```

**為何用 DBSCAN 而非 K-means？**
- K-means 需預設 k，假設球形群集
- 詐騙集團在特徵空間形成**不規則高密度島嶼**（同撥號中心 → 相似的 Report_Count / Financial_Loss 模式）
- DBSCAN 自動找到這些島嶼，稀疏區域標記為雜訊（-1）→ 通常是正常通話

`fraud_100000_dataset.csv` 的地面真值群集：

| Cluster_ID | 數量 | 說明 |
|---|---|---|
| C_NORMAL | 90,000 | 正常來電 |
| C_88X_RING | 4,000 | 協同詐騙集團（88x 前綴熱鍵） |
| C_DEMO_TARGET | 4,000 | 被鎖定受害者 |
| C_POISON_BOT | 2,000 | 機器人帳號（管線中過濾掉） |

### 3.2 t-SNE 視覺化

```python
from src.ml.unsupervised import run_tsne_and_plot

# 產生雙面板 t-SNE 圖
# 左：詐騙標籤分佈  右：Cluster_ID / DBSCAN 群集
X_tsne = run_tsne_and_plot(
    X, y,
    ground_truth_clusters=cluster_ids,  # 或傳入 dbscan_labels
    n_samples=5000,   # 子採樣加速
    save_dir="."
)
# 儲存 tsne_visualization.png
```

### 3.3 一鍵執行完整非監督式管線

```python
from src.ml.unsupervised import run_unsupervised_analysis
from src.processing.data_pipeline import clean_and_prepare_data, LARGE_DATASET

X, y, scaler, cluster_ids = clean_and_prepare_data(LARGE_DATASET)
results = run_unsupervised_analysis(X, y, cluster_ids=cluster_ids, save_dir=".")
```

---

## 階段四：嚴謹驗證與指標優化（科學家素養）

**核心套件：** `sklearn.metrics`, `imbalanced-learn`, `cleanlab`  
**實作檔案：** `src/ml/scam_classifier.py`, `src/ml/data_refinement.py`

### 4.1 分層 K-Fold 驗證

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# StratifiedKFold 確保每折的詐騙比例一致（10%），避免驗證偏差
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X, y, cv=skf, scoring='f1')
print(f"5-Fold CV F1: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
```

### 4.2 Precision-Recall Curve（防詐正確指標）

```python
from src.ml.scam_classifier import plot_precision_recall_curve

y_proba = xgb_model.predict_proba(X_test)[:, 1]
auprc = plot_precision_recall_curve(y_test, y_proba, save_path="pr_curve.png")
```

**為何 AUPRC > ROC-AUC？**

在 90:10 的不均衡資料中，ROC-AUC 因計入大量 True Negative 而虛高。  
AUPRC 只關注少數詐騙類別的排名精準度，才是防詐場景的正確指標。

### 4.3 SMOTE + cleanlab 數據精煉

```python
from src.ml.data_refinement import DataRefinementPipeline

pipeline = DataRefinementPipeline(random_state=42)

# Phase 1: SMOTE 過採樣（1% → 50% 詐騙）
X_balanced, y_balanced = pipeline.apply_smote(X_train, y_train)

# Phase 2: cleanlab 找出標籤錯誤（用 XGBoost cross-val 預測作為 pred_probs）
X_clean, y_clean = pipeline.apply_cleanlab(X_balanced, y_balanced)
# pipeline.noisy_label_indices → 可疑標籤的索引
```

**面試關鍵論述：**
> 「Data-centric AI 的核心是『垃圾進，垃圾出』。SMOTE 解決樣本不足問題，cleanlab 用 confident learning 找出 mislabeled 樣本。在 100k 資料集中，C_POISON_BOT 類型的帳號會惡意污染標籤，cleanlab 能自動偵測這些噪音。」

---

## 階段五：模型落地與 API 化（產品化思維）

**核心套件：** `FastAPI`, `uvicorn`, `joblib`  
**實作檔案：** `src/api/detection_api.py`

### 5.1 FastAPI 服務化

```python
# 啟動 API（含 Swagger UI）
uvicorn src.api.detection_api:app --reload --host 0.0.0.0 --port 8000

# API 端點
# POST /register_user      → 新增用戶
# POST /report_fraud        → 提交詐騙回報
# GET  /phone_risk/{phone}  → 查詢號碼風險
# GET  /leaderboard         → 守護者排行榜
# GET  /docs               → Swagger UI
```

### 5.2 Docker 一鍵啟動完整架構

```bash
# 完整生產架構：Kafka + Spark + Redis + MLflow + FastAPI
docker-compose up -d

# 服務端口
# Kafka UI:      http://localhost:8080
# MLflow:        http://localhost:5000
# FastAPI:       http://localhost:8000/docs
# RedisInsight:  http://localhost:8001
```

---

## 階段六：Data Skew 處理（資料工程進階）

> **📌 本階段為資料工程能力的核心展示項目之一**

**實作檔案：** `src/processing/spark_etl.py`

### 6.1 Salting Technique

在全球防詐場景中，特定詐騙集中的號碼（如 C_88X_RING）被大量回報，
造成 Spark Partition 嚴重傾斜，少數 Worker 負載是其他節點的 100x 以上。

```python
from src.processing.spark_etl import AntiFraudETL

etl = AntiFraudETL()
results = etl.run(n_records=50_000)

print(f"Pre-salt skew ratio:  {results['pre_salt_skew_ratio']:.1f}x")
# → 100.7x（熱鍵集中所有流量）
print(f"Post-salt skew ratio: {results['post_salt_skew_ratio']:.2f}x")
# → 2.14x（salting 均勻分散後）
```

**Salting 原理：**

```python
# 原始 Key（熱鍵）
key = "+886-800-SCAM"  # 全部流量集中到同一個 Partition

# Salt 後的 Key（分散到 32 個 Partition）
import random
salt = random.randint(0, 31)
salted_key = f"{key}__salt_{salt}"
# "+886-800-SCAM__salt_17" → Partition 17
# "+886-800-SCAM__salt_3"  → Partition 3
# ...分散到 32 個 Partition，skew ratio 從 100.7x → 2.14x
```

---

## 階段七：A/B 測試框架（實驗設計）

> **📌 核心工作流程：每次演算法升級都需要嚴謹的 A/B 實驗**

**實作檔案：** `src/ml/ab_testing.py`

### 7.1 Power Analysis（先做再測）

```python
from src.ml.ab_testing import ABTestingFramework

ab = ABTestingFramework()

# 先計算需要多少樣本才能達到 80% 統計檢定力
sample_size = ab.power_analysis(
    baseline_rate=0.673,    # 目前命中率
    min_detectable_effect=0.02,  # 最小可偵測效果 +2pp
    alpha=0.05,
    power=0.80
)
print(f"需要樣本數: {sample_size} per arm")
```

### 7.2 統計顯著性 + 效果量

```python
# 執行完整 A/B 測試（結合頻率派 + 貝葉斯）
results = ab.run_demo()
hit_rate_result = results["hit_rate"]

print(f"p-value:   {hit_rate_result.p_value:.4f}")   # 0.031 < 0.05 → 顯著
print(f"Cohen's d: {hit_rate_result.cohen_d:.3f}")    # 效果量（面試必問）
print(f"95% CI:    [{hit_rate_result.ci_lower:.3f}, {hit_rate_result.ci_upper:.3f}]")
print(f"顯著:      {hit_rate_result.is_significant}")
```

**面試關鍵論述：**
> 「p-value 只告訴你『效果不為零』，不告訴你效果有多大。Cohen's d 才是 business 決策的依據。d=0.14 表示中度效果，加上 CI=[0.001, 0.029]，代表命中率的實際提升範圍。這才是向產品團隊報告的正確方式。」

---

## 階段八：模型監控與閉環（MLOps）

> **📌 展示「閉環監控」能力**

**實作檔案：** `src/monitoring/model_monitor.py`

### 8.1 PSI 分佈漂移檢測

```python
from src.monitoring.model_monitor import ModelMonitor, run_demo as monitor_demo

# 模擬 30 天的模型漂移
results = monitor_demo()

print(f"PSI Score:         {results['psi_score']:.4f}")
# PSI < 0.10 → 穩定
# PSI 0.10-0.25 → 需調查
# PSI > 0.25 → 重新訓練！
print(f"Drift Severity:    {results['drift_severity']}")
print(f"Retrain Triggered: {results['retraining_triggered']}")
```

**為何監控 PSI？**
- 詐騙模式每週演進，3 個月前的模型可能已對新攻擊「失明」
- PSI 比較「當前預測分佈」vs「基準分佈」，偏移超過 0.25 自動觸發重新訓練
- 這是防詐系統的「免疫系統」

### 8.2 閉環架構

```
新資料入庫
    ↓
Kafka 串流 (src/ingestion/kafka_producer.py)
    ↓
Spark ETL + Data Skew Salting
    ↓
特徵工程 + Guardian Score 更新
    ↓
XGBoost 預測（FastAPI 即時服務）
    ↓
PSI 監控 → 漂移告警
    ↓ (drift > 0.25)
自動觸發重新訓練 → MLflow 版本管理
    ↑___________________________|
```

---

## 階段九：Guardian Score 信譽引擎（貝葉斯創新）

> **📌 Archangel 的核心差異化功能，展現貝葉斯思維**

**實作檔案：** `src/feature_engineering/guardian_score.py`

### 9.1 Beta 分佈即時更新

```python
from src.feature_engineering.guardian_score import GuardianScoreEngine

engine = GuardianScoreEngine()
engine.register_user("alice", "fp_abc", "TW")

# 每次回報驗證後，貝葉斯更新 Beta(α, β)
# 初始: Beta(2, 2) → accuracy_rate = 0.5（中立先驗）
# 每次正確: α += 1（獲得信任）
# 每次錯誤: β += 1（降低信任）

result = engine.submit_report("alice", "+886-800-SCAM", "投資詐騙")
print(f"決策: {result['decision']}")        # REVIEW / BLOCK / SAFE
print(f"加權詐騙分數: {result['weighted_scam_score']:.3f}")
```

### 9.2 四階信譽等級

| 等級 | 分數門檻 | 回報權重 | 說明 |
|---|---|---|---|
| 平民 (CIVILIAN) | 0.0 | 0.10 | 新用戶，低信任 |
| 騎士 (KNIGHT) | 0.40 | 0.35 | 有回報記錄 |
| 守護者 (GUARDIAN) | 0.65 | 0.65 | 可信賴的舉報者 |
| 大天使 (ARCHANGEL) | 0.85 | 1.00 | 最高信任，可即時封鎖 |

---

## 快速執行指令

```bash
# 啟動 conda 環境
conda activate condaml

# 一鍵執行完整 Demo
python run_demo.py

# 個別模組
python -c "
from src.processing.data_pipeline import clean_and_prepare_data, LARGE_DATASET
from src.ml.scam_classifier import train_xgboost_and_evaluate, plot_shap_summary, plot_precision_recall_curve
from src.ml.unsupervised import run_unsupervised_analysis

X, y, scaler, cluster_ids = clean_and_prepare_data(LARGE_DATASET)
model, features, metrics = train_xgboost_and_evaluate(X, y)
plot_shap_summary(model, metrics['_X_test'])
plot_precision_recall_curve(metrics['_y_test'], metrics['_y_proba'])
run_unsupervised_analysis(X, y, cluster_ids=cluster_ids)
"

# Jupyter Notebook 互動式
jupyter notebook notebooks/

# FastAPI Swagger UI
uvicorn src.api.detection_api:app --reload --port 8000

# 執行測試
pytest tests/ -v

# Docker 完整架構
docker-compose up -d
```

---

## 環境設定

所有套件已安裝在 `condaml` conda 環境中：

- ✅ xgboost (3.2.0)
- ✅ shap (0.51.0)
- ✅ scikit-learn (1.8.0)
- ✅ scipy (1.17.1)
- ✅ pandas (3.0.1)
- ✅ numpy (2.2.6)
- ✅ imbalanced-learn (0.14.1)
- ✅ cleanlab
- ✅ FastAPI (0.135.1)
- ✅ uvicorn (0.41.0)
- ✅ joblib (1.5.3)
- ✅ matplotlib, seaborn
- ✅ pytest

**啟動環境：**
```bash
conda activate condaml
```

---

## 專案目錄結構

```
Archangel/
├── src/
│   ├── processing/
│   │   ├── data_pipeline.py      # 雙 schema 清洗 + RobustScaler + Z-score
│   │   └── spark_etl.py          # Data Skew Salting (100.7x → 2.14x)
│   ├── ml/
│   │   ├── scam_classifier.py    # XGBoost + SHAP + PR Curve + joblib
│   │   ├── ab_testing.py         # A/B Testing + Cohen's d + Power Analysis
│   │   ├── data_refinement.py    # SMOTE + cleanlab
│   │   └── unsupervised.py       # DBSCAN + t-SNE  ← 新增
│   ├── monitoring/
│   │   └── model_monitor.py      # PSI 漂移偵測 + 自動重新訓練
│   ├── feature_engineering/
│   │   └── guardian_score.py     # Bayesian Beta Distribution 信譽引擎
│   ├── ingestion/
│   │   └── kafka_producer.py     # Kafka 串流模擬
│   └── api/
│       └── detection_api.py      # FastAPI REST API
├── tests/
│   └── test_guardian_score.py
├── configs/
│   └── pipeline_config.yaml
├── fraud_100000_dataset.csv       # 主要資料集（100k 筆）
├── fraud_1000_dataset.csv         # 舊版資料集（1k 筆）
├── run_demo.py                    # 一鍵執行完整 Demo
├── docker-compose.yml             # Kafka + Spark + Redis + MLflow
└── Dockerfile
```

---

## 面試準備重點

1. **技術深度**：解釋每個步驟的理論依據（為何 RobustScaler > StandardScaler？為何 AUPRC > ROC-AUC？）
2. **業務理解**：說明為何在防詐場景中 F1-Score 比 Accuracy 重要；PSI 對業務意味著什麼
3. **可解釋性**：用 SHAP 展示模型決策過程，能指出「是哪些特徵讓這個號碼被判斷為詐騙」
4. **產品化思維**：展示 Docker 一鍵啟動、FastAPI 即時預測、MLflow 版本管理
5. **科學嚴謹性**：A/B Testing 前先做 Power Analysis，報告 Cohen's d 效果量而非只看 p-value
6. **Data-centric AI**：SMOTE + cleanlab 的組合展現「先修數據，再調模型」的思維

---

**祝面試順利！** 🚀
