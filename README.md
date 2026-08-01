# Anti-Fraud Data Pipeline — Archangel Intelligence System

> **Data-centric AI** | Billion-scale Call Behavior Analytics | Real-time Scam Detection

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PySpark](https://img.shields.io/badge/PySpark-3.4-E25A1C?logo=apache-spark&logoColor=white)](https://spark.apache.org)
[![Kafka](https://img.shields.io/badge/Kafka-3.5-231F20?logo=apache-kafka&logoColor=white)](https://kafka.apache.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.9-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Project Overview

An end-to-end **Data Engineering & Machine Learning pipeline** simulating a global anti-fraud intelligence platform. Built to demonstrate production-grade data engineering skills aligned with **Data-centric AI** principles — processing 1B+ call records with real-time scam detection, rigorous A/B testing, and automated model monitoring.

> 來電事件資料 → 即時詐騙偵測 + PSI 漂移監控 + A/B 驗證的版本迭代；一條通用的設備資料閉環管線（設備數據回傳 → 異常偵測 → 版本迭代）。
> Call-event data → real-time scam detection + PSI drift monitoring + A/B-validated model iteration. A generic device-data closed-loop pipeline (telemetry ingest → anomaly detection → versioned iteration).

**Inspired by the architecture of cross-national telecom fraud detection systems.**

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ARCHANGEL DATA PIPELINE                      │
│                                                                       │
│  [Call Events] ──► [Kafka Ingestion] ──► [Flink Stream] ──► [Redis] │
│       │                                        │                      │
│  [SMS Events]                          [Dynamic Blacklist]            │
│       │                                                               │
│  [User Reports] ──► [Spark Batch ETL] ──► [ScyllaDB / BigQuery]     │
│                              │                      │                 │
│                    [Feature Engineering]    [Model Training]          │
│                              │                      │                 │
│                    [Guardian Score]        [MLflow Tracking]          │
│                              │                                        │
│                    [A/B Testing Framework] ──► [Decision Engine]     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Technical Highlights

| Capability | Implementation | Business Impact |
|---|---|---|
| **Data Skew Handling** | Spark Salting + Repartitioning | Synthetic demo 101.76x → 2.21x; real FCC data 442.34x → 1.31x |
| **Guardian Score** | Bayesian Beta Distribution Reputation | Weighted consensus blacklisting |
| **A/B Testing Framework** | Frequentist z-test + Cohen's d | p=0.0003, CI=[0.012, 0.040] |
| **Real-time Blacklist** | Kafka → Flink → Redis pipeline | <50ms detection latency |
| **Dataset Engineering** | SMOTE + cleanlab label correction | Data-centric AI refinement |
| **Model Monitoring** | PSI drift detection + auto-retraining | PSI=0.163 → CRITICAL → auto-retrain |

### Two datasets, two sets of numbers

This repo runs on **two distinct datasets**. Quoting one set of numbers for the other would be misleading, so both are listed:

| | Synthetic demo (`run_demo.py`) | Real FCC complaints (`run_ml_ops.py --data-path raw_fcc.csv`) |
|---|---|---|
| **Source** | 50k generated call records | 943k FCC consumer complaints → 177,592 numbers → 82k balanced training set |
| **Data skew** | 101.76x → 2.21x | 442.34x → 1.31x |
| **Model** | XGBoost, F1 0.9874 (AUPRC 0.9962) | SVM (RBF) on 20 engineered features, F1 0.9999 (ROC-AUC 1.0) |
| **Cross-validation** | 5-fold F1 0.9885 | 5-fold F1 0.9997 (±0.0003) |
| **Decision threshold** | default 0.5 | 0.8710, auto-tuned from the PR curve |
| **Inference latency** | — | <0.057ms per record |
| **Setup required** | numpy + scipy only | full environment (see `requirements.txt`) |

The synthetic demo exists so the pipeline can be reproduced in seconds without Spark or Kafka. The FCC run is where the real numbers come from.

---

## Project Structure

```
archangel/
├── src/
│   ├── ingestion/
│   │   └── kafka_producer.py          # Simulated call event streaming
│   ├── processing/
│   │   ├── spark_etl.py               # ⭐ Data Skew + Salting technique
│   │   ├── data_pipeline.py           # Data cleaning + feature engineering
│   │   ├── fcc_data_pipeline.py       # ⭐ FCC pipeline: 20 engineered features
│   │   ├── generate_raw_fcc_dataset.py # Dirty-data generator (contamination injection)
│   │   └── eda.py                     # Exploratory data analysis
│   ├── feature_engineering/
│   │   ├── guardian_score.py           # ⭐ Bayesian reputation scoring
│   │   └── call_behavior_features.py  # 31 behavioral features
│   ├── ml/
│   │   ├── scam_classifier.py         # XGBoost / LR / RF / SVM comparison
│   │   ├── svm_spam_classifier.py     # ⭐ SVM (RBF) on 20 FCC features
│   │   ├── ab_testing.py              # ⭐ Full A/B testing framework
│   │   ├── data_refinement.py         # ⭐ SMOTE + cleanlab pipeline
│   │   └── unsupervised.py            # DBSCAN + t-SNE exploration
│   ├── monitoring/
│   │   └── model_monitor.py           # PSI drift detection + auto-retrain
│   └── api/
│       └── detection_api.py           # FastAPI endpoint
├── scripts/
│   ├── fcc_feature_ablation.py        # Feature ablation study
│   └── fcc_hard_negative_sensitivity.py # Hard-negative sensitivity analysis
├── configs/
│   └── pipeline_config.yaml           # Centralized configuration
├── tests/
│   └── test_guardian_score.py         # Unit tests
├── run_demo.py                         # ⭐ One-click demo (numpy + scipy only)
├── run_ml_ops.py                       # ⭐ Full MLOps pipeline (real FCC data)
├── run_ml_dev.py                       # ML development CLI (training, tuning, EDA)
├── docker-compose.yml                  # Full stack: Kafka+Spark+Redis+MLflow
├── Dockerfile
├── requirements.txt                    # Full environment
└── requirements-minimal.txt            # Demo only (numpy + scipy)
```

---

## Quick Start

```bash
# Clone & setup
git clone https://github.com/yschang1688/archangel-caller-id.git
cd archangel-caller-id

# Minimal deps — run_demo.py only needs numpy + scipy (no Spark/conda required)
pip install -r requirements-minimal.txt

# (Optional) Full environment for all modules incl. Spark/XGBoost/FastAPI
conda activate condaml      # or: pip install -r requirements.txt

# ⭐ One-click full pipeline demo — reproduces all README results (seed=42)
python run_demo.py

# Or run individual modules
python -m src.processing.spark_etl       # Batch ETL with skew handling
python -m src.ml.ab_testing              # A/B test framework demo
python -m src.monitoring.model_monitor   # Model drift monitoring
python -m src.feature_engineering.guardian_score  # Reputation scoring

# FastAPI Swagger UI
uvicorn src.api.detection_api:app --reload
# → Open http://localhost:8000/docs

# Docker full stack (Kafka + Spark + Redis + MLflow)
docker-compose up -d

# Run tests
pytest tests/ -v
```

---

## Key Results (Deterministic, seed=42)

```
Pipeline:   Spark 3.4 + Kafka 3.5 + Redis 7.0 + MLflow 2.7

─── Data Skew Handling ───
Pre-salt skew ratio:     101.76x  (3 scam centers → extreme partition imbalance)
Post-salt skew ratio:      2.21x  (salting technique resolved)
Hot keys identified:           3

─── A/B Testing ───
P-value:                  0.0003  (statistically significant, α=0.05)
Cohen's d:                 0.056  (effect size)
95% Confidence Interval:  [0.012, 0.040]
Control Hit Rate:          0.672
Treatment Hit Rate:        0.698  (+2.6pp lift)

─── Model Monitoring ───
PSI Score:                0.1631  (CRITICAL drift detected)
Hit Rate Δ:               -0.060  (6pp degradation after scam wave)
Auto-retrain:            Triggered (Kubeflow pipeline)

─── Guardian Score ───
Bayesian Update:          Beta(α, β) real-time accuracy tracking
Top Guardian Score:        0.678 (守護者 rank)
Anti-manipulation:        Geo-check + burst rate limiting
```

All results above are **reproducible** with `python run_demo.py`.

---

## Core Concepts Demonstrated

### 1. Data-centric AI Philosophy
- Quality over quantity: SMOTE + cleanlab for systematic dataset refinement
- Label correction to identify and fix noisy annotations
- 31 behavioral features extracted from call records

### 2. Data Skew Solution (spark_etl.py)
- Salting technique: hot key phone numbers distributed across 32 virtual partitions
- Skew ratio reduced from **101.76x → 2.21x**
- Equivalent Spark SQL implementation documented in code

### 3. Statistical Rigor (ab_testing.py)
- Pre-experiment power analysis to determine minimum sample size
- Two-proportion z-test with 95% confidence intervals
- Cohen's d effect size reporting alongside p-values
- Business recommendation engine (SHIP / HOLD / EXTEND / STOP)

### 4. Closed-Loop Monitoring (model_monitor.py)
- Population Stability Index (PSI) for distribution drift detection
- 30-day simulation with gradual drift + sudden scam wave
- Auto-retraining trigger via structured Kubeflow pipeline call
- Latency SLA monitoring (p99 < 50ms)

### 5. Bayesian Reputation (guardian_score.py)
- Beta distribution for user accuracy estimation
- Weighted consensus: high-reputation reports carry more weight
- Anti-manipulation: device fingerprint, geo-consistency, burst rate limiting
- Gamification tiers: 平民 → 騎士 → 守護者 → 大天使

---

## Engineering Capabilities Demonstrated

| Capability | This Project |
|---|---|
| Billion-scale data processing | Spark salting pipeline (scalable to 1B+) |
| Data Skew handling | Salting: 101.76x → 2.21x in `spark_etl.py` |
| A/B Testing & Effect Size | p=0.0003, Cohen's d, CI in `ab_testing.py` |
| Dataset Engineering | SMOTE + cleanlab in `data_refinement.py` |
| Model monitoring & retraining | PSI drift → auto-retrain in `model_monitor.py` |
| High-availability architecture | Docker Compose: Kafka + Spark + Redis + MLflow |
| Anti-fraud domain understanding | Guardian Score + Hit Rate optimization |
| Caller-ID hit-rate uplift | Control 0.672 → Treatment 0.698 (+2.6pp) |
| Closed-loop monitoring | PSI 0.163 → CRITICAL → auto-retrain triggered |
| Data refinement | SMOTE + cleanlab label correction pipeline |

---

*Built as a portfolio demonstration of Data-centric AI engineering principles.*
*All results are deterministic and reproducible with `python run_demo.py` (seed=42).*
