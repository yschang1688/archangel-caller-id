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
| **Data Skew Handling** | Spark Salting + Repartitioning | Skew ratio: 101.76x → 2.21x |
| **Guardian Score** | Bayesian Beta Distribution Reputation | Weighted consensus blacklisting |
| **A/B Testing Framework** | Frequentist z-test + Cohen's d | p=0.0003, CI=[0.012, 0.040] |
| **Real-time Blacklist** | Kafka → Flink → Redis pipeline | <50ms detection latency |
| **Dataset Engineering** | SMOTE + cleanlab label correction | Data-centric AI refinement |
| **Model Monitoring** | PSI drift detection + auto-retraining | PSI=0.163 → CRITICAL → auto-retrain |

---

## Project Structure

```
archangel/
├── src/
│   ├── ingestion/
│   │   └── kafka_producer.py          # Simulated call event streaming
│   ├── processing/
│   │   ├── spark_etl.py               # ⭐ Data Skew + Salting technique
│   │   └── data_pipeline.py           # Data cleaning + feature engineering
│   ├── feature_engineering/
│   │   ├── guardian_score.py           # ⭐ Bayesian reputation scoring
│   │   └── call_behavior_features.py  # 31 behavioral features
│   ├── ml/
│   │   ├── scam_classifier.py         # XGBoost classifier + baseline
│   │   ├── ab_testing.py              # ⭐ Full A/B testing framework
│   │   └── data_refinement.py         # ⭐ SMOTE + cleanlab pipeline
│   ├── monitoring/
│   │   └── model_monitor.py           # PSI drift detection + auto-retrain
│   └── api/
│       └── detection_api.py           # FastAPI endpoint
├── configs/
│   └── pipeline_config.yaml           # Centralized configuration
├── tests/
│   └── test_guardian_score.py         # Unit tests
├── notebooks/                          # EDA notebooks
├── run_demo.py                         # ⭐ One-click pipeline demo
├── docker-compose.yml                  # Full stack: Kafka+Spark+Redis+MLflow
├── Dockerfile
├── fraud_1000_dataset.csv              # 10K sample dataset
└── requirements.txt
```

---

## Quick Start

```bash
# Clone & setup
git clone https://github.com/yourname/archangel.git
cd archangel

# Activate conda environment (Python 3.11)
conda activate condaml

# ⭐ One-click full pipeline demo
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

## Alignment with ISL Data Research Engineer Role

| JD Requirement | This Project |
|---|---|
| Billion-scale data processing | Spark salting pipeline (scalable to 1B+) |
| Data Skew handling | Salting: 101.76x → 2.21x in `spark_etl.py` |
| A/B Testing & Effect Size | p=0.0003, Cohen's d, CI in `ab_testing.py` |
| Dataset Engineering | SMOTE + cleanlab in `data_refinement.py` |
| Model monitoring & retraining | PSI drift → auto-retrain in `model_monitor.py` |
| High-availability architecture | Docker Compose: Kafka + Spark + Redis + MLflow |
| Anti-fraud domain understanding | Guardian Score + Hit Rate optimization |
| Hit Rate 提升來電識別率 | Control 0.672 → Treatment 0.698 (+2.6pp) |
| 閉環監控 | PSI 0.163 → CRITICAL → auto-retrain triggered |
| 數據精煉 | SMOTE + cleanlab label correction pipeline |

---

*Built as a portfolio demonstration of Data-centric AI engineering principles.*
*All results are deterministic and reproducible with `python run_demo.py` (seed=42).*
