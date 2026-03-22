#!/usr/bin/env python3
"""
run_demo.py — Archangel Intelligence System: One-Click Pipeline Demo
=====================================================================
Executes all core modules sequentially with deterministic output.
Designed for terminal presentation to ISL Data Research Engineer interview.

Usage:
    python run_demo.py          # Full demo (all 4 modules)
    python run_demo.py --quick  # Quick mode (skip data refinement)

Modules:
    1. Spark ETL — Data Skew Salting Technique
    2. A/B Testing — Statistical Rigor with Cohen's d
    3. Model Monitor — PSI Drift Detection & Auto-Retrain
    4. Guardian Score — Bayesian Reputation System
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


def print_banner():
    """Print the demo banner."""
    print("\n")
    print("═" * 68)
    print("  🛡️  ARCHANGEL INTELLIGENCE SYSTEM — Full Pipeline Demo")
    print("  ─────────────────────────────────────────────────────────")
    print("  Data-centric AI  |  Anti-Fraud  |  ISL Data Research Engineer")
    print("  Deterministic seed: 42  |  All results are reproducible")
    print("═" * 68)


def print_module_header(num: int, total: int, title: str, subtitle: str):
    """Print a module section header."""
    print(f"\n\n{'▓' * 68}")
    print(f"  ▶ Module {num}/{total}: {title}")
    print(f"    {subtitle}")
    print(f"{'▓' * 68}")


def print_summary(results: dict):
    """Print the final summary table with key results."""
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


def run_full_demo(quick: bool = False):
    """Execute the complete demo pipeline."""
    # Configure logging to stdout (instead of stderr) for consistent output order
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
    from src.ml.ab_testing import run_demo as ab_demo
    from src.monitoring.model_monitor import run_demo as monitor_demo
    from src.feature_engineering.guardian_score import run_demo as guardian_demo

    # Re-enable logging
    logging.disable(logging.NOTSET)

    print_banner()

    all_results = {}
    t_start = time.perf_counter()

    total_modules = 4

    # ═══════════════════════════════════════════════════════════════════════
    # Module 1: Spark ETL — Data Skew Salting
    # ═══════════════════════════════════════════════════════════════════════
    print_module_header(1, total_modules,
                        "Spark ETL — Data Skew Salting",
                        "Demonstrates salting technique to resolve partition skew")
    etl = AntiFraudETL()
    etl_results = etl.run(n_records=50_000)
    all_results["etl"] = etl_results

    # ═══════════════════════════════════════════════════════════════════════
    # Module 2: A/B Testing — Statistical Rigor
    # ═══════════════════════════════════════════════════════════════════════
    print_module_header(2, total_modules,
                        "A/B Testing — Statistical Rigor",
                        "Power analysis → z-test → Cohen's d → Business decision")

    ab_results = ab_demo()
    all_results["ab_testing"] = ab_results

    # ═══════════════════════════════════════════════════════════════════════
    # Module 3: Model Monitor — Drift Detection
    # ═══════════════════════════════════════════════════════════════════════
    print_module_header(3, total_modules,
                        "Model Monitor — PSI Drift Detection",
                        "30-day simulation with auto-retraining trigger")

    monitor_results = monitor_demo()
    all_results["model_monitor"] = monitor_results

    # ═══════════════════════════════════════════════════════════════════════
    # Module 4: Guardian Score — Bayesian Reputation
    # ═══════════════════════════════════════════════════════════════════════
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
    print(f"\n  💡 Next steps:")
    print(f"     • Jupyter Notebook:  jupyter notebook notebooks/")
    print(f"     • FastAPI Swagger:   uvicorn src.api.detection_api:app --reload")
    print(f"     • Docker full stack: docker-compose up -d")
    print(f"     • Run tests:        pytest tests/ -v")
    print()

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Archangel Intelligence System — Pipeline Demo"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: skip data refinement pipeline"
    )
    args = parser.parse_args()

    results = run_full_demo(quick=args.quick)
    return results


if __name__ == "__main__":
    main()
