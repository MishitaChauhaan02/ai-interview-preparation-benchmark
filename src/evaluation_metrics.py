"""
Quantitative Evaluation Metrics Engine.

Calculates scoring alignment metrics against reference scores:
- Mean Absolute Error (MAE) per dimension & overall
- Exact agreement percentage (%)
- Within-one-point agreement percentage (%)
- Response latency (Median, 95th percentile)
- Token usage & estimated API cost ($ USD)
- Success rates and call counts

Includes rigorous research integrity warnings distinguishing provisional
reference scores from certified human ground truth.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.config import (
    OUTPUT_METRICS_CSV,
    OUTPUT_RAW_CSV,
    SCORING_DIMENSIONS,
)

# Research Integrity Advisory Header
RESEARCH_INTEGRITY_NOTICE = (
    "NOTE ON REFERENCE SCORES & PROVENANCE:\n"
    "Reference scores present in this dataset are treated as PROVISIONAL REFERENCE BENCHMARKS.\n"
    "Unless verified through multi-annotator inter-rater reliability studies (e.g., Cohen's Kappa),\n"
    "these scores must not be represented as gold-standard ground truth in academic literature."
)


def compute_group_metrics(df_group: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute full suite of evaluation and operational metrics for a specific model & approach.
    """
    metrics: Dict[str, Any] = {}

    total_records = len(df_group)
    successful_df = df_group[df_group["status"] == "SUCCESS"].copy()
    valid_count = len(successful_df)

    metrics["total_calls"] = total_records
    metrics["successful_calls"] = valid_count
    metrics["failed_calls"] = total_records - valid_count
    metrics["success_rate_pct"] = round((valid_count / total_records * 100.0) if total_records > 0 else 0.0, 2)

    if valid_count == 0:
        return metrics

    # Latency metrics
    latencies = successful_df["latency_sec"].dropna().to_numpy()
    if len(latencies) > 0:
        metrics["median_latency_sec"] = round(float(np.median(latencies)), 3)
        metrics["p95_latency_sec"] = round(float(np.percentile(latencies, 95)), 3) if len(latencies) >= 5 else round(float(np.max(latencies)), 3)
    else:
        metrics["median_latency_sec"] = np.nan
        metrics["p95_latency_sec"] = np.nan

    # Token and cost metrics
    metrics["mean_prompt_tokens"] = round(float(successful_df["prompt_tokens"].mean()), 1)
    metrics["mean_completion_tokens"] = round(float(successful_df["completion_tokens"].mean()), 1)
    metrics["mean_total_tokens"] = round(float(successful_df["total_tokens"].mean()), 1)
    metrics["total_estimated_cost_usd"] = round(float(successful_df["estimated_cost_usd"].sum()), 5)

    # Dimensional MAE and Agreement
    dim_maes: List[float] = []
    dim_exacts: List[float] = []
    dim_withins: List[float] = []

    for dim in SCORING_DIMENSIONS:
        model_col = f"score_{dim}"
        ref_col = f"ref_{dim}"

        if model_col in successful_df.columns and ref_col in successful_df.columns:
            # Filter rows where both model score and ref score are non-null
            valid_pairs = successful_df[[model_col, ref_col]].dropna()
            if len(valid_pairs) > 0:
                y_pred = valid_pairs[model_col].astype(float).to_numpy()
                y_ref = valid_pairs[ref_col].astype(float).to_numpy()

                # MAE
                mae = mean_absolute_error(y_ref, y_pred)
                metrics[f"{dim}_mae"] = round(float(mae), 3)
                dim_maes.append(mae)

                # Exact Agreement %
                exact_pct = np.mean(y_pred == y_ref) * 100.0
                metrics[f"{dim}_exact_pct"] = round(float(exact_pct), 2)
                dim_exacts.append(exact_pct)

                # Within-One-Point Agreement %
                within_one_pct = np.mean(np.abs(y_pred - y_ref) <= 1.0) * 100.0
                metrics[f"{dim}_within_one_pct"] = round(float(within_one_pct), 2)
                dim_withins.append(within_one_pct)
            else:
                metrics[f"{dim}_mae"] = np.nan
                metrics[f"{dim}_exact_pct"] = np.nan
                metrics[f"{dim}_within_one_pct"] = np.nan
        else:
            metrics[f"{dim}_mae"] = np.nan
            metrics[f"{dim}_exact_pct"] = np.nan
            metrics[f"{dim}_within_one_pct"] = np.nan

    # Overall scoring metrics across all 4 dimensions
    if dim_maes:
        metrics["overall_mae"] = round(float(np.mean(dim_maes)), 3)
        metrics["overall_exact_pct"] = round(float(np.mean(dim_exacts)), 2)
        metrics["overall_within_one_pct"] = round(float(np.mean(dim_withins)), 2)
    else:
        metrics["overall_mae"] = np.nan
        metrics["overall_exact_pct"] = np.nan
        metrics["overall_within_one_pct"] = np.nan

    return metrics


def calculate_and_save_metrics(
    raw_csv: Path = OUTPUT_RAW_CSV,
    metrics_csv: Path = OUTPUT_METRICS_CSV,
) -> pd.DataFrame:
    """
    Read model_outputs.csv, compute all comparative metrics, and write model_metrics.csv.
    """
    if not raw_csv.exists() or raw_csv.stat().st_size == 0:
        print(f"[WARN] No experimental output file found at '{raw_csv}'. Metrics cannot be computed.")
        return pd.DataFrame()

    df = pd.read_csv(raw_csv)
    if df.empty:
        print(f"[WARN] '{raw_csv}' contains no data rows.")
        return pd.DataFrame()

    # Determine grouping columns: model and prompt_approach (if present)
    group_cols = ["model"]
    if "prompt_approach" in df.columns:
        group_cols.append("prompt_approach")

    results_list: List[Dict[str, Any]] = []

    for group_keys, sub_df in df.groupby(group_cols):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        row_meta = dict(zip(group_cols, group_keys))
        computed = compute_group_metrics(sub_df)
        combined = {**row_meta, **computed}
        results_list.append(combined)

    metrics_df = pd.DataFrame(results_list)

    # Sort predictably: by model and prompt approach
    sort_cols = [c for c in ["model", "prompt_approach"] if c in metrics_df.columns]
    if sort_cols:
        metrics_df.sort_values(by=sort_cols, inplace=True)

    # Save to CSV
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\n[INFO] Successfully calculated and saved quantitative metrics to '{metrics_csv}'.")

    # Display clean research summary table in console
    display_summary_table(metrics_df)

    return metrics_df


def display_summary_table(metrics_df: pd.DataFrame) -> None:
    """Print an aligned, publication-grade console table summarizing the metrics."""
    if metrics_df.empty:
        return

    print("\n" + "=" * 90)
    print("                     LLM EVALUATOR BENCHMARK METRICS SUMMARY")
    print("=" * 90)
    print(RESEARCH_INTEGRITY_NOTICE)
    print("-" * 90)

    display_cols = [
        "model",
        "prompt_approach",
        "successful_calls",
        "overall_mae",
        "overall_exact_pct",
        "overall_within_one_pct",
        "median_latency_sec",
        "total_estimated_cost_usd",
    ]
    available_cols = [c for c in display_cols if c in metrics_df.columns]
    summary_view = metrics_df[available_cols].rename(
        columns={
            "model": "Model",
            "prompt_approach": "Approach",
            "successful_calls": "N (evals)",
            "overall_mae": "Overall MAE",
            "overall_exact_pct": "Exact %",
            "overall_within_one_pct": "±1 Pt %",
            "median_latency_sec": "Med Lat (s)",
            "total_estimated_cost_usd": "Cost ($)",
        }
    )
    print(summary_view.to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    calculate_and_save_metrics()
