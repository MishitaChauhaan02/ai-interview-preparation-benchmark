"""
Visualization Engine for LLM Interview Evaluation Benchmarks.

Generates research paper-ready figures using Matplotlib and Seaborn:
1. comparison_mae.png: Overall MAE comparison across models
2. dimension_mae.png: MAE comparison broken down by scoring dimension
3. latency_comparison.png: Response latency distributions (median & p95)
4. agreement_comparison.png: Exact vs within-one-point agreement percentages

STRICT RESEARCH INTEGRITY RULE:
Plots are generated strictly from measured experimental data.
If data is missing or incomplete, a descriptive advisory is emitted,
and no fabricated or synthetic placeholder data is ever plotted.
"""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    CHART_AGREEMENT,
    CHART_COMPARISON_MAE,
    CHART_DIMENSION_MAE,
    CHART_LATENCY,
    CHART_PROMPT_COMPARISON,
    OUTPUT_METRICS_CSV,
    OUTPUT_RAW_CSV,
    RESULTS_DIR,
    SCORING_DIMENSIONS,
)

# Apply consistent, modern publication styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update(
    {
        "figure.autolayout": True,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
    }
)

FOOTNOTE_PROVISIONAL = (
    "* Reference baseline scores are provisional benchmarks. Results reflect relative alignment."
)


def plot_overall_mae(metrics_df: pd.DataFrame, output_path: Path = CHART_COMPARISON_MAE) -> None:
    """Generate bar chart comparing overall MAE across models and prompt approaches."""
    if metrics_df.empty or "overall_mae" not in metrics_df.columns:
        print("[WARN] Skipping overall MAE chart: 'overall_mae' data unavailable.")
        return

    valid_df = metrics_df.dropna(subset=["overall_mae"]).copy()
    if valid_df.empty:
        print("[WARN] No valid overall MAE values to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    has_approaches = "prompt_approach" in valid_df.columns and valid_df["prompt_approach"].nunique() > 1
    hue_col = "prompt_approach" if has_approaches else None

    barplot = sns.barplot(
        data=valid_df,
        x="model",
        y="overall_mae",
        hue="prompt_approach" if has_approaches else "model",
        palette="crest" if has_approaches else "mako",
        legend=True if has_approaches else False,
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_title("Overall Mean Absolute Error (MAE) Across Evaluator Models", pad=12)
    ax.set_xlabel("Evaluator Model")
    ax.set_ylabel("Overall MAE (Lower is Closer to Reference)")
    ax.set_ylim(0, max(2.5, valid_df["overall_mae"].max() * 1.25))

    # Annotate numeric values on bars
    for p in barplot.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                xytext=(0, 4),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )

    fig.text(0.5, -0.02, FOOTNOTE_PROVISIONAL, ha="center", fontsize=8, color="#555555", style="italic")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved overall MAE comparison chart to '{output_path}'.")


def plot_dimension_mae(metrics_df: pd.DataFrame, output_path: Path = CHART_DIMENSION_MAE) -> None:
    """Generate grouped bar chart comparing MAE across Clarity, Relevance, Structure, and Confidence."""
    dim_cols = [f"{d}_mae" for d in SCORING_DIMENSIONS]
    available_dims = [c for c in dim_cols if c in metrics_df.columns]

    if not available_dims or metrics_df.empty:
        print("[WARN] Skipping dimension MAE chart: dimensional data unavailable.")
        return

    # Reshape dataframe to long format for Seaborn
    id_vars = ["model"]
    if "prompt_approach" in metrics_df.columns:
        id_vars.append("prompt_approach")

    melted = pd.melt(
        metrics_df,
        id_vars=id_vars,
        value_vars=available_dims,
        var_name="dimension_raw",
        value_name="mae",
    ).dropna(subset=["mae"])

    melted["Dimension"] = melted["dimension_raw"].apply(lambda x: x.replace("_mae", "").capitalize())

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Group by Dimension on X, Model as hue
    barplot = sns.barplot(
        data=melted,
        x="Dimension",
        y="mae",
        hue="model",
        palette="viridis",
        ax=ax,
        edgecolor="black",
        linewidth=0.7,
    )

    ax.set_title("Mean Absolute Error by Evaluation Dimension", pad=12)
    ax.set_xlabel("Evaluation Dimension")
    ax.set_ylabel("MAE vs Provisional Reference Score")
    ax.set_ylim(0, max(2.5, melted["mae"].max() * 1.25))
    ax.legend(title="Model", frameon=True)

    for p in barplot.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=8,
                rotation=0,
            )

    fig.text(0.5, -0.02, FOOTNOTE_PROVISIONAL, ha="center", fontsize=8, color="#555555", style="italic")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved dimensional MAE chart to '{output_path}'.")


def plot_latency(raw_df: pd.DataFrame, output_path: Path = CHART_LATENCY) -> None:
    """Generate latency comparison boxplot and strip plot showing response time distribution."""
    if raw_df.empty or "latency_sec" not in raw_df.columns:
        print("[WARN] Skipping latency chart: 'latency_sec' data unavailable.")
        return

    valid_df = raw_df[(raw_df["status"] == "SUCCESS") & raw_df["latency_sec"].notna()].copy()
    if valid_df.empty:
        print("[WARN] No valid latency observations available to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    has_approaches = "prompt_approach" in valid_df.columns and valid_df["prompt_approach"].nunique() > 1
    hue_col = "prompt_approach" if has_approaches else None

    sns.boxplot(
        data=valid_df,
        x="model",
        y="latency_sec",
        hue="prompt_approach" if has_approaches else "model",
        palette="Set2",
        legend=True if has_approaches else False,
        ax=ax,
        boxprops=dict(alpha=0.7),
        showmeans=True,
        meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white"),
    )

    sns.stripplot(
        data=valid_df,
        x="model",
        y="latency_sec",
        hue=hue_col,
        dodge=True if hue_col else False,
        jitter=0.2,
        size=5,
        color=".25",
        alpha=0.6,
        ax=ax,
    )

    # Clean up duplicated legend from stripplot
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        n_unique = valid_df["prompt_approach"].nunique() if hue_col else 1
        ax.legend(handles[:n_unique], labels[:n_unique], title="Approach" if hue_col else None, frameon=True)

    ax.set_title("API Response Latency Distribution by Model (Seconds)", pad=12)
    ax.set_xlabel("Model")
    ax.set_ylabel("Latency (Seconds)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved latency comparison chart to '{output_path}'.")


def plot_agreement_rates(metrics_df: pd.DataFrame, output_path: Path = CHART_AGREEMENT) -> None:
    """Generate side-by-side comparison of Exact Agreement % vs Within-One-Point Agreement %."""
    if metrics_df.empty or "overall_exact_pct" not in metrics_df.columns:
        print("[WARN] Skipping agreement chart: agreement metrics unavailable.")
        return

    valid_df = metrics_df.dropna(subset=["overall_exact_pct", "overall_within_one_pct"]).copy()
    if valid_df.empty:
        print("[WARN] No valid agreement values to plot.")
        return

    # Melt to long format
    id_vars = ["model"]
    if "prompt_approach" in valid_df.columns:
        id_vars.append("prompt_approach")

    melted = pd.melt(
        valid_df,
        id_vars=id_vars,
        value_vars=["overall_exact_pct", "overall_within_one_pct"],
        var_name="metric_type",
        value_name="percentage",
    )

    metric_labels = {
        "overall_exact_pct": "Exact Agreement (%)",
        "overall_within_one_pct": "Within ±1 Point Agreement (%)",
    }
    melted["Agreement Metric"] = melted["metric_type"].map(metric_labels)

    fig, ax = plt.subplots(figsize=(9, 5))

    barplot = sns.barplot(
        data=melted,
        x="model",
        y="percentage",
        hue="Agreement Metric",
        palette="Blues_r",
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_title("Scoring Agreement Rates Across Models", pad=12)
    ax.set_xlabel("Evaluator Model")
    ax.set_ylabel("Agreement Percentage (%)")
    ax.set_ylim(0, 105)
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.legend(title="Metric", frameon=True)

    for p in barplot.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.1f}%",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="bold",
            )

    fig.text(0.5, -0.02, FOOTNOTE_PROVISIONAL, ha="center", fontsize=8, color="#555555", style="italic")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved agreement comparison chart to '{output_path}'.")


def generate_all_plots(
    raw_csv: Path = OUTPUT_RAW_CSV,
    metrics_csv: Path = OUTPUT_METRICS_CSV,
) -> None:
    """
    Orchestrate generation of all research figures.
    Guarantees integrity: only plots genuine, un-fabricated data.
    """
    print("\n--- Generating Research Visualizations ---")
    if not metrics_csv.exists() or not raw_csv.exists():
        print(
            f"[WARN] Cannot generate plots because results files are missing:\n"
            f" - Raw data: {raw_csv} (exists: {raw_csv.exists()})\n"
            f" - Metrics: {metrics_csv} (exists: {metrics_csv.exists()})\n"
            f"Run an evaluation stage first to produce valid experimental data."
        )
        return

    metrics_df = pd.read_csv(metrics_csv)
    raw_df = pd.read_csv(raw_csv)

    if metrics_df.empty or raw_df.empty:
        print("[WARN] Result files are empty. No visualizations generated.")
        return

    plot_overall_mae(metrics_df, CHART_COMPARISON_MAE)
    plot_dimension_mae(metrics_df, CHART_DIMENSION_MAE)
    plot_latency(raw_df, CHART_LATENCY)
    plot_agreement_rates(metrics_df, CHART_AGREEMENT)
    print("--- Research Visualizations Complete ---\n")


if __name__ == "__main__":
    generate_all_plots()
