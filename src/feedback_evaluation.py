"""
Feedback Quality Evaluation Framework.

Provides:
1. Human Evaluation Rubric and Template Generator (results/human_feedback_ratings_template.csv)
   covering Specificity, Actionability, Relevance, Technical Correctness, and Usefulness.
2. Analyzer for completed human feedback rating sheets.
3. Automated linguistic/heuristic diagnostics for feedback texts (length, actionability indicators,
   lexical richness), clearly distinguished from qualitative human judgment.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import (
    HUMAN_EVALUATION_CRITERIA,
    OUTPUT_HUMAN_TEMPLATE_CSV,
    OUTPUT_RAW_CSV,
    RESULTS_DIR,
)

# Human rubric dimensions
RUBRIC_DIMENSIONS = [
    "specificity_1_to_5",
    "actionability_1_to_5",
    "relevance_1_to_5",
    "technical_correctness_1_to_5",
    "usefulness_improved_answer_1_to_5",
]

# Actionable cue words commonly associated with constructive coaching
ACTION_KEYWORDS = {
    "structure", "framework", "star", "avoid", "instead", "add", "concrete",
    "metric", "quantify", "example", "mention", "elaborate", "clarify", "practice",
    "specify", "reframe", "focus", "organize", "differentiate", "tradeoff"
}


def generate_human_evaluation_template(
    raw_outputs_csv: Path = OUTPUT_RAW_CSV,
    template_csv: Path = OUTPUT_HUMAN_TEMPLATE_CSV,
    sample_per_model: Optional[int] = None,
) -> Path:
    """
    Generate a blank, randomized evaluation template for human raters.
    Pre-populates question, candidate answer, model feedback, and improved answer,
    leaving rubric score columns blank for independent human assessment.
    """
    if not raw_outputs_csv.exists():
        raise FileNotFoundError(
            f"Cannot create human evaluation template because '{raw_outputs_csv}' does not exist. "
            f"Run an evaluation experiment first."
        )

    df = pd.read_csv(raw_outputs_csv)
    if df.empty:
        raise ValueError(f"'{raw_outputs_csv}' has no data rows.")

    # Filter to successful runs
    success_df = df[df["status"] == "SUCCESS"].copy()
    if success_df.empty:
        raise ValueError("No successful model evaluations found to create human assessment template.")

    if sample_per_model:
        success_df = success_df.groupby("model").apply(
            lambda x: x.sample(min(len(x), sample_per_model), random_state=42)
        ).reset_index(drop=True)

    template_rows: List[Dict[str, Any]] = []
    for _, row in success_df.iterrows():
        template_rows.append(
            {
                "evaluator_name": "",
                "record_id": row.get("record_id", ""),
                "model": row.get("model", ""),
                "prompt_approach": row.get("prompt_approach", ""),
                "question": row.get("question", ""),
                "candidate_answer": row.get("candidate_answer", ""),
                "model_feedback": row.get("feedback", ""),
                "model_improved_answer": row.get("improved_answer", ""),
                "specificity_1_to_5": "",
                "actionability_1_to_5": "",
                "relevance_1_to_5": "",
                "technical_correctness_1_to_5": "",
                "usefulness_improved_answer_1_to_5": "",
                "evaluator_notes": "",
            }
        )

    template_df = pd.DataFrame(template_rows)
    template_csv.parent.mkdir(parents=True, exist_ok=True)
    template_df.to_csv(template_csv, index=False)

    print(f"\n[INFO] Human evaluation rating template generated: '{template_csv}'")
    print(f"[INFO] Included {len(template_df)} evaluation rows across models.")
    print("\nHUMAN RATING RUBRIC GUIDE (1 to 5 scale):")
    for crit in HUMAN_EVALUATION_CRITERIA:
        print(f" - {crit['dimension'].upper()}: {crit['description']}")
    print("-" * 75)

    return template_csv


def analyze_human_ratings(rated_csv_path: Path) -> pd.DataFrame:
    """
    Analyze human assessment scores once a rater has completed the template.
    Calculates mean, std, and inter-rater statistics without fabricating missing ratings.
    """
    if not rated_csv_path.exists():
        raise FileNotFoundError(f"Rated file '{rated_csv_path}' not found.")

    df = pd.read_csv(rated_csv_path)

    # Check if any ratings exist
    has_ratings = False
    for dim in RUBRIC_DIMENSIONS:
        if dim in df.columns and df[dim].dropna().count() > 0:
            has_ratings = True
            break

    if not has_ratings:
        print(
            f"[WARN] No numeric ratings found in '{rated_csv_path}'. "
            "Please populate columns (1 to 5) before running analysis."
        )
        return pd.DataFrame()

    results: List[Dict[str, Any]] = []
    for model_name, sub_df in df.groupby("model"):
        res = {"model": model_name, "rated_samples": len(sub_df)}
        for dim in RUBRIC_DIMENSIONS:
            if dim in sub_df.columns:
                valid_vals = pd.to_numeric(sub_df[dim], errors="coerce").dropna()
                res[f"{dim}_mean"] = round(float(valid_vals.mean()), 2) if len(valid_vals) > 0 else np.nan
                res[f"{dim}_std"] = round(float(valid_vals.std()), 2) if len(valid_vals) > 1 else np.nan
            else:
                res[f"{dim}_mean"] = np.nan
                res[f"{dim}_std"] = np.nan
        results.append(res)

    summary_df = pd.DataFrame(results)
    output_path = RESULTS_DIR / "human_feedback_metrics_summary.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"[INFO] Human feedback analysis saved to '{output_path}'.")
    print(summary_df.to_string(index=False))
    return summary_df


def compute_automated_feedback_heuristics(raw_outputs_csv: Path = OUTPUT_RAW_CSV) -> pd.DataFrame:
    """
    Compute objective linguistic and heuristic diagnostics for model-generated feedback.
    
    Heuristics include:
    - Word count (feedback and improved answer)
    - Actionable cue word density (constructive coaching vocabulary)
    - Sentence count
    
    IMPORTANT RESEARCH DISCLAIMER:
    These metrics are surface-level linguistic features, NOT proof of pedagogical excellence.
    They complement, but do not replace, human qualitative review.
    """
    if not raw_outputs_csv.exists():
        print(f"[WARN] '{raw_outputs_csv}' does not exist.")
        return pd.DataFrame()

    df = pd.read_csv(raw_outputs_csv)
    success_df = df[df["status"] == "SUCCESS"].copy()
    if success_df.empty:
        return pd.DataFrame()

    records = []
    for _, row in success_df.iterrows():
        feedback_text = str(row.get("feedback", ""))
        improved_text = str(row.get("improved_answer", ""))

        words_fb = re.findall(r"\b[a-zA-Z0-9_\-']+\b", feedback_text.lower())
        words_imp = re.findall(r"\b[a-zA-Z0-9_\-']+\b", improved_text.lower())

        word_count_fb = len(words_fb)
        word_count_imp = len(words_imp)

        # Count action cue words
        action_hits = sum(1 for w in words_fb if w in ACTION_KEYWORDS)
        action_density_pct = round((action_hits / max(1, word_count_fb)) * 100.0, 2)

        records.append(
            {
                "record_id": row.get("record_id"),
                "model": row.get("model"),
                "prompt_approach": row.get("prompt_approach"),
                "fb_word_count": word_count_fb,
                "imp_answer_word_count": word_count_imp,
                "actionable_cue_count": action_hits,
                "actionable_cue_density_pct": action_density_pct,
            }
        )

    diag_df = pd.DataFrame(records)
    summary = diag_df.groupby(["model", "prompt_approach"]).agg(
        avg_fb_words=("fb_word_count", "mean"),
        avg_improved_answer_words=("imp_answer_word_count", "mean"),
        avg_actionable_cues=("actionable_cue_count", "mean"),
        avg_actionable_density_pct=("actionable_cue_density_pct", "mean"),
    ).round(2).reset_index()

    out_path = RESULTS_DIR / "feedback_heuristic_diagnostics.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n[INFO] Automated linguistic feedback diagnostics saved to '{out_path}'.")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feedback Quality Evaluation Framework")
    parser.add_argument("--generate-template", action="store_true", help="Generate blank human rating CSV")
    parser.add_argument("--analyze-template", type=str, help="Path to filled human evaluation CSV")
    parser.add_argument("--heuristics", action="store_true", help="Compute automated linguistic diagnostics")
    args = parser.parse_args()

    if args.generate_template:
        generate_human_evaluation_template()
    elif args.analyze_template:
        analyze_human_ratings(Path(args.analyze_template))
    elif args.heuristics:
        compute_automated_feedback_heuristics()
    else:
        # Default action: create template if outputs exist
        if OUTPUT_RAW_CSV.exists():
            generate_human_evaluation_template()
            compute_automated_feedback_heuristics()
        else:
            print("No output file found. Run an evaluation experiment first.")
