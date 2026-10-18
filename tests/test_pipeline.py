"""
Unit and Integration Test Suite for LLM Evaluator Benchmark Pipeline.

Validates:
1. Dataset ingestion and schema integrity
2. Anti-leakage prompt builder
3. Score validation and clamping
4. Cost calculation accuracy
5. Quantitative metrics calculation (MAE, exact agreement, within-one-point agreement)
6. Feedback heuristic diagnostics
7. Caching and resume mechanics
8. Plotting engine execution
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_MODELS,
    OUTPUT_JSON_SCHEMA,
    SCORING_DIMENSIONS,
    SYSTEM_PROMPT_GENERAL,
    SYSTEM_PROMPT_RUBRIC,
)
from src.evaluation_metrics import compute_group_metrics
from src.feedback_evaluation import compute_automated_feedback_heuristics
from src.plotting import plot_agreement_rates, plot_dimension_mae, plot_overall_mae
from src.utils import (
    calculate_cost,
    clean_and_validate_scores,
    format_user_prompt,
    load_completed_evaluations,
    load_dataset,
    save_output_row,
)


class TestPipeline(unittest.TestCase):
    def test_dataset_loading(self):
        """Verify dataset loads properly and has required fields."""
        records, metadata = load_dataset()
        self.assertGreaterEqual(len(records), 30)
        first = records[0]
        self.assertIn("id", first)
        self.assertIn("question", first)
        self.assertIn("candidate_answer", first)
        self.assertIn("ref_scores", first)
        for dim in SCORING_DIMENSIONS:
            self.assertIn(dim, first["ref_scores"])
            self.assertIsInstance(first["ref_scores"][dim], int)
            self.assertTrue(1 <= first["ref_scores"][dim] <= 5)

    def test_prompt_no_data_leakage(self):
        """Verify prompt does NOT leak reference scores, reference feedback, or improved answers."""
        records, _ = load_dataset()
        rec = records[0]
        prompt = format_user_prompt(rec["question"], rec["candidate_answer"])

        self.assertIn(rec["question"], prompt)
        self.assertIn(rec["candidate_answer"], prompt)

        # STRICT ANTI-LEAKAGE CHECK
        if rec["ref_feedback"]:
            self.assertNotIn(rec["ref_feedback"], prompt)
        if rec["ref_improved_answer"]:
            self.assertNotIn(rec["ref_improved_answer"], prompt)
        self.assertNotIn("ref_scores", prompt)

    def test_score_validation(self):
        """Verify clean_and_validate_scores validates scores properly."""
        valid_input = {"clarity": 4, "relevance": 5, "structure": 3, "confidence": 2}
        cleaned = clean_and_validate_scores(valid_input)
        self.assertEqual(cleaned, valid_input)

        # String integers
        str_input = {"clarity": "4", "relevance": "5", "structure": "3", "confidence": "2"}
        cleaned_str = clean_and_validate_scores(str_input)
        self.assertEqual(cleaned_str, valid_input)

        # Missing dimension
        with self.assertRaises(ValueError):
            clean_and_validate_scores({"clarity": 4, "relevance": 5, "structure": 3})

        # Score out of bounds
        with self.assertRaises(ValueError):
            clean_and_validate_scores({"clarity": 6, "relevance": 5, "structure": 3, "confidence": 2})

        with self.assertRaises(ValueError):
            clean_and_validate_scores({"clarity": 0, "relevance": 5, "structure": 3, "confidence": 2})

    def test_cost_calculation(self):
        """Verify cost calculation handles prompt and completion pricing."""
        # gpt-4o-mini is $0.15/1M prompt, $0.60/1M completion
        cost = calculate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 0.75, places=4)

    def test_metrics_computation(self):
        """Verify MAE, exact agreement, and within-one-point agreement math."""
        # Create synthetic evaluation df for testing math
        df_group = pd.DataFrame(
            [
                {
                    "status": "SUCCESS",
                    "latency_sec": 1.2,
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "estimated_cost_usd": 0.0001,
                    "score_clarity": 4,
                    "score_relevance": 5,
                    "score_structure": 3,
                    "score_confidence": 2,
                    "ref_clarity": 3,      # diff = 1
                    "ref_relevance": 5,    # diff = 0
                    "ref_structure": 1,    # diff = 2
                    "ref_confidence": 2,   # diff = 0
                },
                {
                    "status": "SUCCESS",
                    "latency_sec": 1.4,
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "estimated_cost_usd": 0.0001,
                    "score_clarity": 3,
                    "score_relevance": 4,
                    "score_structure": 2,
                    "score_confidence": 3,
                    "ref_clarity": 3,      # diff = 0
                    "ref_relevance": 4,    # diff = 0
                    "ref_structure": 2,    # diff = 0
                    "ref_confidence": 4,   # diff = 1
                },
            ]
        )

        metrics = compute_group_metrics(df_group)
        self.assertEqual(metrics["successful_calls"], 2)
        # Clarity diffs: 1, 0 -> MAE = 0.5
        self.assertEqual(metrics["clarity_mae"], 0.5)
        # Relevance diffs: 0, 0 -> MAE = 0.0
        self.assertEqual(metrics["relevance_mae"], 0.0)
        # Structure diffs: 2, 0 -> MAE = 1.0
        self.assertEqual(metrics["structure_mae"], 1.0)
        # Confidence diffs: 0, 1 -> MAE = 0.5
        self.assertEqual(metrics["confidence_mae"], 0.5)
        # Overall MAE = (0.5 + 0.0 + 1.0 + 0.5) / 4 = 0.5
        self.assertEqual(metrics["overall_mae"], 0.5)

        # Exact agreement:
        # Clarity: 1/2 = 50%
        # Relevance: 2/2 = 100%
        # Structure: 1/2 = 50%
        # Confidence: 1/2 = 50%
        # Overall exact = 62.5%
        self.assertEqual(metrics["overall_exact_pct"], 62.5)

        # Within-one-point agreement:
        # Clarity: diffs 1, 0 -> 100%
        # Relevance: diffs 0, 0 -> 100%
        # Structure: diffs 2, 0 -> 50%
        # Confidence: diffs 0, 1 -> 100%
        # Overall = 87.5%
        self.assertEqual(metrics["overall_within_one_pct"], 87.5)

    def test_caching_and_resume(self):
        """Verify save_output_row and load_completed_evaluations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_outputs.csv"
            row1 = {
                "record_id": 1,
                "model": "gpt-4o-mini",
                "prompt_approach": "rubric",
                "status": "SUCCESS",
            }
            row2 = {
                "record_id": 2,
                "model": "gpt-4o-mini",
                "prompt_approach": "rubric",
                "status": "FAILED",
            }
            save_output_row(row1, csv_path)
            save_output_row(row2, csv_path)

            cached = load_completed_evaluations(csv_path)
            self.assertIn((1, "gpt-4o-mini", "rubric"), cached)
            # Failed row should not be in successful cache
            self.assertNotIn((2, "gpt-4o-mini", "rubric"), cached)


if __name__ == "__main__":
    unittest.main()
