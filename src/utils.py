"""
Utility functions for dataset ingestion, data leakage prevention, cost estimation,
caching/resume mechanics, and safe error logging.
"""

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.config import (
    DEFAULT_DATASET_PATH,
    FALLBACK_DATASET_PATH,
    MODEL_PRICING_PER_1M,
    OUTPUT_ERRORS_CSV,
    OUTPUT_RAW_CSV,
    SCORE_MAX,
    SCORE_MIN,
    SCORING_DIMENSIONS,
    USER_EVALUATION_PROMPT_TEMPLATE,
)


def load_dataset(dataset_path: Optional[Path] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Safely load and validate the interview dataset.

    Inspects default path ('data/interview_dataset.json') and falls back
    to 'data/interview_training_data_all.json' if needed. Handles both
    top-level lists and wrapped objects with a 'records' key.

    Returns:
        Tuple of (validated_records_list, dataset_metadata_dict)
    """
    target_path = dataset_path or DEFAULT_DATASET_PATH

    if not target_path.exists():
        if FALLBACK_DATASET_PATH.exists():
            print(
                f"[INFO] '{target_path.name}' was not found. "
                f"Automatically loading fallback dataset from '{FALLBACK_DATASET_PATH.name}'."
            )
            target_path = FALLBACK_DATASET_PATH
        else:
            raise FileNotFoundError(
                f"Dataset not found at '{target_path}'. Please ensure 'data/interview_dataset.json' "
                f"or 'data/interview_training_data_all.json' exists in the workspace."
            )

    with open(target_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    metadata: Dict[str, Any] = {
        "loaded_from": str(target_path),
        "dataset_name": "Unknown",
        "version": "Unknown",
        "description": "",
    }

    if isinstance(raw_data, dict):
        metadata["dataset_name"] = raw_data.get("dataset_name", "AI Interview Dataset")
        metadata["version"] = raw_data.get("version", "1.0")
        metadata["description"] = raw_data.get("description", "")
        metadata["score_scale"] = raw_data.get("score_scale", "1-5")
        records_raw = raw_data.get("records", [])
    elif isinstance(raw_data, list):
        records_raw = raw_data
    else:
        raise ValueError(f"Unrecognized JSON structure in {target_path}. Expected object or array.")

    validated_records: List[Dict[str, Any]] = []
    for idx, item in enumerate(records_raw):
        # Basic validation of mandatory fields
        record_id = item.get("id", idx + 1)
        question = item.get("question", "").strip()
        candidate_answer = item.get("candidate_answer", "").strip()

        if not question or not candidate_answer:
            print(f"[WARN] Skipping record index {idx} (ID: {record_id}): empty question or answer.")
            continue

        # Extract reference scores if present (labeled strictly as provisional baseline)
        ref_scores_raw = item.get("scores", {})
        ref_scores: Dict[str, Optional[int]] = {}
        for dim in SCORING_DIMENSIONS:
            val = ref_scores_raw.get(dim)
            if val is not None:
                try:
                    ival = int(round(float(val)))
                    ref_scores[dim] = max(SCORE_MIN, min(SCORE_MAX, ival))
                except (ValueError, TypeError):
                    ref_scores[dim] = None
            else:
                ref_scores[dim] = None

        validated_record = {
            "id": record_id,
            "role": item.get("role", "General"),
            "category": item.get("category", "General"),
            "difficulty": item.get("difficulty", "medium"),
            "question": question,
            "candidate_answer": candidate_answer,
            "answer_quality": item.get("answer_quality", "unknown"),
            "ref_scores": ref_scores,
            "ref_feedback": item.get("feedback", ""),
            "ref_improved_answer": item.get("improved_answer", ""),
        }
        validated_records.append(validated_record)

    metadata["total_records"] = len(validated_records)
    return validated_records, metadata


def format_user_prompt(question: str, candidate_answer: str) -> str:
    """
    Format the user prompt for LLM evaluation.

    CRITICAL SECURITY & RESEARCH INTEGRITY:
    Only the question and candidate answer are included. Reference scores,
    reference feedback, and reference improved answers are STRICTLY excluded
    to prevent prompt leakage.
    """
    return USER_EVALUATION_PROMPT_TEMPLATE.format(
        question=question.strip(),
        candidate_answer=candidate_answer.strip(),
    )


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate estimated API cost in USD based on model pricing per 1M tokens.
    """
    pricing = MODEL_PRICING_PER_1M.get(model, MODEL_PRICING_PER_1M["default"])
    cost_prompt = (prompt_tokens / 1_000_000.0) * pricing["prompt"]
    cost_completion = (completion_tokens / 1_000_000.0) * pricing["completion"]
    return round(cost_prompt + cost_completion, 6)


def clean_and_validate_scores(raw_scores: Any) -> Dict[str, int]:
    """
    Validate and parse scores dictionary from LLM JSON response.
    Ensures clarity, relevance, structure, and confidence are present
    and strictly between SCORE_MIN and SCORE_MAX.
    """
    if not isinstance(raw_scores, dict):
        raise ValueError(f"Scores must be a JSON dictionary, got {type(raw_scores).__name__}")

    cleaned: Dict[str, int] = {}
    missing_dims = []

    for dim in SCORING_DIMENSIONS:
        if dim not in raw_scores:
            missing_dims.append(dim)
            continue
        try:
            val = int(round(float(raw_scores[dim])))
            if val < SCORE_MIN or val > SCORE_MAX:
                raise ValueError(
                    f"Dimension '{dim}' score {val} is outside allowed range [{SCORE_MIN}, {SCORE_MAX}]."
                )
            cleaned[dim] = val
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid numeric score for '{dim}': {raw_scores[dim]} ({e})")

    if missing_dims:
        raise ValueError(f"Missing required scoring dimensions: {missing_dims}")

    return cleaned


def load_completed_evaluations(output_csv: Path = OUTPUT_RAW_CSV) -> Set[Tuple[int, str, str]]:
    """
    Load set of (record_id, model, prompt_approach) that have already succeeded.
    Enables resuming interrupted runs without repeating successful API calls.
    """
    completed: Set[Tuple[int, str, str]] = set()
    if not output_csv.exists():
        return completed

    try:
        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "SUCCESS":
                    try:
                        rec_id = int(row.get("record_id", -1))
                        model = str(row.get("model", "")).strip()
                        prompt_approach = str(row.get("prompt_approach", "")).strip()
                        if rec_id >= 0 and model:
                            completed.add((rec_id, model, prompt_approach))
                    except ValueError:
                        continue
    except Exception as e:
        print(f"[WARN] Error reading cache from '{output_csv}': {e}. Resuming from scratch.")

    return completed


def save_output_row(row_data: Dict[str, Any], output_csv: Path = OUTPUT_RAW_CSV) -> None:
    """
    Append an evaluation output row immediately to model_outputs.csv.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "model",
        "prompt_approach",
        "status",
        "latency_sec",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "score_clarity",
        "score_relevance",
        "score_structure",
        "score_confidence",
        "feedback",
        "improved_answer",
        "ref_clarity",
        "ref_relevance",
        "ref_structure",
        "ref_confidence",
        "role",
        "category",
        "difficulty",
        "question",
        "candidate_answer",
        "timestamp",
    ]

    file_exists = output_csv.exists() and output_csv.stat().st_size > 0
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row_data.get(k, "") for k in fieldnames})


def log_error(
    record_id: Optional[int],
    model: str,
    prompt_approach: str,
    error_type: str,
    error_message: str,
    output_csv: Path = OUTPUT_ERRORS_CSV,
) -> None:
    """
    Log an API or parsing error to errors.csv.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "record_id",
        "model",
        "prompt_approach",
        "error_type",
        "error_message",
    ]
    file_exists = output_csv.exists() and output_csv.stat().st_size > 0
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "record_id": record_id if record_id is not None else -1,
                "model": model,
                "prompt_approach": prompt_approach,
                "error_type": error_type,
                "error_message": str(error_message).replace("\n", " ").strip(),
            }
        )


def save_json(data: Any, filepath: Path) -> None:
    """Safely write JSON with formatting."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
