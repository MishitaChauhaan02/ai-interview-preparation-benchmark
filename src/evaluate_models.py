"""
Main LLM Evaluation & Experiment Orchestrator.

Implements:
1. Stage A: Pilot Experiment (5 records across configured models)
2. Stage B: Full Experiment (all valid dataset records with resume capability)
3. Prompting Strategy Experiment (Approach A: General vs Approach B: Rubric-guided)
4. Robust rate-limit exponential backoff, schema validation, and error logging
5. Automatic metric calculation, human template generation, and plotting
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APIError, AuthenticationError, NotFoundError, RateLimitError

from src.config import (
    BACKOFF_FACTOR,
    DEFAULT_MODELS,
    INITIAL_BACKOFF_SECONDS,
    MAX_RETRIES,
    OUTPUT_CONFIG_JSON,
    OUTPUT_ERRORS_CSV,
    OUTPUT_RAW_CSV,
    PILOT_SAMPLE_SIZE,
    PROJECT_ROOT,
    RANDOM_SEED,
    REQUEST_TIMEOUT_SECONDS,
    RESULTS_DIR,
    SYSTEM_PROMPT_GENERAL,
    SYSTEM_PROMPT_RUBRIC,
    TEMPERATURE,
)
from src.evaluation_metrics import calculate_and_save_metrics
from src.feedback_evaluation import compute_automated_feedback_heuristics, generate_human_evaluation_template
from src.plotting import generate_all_plots
from src.utils import (
    calculate_cost,
    clean_and_validate_scores,
    format_user_prompt,
    load_completed_evaluations,
    load_dataset,
    log_error,
    save_json,
    save_output_row,
)


def get_openai_client() -> OpenAI:
    """
    Initialize authenticated OpenAI client using environment variables.
    Fails with informative error guidance if API key is not configured.
    """
    # Load .env from project root
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or api_key.strip() == "" or "your_openai_api_key_here" in api_key:
        print("\n" + "!" * 80)
        print("CRITICAL CONFIGURATION ERROR: OPENAI_API_KEY is not configured.")
        print("Please follow these steps to configure your environment:")
        print("  1. Open the '.env' file in the project root.")
        print("  2. Set your key: OPENAI_API_KEY=sk-proj-...")
        print("  3. Never commit '.env' to git (it is listed in .gitignore).")
        print("!" * 80 + "\n")
        raise ValueError("Missing or invalid OPENAI_API_KEY in environment.")

    return OpenAI(api_key=api_key.strip(), timeout=REQUEST_TIMEOUT_SECONDS)


def evaluate_single_answer(
    client: OpenAI,
    model: str,
    prompt_approach: str,
    question: str,
    candidate_answer: str,
    record_id: int,
) -> Dict[str, Any]:
    """
    Evaluate a candidate's answer with deterministic LLM settings.
    Implements exponential backoff retries for transient errors.
    Strictly excludes reference answers/scores from prompt.
    """
    system_prompt = SYSTEM_PROMPT_RUBRIC if prompt_approach == "rubric" else SYSTEM_PROMPT_GENERAL
    user_prompt = format_user_prompt(question, candidate_answer)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    retries = 0
    backoff = INITIAL_BACKOFF_SECONDS

    while retries <= MAX_RETRIES:
        try:
            start_time = time.perf_counter()

            # Call OpenAI Chat Completions API with JSON mode & deterministic parameters
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                seed=RANDOM_SEED,
                response_format={"type": "json_object"},
            )

            latency = round(time.perf_counter() - start_time, 4)

            # Extract raw response text
            choice = response.choices[0]
            raw_content = choice.message.content or ""

            # Extract token usage
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0
            estimated_cost = calculate_cost(model, prompt_tokens, completion_tokens)

            # Parse and validate JSON structure
            try:
                parsed_json = json.loads(raw_content)
            except json.JSONDecodeError as jde:
                raise ValueError(f"Malformed JSON from model: {jde}. Raw snippet: {raw_content[:150]}")

            # Validate scores
            raw_scores = parsed_json.get("scores", {})
            validated_scores = clean_and_validate_scores(raw_scores)

            feedback_text = str(parsed_json.get("feedback", "")).strip()
            improved_answer_text = str(parsed_json.get("improved_answer", "")).strip()

            if not feedback_text:
                raise ValueError("Model response missing 'feedback' text.")
            if not improved_answer_text:
                raise ValueError("Model response missing 'improved_answer' text.")

            return {
                "status": "SUCCESS",
                "latency_sec": latency,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": estimated_cost,
                "scores": validated_scores,
                "feedback": feedback_text,
                "improved_answer": improved_answer_text,
            }

        except NotFoundError as nfe:
            # Model is unavailable or invalid ID; do NOT retry
            err_msg = f"Model '{model}' is not accessible on this OpenAI account or does not exist ({nfe})."
            log_error(record_id, model, prompt_approach, "NotFoundError", err_msg)
            return {"status": "FAILED", "error": err_msg}

        except AuthenticationError as ae:
            err_msg = f"Authentication failure with OpenAI API: {ae}"
            log_error(record_id, model, prompt_approach, "AuthenticationError", err_msg)
            raise ae

        except (RateLimitError, APIConnectionError) as retryable_err:
            retries += 1
            if retries > MAX_RETRIES:
                err_msg = f"Exceeded max retries ({MAX_RETRIES}) on {type(retryable_err).__name__}: {retryable_err}"
                log_error(record_id, model, prompt_approach, type(retryable_err).__name__, err_msg)
                return {"status": "FAILED", "error": err_msg}

            print(
                f"[RETRY {retries}/{MAX_RETRIES}] {type(retryable_err).__name__} for model '{model}' "
                f"on record {record_id}. Sleeping {backoff:.1f}s before retry..."
            )
            time.sleep(backoff)
            backoff *= BACKOFF_FACTOR

        except Exception as generic_err:
            # Schema validation or other non-retryable error
            err_msg = f"{type(generic_err).__name__}: {generic_err}"
            log_error(record_id, model, prompt_approach, type(generic_err).__name__, err_msg)
            return {"status": "FAILED", "error": err_msg}

    return {"status": "FAILED", "error": "Unknown error in retry loop"}


def run_experiment(
    stage_name: str,
    records: List[Dict[str, Any]],
    models: List[str],
    prompt_approaches: List[str],
    force_rerun: bool = False,
) -> Tuple[int, int]:
    """
    Execute evaluations over the designated records, models, and prompt approaches.
    Saves outputs immediately to results/model_outputs.csv and skips cached runs.
    """
    client = get_openai_client()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    completed_cache = set() if force_rerun else load_completed_evaluations(OUTPUT_RAW_CSV)
    if completed_cache:
        print(f"[CACHE] Found {len(completed_cache)} previously completed evaluations in '{OUTPUT_RAW_CSV.name}'.")

    total_combinations = len(records) * len(models) * len(prompt_approaches)
    print("\n" + "=" * 80)
    print(f"STARTING EXPERIMENT: {stage_name.upper()}")
    print(f"Records to evaluate: {len(records)}")
    print(f"Configured models:   {models}")
    print(f"Prompt approaches:   {prompt_approaches}")
    print(f"Total evaluations:   {total_combinations}")
    print("=" * 80 + "\n")

    eval_count = 0
    success_count = 0
    failure_count = 0
    skipped_count = 0

    for rec in records:
        rec_id = rec["id"]
        question = rec["question"]
        candidate_answer = rec["candidate_answer"]
        ref_scores = rec["ref_scores"]

        for model in models:
            for approach in prompt_approaches:
                eval_count += 1
                cache_key = (rec_id, model, approach)

                if cache_key in completed_cache:
                    skipped_count += 1
                    continue

                print(
                    f"[{eval_count}/{total_combinations}] Evaluating Record #{rec_id} "
                    f"| Model: {model} | Approach: {approach} ... ",
                    end="",
                    flush=True,
                )

                result = evaluate_single_answer(
                    client=client,
                    model=model,
                    prompt_approach=approach,
                    question=question,
                    candidate_answer=candidate_answer,
                    record_id=rec_id,
                )

                timestamp_str = datetime.now(timezone.utc).isoformat()

                if result["status"] == "SUCCESS":
                    success_count += 1
                    scores = result["scores"]
                    print(
                        f"SUCCESS (Lat: {result['latency_sec']}s, Cost: ${result['estimated_cost_usd']}) "
                        f"Scores: C={scores['clarity']} R={scores['relevance']} "
                        f"S={scores['structure']} Cf={scores['confidence']}"
                    )

                    out_row = {
                        "record_id": rec_id,
                        "model": model,
                        "prompt_approach": approach,
                        "status": "SUCCESS",
                        "latency_sec": result["latency_sec"],
                        "prompt_tokens": result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "total_tokens": result["total_tokens"],
                        "estimated_cost_usd": result["estimated_cost_usd"],
                        "score_clarity": scores["clarity"],
                        "score_relevance": scores["relevance"],
                        "score_structure": scores["structure"],
                        "score_confidence": scores["confidence"],
                        "feedback": result["feedback"],
                        "improved_answer": result["improved_answer"],
                        "ref_clarity": ref_scores.get("clarity", ""),
                        "ref_relevance": ref_scores.get("relevance", ""),
                        "ref_structure": ref_scores.get("structure", ""),
                        "ref_confidence": ref_scores.get("confidence", ""),
                        "role": rec.get("role", ""),
                        "category": rec.get("category", ""),
                        "difficulty": rec.get("difficulty", ""),
                        "question": question,
                        "candidate_answer": candidate_answer,
                        "timestamp": timestamp_str,
                    }
                    save_output_row(out_row, OUTPUT_RAW_CSV)
                else:
                    failure_count += 1
                    print(f"FAILED ({result.get('error')})")
                    failed_row = {
                        "record_id": rec_id,
                        "model": model,
                        "prompt_approach": approach,
                        "status": "FAILED",
                        "latency_sec": "",
                        "prompt_tokens": "",
                        "completion_tokens": "",
                        "total_tokens": "",
                        "estimated_cost_usd": "",
                        "score_clarity": "",
                        "score_relevance": "",
                        "score_structure": "",
                        "score_confidence": "",
                        "feedback": "",
                        "improved_answer": "",
                        "ref_clarity": ref_scores.get("clarity", ""),
                        "ref_relevance": ref_scores.get("relevance", ""),
                        "ref_structure": ref_scores.get("structure", ""),
                        "ref_confidence": ref_scores.get("confidence", ""),
                        "role": rec.get("role", ""),
                        "category": rec.get("category", ""),
                        "difficulty": rec.get("difficulty", ""),
                        "question": question,
                        "candidate_answer": candidate_answer,
                        "timestamp": timestamp_str,
                    }
                    save_output_row(failed_row, OUTPUT_RAW_CSV)

    print(
        f"\n[SUMMARY] Completed {stage_name}: "
        f"{success_count} successful, {failure_count} failed, {skipped_count} skipped (cached)."
    )

    return success_count, failure_count


def test_model_connectivity(models: List[str]) -> Dict[str, bool]:
    """Pre-flight connectivity check to verify model access and prevent mid-run failures."""
    print("\n--- Verifying OpenAI Model Connectivity ---")
    client = get_openai_client()
    status_map = {}
    for m in models:
        try:
            resp = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=2,
                temperature=0.0,
            )
            print(f"  [✓] Model '{m}' is accessible and operational.")
            status_map[m] = True
        except Exception as e:
            print(f"  [✗] Model '{m}' failed connectivity test: {type(e).__name__} ({e})")
            status_map[m] = False
    print("-------------------------------------------\n")
    return status_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Technical Interview Evaluation Benchmarking Experiment Runner"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pilot",
        action="store_true",
        help=f"Run Stage A Pilot Experiment on the first {PILOT_SAMPLE_SIZE} records.",
    )
    group.add_argument(
        "--full",
        action="store_true",
        help="Run Stage B Full Experiment on all records in the dataset.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Custom number of dataset records to evaluate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"List of OpenAI model IDs to evaluate (default: {DEFAULT_MODELS}).",
    )
    parser.add_argument(
        "--prompt-approach",
        choices=["rubric", "general", "both"],
        default="rubric",
        help="Prompting strategy: 'rubric' (Approach B), 'general' (Approach A), or 'both' for comparative experiment.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore cached evaluations and force re-running all API calls.",
    )
    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Run connectivity check on configured models and exit.",
    )

    args = parser.parse_args()

    # If connection test requested
    if args.test_connection:
        test_model_connectivity(args.models)
        return

    # Ingest dataset
    records, dataset_meta = load_dataset()
    print(f"[DATASET] Loaded {len(records)} valid records from '{dataset_meta['loaded_from']}'.")

    # Determine subset of records
    if args.pilot:
        stage_name = "Stage A — Pilot Experiment"
        selected_records = records[:PILOT_SAMPLE_SIZE]
    elif args.full:
        stage_name = "Stage B — Full Experiment"
        selected_records = records
    elif args.sample_size:
        stage_name = f"Custom Sample ({args.sample_size} records)"
        selected_records = records[: args.sample_size]
    else:
        # Default to pilot if none specified for safety
        stage_name = f"Stage A — Pilot Experiment (Default, {PILOT_SAMPLE_SIZE} records)"
        selected_records = records[:PILOT_SAMPLE_SIZE]

    # Determine prompt approaches
    if args.prompt_approach == "both":
        prompt_approaches = ["rubric", "general"]
    else:
        prompt_approaches = [args.prompt_approach]

    # Save experiment configuration artifact
    experiment_config = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage_name,
        "sample_size": len(selected_records),
        "models": args.models,
        "prompt_approaches": prompt_approaches,
        "temperature": TEMPERATURE,
        "seed": RANDOM_SEED,
        "dataset_metadata": dataset_meta,
        "deterministic_disclaimer": (
            "Deterministic settings (temperature=0.0, seed=42) minimize output variance "
            "but do not guarantee bit-level identity across OpenAI server clusters."
        ),
    }
    save_json(experiment_config, OUTPUT_CONFIG_JSON)

    # Execute experiment
    run_experiment(
        stage_name=stage_name,
        records=selected_records,
        models=args.models,
        prompt_approaches=prompt_approaches,
        force_rerun=args.force_rerun,
    )

    # Quantitative Evaluation Metrics Generation
    metrics_df = calculate_and_save_metrics()

    # Generate Human Rating Template and Feedback Linguistic Diagnostics
    try:
        generate_human_evaluation_template()
        compute_automated_feedback_heuristics()
    except Exception as e:
        print(f"[WARN] Human template generation notice: {e}")

    # Generate Visualizations
    try:
        generate_all_plots()
    except Exception as e:
        print(f"[WARN] Error during chart generation: {e}")

    print("\n[COMPLETE] Experiment workflow finished.")
    print(f"Results and artifacts saved in: '{RESULTS_DIR}/'")


if __name__ == "__main__":
    main()
