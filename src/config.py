"""
Configuration module for the AI Interview Preparation LLM Benchmarking Study.

Centralizes model definitions, evaluation dimensions, pricing tiers, directory
paths, prompt templates, and experimental parameters.
"""

from pathlib import Path
from typing import Dict, List, Any

# ==============================================================================
# 1. DIRECTORIES AND FILE PATHS
# ==============================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# Primary and fallback dataset locations
DEFAULT_DATASET_PATH: Path = DATA_DIR / "interview_dataset.json"
FALLBACK_DATASET_PATH: Path = DATA_DIR / "interview_training_data_all.json"

# Output artifact paths within results/
OUTPUT_RAW_CSV: Path = RESULTS_DIR / "model_outputs.csv"
OUTPUT_METRICS_CSV: Path = RESULTS_DIR / "model_metrics.csv"
OUTPUT_ERRORS_CSV: Path = RESULTS_DIR / "errors.csv"
OUTPUT_CONFIG_JSON: Path = RESULTS_DIR / "experiment_config.json"
OUTPUT_HUMAN_TEMPLATE_CSV: Path = RESULTS_DIR / "human_feedback_ratings_template.csv"

# Visualization chart paths within results/
CHART_COMPARISON_MAE: Path = RESULTS_DIR / "comparison_mae.png"
CHART_DIMENSION_MAE: Path = RESULTS_DIR / "dimension_mae.png"
CHART_LATENCY: Path = RESULTS_DIR / "latency_comparison.png"
CHART_AGREEMENT: Path = RESULTS_DIR / "agreement_comparison.png"
CHART_PROMPT_COMPARISON: Path = RESULTS_DIR / "prompt_approach_comparison.png"

# ==============================================================================
# 2. MODEL CONFIGURATION & PRICING
# ==============================================================================
# Candidate models designated for the study
DEFAULT_MODELS: List[str] = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1",
]

# Estimated API pricing per 1,000,000 tokens (USD) as of standard OpenAI catalog
# Prompt (input) and Completion (output) token costs
MODEL_PRICING_PER_1M: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {
        "prompt": 0.15,
        "completion": 0.60,
    },
    "gpt-4.1-mini": {
        "prompt": 0.40,
        "completion": 1.60,
    },
    "gpt-4.1": {
        "prompt": 2.00,
        "completion": 8.00,
    },
    "gpt-4o": {
        "prompt": 2.50,
        "completion": 10.00,
    },
    "default": {
        "prompt": 0.50,
        "completion": 1.50,
    },
}

# ==============================================================================
# 3. EXPERIMENTAL PARAMETERS
# ==============================================================================
# Deterministic generation settings
# Note: While temperature=0.0 and seed=42 significantly constrain variance, OpenAI
# does not guarantee 100% bit-level reproducibility due to distributed hardware execution.
TEMPERATURE: float = 0.0
RANDOM_SEED: int = 42

# Sample sizes
PILOT_SAMPLE_SIZE: int = 5

# Retry and timeout parameters
MAX_RETRIES: int = 4
INITIAL_BACKOFF_SECONDS: float = 2.0
BACKOFF_FACTOR: float = 2.0
REQUEST_TIMEOUT_SECONDS: float = 60.0

# ==============================================================================
# 4. SCORING DIMENSIONS & RUBRICS
# ==============================================================================
SCORING_DIMENSIONS: List[str] = ["clarity", "relevance", "structure", "confidence"]
SCORE_MIN: int = 1
SCORE_MAX: int = 5

# JSON Schema for LLM Structured Outputs
OUTPUT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {
                "clarity": {"type": "integer", "minimum": 1, "maximum": 5},
                "relevance": {"type": "integer", "minimum": 1, "maximum": 5},
                "structure": {"type": "integer", "minimum": 1, "maximum": 5},
                "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["clarity", "relevance", "structure", "confidence"],
            "additionalProperties": False,
        },
        "feedback": {
            "type": "string",
            "description": "Specific, actionable feedback explaining the evaluation and how to improve.",
        },
        "improved_answer": {
            "type": "string",
            "description": "A refined, high-scoring sample answer addressing the interview question.",
        },
    },
    "required": ["scores", "feedback", "improved_answer"],
    "additionalProperties": False,
}

# ==============================================================================
# 5. PROMPT TEMPLATES (APPROACH A vs APPROACH B)
# ==============================================================================
# Approach A: General Instruction
SYSTEM_PROMPT_GENERAL: str = (
    "You are an expert technical interviewer and interview coach. "
    "Evaluate the candidate's answer to the given interview question.\n\n"
    "Score the candidate's answer on four dimensions from 1 (poor) to 5 (excellent):\n"
    "1. clarity: Is the answer easy to follow and clearly articulated?\n"
    "2. relevance: Does the answer directly address what was asked?\n"
    "3. structure: Is the response logically organized and coherent?\n"
    "4. confidence: Does the phrasing demonstrate self-assurance and technical authority?\n\n"
    "Provide constructive, actionable feedback and suggest an improved sample answer.\n\n"
    "You must respond with valid JSON strictly conforming to this schema:\n"
    "{\n"
    '  "scores": {\n'
    '    "clarity": 1,\n'
    '    "relevance": 1,\n'
    '    "structure": 1,\n'
    '    "confidence": 1\n'
    "  },\n"
    '  "feedback": "Specific, actionable feedback",\n'
    '  "improved_answer": "A stronger example answer"\n'
    "}"
)

# Approach B: Detailed Rubric-Guided Instruction
SYSTEM_PROMPT_RUBRIC: str = (
    "You are an expert technical interviewer and interview coach conducting a rigorous evaluation. "
    "Evaluate the candidate's answer based on the following explicit 1-5 scoring rubric.\n\n"
    "SCORING RUBRIC (Integer score 1 to 5 for each dimension):\n\n"
    "1. CLARITY (Articulation, conciseness, communication):\n"
    "  1: Incoherent, excessive filler words, jumbled ideas, nearly impossible to comprehend.\n"
    "  2: Hard to follow, ambiguous phrasing, excessive technical jargon without context or rambling.\n"
    "  3: Understandable but unpolished; occasional filler or slightly vague statements.\n"
    "  4: Clear, concise, articulate, well-explained with minimal fluff.\n"
    "  5: Exceptionally articulate, crystal-clear mental model, effortless to understand.\n\n"
    "2. RELEVANCE (Directness, topical focus, completeness):\n"
    "  1: Completely off-topic or fails to answer the question asked.\n"
    "  2: Only tangentially addresses the prompt; misses core technical or behavioral objective.\n"
    "  3: Partially answers the question, but leaves key aspects unaddressed or diluted.\n"
    "  4: Directly answers the question and satisfies all primary criteria.\n"
    "  5: Perfectly targeted, addresses explicit and implicit nuances, thorough without digression.\n\n"
    "3. STRUCTURE (Logical flow, framework utilization e.g., STAR, categorization):\n"
    "  1: Disorganized stream-of-consciousness, no recognizable sequence or logical order.\n"
    "  2: Weak organization; ideas jump abruptly without clear transitions.\n"
    "  3: Basic structure present (e.g. beginning and end), but transitions or milestones are choppy.\n"
    "  4: Well-organized; logical progression (e.g. Situation-Task-Action-Result or definition-tradeoff-example).\n"
    "  5: Flawless structural architecture, seamless transitions, exemplary pedagogical progression.\n\n"
    "4. CONFIDENCE (Phrasing, technical ownership, presence, authority):\n"
    "  1: Extreme hesitation, frequent disclaimers ('I don't really know', 'maybe'), timid tone.\n"
    "  2: Apologetic or tentative; second-guesses statements; lacks conviction.\n"
    "  3: Moderate confidence; neutral tone but lacks distinct conviction or personal ownership.\n"
    "  4: Firm, professional, takes ownership of actions and technical conclusions.\n"
    "  5: Poised, authoritative yet humble, radiates mastery and executive presence through wording.\n\n"
    "FEEDBACK CRITERIA:\n"
    "- Identify concrete strengths and weaknesses observed in the candidate's response.\n"
    "- Provide clear, actionable steps for improvement rather than generic praise.\n"
    "- Suggest an improved, realistic answer that the candidate could deliver in an interview.\n\n"
    "You must return ONLY a valid JSON object matching this structure:\n"
    "{\n"
    '  "scores": {\n'
    '    "clarity": 1,\n'
    '    "relevance": 1,\n'
    '    "structure": 1,\n'
    '    "confidence": 1\n'
    "  },\n"
    '  "feedback": "Specific, actionable feedback",\n'
    '  "improved_answer": "A stronger example answer"\n'
    "}"
)

# User evaluation prompt template (Question and Answer ONLY — strictly NO reference leak)
USER_EVALUATION_PROMPT_TEMPLATE: str = (
    "Evaluate the following interview response:\n\n"
    "INTERVIEW QUESTION:\n{question}\n\n"
    "CANDIDATE ANSWER:\n{candidate_answer}\n\n"
    "Provide your evaluation scores (1-5), feedback, and improved sample answer in JSON format."
)

# ==============================================================================
# 6. HUMAN EVALUATION RUBRIC CRITERIA
# ==============================================================================
HUMAN_EVALUATION_CRITERIA: List[Dict[str, str]] = [
    {
        "dimension": "specificity",
        "description": "How specific to the candidate's actual words and errors is the feedback? (1=pure generic template, 5=highly customized to the exact response)",
    },
    {
        "dimension": "actionability",
        "description": "Does the feedback provide concrete, actionable advice on what to say or do differently? (1=vague critique, 5=clear, implementable guidance)",
    },
    {
        "dimension": "relevance",
        "description": "Is the feedback aligned with the interview question and role requirements? (1=irrelevant, 5=spot-on)",
    },
    {
        "dimension": "technical_correctness",
        "description": "Is the technical critique factually accurate and sound? (1=factually incorrect advice, 5=flawless technical evaluation)",
    },
    {
        "dimension": "usefulness_improved_answer",
        "description": "How effective and realistic is the suggested improved answer as an interview response? (1=poor/unrealistic, 5=interview-ready gold standard)",
    },
]
