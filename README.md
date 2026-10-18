# Evaluating Large Language Models as Technical Interview Evaluators: A Comparative Empirical Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI API](https://img.shields.io/badge/OpenAI-API-green.svg)](https://platform.openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A rigorous empirical benchmarking framework and engineering minor project comparing Large Language Models (`gpt-4o-mini`, `gpt-4.1-mini`, `gpt-4.1`, etc.) on evaluating candidate technical interview answers across multi-dimensional criteria (Clarity, Relevance, Structure, and Confidence), providing actionable coaching feedback, and proposing interview-ready sample answers.

---

## Table of Contents

1. [Executive Summary & Research Objective](#1-executive-summary--research-objective)
2. [Research Questions (RQs)](#2-research-questions-rqs)
3. [Evaluation Dimensions & Rubric](#3-evaluation-dimensions--rubric)
4. [Dataset Architecture & Provenance](#4-dataset-architecture--provenance)
5. [System Architecture & Project Layout](#5-system-architecture--project-layout)
6. [Installation & macOS Setup](#6-installation--macos-setup)
7. [API Key Configuration](#7-api-key-configuration)
8. [Reproducibility & Execution Commands](#8-reproducibility--execution-commands)
   - [Pre-flight Connectivity Check](#pre-flight-connectivity-check)
   - [Stage A: Pilot Experiment](#stage-a-pilot-experiment)
   - [Stage B: Full Experiment](#stage-b-full-experiment)
   - [Optional Prompting Strategy Experiment](#optional-prompting-strategy-experiment)
   - [Human Feedback Evaluation Framework](#human-feedback-evaluation-framework)
   - [Standalone Chart Generation](#standalone-chart-generation)
9. [Quantitative Metrics & Mathematical Definitions](#9-quantitative-metrics--mathematical-definitions)
10. [Interpreting Experimental Results](#10-interpreting-experimental-results)
11. [Research Integrity, Disclosures & Limitations](#11-research-integrity-disclosures--limitations)

---

## 1. Executive Summary & Research Objective

Automating technical interview feedback using foundation models promises scalable personalized coaching for job applicants. However, deploying Large Language Models (LLMs) as evaluators requires systematic validation rather than subjective demonstrations. 

This research project evaluates:
- How accurately different frontier and cost-efficient LLMs align with benchmark interview evaluation scores across distinct communication dimensions.
- How prompt architecture (unconstrained general instructions vs. detailed rubric-guided definitions) influences scoring consistency, error rates, and qualitative feedback actionability.
- The cost-to-performance frontier, balancing response latency, API expenditure, and scoring reliability for real-world interview preparation platforms.

---

## 2. Research Questions (RQs)

- **RQ1 (Dimensional Alignment):** How closely do model-assigned scores align with reference scores across **Clarity**, **Relevance**, **Structure**, and **Confidence** measured by Mean Absolute Error (MAE), Exact Agreement, and Within-One-Point Agreement?
- **RQ2 (Cost-Performance Trade-off):** Does higher model complexity (e.g., `gpt-4.1` vs. `gpt-4o-mini` / `gpt-4.1-mini`) produce statistically significant improvements in scoring consistency commensurate with increased latency and token pricing?
- **RQ3 (Prompt Engineering Impact):** Does explicit rubric anchoring (Approach B) mitigate score inflation and variance compared to general open-ended evaluation prompting (Approach A)?
- **RQ4 (Feedback Quality):** To what extent do model-generated feedback texts provide specific, actionable, and technically correct coaching guidance rather than generic encouragement?

---

## 3. Evaluation Dimensions & Rubric

Each candidate response is scored on a discrete integer scale from **1 (Poor)** to **5 (Exceptional)** across four dimensions:

| Dimension | Description | Score 1 Anchor | Score 5 Anchor |
| :--- | :--- | :--- | :--- |
| **Clarity** | Articulation, conciseness, communication flow, lack of filler words. | Incoherent, heavy filler ('um', 'like'), disorganized thought. | Crystal-clear mental model, highly articulate, zero extraneous filler. |
| **Relevance** | Directness, focus, addressing core technical/behavioral concepts. | Completely off-topic or evasive; misses core intent. | Directly targets question, addresses nuanced edge cases. |
| **Structure** | Organization, logical sequencing, framework usage (e.g., STAR method). | Stream-of-consciousness, disjointed fragments. | Flawless structural progression (Situation-Task-Action-Result / definition-tradeoff-example). |
| **Confidence** | Technical ownership, authority, conviction conveyed in phrasing. | Apologetic, timid, second-guessing ('maybe', 'I don't know'). | Poised, authoritative yet humble, radiates mastery through wording. |

Additionally, every model produces:
- **`feedback`**: Actionable critique highlighting strengths, vulnerabilities, and targeted improvement strategies.
- **`improved_answer`**: A high-scoring sample response demonstrating how to address the prompt effectively.

---

## 4. Dataset Architecture & Provenance

The benchmark dataset is located at `data/interview_dataset.json` (with automatic fallback to `data/interview_training_data_all.json`). It contains **33 curated interview records** spanning Software Engineering, Technical, Behavioral, and HR domains across varying difficulties (easy, medium, hard).

### Record Structure
```json
{
  "id": 1,
  "role": "Software Engineer",
  "category": "Behavioral",
  "difficulty": "easy",
  "question": "Tell me about yourself.",
  "candidate_answer": "Um, I am a fresher, I did my B.Tech and I like coding. I know Java and some Python. I want a job in a good company.",
  "answer_quality": "weak",
  "feedback": "Your answer is too short and generic. Avoid fillers like 'um'...",
  "scores": {
    "clarity": 2,
    "relevance": 2,
    "structure": 1,
    "confidence": 2
  },
  "improved_answer": "I'm a recent B.Tech graduate in Computer Science with hands-on experience in Java and Python..."
}
```

> [!IMPORTANT]
> **Data Leakage Safeguard:** During evaluation, the prompt generator extracts **only** `question` and `candidate_answer`. The reference scores, reference feedback, and reference improved answer are strictly excluded from the prompt payload to guarantee unbiased evaluation.

---

## 5. System Architecture & Project Layout

```
ai-interview-preparation/
├── .env                                  # Local environment credentials (ignored by Git)
├── .env.example                          # Sample environment configuration template
├── .gitignore                            # Git security rules
├── requirements.txt                      # Project Python dependencies
├── README.md                             # Research documentation & instructions
├── data/
│   ├── interview_dataset.json            # Active benchmark dataset (33 records)
│   └── interview_training_data_all.json  # Raw source dataset
├── results/                              # Generated experimental artifacts
│   ├── model_outputs.csv                 # Granular model responses & latencies
│   ├── model_metrics.csv                 # Aggregated quantitative metrics
│   ├── human_feedback_ratings_template.csv # Blank human evaluation sheet
│   ├── errors.csv                        # Detailed error logs
│   ├── experiment_config.json            # Run parameters and metadata
│   ├── comparison_mae.png                # Overall MAE comparison chart
│   ├── dimension_mae.png                 # Dimensional MAE breakdown chart
│   ├── latency_comparison.png            # Response latency boxplot
│   └── agreement_comparison.png          # Exact and ±1 pt agreement rates
├── src/
│   ├── __init__.py                       # Package initializer
│   ├── config.py                         # Central configuration, models, rubrics, pricing
│   ├── evaluate_models.py                # Main experiment runner (Pilot, Full, CLI)
│   ├── evaluation_metrics.py             # MAE, exact agreement, latency & cost calculations
│   ├── feedback_evaluation.py            # Human evaluation rubric & linguistic diagnostics
│   ├── plotting.py                       # High-resolution Seaborn publication charts
│   └── utils.py                          # Ingestion, validation, caching & cost helpers
└── tests/
    └── test_pipeline.py                  # Unit and integration test suite
```

---

## 6. Installation & macOS Setup

Open Terminal on macOS and navigate to the project directory:

```bash
cd /Users/mishitachauhan/ai-interview-preparation
```

### 1. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Project Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Offline Pipeline Tests
Run the unit test suite to verify data ingestion, mathematical routines, anti-leakage checks, and caching mechanics:
```bash
python -m unittest tests/test_pipeline.py
```

---

## 7. API Key Configuration

1. Create or open your local `.env` file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` in any text editor and paste your OpenAI API key:
   ```env
   OPENAI_API_KEY=sk-proj-yourActualKeyHere
   ```
   > [!CAUTION]
   > Never commit `.env` to Git. It is automatically ignored in `.gitignore`.

---

## 8. Reproducibility & Execution Commands

All commands below should be run with the virtual environment activated (`source .venv/bin/activate`).

### Pre-flight Connectivity Check
Verify that your API key is valid and test model accessibility without performing full evaluations:
```bash
python -m src.evaluate_models --test-connection
```

---

### Stage A: Pilot Experiment
Evaluates the first 5 records across all candidate models (`gpt-4o-mini`, `gpt-4.1-mini`, `gpt-4.1`). This verifies API connectivity, JSON response schema validation, and saves preliminary metrics.
```bash
python -m src.evaluate_models --pilot
```
*Outputs generated:* `results/model_outputs.csv`, `results/model_metrics.csv`, `results/experiment_config.json`, and all visual charts.

---

### Stage B: Full Experiment
Runs the comprehensive evaluation across all 33 valid records in the dataset.
- Features automatic **resume capability**: if an API call was previously recorded in `results/model_outputs.csv`, it is skipped to prevent duplicate expenditure.
- Implements exponential backoff retries for rate limits.
```bash
python -m src.evaluate_models --full
```

To force a fresh re-evaluation from scratch, append `--force-rerun`:
```bash
python -m src.evaluate_models --full --force-rerun
```

---

### Optional Prompting Strategy Experiment
Compares **Approach A (General Prompting)** with **Approach B (Rubric-Guided Prompting)** across models:
```bash
python -m src.evaluate_models --pilot --prompt-approach both
```
Or across the full dataset:
```bash
python -m src.evaluate_models --full --prompt-approach both
```

---

### Human Feedback Evaluation Framework
To evaluate feedback quality qualitatively without assuming reference feedback is the only correct answer:

1. **Generate the blank human evaluation template:**
   ```bash
   python -m src.feedback_evaluation --generate-template
   ```
   This creates `results/human_feedback_ratings_template.csv`.

2. **Supply ratings (1 to 5):**
   Open the CSV and fill in ratings for `specificity_1_to_5`, `actionability_1_to_5`, `relevance_1_to_5`, `technical_correctness_1_to_5`, and `usefulness_improved_answer_1_to_5`.

3. **Compute human evaluation statistics:**
   ```bash
   python -m src.feedback_evaluation --analyze-template results/human_feedback_ratings_template.csv
   ```

4. **Compute automated linguistic feedback diagnostics (word counts, action-cue density):**
   ```bash
   python -m src.feedback_evaluation --heuristics
   ```

---

### Standalone Chart Generation
Re-generate high-resolution PNG charts from existing `model_outputs.csv` and `model_metrics.csv` at any time:
```bash
python -m src.plotting
```

---

## 9. Quantitative Metrics & Mathematical Definitions

### 1. Mean Absolute Error (MAE)
Measures the average magnitude of absolute difference between the model's assigned score $\hat{y}_i$ and the reference score $y_i$ for record $i \in \{1, \dots, N\}$:
$$\text{MAE}_d = \frac{1}{N} \sum_{i=1}^{N} |\hat{y}_{i,d} - y_{i,d}|$$
$$\text{Overall MAE} = \frac{1}{4} \sum_{d \in \{\text{clarity, relevance, structure, confidence}\}} \text{MAE}_d$$
*Interpretation:* An MAE of $0.0$ indicates perfect agreement. An MAE $\le 0.5$ indicates strong alignment on a 5-point discrete scale.

### 2. Exact Agreement Percentage (%)
The proportion of evaluation records where the model's score matches the reference score identically:
$$\text{Exact Agreement}_d = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\hat{y}_{i,d} = y_{i,d}) \times 100\%$$

### 3. Within-One-Point Agreement Percentage (%)
The proportion of records where the model's score is within $\pm 1$ point of the reference score (standard tolerance in educational psychometrics):
$$\text{Within-1-Point Agreement}_d = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(|\hat{y}_{i,d} - y_{i,d}| \le 1) \times 100\%$$

### 4. Operational & Latency Metrics
- **Median Latency ($t_{50}$):** 50th percentile of response turnaround time in seconds.
- **95th Percentile Latency ($t_{95}$):** Tail latency indicator representing worst-case turnaround under standard conditions.
- **Estimated Cost (\$ USD):** Computed using OpenAI published token pricing:
  $$\text{Cost} = \left(\frac{T_{\text{prompt}}}{10^6} \times P_{\text{prompt}}\right) + \left(\frac{T_{\text{completion}}}{10^6} \times P_{\text{completion}}\right)$$

---

## 10. Interpreting Experimental Results

When evaluating models for an interview preparation system:
1. **Low MAE + High Within-1-Point Agreement ($\ge 90\%$):** Indicates the model reliably distinguishes weak, average, and strong interview answers without erratic outliers.
2. **Dimensional Variance:** Models frequently perform better on *Clarity* and *Relevance* than *Confidence*. Phrasing nuances that convey confidence (e.g., passive vs. active voice) are more subtle and prone to divergent interpretations.
3. **Rubric Guidance (Approach B):** Detailed rubric criteria typically constrain score variance, compressing MAE compared to unconstrained general prompting.
4. **Efficiency Frontier:** If `gpt-4o-mini` achieves within 0.15 MAE of `gpt-4.1` while costing $15\times$ less and responding in under 2 seconds, it represents the superior candidate for interactive, real-time coaching applications.

---

## 11. Research Integrity, Disclosures & Limitations

In accordance with academic research standards:

1. **Provisional Reference Scores Notice:**
   The reference scores provided in `data/interview_dataset.json` represent a standardized baseline benchmark. Because their formal human annotation provenance and inter-rater reliability coefficients (e.g., Krippendorff's Alpha or Fleiss' Kappa) are uncertified, they are designated as **provisional reference scores**. They should be presented in academic publications as a comparative benchmark rather than infallible ground truth.

2. **LLM Evaluator Limitations:**
   - **Self-Enhancement Bias:** LLMs evaluated against LLM-generated reference texts may exhibit bias toward stylistic verbosity.
   - **Non-Determinism:** While `temperature=0.0` and `seed=42` minimize output entropy, OpenAI does not guarantee identical bit-level reproducibility across different infrastructure clusters or model snapshot revisions.

3. **No Fabricated Claims:**
   All tables, charts, and metric values reported in this repository are derived exclusively from actual API calls executed against the benchmark dataset.
