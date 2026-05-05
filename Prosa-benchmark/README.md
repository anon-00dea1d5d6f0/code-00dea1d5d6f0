# Prosa

Brazilian Portuguese LLM evaluation benchmark, built from 1,000 real user multi-turn conversations sourced from [WildChat](https://huggingface.co/datasets/allenai/WildChat). Candidate model responses are scored against per-question binary rubrics (pass/fail) by an LLM judge, with a multi-judge post-hoc filter that removes low-quality rubrics before the final scoring.

This repository implements the *Evaluation Protocol* (Section 4) and *Results* (Section 5) of the paper, and ships the frozen artefacts needed to reproduce the leaderboard or to evaluate any new candidate.

## Pipeline overview

| Stage | Script | Paper section | What it does |
|---|---|---|---|
| 1 | `gen_answer.py` | — | Generates the candidate model's responses to the 1,000 questions. |
| 2 | `gen_rubrics.py` | §4.1 | Generates per-question binary rubrics (RRD F.1 prompt). Already frozen in `question.jsonl`. |
| 3 | `gen_score.py` | §4.2 | Scores each response with binary YES/NO verdicts against the question's rubrics (RRD F.3 prompt). |
| 4 | `filter_rubrics.py` | §4.3 | Multi-judge post-hoc filtering pipeline that removes low-quality rubrics. |
| 5 | `show_score.py` | §5.2 | Aggregates the verdicts into the final leaderboard. |

The benchmark also ships a holistic 1–10 scoring path (`gen_score_holistic.py` + `show_score_holistic.py`) used as the §5.1 baseline in the validation comparison.

## Frozen artefacts

Every artefact is shipped frozen. Reproducing the paper's leaderboard requires no API calls; only adding a new candidate does.

| Artefact | Paper section | Location |
|---|---|---|
| 1,000 questions + 12,920 raw rubrics + filter outcomes | §3, §4.1, §4.3 | `prosa/data/prosa/question.jsonl` |
| Cached responses for the 16 candidates and the rubric-generation reference model | §4.4 | `prosa/data/prosa/model_answer/` |
| Holistic baseline verdicts (16 × 3) | §5.1 | `prosa/data/prosa/score_holistic/` |
| Rubric verdicts (16 × 3) | §4.2, §5.2 | `prosa/data/prosa/score_rubric/` |
| Three independent runs of Qwen3-30B × GPT-4.1 for the *unstable* filter | §4.3 | `score_rubric/` (run 1), `score_rubric_run2/`, `score_rubric_run3/` |

## Installation

```bash
conda create -n prosa python=3.10 -y
conda activate prosa
pip install -r requirements.txt
```

All API calls go through an OpenAI-compatible endpoint. For OpenAI itself, set `OPENAI_API_KEY`. For any other provider (OpenRouter, vLLM, DeepInfra, Google's `generativelanguage.googleapis.com/v1beta/openai/`, locally hosted models, etc.), pass `--api-base` and `--api-key` to the relevant script.

## Reproducing the paper

To print the leaderboard from the frozen verdicts (no API calls):

```bash
python -m prosa.show_score --filtered    # rubric leaderboard (Table 5)
python -m prosa.show_score_holistic      # holistic baseline (§5.1)
```

## Evaluating a new candidate

Three steps. All commands run from the repo root.

### 1. Generate the candidate's responses

```bash
python -m prosa.gen_answer \
    --model gpt-4.1-2025-04-14 \
    --parallel 50

# OpenAI-compatible endpoint
python -m prosa.gen_answer \
    --model Qwen/Qwen3-235B-A22B-Instruct \
    --model-id Qwen3-235B-A22B-Instruct \
    --api-base https://openrouter.ai/api/v1 \
    --api-key "$OPENROUTER_API_KEY" \
    --parallel 20
```

Saves to `prosa/data/prosa/model_answer/{model}.jsonl`.

Useful flags: `--temperature 0.7`, `--max-tokens 4096`, `--no-resume` (regenerate everything), `--reasoning-effort low|medium|high` (for reasoning models).

### 2. Score the responses

Rubric scoring (canonical, RRD F.3 prompt — recommended):

```bash
python -m prosa.gen_score \
    --model gpt-4.1-2025-04-14 \
    --judge-model gemini-3-flash-preview \
    --parallel 50
```

Saves to `prosa/data/prosa/score_rubric/{model}_by_{judge}.jsonl`.

Holistic scoring (1–10 baseline, used in the paper's protocol comparison):

```bash
python -m prosa.gen_score_holistic \
    --model gpt-4.1-2025-04-14 \
    --judge-model gpt-4.1-2025-04-14 \
    --parallel 50
```

Saves to `prosa/data/prosa/score_holistic/{model}_by_{judge}.jsonl`.

### 3. Display the leaderboard

```bash
# Filtered rubric scoring (canonical)
python -m prosa.show_score --filtered

# Without filtering
python -m prosa.show_score

# Holistic baseline
python -m prosa.show_score_holistic
```

## Re-running the rubric filter

The frozen `checklist_filtered_indices` in `question.jsonl` come from the multi-judge filter described in the paper. To re-run the filter (e.g., after adding new candidates):

```bash
python -m prosa.filter_rubrics            # apply all filters
python -m prosa.filter_rubrics --dry-run  # statistics only
python -m prosa.filter_rubrics --no-trivial --no-misaligned   # disable specific filters
```

The five filters:

1. **Trivial** — every candidate passes the rubric (majority ≥ 2/3 judges).
2. **Impossible** — every candidate fails (majority ≥ 2/3 judges).
3. **Misaligned** — the two top-scoring candidates fail while the bottom-scoring candidate passes (majority ≥ 2/3 judges).
4. **Unstable** — the rubric's verdict flips across three independent runs of the same (candidate, judge) pair at temperature 0.
5. **Min rubrics** — questions with no surviving rubric are excluded from scoring.

Filter output is written back to `question.jsonl`:

- `checklist` — original rubrics generated in Stage 2 (never modified).
- `checklist_filtered` — surviving rubrics, renumbered.
- `checklist_filtered_indices` — 0-based indices into `evaluations[]` of the score files.
- `question_excluded` — `true` if the question has no rubrics after filtering.

## Repository layout

```
Prosa-benchmark/
├── README.md
├── requirements.txt
└── prosa/
    ├── __init__.py
    ├── conversation.py        # OpenAI-style conversation template
    ├── common.py              # OpenAI-compatible client + helpers
    ├── prosa_utils.py         # Shared loaders
    ├── gen_answer.py          # Stage 1 — generate model responses
    ├── gen_rubrics.py         # Stage 2 — generate rubrics (RRD F.1)
    ├── gen_score.py           # Stage 3 — rubric scoring (RRD F.3, canonical)
    ├── gen_score_holistic.py  # Stage 3 baseline — holistic 1–10 score
    ├── filter_rubrics.py      # Stage 4 — multi-judge rubric filter
    ├── show_score.py          # Stage 5 — rubric leaderboard
    ├── show_score_holistic.py # Stage 5 baseline — holistic leaderboard
    ├── update_checklists.py   # Replace the rubric set in question.jsonl
    └── data/prosa/
        ├── question.jsonl              # 1,000 questions + 12,920 rubrics + filter outcomes
        ├── model_answer/               # Cached candidate + reference responses
        ├── score_holistic/             # Baseline holistic verdicts (16 × 3)
        ├── score_rubric/               # Canonical rubric verdicts (16 × 3)
        ├── score_rubric_run2/          # Stability run 2 (Qwen3-30B × GPT-4.1)
        └── score_rubric_run3/          # Stability run 3 (Qwen3-30B × GPT-4.1)
```

## Data format

### `question.jsonl`

```json
{
    "conversation_hash": "331893ee1b0634d0ae8c573e050e0705",
    "category": "Information seeking",
    "conversation_input": [
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "user",      "content": "last user query"}
    ],
    "last_query": "last user query",
    "history_text": "USER: ...\n\nASSISTANT: ...\n\n",
    "num_turns": 2,
    "difficulty": {"intent": "...", "knowledge": "...", "level": "medium"},
    "checklist": "1. Does the response mention X?\n2. Is the format correct?",
    "checklist_filtered": "1. Is the format correct?",
    "checklist_filtered_indices": [1],
    "question_excluded": false
}
```

### `model_answer/{model}.jsonl`

```json
{"conversation_hash": "...", "output": ["model response"]}
```

### `score_holistic/{model}_by_{judge}.jsonl`

```json
{"conversation_hash": "...", "score": 7.5}
```

`score` is the judge's 1–10 holistic rating (paper §5.1 baseline).

### `score_rubric/{model}_by_{judge}.jsonl`

```json
{
    "conversation_hash": "...",
    "evaluations": ["YES", "NO", "YES"],
    "n_rubrics": 3,
    "n_valid": 3,
    "n_passed": 2,
    "pass_rate": 0.6667,
    "score": 6.667
}
```

`evaluations` lists one YES/NO verdict per rubric (in `checklist` order). `score` is `pass_rate × 10`. The other fields are derivable from `evaluations` but are pre-computed for convenience.
