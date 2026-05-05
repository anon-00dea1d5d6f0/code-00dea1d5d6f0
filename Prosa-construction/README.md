# Prosa: Conversation Filtering Pipeline

Conversation filtering pipeline for **Prosa**, a Brazilian Portuguese real-user multi-turn chat benchmark for LLM-as-a-judge evaluation. This code reduces the WildChat-4.8M corpus down to the 1,000 curated conversations that form the prompt set of Prosa.

## Pipeline overview

| Step | Filter | Type | Output count |
|---|---|---|---|
| Raw | WildChat-4.8M | — | 3,199,860 |
| 1 | Base Filter (Portuguese, unique IP, Brazil, ≤5 user turns, no REDACTED, no empty content) | Rule-based | 3,889 |
| 2 | Token Filter (Qwen3-Embedding-8B tokenizer, length in [50, 8192]) | Rule-based | 2,537 |
| 3 | Semantic Deduplication (cosine ≥ 0.90 with the same encoder) | Rule-based | 2,097 |
| 4 | Difficulty Filter (gpt-4.1 and Qwen3-235B; retain at least one judge labelling the task as medium or above) | LLM-based | 1,420 |
| 5 | Topic Filter (gpt-4.1, 12 Magpie-inspired categories; drop "Others") | LLM-based | 1,399 |
| 6 | Nonsensical Filter (gpt-4.1; remove conversations without a clear, evaluable user request) | LLM-based | 1,355 |
| 7 | Random sample (seed = 42) | — | 1,000 |

## Requirements

Python 3.10+ and the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The LLM-based steps (4–6) call the OpenAI API; export your key before running them:

```bash
export OPENAI_API_KEY=...
```

Step 4 also uses an open-weights model accessed via OpenAI-compatible endpoint (Qwen3-235B-A22B-Instruct-2507); see the script's `--help` for the relevant flags.

## Source data

The raw corpus is `WildChat-4.8M` (86 parquet shards):

```
https://huggingface.co/datasets/allenai/WildChat-4.8M
```

By default Step 1 reads the raw shards from `<repo>/wildchat_raw/`. Override with `WILDCHAT_DATA_DIR=/path/to/raw/parquets`.

## Usage

Run the steps sequentially. Each script writes its parquet output under `OUTPUT_BASE` (the repo directory by default); subsequent steps consume the previous step's output.

```bash
# Optional: override locations
export PROSA_OUTPUT_BASE=/path/to/working/dir   # default: this repo
export WILDCHAT_DATA_DIR=/path/to/raw/parquets  # default: ./wildchat_raw
```

```bash
python filter_step1_base.py
python filter_step2_tokenize_and_filter.py
python filter_step3_embeddings_and_dedup.py
python filter_step4_difficulty.py
python filter_step5_topics.py
python filter_step6_nonsensical.py
python filter_step7_sample_1000.py
```

The final benchmark prompt set is written to `07_sample_1000/wildchat_sample_1000.parquet`.

## Frozen output

For convenience and reproducibility we ship the frozen output of our run at:

- `06_nonsensical_filtered/wildchat_nonsensical_filtered.parquet` (1,355 rows; output of step 6, before random sampling)
- `07_sample_1000/wildchat_sample_1000.parquet` (1,000 rows; same `conversation_hash` set used across all experiments in the paper)
