#!/usr/bin/env python3
"""
Step 4: Difficulty classification and combined filter.

Sub-steps:
  4a. Difficulty classification — uses an LLM via OpenAI-compatible API
      to classify the difficulty of the user's last query. Must be run
      once per model (e.g., Qwen3-235B + GPT-4.1).

  4b. Combined filter — keeps only examples where at least one of the
      two models classified the query as 'medium', 'hard', or 'very hard'.

Usage:
    # 4a — Classification (run once per model)
    python filter_step4_difficulty.py classify \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
        --api-key token-abc123 \
        --max-concurrent 50

    python filter_step4_difficulty.py classify \
        --base-url https://api.openai.com/v1 \
        --model gpt-4.1-2025-04-14 \
        --api-key $OPENAI_API_KEY \
        --max-concurrent 50

    # 4b — Combined filter
    python filter_step4_difficulty.py filter

Produces step4_stats.json with statistics for the report update.
"""

import os
import json
import asyncio
import argparse
import re
import time
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI


# ── Configuration ────────────────────────────────────────────────────────────
OUTPUT_BASE = os.environ.get("PROSA_OUTPUT_BASE", os.path.dirname(os.path.abspath(__file__)))
STEP3_FILE = os.path.join(OUTPUT_BASE, "03_embeddings_dedup", "wildchat_embeddings_dedup.parquet")
CLASSIFY_DIR = os.path.join(OUTPUT_BASE, "04_difficulty")
FILTER_DIR = os.path.join(OUTPUT_BASE, "04_difficulty_filtered")
FILTER_OUTPUT = os.path.join(FILTER_DIR, "wildchat_difficulty_filtered.parquet")

KEEP_LEVELS = {"medium", "hard", "very hard"}


# ── Prompt templates ─────────────────────────────────────────────────────────

PROMPT_WITH_HISTORY = """# Instruction
You first need to identify the given user intent and then label the difficulty level of the user
query based on the content of the user query and the conversation history.

## History
'''{user_history}'''

## User Query
'''{input}'''

## Output Format
Given the user query and the conversation history, in your output, you first need to identify the user intent and the
knowledge needed to solve the task in the user query. Then, rate the difficulty level of
the user query as 'very easy', 'easy', 'medium', 'hard', or 'very hard'.
Now, please output the user intent and difficulty level below in a json format by filling in the
placeholders in [...]:
'''
{{
"intent": "The user wants to [....]",
"knowledge": "To solve this problem, the models need to know [....]",
"difficulty": "[very easy/easy/medium/hard/very hard]"
}}
'''"""

PROMPT_WITHOUT_HISTORY = """# Instruction
You first need to identify the given user intent and then label the difficulty level of the user
query based on the content of the user query.

## User Query
'''{input}'''

## Output Format
Given the user query, in your output, you first need to identify the user intent and the
knowledge needed to solve the task in the user query. Then, rate the difficulty level of
the user query as 'very easy', 'easy', 'medium', 'hard', or 'very hard'.
Now, please output the user intent and difficulty level below in a json format by filling in the
placeholders in [...]:
'''
{{
"intent": "The user wants to [....]",
"knowledge": "To solve this problem, the models need to know [....]",
"difficulty": "[very easy/easy/medium/hard/very hard]"
}}
'''"""

MAX_RETRIES = 3
RETRY_DELAY = 2


# ── Helpers ──────────────────────────────────────────────────────────────────

def fmt(n: int) -> str:
    return f"{n:,}"


def build_history_and_query(conversation: list) -> tuple[str, str]:
    """Build the plain-text history and the user's last query."""
    history = ""
    if len(conversation) > 0:
        for x in conversation[:-1]:
            if x["role"] == "user":
                history += "USER: " + x["content"] + "\n\n"
            elif x["role"] == "assistant":
                history += "ASSISTANT: " + x["content"] + "\n\n"
        last_query = conversation[-1]["content"]
    else:
        last_query = ""
    return history.strip(), last_query


def build_prompt(history: str, query: str) -> str:
    """Choose the correct template and build the prompt."""
    if history:
        return PROMPT_WITH_HISTORY.format(user_history=history, input=query)
    else:
        return PROMPT_WITHOUT_HISTORY.format(input=query)


def parse_response(response_text: str) -> dict:
    """Extract the JSON from the LLM response."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    patterns = [
        r"```json?\s*(\{[^`]+\})\s*```",
        r"'''?\s*(\{[^']+\})\s*'''?",
        r'(\{[^{}]*"intent"[^{}]*"knowledge"[^{}]*"difficulty"[^{}]*\})',
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    return {
        "intent": "parse_error",
        "knowledge": "parse_error",
        "difficulty": "parse_error",
        "raw_response": response_text[:500],
    }


def extract_response_metadata(response) -> dict:
    """Extract metadata from the API response."""
    meta = {
        "id": response.id,
        "model": response.model,
        "created": response.created,
        "finish_reason": response.choices[0].finish_reason if response.choices else None,
    }
    if response.usage:
        meta["prompt_tokens"] = response.usage.prompt_tokens
        meta["completion_tokens"] = response.usage.completion_tokens
        meta["total_tokens"] = response.usage.total_tokens
    return meta


def extract_difficulty(entry) -> str:
    """Extract the difficulty level from a difficulty field."""
    if isinstance(entry, dict):
        return entry.get("difficulty", "unknown")
    return "unknown"


# ── Async classification ──────────────────────────────────────────────────────

async def classify_example(
    client: AsyncOpenAI,
    model: str,
    history: str,
    query: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Classify the difficulty of a query via API."""
    if not query or not query.strip():
        return {
            "intent": "empty_query",
            "knowledge": "empty_query",
            "difficulty": "empty_query",
        }

    prompt = build_prompt(history, query)

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = response.choices[0].message.content
                result = parse_response(response_text)
                result["_response_metadata"] = extract_response_metadata(response)
                return result

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    return {
                        "intent": "api_error",
                        "knowledge": "api_error",
                        "difficulty": "api_error",
                        "error": str(e)[:200],
                    }

    return {
        "intent": "unknown_error",
        "knowledge": "unknown_error",
        "difficulty": "unknown_error",
    }


async def process_all(
    df: pd.DataFrame,
    client: AsyncOpenAI,
    model: str,
    max_concurrent: int,
) -> list[dict]:
    """Process all examples in parallel."""
    semaphore = asyncio.Semaphore(max_concurrent)

    # Prepare histories and queries
    histories = []
    queries = []
    for conv in df["conversation"]:
        h, q = build_history_and_query(conv)
        histories.append(h)
        queries.append(q)

    n_with_history = sum(1 for h in histories if h)
    n_without = len(histories) - n_with_history
    print(f"  With history:     {fmt(n_with_history)}")
    print(f"  Without history:  {fmt(n_without)}")

    tasks = [
        classify_example(client, model, h, q, semaphore)
        for h, q in zip(histories, queries)
    ]

    results = await tqdm_asyncio.gather(*tasks, desc="Classifying")
    return results


# ── Subcommand: classify ──────────────────────────────────────────────────────

def run_classify(args):
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   STEP 4a: DIFFICULTY CLASSIFICATION — WILDCHAT                    ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t_start = time.time()

    # Configuration
    print(f"\n  Model:            {args.model}")
    print(f"  API URL:          {args.base_url}")
    print(f"  Max concurrent:   {args.max_concurrent}")

    # Determine input/output
    # If a previous classification exists in CLASSIFY_DIR, use it as input
    model_short = args.model.split("/")[-1] if "/" in args.model else args.model
    field_name = f"difficulty_{model_short}"

    existing_files = sorted(
        [f for f in os.listdir(CLASSIFY_DIR) if f.endswith(".parquet")]
    ) if os.path.exists(CLASSIFY_DIR) else []

    if existing_files:
        input_file = os.path.join(CLASSIFY_DIR, existing_files[-1])
        print(f"\n  Previous classification found: {input_file}")
    else:
        input_file = STEP3_FILE

    print(f"  Input:            {input_file}")

    # Load dataset
    print(f"\nLoading dataset: {input_file}")
    df = pd.read_parquet(input_file)
    total = len(df)
    print(f"Total examples: {fmt(total)}")

    # Check if column already exists
    if field_name in df.columns:
        print(f"\n  WARNING: Field '{field_name}' already exists in dataset!")
        print(f"  Overwriting previous classification...")

    # API client
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)

    # Process
    print(f"\nProcessing {fmt(total)} examples...")
    results = asyncio.run(process_all(df, client, args.model, args.max_concurrent))

    # Store results in the model-named field
    print(f"\nStoring results in field '{field_name}'...")
    eval_entries = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for r in results:
        entry = {
            "model": args.model,
            "intent": r.get("intent", "error"),
            "knowledge": r.get("knowledge", "error"),
            "difficulty": r.get("difficulty", "error"),
        }
        if "raw_response" in r:
            entry["raw_response"] = r["raw_response"]
        if "error" in r:
            entry["error"] = r["error"]
        if "_response_metadata" in r:
            entry["response_metadata"] = r["_response_metadata"]
            total_prompt_tokens += r["_response_metadata"].get("prompt_tokens", 0)
            total_completion_tokens += r["_response_metadata"].get("completion_tokens", 0)
        eval_entries.append(entry)

    df[field_name] = eval_entries

    # Statistics
    difficulties = [r.get("difficulty", "error") for r in results]
    diff_counts = pd.Series(difficulties).value_counts()

    print(f"\nDifficulty distribution:")
    for level, count in diff_counts.items():
        pct = count / total * 100
        print(f"  {level}: {fmt(count)} ({pct:.2f}%)")

    # Errors
    n_parse_err = sum(1 for d in difficulties if d == "parse_error")
    n_api_err = sum(1 for d in difficulties if d == "api_error")
    n_empty = sum(1 for d in difficulties if d == "empty_query")
    total_tokens = total_prompt_tokens + total_completion_tokens
    print(f"\nTokens:")
    print(f"  Prompt tokens:     {fmt(total_prompt_tokens)}")
    print(f"  Completion tokens: {fmt(total_completion_tokens)}")
    print(f"  Total tokens:      {fmt(total_tokens)}")

    print(f"\n  Parse errors:  {n_parse_err}")
    print(f"  API errors:    {n_api_err}")
    print(f"  Empty queries: {n_empty}")

    # Save
    os.makedirs(CLASSIFY_DIR, exist_ok=True)
    output_file = os.path.join(CLASSIFY_DIR, "wildchat_difficulty.parquet")
    print(f"\nSaving dataset: {output_file}")
    t_save = time.time()
    df.to_parquet(output_file, engine="pyarrow", compression="snappy", index=False)
    save_time = time.time() - t_save

    file_size_mb = os.path.getsize(output_file) / 1024**2
    total_time = time.time() - t_start

    print(f"  Saved in {save_time:.1f}s ({file_size_mb:.1f} MB)")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Examples:         {fmt(total)}")
    print(f"  Model:            {args.model}")
    print(f"  Field:            {field_name}")
    print(f"  Total time:       {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"  File:             {output_file} ({file_size_mb:.1f} MB)")

    # Stats file
    stats_file = os.path.join(OUTPUT_BASE, f"step4_classify_stats_{model_short}.json")
    stats = {
        "timestamp": timestamp,
        "model": args.model,
        "base_url": args.base_url,
        "max_concurrent": args.max_concurrent,
        "field_name": field_name,
        "input_file": input_file,
        "output_file": output_file,
        "output_size_mb": round(file_size_mb, 1),
        "total_examples": total,
        "total_time_s": round(total_time, 1),
        "save_time_s": round(save_time, 1),
        "n_parse_errors": n_parse_err,
        "n_api_errors": n_api_err,
        "n_empty": n_empty,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "difficulty_distribution": {level: int(count) for level, count in diff_counts.items()},
    }

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nStats saved to: {stats_file}")
    print(f"\nClassification with {model_short} complete.")

    # List existing difficulty columns
    diff_cols = [c for c in df.columns if c.startswith("difficulty_")]
    print(f"\nDifficulty columns in dataset: {diff_cols}")
    if len(diff_cols) >= 2:
        print("  -> Ready to run the filter: python filter_step4_difficulty.py filter")


# ── Subcommand: filter ────────────────────────────────────────────────────────

def run_filter(args):
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   STEP 4b: COMBINED DIFFICULTY FILTER — WILDCHAT                   ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t_start = time.time()

    # Load classified dataset
    input_file = os.path.join(CLASSIFY_DIR, "wildchat_difficulty.parquet")
    print(f"\nLoading dataset: {input_file}")
    df = pd.read_parquet(input_file)
    total = len(df)
    print(f"Total examples: {fmt(total)}")

    # Detect difficulty columns
    diff_cols = sorted([c for c in df.columns if c.startswith("difficulty_")])
    print(f"\nDifficulty columns found: {diff_cols}")

    if len(diff_cols) < 2:
        print(f"\n  ERROR: Expected at least 2 difficulty columns, found {len(diff_cols)}.")
        print(f"  Run 'classify' with two models before running 'filter'.")
        return

    field_1 = diff_cols[0]
    field_2 = diff_cols[1]
    print(f"  Model 1: {field_1}")
    print(f"  Model 2: {field_2}")

    # Extract difficulties
    print("\nExtracting difficulties from both models...")
    diff_1 = []
    diff_2 = []
    for _, row in tqdm(df.iterrows(), total=total, desc="Extracting"):
        diff_1.append(extract_difficulty(row[field_1]))
        diff_2.append(extract_difficulty(row[field_2]))

    df["_diff_1"] = diff_1
    df["_diff_2"] = diff_2

    # Statistics before filter
    print("\nDistribution BEFORE filter:")
    print(f"\n  {field_1}:")
    for level, count in df["_diff_1"].value_counts().items():
        pct = count / total * 100
        print(f"    {level}: {fmt(count)} ({pct:.2f}%)")

    print(f"\n  {field_2}:")
    for level, count in df["_diff_2"].value_counts().items():
        pct = count / total * 100
        print(f"    {level}: {fmt(count)} ({pct:.2f}%)")

    # Apply filter: keep if AT LEAST ONE model gave medium/hard/very hard
    print(f"\nApplying filter: keep if at least one model in {KEEP_LEVELS}")
    keep_mask = df["_diff_1"].isin(KEEP_LEVELS) | df["_diff_2"].isin(KEEP_LEVELS)
    df_out = df[keep_mask].reset_index(drop=True)

    kept = len(df_out)
    removed = total - kept
    pct_removed = removed / total * 100 if total > 0 else 0

    print(f"\n  Kept:    {fmt(kept)}")
    print(f"  Removed: {fmt(removed)} ({pct_removed:.2f}%)")

    # Breakdown of removed examples
    removed_mask = ~keep_mask
    df_removed = df[removed_mask]
    n_both_easy = len(df_removed[
        (df_removed["_diff_1"].isin({"very easy", "easy"})) &
        (df_removed["_diff_2"].isin({"very easy", "easy"}))
    ])
    n_empty = len(df_removed[
        (df_removed["_diff_1"] == "empty_query") |
        (df_removed["_diff_2"] == "empty_query")
    ])
    n_other = removed - n_both_easy - n_empty

    print(f"\n  Breakdown of removed:")
    print(f"    Both easy/very easy: {fmt(n_both_easy)}")
    print(f"    Empty query:         {fmt(n_empty)}")
    print(f"    Other:               {fmt(n_other)}")

    # Statistics after filter
    print(f"\nDistribution AFTER filter ({field_1}):")
    for level, count in df_out["_diff_1"].value_counts().items():
        pct = count / kept * 100
        print(f"    {level}: {fmt(count)} ({pct:.2f}%)")

    print(f"\n  Distribution AFTER filter ({field_2}):")
    for level, count in df_out["_diff_2"].value_counts().items():
        pct = count / kept * 100
        print(f"    {level}: {fmt(count)} ({pct:.2f}%)")

    # Drop auxiliary columns
    df_out = df_out.drop(columns=["_diff_1", "_diff_2"])

    # Save
    os.makedirs(FILTER_DIR, exist_ok=True)
    print(f"\nSaving dataset: {FILTER_OUTPUT}")
    t_save = time.time()
    df_out.to_parquet(FILTER_OUTPUT, engine="pyarrow", compression="snappy", index=False)
    save_time = time.time() - t_save

    file_size_mb = os.path.getsize(FILTER_OUTPUT) / 1024**2
    total_time = time.time() - t_start

    print(f"  Saved in {save_time:.1f}s ({file_size_mb:.1f} MB)")

    # Summary
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY — STEP 4")
    print(f"{'=' * 70}")
    print(f"  Input examples:       {fmt(total)}")
    print(f"  Kept examples:        {fmt(kept)}")
    print(f"  Removed examples:     {fmt(removed)} ({pct_removed:.2f}%)")
    print(f"  Criterion:            at least 1 model in {sorted(KEEP_LEVELS)}")
    print(f"  Models:               {field_1}, {field_2}")
    print(f"  Total time:           {total_time:.1f}s")
    print(f"  Output file:          {FILTER_OUTPUT} ({file_size_mb:.1f} MB)")

    # Stats file
    stats_file = os.path.join(OUTPUT_BASE, "step4_filter_stats.json")
    stats = {
        "timestamp": timestamp,
        "input_file": input_file,
        "output_file": FILTER_OUTPUT,
        "output_size_mb": round(file_size_mb, 1),
        "total_input": total,
        "total_kept": kept,
        "total_removed": removed,
        "pct_removed": round(pct_removed, 2),
        "removed_both_easy": n_both_easy,
        "removed_empty_query": n_empty,
        "removed_other": n_other,
        "keep_levels": sorted(KEEP_LEVELS),
        "field_1": field_1,
        "field_2": field_2,
        "total_time_s": round(total_time, 1),
        "save_time_s": round(save_time, 1),
    }

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nStats saved to: {stats_file}")
    print("\nStep 4 complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 4: Difficulty classification and combined filter"
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # classify
    p_classify = subparsers.add_parser("classify", help="Classify difficulty with a single model")
    p_classify.add_argument("--base-url", required=True, help="API base URL (e.g., http://localhost:8000/v1)")
    p_classify.add_argument("--model", required=True, help="Model name (e.g., Qwen/Qwen3-235B-A22B-Instruct-2507)")
    p_classify.add_argument("--api-key", default="no-key", help="API key")
    p_classify.add_argument("--max-concurrent", type=int, default=50, help="Concurrent requests")

    # filter
    p_filter = subparsers.add_parser("filter", help="Filter by combined difficulty")

    args = parser.parse_args()

    if args.command == "classify":
        run_classify(args)
    elif args.command == "filter":
        run_filter(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
