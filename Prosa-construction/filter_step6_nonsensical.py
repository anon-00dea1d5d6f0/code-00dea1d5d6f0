#!/usr/bin/env python3
"""
Step 6: Nonsensical/sensical classification and filter.

Sub-steps:
  6a. Classification — uses an LLM via OpenAI-compatible API to classify
      whether the user's last query is nonsensical or sensical. Uses the
      conversation history (when present) as context.

  6b. Filter — removes examples classified as nonsensical ("yes") or
      with parse_error.

Usage:
    # 6a — Classification
    python filter_step6_nonsensical.py classify \
        --base-url https://api.openai.com/v1 \
        --model gpt-4.1-2025-04-14 \
        --api-key $OPENAI_API_KEY \
        --max-concurrent 50

    # 6b — Filter
    python filter_step6_nonsensical.py filter

Produces step6_stats.json with statistics for the report update.
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
STEP5_FILE = os.path.join(OUTPUT_BASE, "05_topics_filtered", "wildchat_topics_filtered.parquet")
CLASSIFY_DIR = os.path.join(OUTPUT_BASE, "06_nonsensical")
FILTER_DIR = os.path.join(OUTPUT_BASE, "06_nonsensical_filtered")
FILTER_OUTPUT = os.path.join(FILTER_DIR, "wildchat_nonsensical_filtered.parquet")

REMOVE_VALUES = {"yes", "parse_error"}


# ── Prompt templates ─────────────────────────────────────────────────────────

PROMPT_WITH_HISTORY = """# Instruction
You need to decide whether a user query is a nonsensical or sensical task based on the content of the user query and the conversation history.

A query should be marked as nonsensical if it fits one of the categories below:
- gibberish/noise/spam: text without meaningful language (random characters, corrupted text, repetitive junk, spam).
- no intelligible intent: short utterances or fragments that do not form a request or question (e.g., "ok então", "???", "...").
- no actionable intent: there is no clear request, question or actionable intent. Thus, there is no way to judge if the query was done correctly/well (i.e., no definable success criterion).

A query should be marked as sensical if there is any intelligible and actionable user intent, even if it is short, incomplete, ambiguous, or very poorly written.

## History
'''{user_history}'''

## User Query
'''{input}'''

## Output Format
Given the user query and the conversation history, first write a brief assessment of why the user query is nonsencial (yes) or not (no).
Then output a decision by filling in the placeholders in [...]:

```
{{
  "explanation": "[...]",
  "nonsensical_query": "[yes/no]"
}}
```"""

PROMPT_WITHOUT_HISTORY = """# Instruction
You need to decide whether a user query is a nonsensical or sensical task based on the content of the user query.

A query should be marked as nonsensical if it fits one of the categories below:
- gibberish/noise/spam: text without meaningful language (random characters, corrupted text, repetitive junk, spam).
- no intelligible intent: short utterances or fragments that do not form a request or question (e.g., "ok então", "???", "...").
- no actionable intent: there is no clear request, question or actionable intent. Thus, there is no way to judge if the query was done correctly/well (i.e., no definable success criterion).

A query should be marked as sensical if there is any intelligible and actionable user intent, even if it is short, incomplete, ambiguous, or very poorly written.

## User Query
'''{input}'''

## Output Format
Given the user query, first write a brief assessment of why the user query is nonsencial (yes) or not (no).
Then output a decision by filling in the placeholders in [...]:

```
{{
  "explanation": "[...]",
  "nonsensical_query": "[yes/no]"
}}
```"""

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
        r'(\{[^{}]*"explanation"[^{}]*"nonsensical_query"[^{}]*\})',
        r'(\{[^{}]*"nonsensical_query"[^{}]*"explanation"[^{}]*\})',
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    return {
        "explanation": "parse_error",
        "nonsensical_query": "parse_error",
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


# ── Async classification ──────────────────────────────────────────────────────

async def classify_example(
    client: AsyncOpenAI,
    model: str,
    history: str,
    query: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Classify whether a query is nonsensical via API."""
    if not query or not query.strip():
        return {
            "explanation": "empty_query",
            "nonsensical_query": "yes",
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
                        "explanation": "api_error",
                        "nonsensical_query": "api_error",
                        "error": str(e)[:200],
                    }

    return {
        "explanation": "unknown_error",
        "nonsensical_query": "unknown_error",
    }


async def process_all(
    df: pd.DataFrame,
    client: AsyncOpenAI,
    model: str,
    max_concurrent: int,
) -> list[dict]:
    """Process all examples in parallel."""
    semaphore = asyncio.Semaphore(max_concurrent)

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
    print("║   STEP 6a: NONSENSICAL/SENSICAL CLASSIFICATION — WILDCHAT          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t_start = time.time()

    # Configuration
    print(f"\n  Model:            {args.model}")
    print(f"  API URL:          {args.base_url}")
    print(f"  Max concurrent:   {args.max_concurrent}")

    # Input
    input_file = STEP5_FILE
    print(f"  Input:            {input_file}")

    # Load dataset
    print(f"\nLoading dataset: {input_file}")
    df = pd.read_parquet(input_file)
    total = len(df)
    print(f"Total examples: {fmt(total)}")

    # Determine field name
    model_short = args.model.split("/")[-1] if "/" in args.model else args.model
    field_name = f"nonsensical_{model_short}"

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
            "explanation": r.get("explanation", "error"),
            "nonsensical_query": r.get("nonsensical_query", "error"),
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
    classifications = [r.get("nonsensical_query", "error") for r in results]
    class_counts = pd.Series(classifications).value_counts()

    print(f"\nNonsensical/sensical distribution:")
    for cls, count in class_counts.items():
        pct = count / total * 100
        print(f"  {cls}: {fmt(count)} ({pct:.2f}%)")

    # Errors
    n_parse_err = sum(1 for c in classifications if c == "parse_error")
    n_api_err = sum(1 for c in classifications if c == "api_error")
    n_empty = sum(1 for r in results if r.get("explanation") == "empty_query")
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
    output_file = os.path.join(CLASSIFY_DIR, "wildchat_nonsensical.parquet")
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
    stats_file = os.path.join(OUTPUT_BASE, f"step6_classify_stats_{model_short}.json")
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
        "classification_distribution": {cls: int(count) for cls, count in class_counts.items()},
    }

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nStats saved to: {stats_file}")
    print(f"\nNonsensical classification complete.")
    print(f"  -> Run the filter: python filter_step6_nonsensical.py filter")


# ── Subcommand: filter ────────────────────────────────────────────────────────

def run_filter(args):
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   STEP 6b: NONSENSICAL FILTER — WILDCHAT                           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t_start = time.time()

    # Load classified dataset
    input_file = os.path.join(CLASSIFY_DIR, "wildchat_nonsensical.parquet")
    print(f"\nLoading dataset: {input_file}")
    df = pd.read_parquet(input_file)
    total = len(df)
    print(f"Total examples: {fmt(total)}")

    # Detect nonsensical column
    nonsensical_cols = sorted([c for c in df.columns if c.startswith("nonsensical_")])
    print(f"\nNonsensical columns found: {nonsensical_cols}")

    if not nonsensical_cols:
        print("\n  ERROR: No nonsensical column found.")
        print("  Run 'classify' before running 'filter'.")
        return

    nonsensical_field = nonsensical_cols[0]
    print(f"  Using field: {nonsensical_field}")

    # Extract nonsensical_query from each example
    print("\nExtracting nonsensical classification from each example...")
    classifications = []
    for entry in tqdm(df[nonsensical_field], desc="Extracting"):
        if isinstance(entry, dict):
            classifications.append(entry.get("nonsensical_query", ""))
        else:
            classifications.append("")
    df["_nonsensical"] = classifications

    # Count distribution
    class_counts = df["_nonsensical"].value_counts()
    print(f"\nDistribution:")
    for cls, count in class_counts.items():
        marker = " <- REMOVE" if cls in REMOVE_VALUES else ""
        print(f"  {cls}: {fmt(count)}{marker}")

    # Filter: remove nonsensical and parse_error
    to_remove = df["_nonsensical"].isin(REMOVE_VALUES)
    n_nonsensical = int((df["_nonsensical"] == "yes").sum())
    n_parse_error = int((df["_nonsensical"] == "parse_error").sum())

    df_out = df[~to_remove].drop(columns=["_nonsensical"]).reset_index(drop=True)

    kept = len(df_out)
    removed = total - kept
    pct_removed = removed / total * 100 if total > 0 else 0

    print(f"\n  Removed nonsensical:   {fmt(n_nonsensical)}")
    print(f"  Removed parse_error:   {fmt(n_parse_error)}")
    print(f"  Total removed:         {fmt(removed)} ({pct_removed:.2f}%)")
    print(f"  Kept:                  {fmt(kept)}")

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
    print("FINAL SUMMARY — STEP 6")
    print(f"{'=' * 70}")
    print(f"  Input examples:       {fmt(total)}")
    print(f"  Kept examples:        {fmt(kept)}")
    print(f"  Removed examples:     {fmt(removed)} ({pct_removed:.2f}%)")
    print(f"  Removed nonsensical:  {fmt(n_nonsensical)}")
    print(f"  Removed parse_error:  {fmt(n_parse_error)}")
    print(f"  Nonsensical field:    {nonsensical_field}")
    print(f"  Total time:           {total_time:.1f}s")
    print(f"  Output file:          {FILTER_OUTPUT} ({file_size_mb:.1f} MB)")

    # Stats file
    stats_file = os.path.join(OUTPUT_BASE, "step6_filter_stats.json")
    stats = {
        "timestamp": timestamp,
        "input_file": input_file,
        "output_file": FILTER_OUTPUT,
        "output_size_mb": round(file_size_mb, 1),
        "total_input": total,
        "total_kept": kept,
        "total_removed": removed,
        "pct_removed": round(pct_removed, 2),
        "nonsensical_field": nonsensical_field,
        "removed_nonsensical": n_nonsensical,
        "removed_parse_error": n_parse_error,
        "total_time_s": round(total_time, 1),
        "save_time_s": round(save_time, 1),
    }

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nStats saved to: {stats_file}")
    print("\nStep 6 complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 6: Nonsensical/sensical classification and filter"
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # classify
    p_classify = subparsers.add_parser("classify", help="Classify nonsensical/sensical with a single model")
    p_classify.add_argument("--base-url", required=True, help="API base URL")
    p_classify.add_argument("--model", required=True, help="Model name")
    p_classify.add_argument("--api-key", default="no-key", help="API key")
    p_classify.add_argument("--max-concurrent", type=int, default=50, help="Concurrent requests")

    # filter
    p_filter = subparsers.add_parser("filter", help="Filter nonsensical examples")

    args = parser.parse_args()

    if args.command == "classify":
        run_classify(args)
    elif args.command == "filter":
        run_filter(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
