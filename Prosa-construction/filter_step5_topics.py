#!/usr/bin/env python3
"""
Step 5: Topic classification and "Others" filter.

Sub-steps:
  5a. Topic classification — uses an LLM via OpenAI-compatible API to
      classify the topic of the user's last query. Uses the conversation
      history (when present) as context.

  5b. Filter — removes examples whose primary_tag is "Others".

Usage:
    # 5a — Classification
    python filter_step5_topics.py classify \
        --base-url https://api.openai.com/v1 \
        --model gpt-4.1-2025-04-14 \
        --api-key $OPENAI_API_KEY \
        --max-concurrent 50

    # 5b — Filter
    python filter_step5_topics.py filter

Produces step5_stats.json with statistics for the report update.
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
STEP4_FILE = os.path.join(OUTPUT_BASE, "04_difficulty_filtered", "wildchat_difficulty_filtered.parquet")
CLASSIFY_DIR = os.path.join(OUTPUT_BASE, "05_topics")
FILTER_DIR = os.path.join(OUTPUT_BASE, "05_topics_filtered")
FILTER_OUTPUT = os.path.join(FILTER_DIR, "wildchat_topics_filtered.parquet")

REMOVE_TAG = "Others"


# ── Prompt templates ─────────────────────────────────────────────────────────

PROMPT_WITH_HISTORY = """# Instruction
Please label the task tags for the user query based on the content of the user query and the conversation history.

## History
'''{user_history}'''

## User Query
'''{input}'''

## Tagging the user input
Please label the task tags for the user query. You will need to analyze the user query and the conversation history
and select the most relevant task tag from the list below.
all_task_tags = [
"Information seeking", # Users ask for specific information or facts about various topics.
"Reasoning", # Queries require logical thinking, problem−solving, or processing of
complex ideas.
"Planning", # Users need assistance in creating plans or strategies for activities and
projects.
"Editing", # Involves editing, rephrasing, proofreading, or other tasks related to the
composition of general written content.
"Coding & Debugging", # Users seek help with writing, reviewing, or fixing code in
programming.
"Math", # Queries related to mathematical concepts, problems, and calculations.
"Role playing", # Users engage in scenarios requiring ChatGPT to adopt a character or
persona.
"Data analysis", # Requests involve interpreting data, statistics, or performing analytical
tasks.
"Creative writing", # Users seek assistance with crafting stories, poems, or other
creative texts.
"Advice seeking", # Users ask for recommendations or guidance on various personal or
professional issues.
"Brainstorming", # Involves generating ideas, creative thinking, or exploring possibilities.
"Others" # Any queries that do not fit into the above categories or are of a miscellaneous
nature.
]
## Output Format:
Note that you can only select a single primary tag. Other applicable tags can be added to
the list of other tags.
Now, please output your tags below in a json format by filling in the placeholders in <...>:
'''
{{
"primary_tag": "<primary tag>",
"other_tags": ["<tag 1>", "<tag 2>", ... ]
}}
'''"""

PROMPT_WITHOUT_HISTORY = """# Instruction
Please label the task tags for the user query based on the content of the user query.

## User Query
'''{input}'''

## Tagging the user input
Please label the task tags for the user query. You will need to analyze the user query
and select the most relevant task tag from the list below.
all_task_tags = [
"Information seeking", # Users ask for specific information or facts about various topics.
"Reasoning", # Queries require logical thinking, problem−solving, or processing of
complex ideas.
"Planning", # Users need assistance in creating plans or strategies for activities and
projects.
"Editing", # Involves editing, rephrasing, proofreading, or other tasks related to the
composition of general written content.
"Coding & Debugging", # Users seek help with writing, reviewing, or fixing code in
programming.
"Math", # Queries related to mathematical concepts, problems, and calculations.
"Role playing", # Users engage in scenarios requiring ChatGPT to adopt a character or
persona.
"Data analysis", # Requests involve interpreting data, statistics, or performing analytical
tasks.
"Creative writing", # Users seek assistance with crafting stories, poems, or other
creative texts.
"Advice seeking", # Users ask for recommendations or guidance on various personal or
professional issues.
"Brainstorming", # Involves generating ideas, creative thinking, or exploring possibilities.
"Others" # Any queries that do not fit into the above categories or are of a miscellaneous
nature.
]
## Output Format:
Note that you can only select a single primary tag. Other applicable tags can be added to
the list of other tags.
Now, please output your tags below in a json format by filling in the placeholders in <...>:
'''
{{
"primary_tag": "<primary tag>",
"other_tags": ["<tag 1>", "<tag 2>", ... ]
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
        r'(\{[^{}]*"primary_tag"[^{}]*"other_tags"[^{}]*\})',
        r'(\{[^{}]*"primary_tag"[^{}]*\})',
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    return {
        "primary_tag": "parse_error",
        "other_tags": [],
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
    """Classify the topic of a query via API."""
    if not query or not query.strip():
        return {
            "primary_tag": "empty_query",
            "other_tags": [],
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
                        "primary_tag": "api_error",
                        "other_tags": [],
                        "error": str(e)[:200],
                    }

    return {
        "primary_tag": "unknown_error",
        "other_tags": [],
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
    print("║   STEP 5a: TOPIC CLASSIFICATION — WILDCHAT                         ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t_start = time.time()

    # Configuration
    print(f"\n  Model:            {args.model}")
    print(f"  API URL:          {args.base_url}")
    print(f"  Max concurrent:   {args.max_concurrent}")

    # Input
    input_file = STEP4_FILE
    print(f"  Input:            {input_file}")

    # Load dataset
    print(f"\nLoading dataset: {input_file}")
    df = pd.read_parquet(input_file)
    total = len(df)
    print(f"Total examples: {fmt(total)}")

    # Determine field name
    model_short = args.model.split("/")[-1] if "/" in args.model else args.model
    field_name = f"topic_{model_short}"

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
            "primary_tag": r.get("primary_tag", "error"),
            "other_tags": r.get("other_tags", []),
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
    tags = [r.get("primary_tag", "error") for r in results]
    tag_counts = pd.Series(tags).value_counts()

    print(f"\nTopic distribution:")
    for tag, count in tag_counts.items():
        pct = count / total * 100
        print(f"  {tag}: {fmt(count)} ({pct:.2f}%)")

    # Errors
    n_parse_err = sum(1 for t in tags if t == "parse_error")
    n_api_err = sum(1 for t in tags if t == "api_error")
    n_empty = sum(1 for t in tags if t == "empty_query")
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
    output_file = os.path.join(CLASSIFY_DIR, "wildchat_topics.parquet")
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
    stats_file = os.path.join(OUTPUT_BASE, f"step5_classify_stats_{model_short}.json")
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
        "topic_distribution": {tag: int(count) for tag, count in tag_counts.items()},
    }

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nStats saved to: {stats_file}")
    print(f"\nTopic classification complete.")
    print(f"  -> Run the filter: python filter_step5_topics.py filter")


# ── Subcommand: filter ────────────────────────────────────────────────────────

def run_filter(args):
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   STEP 5b: 'OTHERS' TOPIC FILTER — WILDCHAT                        ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t_start = time.time()

    # Load classified dataset
    input_file = os.path.join(CLASSIFY_DIR, "wildchat_topics.parquet")
    print(f"\nLoading dataset: {input_file}")
    df = pd.read_parquet(input_file)
    total = len(df)
    print(f"Total examples: {fmt(total)}")

    # Detect topic column
    topic_cols = sorted([c for c in df.columns if c.startswith("topic_")])
    print(f"\nTopic columns found: {topic_cols}")

    if not topic_cols:
        print("\n  ERROR: No topic column found.")
        print("  Run 'classify' before running 'filter'.")
        return

    topic_field = topic_cols[0]
    print(f"  Using field: {topic_field}")

    # Extract primary_tag from each example
    print("\nExtracting primary_tag from each example...")
    primary_tags = []
    for topic in tqdm(df[topic_field], desc="Extracting"):
        if isinstance(topic, dict):
            primary_tags.append(topic.get("primary_tag", ""))
        else:
            primary_tags.append("")
    df["_primary_tag"] = primary_tags

    # Count distribution
    tag_counts = df["_primary_tag"].value_counts()
    print(f"\nTopic distribution:")
    for tag, count in tag_counts.items():
        marker = " <- REMOVE" if tag == REMOVE_TAG else ""
        print(f"  {tag}: {fmt(count)}{marker}")

    # Filter: remove examples with primary_tag == "Others"
    is_others = df["_primary_tag"] == REMOVE_TAG
    n_others = int(is_others.sum())

    df_out = df[~is_others].drop(columns=["_primary_tag"]).reset_index(drop=True)

    kept = len(df_out)
    removed = total - kept
    pct_removed = removed / total * 100 if total > 0 else 0

    print(f"\n  Removed tag:  \"{REMOVE_TAG}\"")
    print(f"  Kept:         {fmt(kept)}")
    print(f"  Removed:      {fmt(removed)} ({pct_removed:.2f}%)")

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
    print("FINAL SUMMARY — STEP 5")
    print(f"{'=' * 70}")
    print(f"  Input examples:       {fmt(total)}")
    print(f"  Kept examples:        {fmt(kept)}")
    print(f"  Removed examples:     {fmt(removed)} ({pct_removed:.2f}%)")
    print(f"  Removed tag:          \"{REMOVE_TAG}\"")
    print(f"  Topic field:          {topic_field}")
    print(f"  Total time:           {total_time:.1f}s")
    print(f"  Output file:          {FILTER_OUTPUT} ({file_size_mb:.1f} MB)")

    # Stats file
    stats_file = os.path.join(OUTPUT_BASE, "step5_filter_stats.json")
    stats = {
        "timestamp": timestamp,
        "input_file": input_file,
        "output_file": FILTER_OUTPUT,
        "output_size_mb": round(file_size_mb, 1),
        "total_input": total,
        "total_kept": kept,
        "total_removed": removed,
        "pct_removed": round(pct_removed, 2),
        "topic_field": topic_field,
        "removed_tag": REMOVE_TAG,
        "total_time_s": round(total_time, 1),
        "save_time_s": round(save_time, 1),
    }

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nStats saved to: {stats_file}")
    print("\nStep 5 complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Topic classification and 'Others' filter"
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # classify
    p_classify = subparsers.add_parser("classify", help="Classify topics with a single model")
    p_classify.add_argument("--base-url", required=True, help="API base URL")
    p_classify.add_argument("--model", required=True, help="Model name")
    p_classify.add_argument("--api-key", default="no-key", help="API key")
    p_classify.add_argument("--max-concurrent", type=int, default=50, help="Concurrent requests")

    # filter
    p_filter = subparsers.add_parser("filter", help="Filter by 'Others' topic")

    args = parser.parse_args()

    if args.command == "classify":
        run_classify(args)
    elif args.command == "filter":
        run_filter(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
