#!/usr/bin/env python3
"""
Step 2 of the WildChat-4.8M filtering pipeline.

Reads the dataset from Step 1 and:
  2a. Tokenizes each message using Qwen/Qwen3-Embedding-8B (adds n_tokens per msg)
  2b. Filters by total token range (sum of n_tokens across all msgs): [MIN_TOKENS, MAX_TOKENS]

Saves the result and prints detailed statistics.
"""

import os
import time
import json
from datetime import datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# ── Configuration ────────────────────────────────────────────────────────────
OUTPUT_BASE = os.environ.get("PROSA_OUTPUT_BASE", os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_NAME = "Qwen/Qwen3-Embedding-8B"

STEP1_FILE = os.path.join(OUTPUT_BASE, "01_base_filtered", "wildchat_base_filtered.parquet")
STEP2_DIR = os.path.join(OUTPUT_BASE, "02_tokenized_filtered")
STEP2_FILE = os.path.join(STEP2_DIR, "wildchat_tokenized_filtered.parquet")

STATS_FILE = os.path.join(OUTPUT_BASE, "step2_stats.json")

MIN_TOKENS = 50
MAX_TOKENS = 8192


# ── Helpers ──────────────────────────────────────────────────────────────────
def fmt(n) -> str:
    if isinstance(n, float):
        return f"{n:,.1f}"
    return f"{n:,}"


def pct(part: int, total: int) -> str:
    if total == 0:
        return "0.00%"
    return f"{part / total * 100:.2f}%"


def compute_stats(arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return {k: 0 for k in ["mean", "median", "std", "min", "max", "p25", "p75", "p95", "p99"]}
    return {
        "mean": round(float(arr.mean()), 1),
        "median": round(float(np.median(arr)), 1),
        "std": round(float(arr.std()), 1),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "p25": round(float(np.percentile(arr, 25)), 1),
        "p75": round(float(np.percentile(arr, 75)), 1),
        "p95": round(float(np.percentile(arr, 95)), 1),
        "p99": round(float(np.percentile(arr, 99)), 1),
    }


def print_stats(label: str, stats: dict):
    print(f"\n  {label}:")
    print(f"    Mean:    {fmt(stats['mean'])}")
    print(f"    Median:  {fmt(stats['median'])}")
    print(f"    Std:     {fmt(stats['std'])}")
    print(f"    Min:     {fmt(stats['min'])}")
    print(f"    Max:     {fmt(stats['max'])}")
    print(f"    P25:     {fmt(stats['p25'])}")
    print(f"    P75:     {fmt(stats['p75'])}")
    print(f"    P95:     {fmt(stats['p95'])}")
    print(f"    P99:     {fmt(stats['p99'])}")


# ── Sub-step 2a: Per-message tokenization ───────────────────────────────────
def step2a_tokenize_messages(df: pd.DataFrame, tokenizer) -> tuple[pd.DataFrame, dict]:
    """Tokenize each message and add the n_tokens field."""
    print("=" * 70)
    print("SUB-STEP 2a: MESSAGE TOKENIZATION")
    print("=" * 70)

    total_rows = len(df)
    total_msgs = df["conversation"].apply(len).sum()
    print(f"  Total examples: {fmt(total_rows)}")
    print(f"  Total messages to tokenize: {fmt(total_msgs)}")

    print("\nTokenizing messages...")
    t0 = time.time()

    all_token_counts = []
    user_token_counts = []
    assistant_token_counts = []
    conv_total_tokens = []

    for idx in tqdm(range(len(df)), desc="Tokenizing messages", unit="conv"):
        conversation = df.iloc[idx]["conversation"]
        new_conversation = []
        conv_sum = 0
        for msg in conversation:
            content = msg.get("content") or ""
            tokens = tokenizer.encode(content, add_special_tokens=False)
            n_tokens = len(tokens)
            all_token_counts.append(n_tokens)
            conv_sum += n_tokens
            if msg["role"] == "user":
                user_token_counts.append(n_tokens)
            elif msg["role"] == "assistant":
                assistant_token_counts.append(n_tokens)
            new_msg = dict(msg)
            new_msg["n_tokens"] = n_tokens
            new_conversation.append(new_msg)
        conv_total_tokens.append(conv_sum)
        df.at[df.index[idx], "conversation"] = new_conversation

    elapsed = time.time() - t0

    # Store per-conversation token sum for use in the filter
    df["total_tokens"] = conv_total_tokens

    all_arr = np.array(all_token_counts)
    user_arr = np.array(user_token_counts)
    asst_arr = np.array(assistant_token_counts) if assistant_token_counts else np.array([0])
    conv_arr = np.array(conv_total_tokens)

    print(f"\nTokenization complete in {elapsed:.1f}s")
    print(f"  Messages tokenized: {fmt(len(all_token_counts))}")
    print(f"    - User:      {fmt(len(user_token_counts))}")
    print(f"    - Assistant:  {fmt(len(assistant_token_counts))}")
    print(f"  Total tokens: {fmt(int(all_arr.sum()))}")

    print_stats("Tokens per message (all)", compute_stats(all_arr))
    print_stats("Tokens per message (user)", compute_stats(user_arr))
    if len(assistant_token_counts) > 0:
        print_stats("Tokens per message (assistant)", compute_stats(asst_arr))
    print_stats("Total tokens per conversation (sum of n_tokens)", compute_stats(conv_arr))

    stats = {
        "total_msgs_tokenized": len(all_token_counts),
        "total_tokens": int(all_arr.sum()),
        "user_msgs": len(user_token_counts),
        "assistant_msgs": len(assistant_token_counts),
        "stats_all_msgs": compute_stats(all_arr),
        "stats_user_msgs": compute_stats(user_arr),
        "stats_assistant_msgs": compute_stats(asst_arr) if assistant_token_counts else {},
        "stats_conv_total": compute_stats(conv_arr),
        "elapsed_seconds": round(elapsed, 1),
    }

    return df, stats


# ── Sub-step 2b: Token range filter ─────────────────────────────────────────
def step2b_filter_token_range(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Filter conversations by the total sum of n_tokens across all messages."""
    print("\n" + "=" * 70)
    print(f"SUB-STEP 2b: TOTAL TOKEN RANGE FILTER [{MIN_TOKENS}, {fmt(MAX_TOKENS)}]")
    print("=" * 70)

    total_rows = len(df)
    print(f"  Total examples: {fmt(total_rows)}")

    t0 = time.time()

    tok_arr = np.array(df["total_tokens"].tolist())

    # Statistics before filtering
    stats_before = compute_stats(tok_arr)
    print_stats("Total tokens per conversation (before filter)", stats_before)

    # Bucketed distribution (before)
    print(f"\n  Bucketed distribution (before filter):")
    bins = [0, 50, 100, 250, 500, 1000, 2000, 4000, 8192, 16000, float("inf")]
    labels = ["0-49", "50-99", "100-249", "250-499", "500-999", "1000-1999", "2000-3999", "4000-8192", "8193-16000", "16000+"]
    for i in range(len(bins) - 1):
        count = int(((tok_arr >= bins[i]) & (tok_arr < bins[i + 1])).sum())
        p = count / len(tok_arr) * 100
        bar = "█" * int(p / 2)
        print(f"    {labels[i]:>12s}: {fmt(count):>6s} ({p:5.1f}%) {bar}")

    # Count removed
    below_min = int((tok_arr < MIN_TOKENS).sum())
    above_max = int((tok_arr > MAX_TOKENS).sum())

    print(f"\n  Applying filters:")
    print(f"    Removed for < {MIN_TOKENS} tokens:    {fmt(below_min)} ({pct(below_min, total_rows)})")
    print(f"    Removed for > {fmt(MAX_TOKENS)} tokens: {fmt(above_max)} ({pct(above_max, total_rows)})")

    # Filter
    mask = (df["total_tokens"] >= MIN_TOKENS) & (df["total_tokens"] <= MAX_TOKENS)
    df_filtered = df[mask].copy()

    elapsed = time.time() - t0

    filtered_rows = len(df_filtered)
    removed = total_rows - filtered_rows

    # Statistics after filter
    tok_after = np.array(df_filtered["total_tokens"].tolist())
    stats_after = compute_stats(tok_after)

    print_stats("Total tokens per conversation (after filter)", stats_after)

    # Remove auxiliary column
    df_filtered = df_filtered.drop(columns=["total_tokens"])

    stats = {
        "total_input": total_rows,
        "total_output": filtered_rows,
        "removed": removed,
        "removed_below": below_min,
        "removed_above": above_max,
        "pct_removed": pct(removed, total_rows),
        "pct_kept": pct(filtered_rows, total_rows),
        "min_tokens": MIN_TOKENS,
        "max_tokens": MAX_TOKENS,
        "stats_before": stats_before,
        "stats_after": stats_after,
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  Total input:      {fmt(total_rows)}")
    print(f"  Kept:             {fmt(filtered_rows)} ({stats['pct_kept']})")
    print(f"  Removed:          {fmt(removed)} ({stats['pct_removed']})")
    print(f"    - short (<{MIN_TOKENS}):     {fmt(below_min)}")
    print(f"    - long (>{fmt(MAX_TOKENS)}):  {fmt(above_max)}")
    print(f"  Time: {elapsed:.1f}s")

    return df_filtered, stats


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   STEP 2 — TOKENIZATION AND TOKEN FILTER (WILDCHAT)                ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    os.makedirs(STEP2_DIR, exist_ok=True)

    t_total = time.time()

    # Load dataset
    print(f"Loading dataset from: {STEP1_FILE}")
    df = pd.read_parquet(STEP1_FILE)
    print(f"Total examples: {fmt(len(df))}\n")

    # Load tokenizer
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    print("Tokenizer loaded.\n")

    # Sub-steps
    df, stats_2a = step2a_tokenize_messages(df, tokenizer)
    df, stats_2b = step2b_filter_token_range(df)

    # Save final result
    total_final = len(df)
    print("\n" + "=" * 70)
    print("SAVING FINAL RESULT")
    print("=" * 70)
    print(f"  Saving {fmt(total_final)} examples to: {STEP2_FILE}")
    table_out = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table_out, STEP2_FILE, compression="snappy")
    print("  Saved successfully.")

    # Verification
    df_check = pd.read_parquet(STEP2_FILE)
    sample_msg = df_check["conversation"].iloc[0][0]
    assert "n_tokens" in sample_msg, "n_tokens field not found in verification!"
    print(f"  Verification OK: 'n_tokens' field present (sample value: {sample_msg['n_tokens']})")

    total_elapsed = time.time() - t_total

    # Save stats to JSON for later use in the report update
    all_stats = {
        "stats_2a": stats_2a,
        "stats_2b": stats_2b,
        "total_elapsed": round(total_elapsed, 1),
        "tokenizer": TOKENIZER_NAME,
        "output_file": STEP2_FILE,
    }
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"  Stats saved to: {STATS_FILE}")

    # Final summary
    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE")
    print("=" * 70)
    print(f"  Total time:               {total_elapsed:.1f}s")
    print(f"  Tokenizer:                {TOKENIZER_NAME}")
    print(f"  -> 2a. Tokenization:     {fmt(stats_2a['total_msgs_tokenized'])} msgs, {fmt(stats_2a['total_tokens'])} tokens")
    print(f"  -> 2b. Filter [{MIN_TOKENS}, {fmt(MAX_TOKENS)}]: {fmt(stats_2b['total_input'])} -> {fmt(stats_2b['total_output'])} examples")
    print(f"       Removed:             {fmt(stats_2b['removed'])} ({stats_2b['pct_removed']})")
    print(f"         - short (<{MIN_TOKENS}):    {fmt(stats_2b['removed_below'])}")
    print(f"         - long (>{fmt(MAX_TOKENS)}): {fmt(stats_2b['removed_above'])}")
    print(f"  Final result:             {fmt(total_final)} examples")
    print(f"  File:                     {STEP2_FILE}")


if __name__ == "__main__":
    main()
