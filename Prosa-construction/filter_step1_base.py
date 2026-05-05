#!/usr/bin/env python3
"""
Step 1 of the WildChat-4.8M filtering pipeline.

Applies the following filters in sequence, documenting each sub-step:
  1a. Language filter: language == "Portuguese"
  1b. Unique IP filter: drop_duplicates(hashed_ip, keep="first")
  1c. Country filter: country == "Brazil"
  1d. Turn filter: turn <= 5
  1e. Removal of the last assistant message (conversation[:-1])
  1f. Removal of examples with any message having empty content
  1g. Removal of examples flagged as redacted (PII)

Produces a single output parquet file.
"""

import os
import glob
import time
import json
from datetime import datetime
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────────
_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.environ.get("PROSA_OUTPUT_BASE", _REPO_DIR)
DATA_DIR = os.environ.get("WILDCHAT_DATA_DIR", os.path.join(_REPO_DIR, "wildchat_raw"))

STEP1_DIR = os.path.join(OUTPUT_BASE, "01_base_filtered")
STEP1_FILE = os.path.join(STEP1_DIR, "wildchat_base_filtered.parquet")

LANGUAGE_FILTER = "Portuguese"
COUNTRY_FILTER = "Brazil"
MAX_TURNS = 5


# ── Helpers ──────────────────────────────────────────────────────────────────
def fmt(n: int) -> str:
    """Format a number with thousand separators."""
    return f"{n:,}"


def pct(part: int, total: int) -> str:
    """Return a percentage string."""
    if total == 0:
        return "0.00%"
    return f"{part / total * 100:.2f}%"


# ── Sub-step 1a: Language filter ────────────────────────────────────────────
def step1a_filter_portuguese(parquet_files: list) -> tuple[pd.DataFrame, dict]:
    """Read all parquet files and filter by language == Portuguese."""
    print("=" * 70)
    print("SUB-STEP 1a: LANGUAGE FILTER (Portuguese)")
    print("=" * 70)

    total_rows = 0
    filtered_tables = []

    t0 = time.time()
    for i, fpath in enumerate(tqdm(parquet_files, desc="Reading parquets", unit="file")):
        table = pq.read_table(fpath)
        n_rows = table.num_rows
        total_rows += n_rows

        lang_col = table.column("language")
        mask = pa.compute.equal(lang_col, LANGUAGE_FILTER)
        filtered = table.filter(mask)

        if filtered.num_rows > 0:
            filtered_tables.append(filtered)

    elapsed = time.time() - t0

    if filtered_tables:
        combined = pa.concat_tables(filtered_tables)
    else:
        combined = pa.table({})

    df = combined.to_pandas()
    filtered_rows = len(df)
    removed = total_rows - filtered_rows

    stats = {
        "total_input": total_rows,
        "total_output": filtered_rows,
        "removed": removed,
        "pct_removed": pct(removed, total_rows),
        "pct_kept": pct(filtered_rows, total_rows),
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  Original total:   {fmt(total_rows)}")
    print(f"  Kept (PT):        {fmt(filtered_rows)} ({stats['pct_kept']})")
    print(f"  Removed:          {fmt(removed)} ({stats['pct_removed']})")
    print(f"  Time: {elapsed:.1f}s")

    return df, stats


# ── Sub-step 1b: Unique IP filter ───────────────────────────────────────────
def step1b_filter_unique_ips(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove duplicate IPs, keeping the first occurrence."""
    print("\n" + "=" * 70)
    print("SUB-STEP 1b: UNIQUE IP FILTER")
    print("=" * 70)

    total_rows = len(df)
    unique_ips_before = df["hashed_ip"].nunique()
    print(f"  Total examples:        {fmt(total_rows)}")
    print(f"  Unique hashed_ip:      {fmt(unique_ips_before)}")
    print(f"  Repeated hashed_ip:    {fmt(total_rows - unique_ips_before)}")

    t0 = time.time()
    df_filtered = df.drop_duplicates(subset="hashed_ip", keep="first").copy()
    elapsed = time.time() - t0

    filtered_rows = len(df_filtered)
    removed = total_rows - filtered_rows

    stats = {
        "total_input": total_rows,
        "total_output": filtered_rows,
        "removed": removed,
        "pct_removed": pct(removed, total_rows),
        "pct_kept": pct(filtered_rows, total_rows),
        "unique_ips_before": unique_ips_before,
        "unique_ips_after": df_filtered["hashed_ip"].nunique(),
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  Kept (unique IP):    {fmt(filtered_rows)} ({stats['pct_kept']})")
    print(f"  Removed:             {fmt(removed)} ({stats['pct_removed']})")
    print(f"  Time: {elapsed:.1f}s")

    return df_filtered, stats


# ── Sub-step 1c: Country filter ─────────────────────────────────────────────
def step1c_filter_brazil(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep only conversations originating from Brazil."""
    print("\n" + "=" * 70)
    print("SUB-STEP 1c: COUNTRY FILTER (Brazil)")
    print("=" * 70)

    total_rows = len(df)
    print(f"  Total examples: {fmt(total_rows)}")

    # Top country distribution before filtering
    country_dist = df["country"].value_counts().head(10)
    print(f"\n  Top 10 countries (before filter):")
    for country, count in country_dist.items():
        marker = " <--" if country == COUNTRY_FILTER else ""
        print(f"    {country:20s}: {fmt(count):>8s} ({pct(count, total_rows)}){marker}")

    t0 = time.time()
    df_filtered = df[df["country"] == COUNTRY_FILTER].copy()
    elapsed = time.time() - t0

    filtered_rows = len(df_filtered)
    removed = total_rows - filtered_rows

    stats = {
        "total_input": total_rows,
        "total_output": filtered_rows,
        "removed": removed,
        "pct_removed": pct(removed, total_rows),
        "pct_kept": pct(filtered_rows, total_rows),
        "country_distribution": {str(k): int(v) for k, v in country_dist.items()},
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  Kept (Brazil):     {fmt(filtered_rows)} ({stats['pct_kept']})")
    print(f"  Removed:           {fmt(removed)} ({stats['pct_removed']})")
    print(f"  Time: {elapsed:.1f}s")

    return df_filtered, stats


# ── Sub-step 1d: Turn filter ────────────────────────────────────────────────
def step1d_filter_max_turns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove conversations with more than MAX_TURNS turns."""
    print("\n" + "=" * 70)
    print(f"SUB-STEP 1d: MAXIMUM TURN FILTER (≤ {MAX_TURNS})")
    print("=" * 70)

    total_rows = len(df)
    print(f"  Total examples: {fmt(total_rows)}")

    # Turn distribution before filtering
    turn_dist = df["turn"].value_counts().sort_index()
    print(f"\n  Turn distribution (before filter):")
    for turn_val, count in turn_dist.items():
        marker = " ✓" if turn_val <= MAX_TURNS else " ✗"
        print(f"    {turn_val:>3d} turns: {fmt(count):>8s} ({pct(count, total_rows)}){marker}")

    t0 = time.time()
    df_filtered = df[df["turn"] <= MAX_TURNS].copy()
    elapsed = time.time() - t0

    filtered_rows = len(df_filtered)
    removed = total_rows - filtered_rows

    stats = {
        "total_input": total_rows,
        "total_output": filtered_rows,
        "removed": removed,
        "pct_removed": pct(removed, total_rows),
        "pct_kept": pct(filtered_rows, total_rows),
        "max_turns": MAX_TURNS,
        "turn_distribution": {int(k): int(v) for k, v in turn_dist.items()},
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  Kept (≤{MAX_TURNS} turns): {fmt(filtered_rows)} ({stats['pct_kept']})")
    print(f"  Removed:              {fmt(removed)} ({stats['pct_removed']})")
    print(f"  Time: {elapsed:.1f}s")

    return df_filtered, stats


# ── Sub-step 1e: Removal of last assistant message ──────────────────────────
def step1e_remove_last_assistant(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove the last (assistant) message from each conversation."""
    print("\n" + "=" * 70)
    print("SUB-STEP 1e: REMOVAL OF LAST ASSISTANT MESSAGE")
    print("=" * 70)

    total_rows = len(df)
    print(f"  Total examples: {fmt(total_rows)}")

    # Check that all end with assistant
    last_roles = df["conversation"].apply(lambda c: c[-1]["role"])
    ends_with_assistant = (last_roles == "assistant").sum()
    ends_with_other = total_rows - ends_with_assistant
    print(f"\n  End with 'assistant':  {fmt(ends_with_assistant)} ({pct(ends_with_assistant, total_rows)})")
    print(f"  End with other role:   {fmt(ends_with_other)}")

    # Count messages before
    total_msgs_before = df["conversation"].apply(len).sum()

    t0 = time.time()
    df["conversation"] = df["conversation"].apply(lambda c: c[:-1])
    elapsed = time.time() - t0

    # Count messages after
    total_msgs_after = df["conversation"].apply(len).sum()
    msgs_removed = total_msgs_before - total_msgs_after

    # Check that all now end with user
    last_roles_after = df["conversation"].apply(lambda c: c[-1]["role"])
    ends_with_user_after = (last_roles_after == "user").sum()

    # Message distribution after removal
    msg_dist = df["conversation"].apply(len).value_counts().sort_index()
    print(f"\n  Messages per conversation (after removal):")
    for n_msgs, count in msg_dist.items():
        print(f"    {n_msgs:>2d} msgs: {fmt(count):>8s} ({pct(count, total_rows)})")

    stats = {
        "total_input": total_rows,
        "total_output": total_rows,
        "ends_with_assistant": int(ends_with_assistant),
        "ends_with_user_after": int(ends_with_user_after),
        "total_msgs_before": int(total_msgs_before),
        "total_msgs_after": int(total_msgs_after),
        "msgs_removed": int(msgs_removed),
        "avg_msgs_before": round(total_msgs_before / total_rows, 2),
        "avg_msgs_after": round(total_msgs_after / total_rows, 2),
        "msg_distribution": {int(k): int(v) for k, v in msg_dist.items()},
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  Examples:              {fmt(total_rows)} (unchanged)")
    print(f"  Messages removed:      {fmt(msgs_removed)}")
    print(f"  Avg msgs/conversation: {stats['avg_msgs_before']} → {stats['avg_msgs_after']}")
    print(f"  End with 'user':       {fmt(ends_with_user_after)} ({pct(ends_with_user_after, total_rows)})")
    print(f"  Time: {elapsed:.1f}s")

    return df, stats


# ── Sub-step 1f: Removal of examples with empty message content ─────────────
def step1f_remove_empty_content(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove examples where any message has empty or None content."""
    print("\n" + "=" * 70)
    print("SUB-STEP 1f: REMOVAL OF EXAMPLES WITH EMPTY MESSAGE CONTENT")
    print("=" * 70)

    total_rows = len(df)
    print(f"  Total examples: {fmt(total_rows)}")

    t0 = time.time()

    def has_empty_content(conversation) -> bool:
        if conversation is None:
            return True
        if isinstance(conversation, np.ndarray):
            conversation = conversation.tolist()
        if not isinstance(conversation, list) or len(conversation) == 0:
            return True
        for turn in conversation:
            if not isinstance(turn, dict):
                continue
            content = turn.get("content", "")
            if content == "" or content is None:
                return True
        return False

    mask_empty = df["conversation"].apply(has_empty_content)
    n_with_empty = int(mask_empty.sum())

    df_filtered = df[~mask_empty].copy()
    elapsed = time.time() - t0

    filtered_rows = len(df_filtered)
    removed = total_rows - filtered_rows

    stats = {
        "total_input": total_rows,
        "total_output": filtered_rows,
        "removed": removed,
        "pct_removed": pct(removed, total_rows),
        "pct_kept": pct(filtered_rows, total_rows),
        "n_with_empty_content": n_with_empty,
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  Examples with empty content: {fmt(n_with_empty)}")
    print(f"  Kept:                       {fmt(filtered_rows)} ({stats['pct_kept']})")
    print(f"  Removed:                    {fmt(removed)} ({stats['pct_removed']})")
    print(f"  Time: {elapsed:.1f}s")

    return df_filtered, stats


# ── Sub-step 1g: Removal of redacted examples (PII) ─────────────────────────
def step1g_filter_redacted(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove conversations flagged as redacted (PII detected/anonymized by WildChat)."""
    print("\n" + "=" * 70)
    print("SUB-STEP 1g: REDACTED EXAMPLES FILTER (PII)")
    print("=" * 70)

    total_rows = len(df)
    print(f"  Total examples: {fmt(total_rows)}")

    # Distribution
    redacted_counts = df["redacted"].value_counts()
    for val, count in redacted_counts.items():
        marker = " <- REMOVE" if val else ""
        print(f"    redacted={val}: {fmt(count)} ({pct(count, total_rows)}){marker}")

    t0 = time.time()
    is_redacted = df["redacted"] == True
    df_filtered = df[~is_redacted].copy()
    elapsed = time.time() - t0

    filtered_rows = len(df_filtered)
    removed = total_rows - filtered_rows

    stats = {
        "total_input": total_rows,
        "total_output": filtered_rows,
        "removed": removed,
        "pct_removed": pct(removed, total_rows),
        "pct_kept": pct(filtered_rows, total_rows),
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  Kept (non-redacted):     {fmt(filtered_rows)} ({stats['pct_kept']})")
    print(f"  Removed:                 {fmt(removed)} ({stats['pct_removed']})")
    print(f"  Time: {elapsed:.1f}s")

    return df_filtered, stats

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   FILTERING PIPELINE — WILDCHAT-4.8M -> PORTUGUESE (BRAZIL)       ║")
    print("║   Step 1: Base Filtering                                           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    os.makedirs(STEP1_DIR, exist_ok=True)

    t_total = time.time()

    # Discover parquet files
    parquet_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    num_parquet_files = len(parquet_files)
    print(f"Parquet files found: {num_parquet_files}")
    print()

    # Run sub-steps
    df, stats_1a = step1a_filter_portuguese(parquet_files)
    df, stats_1b = step1b_filter_unique_ips(df)
    df, stats_1c = step1c_filter_brazil(df)
    df, stats_1d = step1d_filter_max_turns(df)
    df, stats_1e = step1e_remove_last_assistant(df)
    df, stats_1f = step1f_remove_empty_content(df)
    df, stats_1g = step1g_filter_redacted(df)

    # Save final result
    total_final = len(df)
    print("\n" + "=" * 70)
    print("SAVING FINAL RESULT")
    print("=" * 70)
    print(f"  Saving {fmt(total_final)} examples to: {STEP1_FILE}")
    table_out = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table_out, STEP1_FILE, compression="snappy")
    print("  Saved successfully.")

    total_elapsed = time.time() - t_total

    # Final summary
    total_original = stats_1a["total_input"]
    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE")
    print("=" * 70)
    print(f"  Total time:            {total_elapsed:.1f}s")
    print(f"  Original dataset:      {fmt(total_original)} examples")
    print(f"  -> 1a. Portuguese:    {fmt(stats_1a['total_output'])} examples")
    print(f"  -> 1b. Unique IP:     {fmt(stats_1b['total_output'])} examples")
    print(f"  -> 1c. Brazil:        {fmt(stats_1c['total_output'])} examples")
    print(f"  -> 1d. <={MAX_TURNS} turns:      {fmt(stats_1d['total_output'])} examples")
    print(f"  -> 1e. No assistant:  {fmt(stats_1e['total_output'])} examples (msgs removed: {fmt(stats_1e['msgs_removed'])})")
    print(f"  -> 1f. No empty content: {fmt(stats_1f['total_output'])} examples")
    print(f"  -> 1g. No PII (redacted): {fmt(stats_1g['total_output'])} examples")
    print(f"  Final result:          {fmt(total_final)} examples")
    print(f"  File:                  {STEP1_FILE}")


if __name__ == "__main__":
    main()
