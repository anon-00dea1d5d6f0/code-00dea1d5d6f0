#!/usr/bin/env python3
"""
Step 7: Random sampling of 1000 examples.

Randomly samples 1000 examples from the final dataset (step 6).

Usage:
    python filter_step7_sample_1000.py
"""

import os
import json
import time
from datetime import datetime

import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
OUTPUT_BASE = os.environ.get("PROSA_OUTPUT_BASE", os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(OUTPUT_BASE, "06_nonsensical_filtered", "wildchat_nonsensical_filtered.parquet")
OUTPUT_DIR = os.path.join(OUTPUT_BASE, "07_sample_1000")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "wildchat_sample_1000.parquet")
STATS_FILE = os.path.join(OUTPUT_BASE, "step7_sample_1000_stats.json")

SAMPLE_SIZE = 1000
RANDOM_SEED = 42


def fmt(n: int) -> str:
    return f"{n:,}"


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   STEP 7: RANDOM SAMPLING OF 1000 EXAMPLES — WILDCHAT             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t_start = time.time()

    # Load dataset
    print(f"\nLoading dataset: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    total = len(df)
    print(f"Total examples: {fmt(total)}")

    # Sample
    if total <= SAMPLE_SIZE:
        print(f"\nDataset has {fmt(total)} examples (<= {SAMPLE_SIZE}), using all.")
        df_out = df
        sampled = total
    else:
        print(f"\nSampling {fmt(SAMPLE_SIZE)} examples (seed={RANDOM_SEED})...")
        df_out = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
        sampled = SAMPLE_SIZE

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nSaving dataset: {OUTPUT_FILE}")
    t_save = time.time()
    df_out.to_parquet(OUTPUT_FILE, engine="pyarrow", compression="snappy", index=False)
    save_time = time.time() - t_save

    file_size_mb = os.path.getsize(OUTPUT_FILE) / 1024**2
    total_time = time.time() - t_start

    print(f"  Saved in {save_time:.1f}s ({file_size_mb:.1f} MB)")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Input:            {fmt(total)}")
    print(f"  Sampled:          {fmt(sampled)}")
    print(f"  Seed:             {RANDOM_SEED}")
    print(f"  Total time:       {total_time:.1f}s")
    print(f"  File:             {OUTPUT_FILE} ({file_size_mb:.1f} MB)")

    # Stats file
    stats = {
        "timestamp": timestamp,
        "input_file": INPUT_FILE,
        "output_file": OUTPUT_FILE,
        "output_size_mb": round(file_size_mb, 1),
        "total_input": total,
        "total_sampled": sampled,
        "sample_size": SAMPLE_SIZE,
        "random_seed": RANDOM_SEED,
        "total_time_s": round(total_time, 1),
        "save_time_s": round(save_time, 1),
    }

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nStats saved to: {STATS_FILE}")
    print("\nStep 7 complete.")


if __name__ == "__main__":
    main()
