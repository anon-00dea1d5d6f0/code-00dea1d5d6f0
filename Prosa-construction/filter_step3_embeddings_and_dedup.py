#!/usr/bin/env python3
"""
Step 3: Embedding extraction and deduplication via cosine similarity.

Sub-steps:
  3a. Embedding extraction — uses Qwen/Qwen3-Embedding-8B to generate one
      embedding vector per conversation. Builds a single string per
      conversation following the format:
        USER: <msg1>\n\nASSISTANT: <msg2>\n\n...  + <last message>

  3b. Greedy deduplication — computes the cosine similarity matrix
      across all examples and removes semantic duplicates with threshold 0.90.
      The resulting dataset keeps only one "copy" of each group of
      similar examples.

Produces step3_stats.json with statistics for the report update.
"""

import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────────
OUTPUT_BASE = os.environ.get("PROSA_OUTPUT_BASE", os.path.dirname(os.path.abspath(__file__)))
STEP2_FILE = os.path.join(OUTPUT_BASE, "02_tokenized_filtered", "wildchat_tokenized_filtered.parquet")
OUTPUT_DIR = os.path.join(OUTPUT_BASE, "03_embeddings_dedup")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "wildchat_embeddings_dedup.parquet")
STATS_FILE = os.path.join(OUTPUT_BASE, "step3_stats.json")
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
BATCH_SIZE = 12
SIMILARITY_THRESHOLD = 0.90


def fmt(n: int) -> str:
    return f"{n:,}"


def build_conversation_string(conversation: list) -> str:
    """Build a single string representing the conversation."""
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
    return history + last_query


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   STEP 3: EMBEDDINGS + DEDUPLICATION — WILDCHAT                    ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t_global = time.time()

    # ── Hardware ─────────────────────────────────────────────────────────────
    n_gpus = torch.cuda.device_count()
    gpu_info = []
    print(f"\nAvailable GPUs: {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1024**3
        info = f"{props.name} ({mem_gb:.1f} GB)"
        gpu_info.append(info)
        print(f"  GPU {i}: {info}")

    cuda_version = torch.version.cuda or "N/A"
    torch_version = torch.__version__
    print(f"  CUDA: {cuda_version}")
    print(f"  PyTorch: {torch_version}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    print(f"\nLoading dataset: {STEP2_FILE}")
    df = pd.read_parquet(STEP2_FILE)
    total_input = len(df)
    print(f"Total examples: {fmt(total_input)}")

    # Message distribution
    turn_dist = df["conversation"].apply(len).value_counts().sort_index()
    print("\nMessages per conversation distribution:")
    for n_msgs, count in turn_dist.items():
        print(f"  {n_msgs} msgs: {fmt(count)}")

    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  3a. EMBEDDING EXTRACTION")
    print("─" * 70)

    # ── String construction ──────────────────────────────────────────────────
    print("\nBuilding conversation strings...")
    texts = []
    for idx in tqdm(range(total_input), desc="Preparing texts", unit="conv"):
        conv = df.iloc[idx]["conversation"]
        texts.append(build_conversation_string(conv))

    char_lens = np.array([len(t) for t in texts])
    char_stats = {
        "mean": round(float(char_lens.mean()), 1),
        "median": round(float(np.median(char_lens)), 1),
        "min": int(char_lens.min()),
        "max": int(char_lens.max()),
        "p25": round(float(np.percentile(char_lens, 25)), 1),
        "p75": round(float(np.percentile(char_lens, 75)), 1),
        "p95": round(float(np.percentile(char_lens, 95)), 1),
    }
    print(f"  Mean (chars):    {fmt(int(char_stats['mean']))}")
    print(f"  Median (chars):  {fmt(int(char_stats['median']))}")
    print(f"  Min (chars):     {fmt(char_stats['min'])}")
    print(f"  Max (chars):     {fmt(char_stats['max'])}")
    print(f"  P95 (chars):     {fmt(int(char_stats['p95']))}")

    # ── Model ────────────────────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    import sentence_transformers

    st_version = sentence_transformers.__version__

    print(f"\nLoading model: {MODEL_NAME}")
    t_load = time.time()
    model = SentenceTransformer(
        MODEL_NAME,
        trust_remote_code=True,
    )
    load_time = time.time() - t_load

    max_seq_length = model.max_seq_length
    embedding_dim = model.get_sentence_embedding_dimension()

    print(f"Model loaded in {load_time:.1f}s")
    print(f"  max_seq_length: {max_seq_length}")
    print(f"  embedding_dim:  {embedding_dim}")
    print(f"  dtype:          bfloat16")
    print(f"  sentence-transformers: {st_version}")

    # ── Warm-up ──────────────────────────────────────────────────────────────
    print(f"\nWarm-up ({BATCH_SIZE} examples)...")
    t_warm = time.time()
    _ = model.encode(texts[:BATCH_SIZE], batch_size=BATCH_SIZE, show_progress_bar=False)
    warm_time = time.time() - t_warm
    est_total = warm_time / BATCH_SIZE * total_input
    print(f"  Warm-up: {warm_time:.1f}s for {BATCH_SIZE} examples")
    print(f"  Total estimate: ~{est_total:.0f}s (~{est_total / 60:.1f} min)")

    # ── Encoding ─────────────────────────────────────────────────────────────
    print(f"\nExtracting embeddings for {fmt(total_input)} conversations (batch_size={BATCH_SIZE})...")
    t_enc = time.time()
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
    encode_time = time.time() - t_enc

    peak_vram_emb_gb = torch.cuda.max_memory_allocated() / 1024**3

    speed = total_input / encode_time
    ms_per_example = encode_time / total_input * 1000

    print(f"\n{'=' * 70}")
    print("ENCODING RESULT (3a)")
    print(f"{'=' * 70}")
    print(f"  Embeddings shape:    {embeddings.shape}")
    print(f"  Encoding time:       {encode_time:.1f}s ({encode_time / 60:.1f} min)")
    print(f"  Load time:           {load_time:.1f}s")
    print(f"  Speed:               {speed:.2f} examples/s")
    print(f"  Time per example:    {ms_per_example:.1f} ms")
    print(f"  Peak VRAM:           {peak_vram_emb_gb:.1f} GB")

    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  3b. COSINE SIMILARITY DEDUPLICATION")
    print("─" * 70)

    # ── Normalization ────────────────────────────────────────────────────────
    print("\nExtracting and normalizing embeddings (L2)...")
    emb_np = np.array(embeddings, dtype=np.float32)
    print(f"  Shape: {emb_np.shape}")
    norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
    emb_norm = emb_np / norms

    # ── Similarity matrix ────────────────────────────────────────────────────
    print(f"\nComputing cosine similarity matrix ({fmt(total_input)} x {fmt(total_input)})...")
    emb_gpu = torch.tensor(emb_norm, dtype=torch.float32).cuda()
    t_sim = time.time()
    sim = emb_gpu @ emb_gpu.T
    sim.fill_diagonal_(0)
    sim_time = time.time() - t_sim
    print(f"  Time: {sim_time:.1f}s")

    # ── Greedy deduplication ─────────────────────────────────────────────────
    print(f"\nGreedy deduplication (threshold = {SIMILARITY_THRESHOLD})...")
    t_dedup = time.time()
    keep = torch.ones(total_input, dtype=torch.bool, device="cuda")

    for i in tqdm(range(total_input), desc="Deduplicating", unit="ex"):
        if not keep[i]:
            continue
        similar = sim[i, i + 1:] >= SIMILARITY_THRESHOLD
        keep[i + 1:][similar] = False

    keep_cpu = keep.cpu().numpy()
    dedup_time = time.time() - t_dedup

    n_kept = int(keep_cpu.sum())
    n_removed = total_input - n_kept
    print(f"  Kept:    {fmt(n_kept)}")
    print(f"  Removed: {fmt(n_removed)} ({n_removed / total_input * 100:.2f}%)")
    print(f"  Time: {dedup_time:.1f}s")

    # ── Free GPU ─────────────────────────────────────────────────────────────
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
    del sim, emb_gpu
    torch.cuda.empty_cache()

    # ── Add embedding column to dataset ──────────────────────────────────────
    print("\nAdding 'embedding' column to dataset...")
    df["embedding"] = [emb.tolist() for emb in tqdm(embeddings, desc="Converting", unit="emb")]

    # ── Filter and save dataset ──────────────────────────────────────────────
    print(f"\nFiltering dataset...")
    df_out = df[keep_cpu].reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving dataset: {OUTPUT_FILE}")
    t_save = time.time()
    df_out.to_parquet(OUTPUT_FILE, engine="pyarrow", compression="snappy", index=False)
    save_time = time.time() - t_save

    file_size_mb = os.path.getsize(OUTPUT_FILE) / 1024**2
    total_time = time.time() - t_global

    print(f"  File saved in {save_time:.1f}s ({file_size_mb:.1f} MB)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY — STEP 3")
    print(f"{'=' * 70}")
    print(f"  Input examples:       {fmt(total_input)}")
    print(f"  Kept examples:        {fmt(n_kept)}")
    print(f"  Removed examples:     {fmt(n_removed)} ({n_removed / total_input * 100:.2f}%)")
    print(f"  Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"  Embedding dim:        {embedding_dim}")
    print(f"  Peak VRAM:            {peak_vram_gb:.1f} GB")
    print(f"  Total time:           {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"  Output file:          {OUTPUT_FILE} ({file_size_mb:.1f} MB)")

    # ── Save stats ────────────────────────────────────────────────────────────
    stats = {
        "timestamp": timestamp,
        # 3a - embeddings
        "model": MODEL_NAME,
        "dtype": "bfloat16",
        "batch_size": BATCH_SIZE,
        "max_seq_length": max_seq_length,
        "embedding_dim": embedding_dim,
        "sentence_transformers_version": st_version,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "n_gpus": n_gpus,
        "gpu": gpu_info[0] if gpu_info else "N/A",
        "vram_peak_emb_gb": round(peak_vram_emb_gb, 1),
        "char_mean": char_stats["mean"],
        "char_median": char_stats["median"],
        "char_min": char_stats["min"],
        "char_max": char_stats["max"],
        "char_p25": char_stats["p25"],
        "char_p75": char_stats["p75"],
        "char_p95": char_stats["p95"],
        "load_time_s": round(load_time, 1),
        "encode_time_s": round(encode_time, 1),
        "speed_examples_per_s": round(speed, 2),
        "ms_per_example": round(ms_per_example, 1),
        # 3b - dedup
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "sim_time_s": round(sim_time, 1),
        "dedup_time_s": round(dedup_time, 1),
        "save_time_s": round(save_time, 1),
        "total_time_s": round(total_time, 1),
        "vram_peak_gb": round(peak_vram_gb, 1),
        # counts
        "input_file": STEP2_FILE,
        "output_file": OUTPUT_FILE,
        "output_size_mb": round(file_size_mb, 1),
        "total_input": total_input,
        "total_kept": n_kept,
        "total_removed": n_removed,
        "pct_removed": round(n_removed / total_input * 100, 2),
    }

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nStats saved to: {STATS_FILE}")
    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()
