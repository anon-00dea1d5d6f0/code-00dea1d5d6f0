"""
Display the rubric-scoring leaderboard (paper §5.2 Model ranking).

Reads all score files in data/{bench_name}/score_rubric/ and presents:
  1. Overall ranking (raw 1-10 and adjusted 0-100)
  2. Ranking by category (all models side by side)
  3. Ranking by difficulty level (all models side by side)

The 'score' in each entry is computed as pass_rate * 10, where pass_rate is
the fraction of rubrics the response satisfied (judged YES by the LLM judge
using the RRD F.3 prompt).

Usage:
    python -m prosa.show_score --bench-name prosa
"""

import argparse
import glob
import json
import os
from collections import defaultdict


def load_questions(question_file):
    """Load questions keyed by conversation_hash."""
    questions = {}
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                questions[q["conversation_hash"]] = q
    return questions


def recompute_filtered_scores(all_results, questions):
    """Recompute scores using only rubrics in checklist_filtered_indices.

    Modifies entries in-place: updates score, n_rubrics, n_valid, n_passed, pass_rate.
    Removes entries for questions marked as question_excluded.
    """
    excluded_hashes = {h for h, q in questions.items() if q.get("question_excluded", False)}
    for (model, judge), entries in list(all_results.items()):
        filtered_entries = []
        for e in entries:
            h = e["conversation_hash"]
            if h in excluded_hashes:
                continue
            q = questions.get(h, {})
            indices = q.get("checklist_filtered_indices")
            if indices is None:
                filtered_entries.append(e)
                continue
            evaluations = e.get("evaluations", [])
            filtered_evals = [evaluations[i] for i in indices if i < len(evaluations)]
            n_valid = sum(1 for ev in filtered_evals if ev in ("YES", "NO"))
            n_passed = sum(1 for ev in filtered_evals if ev == "YES")
            pass_rate = n_passed / n_valid if n_valid else None
            e["score"] = pass_rate * 10 if pass_rate is not None else None
            e["n_rubrics"] = len(filtered_evals)
            e["n_valid"] = n_valid
            e["n_passed"] = n_passed
            e["pass_rate"] = pass_rate
            filtered_entries.append(e)
        all_results[(model, judge)] = filtered_entries
    if excluded_hashes:
        print(f"** Excluded {len(excluded_hashes)} questions (question_excluded=true) **\n")


def load_score_files(score_dir):
    """Load all score JSONL files.

    Returns dict: {(model, judge): [{score_entry}, ...]}
    """
    results = {}
    for path in sorted(glob.glob(os.path.join(score_dir, "*.jsonl"))):
        fname = os.path.basename(path)[:-6]  # strip .jsonl
        if "_by_" not in fname:
            continue
        model, judge = fname.rsplit("_by_", 1)
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("score") is not None:
                        entries.append(entry)
        if entries:
            results[(model, judge)] = entries
    return results


def macro_mean(entries):
    """Macro-average: mean of per-example scores (each question weight 1).

    Input: list of score entries. Returns raw score in 0-10 scale, or None.
    """
    scores = [e["score"] for e in entries if e.get("score") is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def micro_mean(entries):
    """Micro-average: sum(n_passed) / sum(n_valid) over all examples.

    Each rubric has equal weight; examples with more rubrics count more.
    Returns raw score in 0-10 scale, or None.
    """
    total_passed = 0
    total_valid = 0
    for e in entries:
        if e.get("n_valid") is not None and e.get("n_passed") is not None:
            total_valid += e["n_valid"]
            total_passed += e["n_passed"]
    if not total_valid:
        return None
    return (total_passed / total_valid) * 10


def compute_mean(scores):
    """Compute raw mean (0-10) from a list of scores (used by category/difficulty)."""
    if not scores:
        return None
    return sum(scores) / len(scores)


def adjusted(raw):
    """Map raw 0-10 to adjusted 0-100."""
    if raw is None:
        return None
    return raw * 10.0


def print_table(headers, rows, col_widths=None):
    """Print a formatted table."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            w = len(str(h))
            for row in rows:
                w = max(w, len(str(row[i])))
            col_widths.append(w + 2)

    header_line = ""
    for i, h in enumerate(headers):
        header_line += str(h).ljust(col_widths[i])
    print(header_line)
    print("-" * len(header_line))

    for row in rows:
        line = ""
        for i, val in enumerate(row):
            line += str(val).ljust(col_widths[i])
        print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Show Prosa rubric-level scoring leaderboard"
    )
    parser.add_argument("--bench-name", type=str, default="prosa")
    parser.add_argument("--score-dir", type=str, default=None,
                        help="Custom directory for score files (default: data/{bench}/score_rubric/)")
    parser.add_argument("--judge", type=str, default="sabia-4",
                        help="Judge whose verdicts to display (default: sabia-4, the paper Table 5 leaderboard)")
    parser.add_argument("--filtered", action="store_true",
                        help="Recompute scores using only checklist_filtered_indices from question.jsonl")
    args = parser.parse_args()

    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", args.bench_name
    )
    question_file = os.path.join(data_dir, "question.jsonl")
    score_dir = args.score_dir if args.score_dir else os.path.join(data_dir, "score_rubric")

    if not os.path.isdir(score_dir):
        print(f"No score directory found: {score_dir}")
        return

    questions = load_questions(question_file)
    all_results = load_score_files(score_dir)
    # Filter to a single judge so the leaderboard is well-defined
    all_results = {(m, j): v for (m, j), v in all_results.items() if j == args.judge}

    if not all_results:
        print("No score files found.")
        return

    if args.filtered:
        print("** Using filtered rubrics (checklist_filtered_indices) **\n")
        recompute_filtered_scores(all_results, questions)

    # Build per-model data: {model: {hash: score}}
    # Also track judge name and full entries per model
    model_scores = {}   # model -> {hash: score}
    model_entries = {}  # model -> list of full entries (for micro-average)
    model_judges = {}   # model -> judge
    for (model, judge), entries in all_results.items():
        model_scores[model] = {e["conversation_hash"]: e["score"] for e in entries}
        model_entries[model] = entries
        model_judges[model] = judge

    # Compute macro and micro scores per model
    model_macro = {m: macro_mean(model_entries[m]) for m in model_entries}
    model_micro = {m: micro_mean(model_entries[m]) for m in model_entries}

    # Sort models by macro score (matches Prosa convention)
    models_ranked = sorted(
        model_macro.keys(), key=lambda m: model_macro[m] or 0, reverse=True
    )

    # ---------------------------------------------------------------
    # 1. Overall ranking (macro + micro side-by-side)
    # ---------------------------------------------------------------
    print("=" * 80)
    print("OVERALL RANKING (RRD F.3 rubric scoring)")
    print("  Macro = mean of per-example pass_rate (each question weight 1)")
    print("  Micro = sum(n_passed) / sum(n_valid) (each rubric weight 1)")
    print("=" * 80)

    overall_rows = []
    for rank, model in enumerate(models_ranked, 1):
        macro = model_macro[model]
        micro = model_micro[model]
        n = len(model_scores[model])
        overall_rows.append((
            rank, model, model_judges[model],
            f"{macro:.2f}" if macro is not None else "-",
            f"{adjusted(macro):.1f}" if macro is not None else "-",
            f"{micro:.2f}" if micro is not None else "-",
            f"{adjusted(micro):.1f}" if micro is not None else "-",
            n,
        ))

    print_table(
        ["#", "Model", "Judge",
         "Macro (0-10)", "Macro (0-100)",
         "Micro (0-10)", "Micro (0-100)", "N"],
        overall_rows,
    )

    # ---------------------------------------------------------------
    # 2. Ranking by category
    # ---------------------------------------------------------------
    print()
    print("=" * 80)
    print("SCORES BY CATEGORY")
    print("=" * 80)

    # Collect all categories
    all_categories = set()
    # model -> cat -> [scores]
    model_cat_scores = defaultdict(lambda: defaultdict(list))
    for model, scores_dict in model_scores.items():
        for h, score in scores_dict.items():
            q = questions.get(h, {})
            cat = q.get("category", "Unknown")
            all_categories.add(cat)
            model_cat_scores[model][cat].append(score)

    categories_sorted = sorted(all_categories)

    # Build table: rows = categories, columns = models (ranked)
    headers = ["Category"] + [m for m in models_ranked]
    rows = []
    for cat in categories_sorted:
        row = [cat]
        for model in models_ranked:
            scores = model_cat_scores[model].get(cat, [])
            raw = compute_mean(scores)
            if raw is not None:
                row.append(f"{adjusted(raw):.1f}")
            else:
                row.append("-")
        rows.append(tuple(row))

    # Auto col widths
    col_widths = [max(len("Category"), max(len(c) for c in categories_sorted)) + 2]
    for m in models_ranked:
        col_widths.append(max(len(m), 8) + 2)

    print_table(headers, rows, col_widths)

    # ---------------------------------------------------------------
    # 3. Ranking by difficulty
    # ---------------------------------------------------------------
    print()
    print("=" * 80)
    print("SCORES BY DIFFICULTY")
    print("=" * 80)

    difficulty_order = ["very easy", "easy", "medium", "hard", "very hard"]

    # model -> level -> [scores]
    model_diff_scores = defaultdict(lambda: defaultdict(list))
    all_levels = set()
    for model, scores_dict in model_scores.items():
        for h, score in scores_dict.items():
            q = questions.get(h, {})
            diff = q.get("difficulty", {})
            level = diff.get("level", "unknown") if isinstance(diff, dict) else str(diff)
            all_levels.add(level)
            model_diff_scores[model][level].append(score)

    # Order: predefined first, then any extras
    levels_sorted = [l for l in difficulty_order if l in all_levels]
    for l in sorted(all_levels):
        if l not in levels_sorted:
            levels_sorted.append(l)

    headers = ["Difficulty"] + [m for m in models_ranked]
    rows = []
    for level in levels_sorted:
        row = [level]
        for model in models_ranked:
            scores = model_diff_scores[model].get(level, [])
            raw = compute_mean(scores)
            if raw is not None:
                row.append(f"{adjusted(raw):.1f}")
            else:
                row.append("-")
        rows.append(tuple(row))

    col_widths = [max(len("Difficulty"), max(len(l) for l in levels_sorted)) + 2]
    for m in models_ranked:
        col_widths.append(max(len(m), 8) + 2)

    print_table(headers, rows, col_widths)

    print()


if __name__ == "__main__":
    main()
