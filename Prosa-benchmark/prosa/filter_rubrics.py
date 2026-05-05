"""
Multi-judge rubric filtering pipeline (paper §4.3 Multi-judge rubric filtering).

Filters rubrics from question.jsonl based on 3-judge majority voting
and intra-judge stability.

Applies five filters (paper §4.3):
  1. Trivial: rubrics where ALL 16 models pass (100% pass rate)
  2. Impossible: rubrics where ALL 16 models fail (0% pass rate)
  3. Misaligned: rubrics where ALL top-K models fail but bottom model passes
  4. Unstable: rubrics that flip YES/NO across repeated runs of the same judge
  5. Min rubrics: questions with fewer than --min-rubrics surviving rubrics
     are excluded from scoring (--min-rubrics 0 to keep all questions).

Filters 1-3: rubric removed if >=2 of 3 judges flag it (majority voting).
Filter 4: rubric removed if ANY of the repeated runs disagrees.

Adds checklist_filtered and checklist_filtered_indices to question.jsonl
without modifying the original checklist field.

Usage:
    python -m prosa.filter_rubrics [--dry-run]
    python -m prosa.filter_rubrics --no-unstable
    python -m prosa.filter_rubrics --no-trivial --no-impossible --no-misaligned
"""

import argparse
import json
import os
import re


DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "prosa")
SCORE_DIR = os.path.join(DATA_DIR, "score_rubric")
QUESTION_FILE = os.path.join(DATA_DIR, "question.jsonl")

JUDGES = [
    "gpt-4.1-2025-04-14",
    "gemini-3-flash-preview",
    "sabia-4",
]

TOP_K = 2   # top 2 strongest models (ALL must fail for misalignment)
BOTTOM_K = 1  # bottom 1 weakest model (must pass for misalignment)
MAJORITY = 2  # need >=2 of 3 judges to flag

# Stability filter defaults
STABILITY_MODEL = "Qwen3-30B-A3B-Instruct-2507"
STABILITY_JUDGE = "gpt-4.1-2025-04-14"
STABILITY_DIRS = [
    os.path.join(DATA_DIR, "score_rubric"),
    os.path.join(DATA_DIR, "score_rubric_run2"),
    os.path.join(DATA_DIR, "score_rubric_run3"),
]


def load_evaluations(score_dir, judges):
    """Load all evaluations: {(model, judge): {hash: [YES/NO list]}}"""
    data = {}
    for path in sorted(os.listdir(score_dir)):
        if not path.endswith(".jsonl"):
            continue
        fname = path[:-6]
        if "_by_" not in fname:
            continue
        model, judge = fname.rsplit("_by_", 1)
        if judge not in judges:
            continue
        evals = {}
        with open(os.path.join(score_dir, path)) as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    evals[e["conversation_hash"]] = e.get("evaluations", [])
        data[(model, judge)] = evals
    return data


def load_stability_runs(dirs, model, judge):
    """Load evaluations from multiple run directories for stability check.

    Returns list of dicts: [{hash: [YES/NO list]}, ...]
    """
    runs = []
    for d in dirs:
        fname = f"{model}_by_{judge}.jsonl"
        path = os.path.join(d, fname)
        if not os.path.exists(path):
            continue
        evals = {}
        with open(path) as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    evals[e["conversation_hash"]] = e.get("evaluations", [])
        runs.append(evals)
    return runs


def get_models_ranked(data, judge):
    """Get models sorted by mean score (descending) for a given judge."""
    model_scores = {}
    for (m, j), evals in data.items():
        if j != judge:
            continue
        all_evals = []
        for ev_list in evals.values():
            all_evals.extend(ev_list)
        yes_count = sum(1 for e in all_evals if e == "YES")
        total = sum(1 for e in all_evals if e in ("YES", "NO"))
        model_scores[m] = yes_count / total if total else 0
    return sorted(model_scores.keys(), key=lambda m: -model_scores[m])


def parse_checklist(checklist_text):
    """Parse numbered markdown list into list of rubric strings."""
    if not checklist_text or not checklist_text.strip():
        return []
    rubrics = []
    current = None
    for line in checklist_text.split("\n"):
        m = re.match(r"^\s*(\d+)\.\s+(.+)$", line)
        if m:
            if current is not None:
                rubrics.append(current.strip())
            current = m.group(2).strip()
        elif current is not None and line.strip():
            current += " " + line.strip()
    if current is not None:
        rubrics.append(current.strip())
    return rubrics


def rebuild_checklist(rubrics):
    """Rebuild numbered markdown list from rubric strings."""
    return "\n".join(f"{i+1}. {r}" for i, r in enumerate(rubrics))


def main():
    parser = argparse.ArgumentParser(description="Filter rubrics using 3-judge majority voting + stability")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print statistics, don't modify question.jsonl")
    parser.add_argument("--score-dir", type=str, default=SCORE_DIR)
    parser.add_argument("--question-file", type=str, default=QUESTION_FILE)
    parser.add_argument("--no-trivial", action="store_true",
                        help="Disable trivial filter (all models pass)")
    parser.add_argument("--no-impossible", action="store_true",
                        help="Disable impossible filter (all models fail)")
    parser.add_argument("--no-misaligned", action="store_true",
                        help="Disable misalignment filter (weak > strong)")
    parser.add_argument("--no-unstable", action="store_true",
                        help="Disable stability filter (intra-judge variance)")
    parser.add_argument("--stability-model", type=str, default=STABILITY_MODEL,
                        help=f"Model for stability check (default: {STABILITY_MODEL})")
    parser.add_argument("--stability-judge", type=str, default=STABILITY_JUDGE,
                        help=f"Judge for stability check (default: {STABILITY_JUDGE})")
    parser.add_argument("--min-rubrics", type=int, default=0,
                        help="Exclude questions with <= this many rubrics after filtering (default: 0)")
    args = parser.parse_args()

    # Load evaluations for filters 1-3
    print("Loading evaluations...")
    data = load_evaluations(args.score_dir, JUDGES)
    models_per_judge = {}
    for j in JUDGES:
        models_per_judge[j] = get_models_ranked(data, j)
        print(f"  {j}: {len(models_per_judge[j])} models")

    # Top and bottom models per judge
    top_models = {j: set(models_per_judge[j][:TOP_K]) for j in JUDGES}
    bottom_models = {j: set(models_per_judge[j][-BOTTOM_K:]) for j in JUDGES}

    for j in JUDGES:
        jn = j.split("-2025")[0]
        print(f"\n  {jn}:")
        print(f"    Top {TOP_K}: {', '.join(top_models[j])}")
        print(f"    Bottom {BOTTOM_K}: {', '.join(bottom_models[j])}")

    # Load stability runs for filter 4
    unstable_rubrics = {}  # hash -> set of unstable indices
    if not args.no_unstable:
        print(f"\nLoading stability runs ({args.stability_model} x {args.stability_judge})...")
        stability_runs = load_stability_runs(STABILITY_DIRS, args.stability_model, args.stability_judge)
        print(f"  Found {len(stability_runs)} runs")

        if len(stability_runs) >= 2:
            # Find common hashes
            common = set(stability_runs[0].keys())
            for r in stability_runs[1:]:
                common &= set(r.keys())

            for h in common:
                evs = [r[h] for r in stability_runs]
                n = min(len(e) for e in evs)
                for i in range(n):
                    votes = [e[i] for e in evs if i < len(e) and e[i] in ("YES", "NO")]
                    if len(votes) >= 2 and len(set(votes)) > 1:
                        # Not unanimous → unstable
                        if h not in unstable_rubrics:
                            unstable_rubrics[h] = set()
                        unstable_rubrics[h].add(i)

            total_unstable = sum(len(v) for v in unstable_rubrics.values())
            print(f"  Unstable rubrics found: {total_unstable} across {len(unstable_rubrics)} questions")
        else:
            print("  WARNING: Need >=2 runs for stability filter. Skipping.")
            args.no_unstable = True

    # Load questions
    print(f"\nLoading questions from {args.question_file}")
    questions = []
    with open(args.question_file) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    print(f"  {len(questions)} questions loaded")

    # For each question, for each rubric, check all filters
    total_rubrics_before = 0
    total_removed_trivial = 0
    total_removed_impossible = 0
    total_removed_misaligned = 0
    total_removed_unstable = 0
    total_removed_total = 0
    total_rubrics_after = 0
    questions_modified = 0

    updated_questions = []

    for q in questions:
        h = q["conversation_hash"]
        checklist_text = q.get("checklist", "")
        rubrics = parse_checklist(checklist_text)
        n_rubrics = len(rubrics)
        total_rubrics_before += n_rubrics

        if n_rubrics == 0:
            updated_questions.append(q)
            continue

        keep = [True] * n_rubrics
        reasons = [None] * n_rubrics

        for i in range(n_rubrics):
            # Collect votes from each judge for filters 1-3
            trivial_votes = 0
            impossible_votes = 0
            misaligned_votes = 0

            for j in JUDGES:
                models = models_per_judge[j]

                passes = 0
                fails = 0
                top_pass_count = 0
                top_fail_count = 0
                top_evaluated = 0
                bottom_pass = False

                for m in models:
                    ev_list = data.get((m, j), {}).get(h, [])
                    if i < len(ev_list) and ev_list[i] in ("YES", "NO"):
                        if ev_list[i] == "YES":
                            passes += 1
                            if m in bottom_models[j]:
                                bottom_pass = True
                            if m in top_models[j]:
                                top_pass_count += 1
                                top_evaluated += 1
                        else:
                            fails += 1
                            if m in top_models[j]:
                                top_fail_count += 1
                                top_evaluated += 1

                total_evaluated = passes + fails
                if total_evaluated == 0:
                    continue

                if passes == total_evaluated:
                    trivial_votes += 1

                if fails == total_evaluated:
                    impossible_votes += 1

                all_top_fail = (top_evaluated > 0 and top_fail_count == top_evaluated)
                if bottom_pass and all_top_fail:
                    misaligned_votes += 1

            # Apply filters (priority order: trivial > impossible > misaligned > unstable)
            if not args.no_trivial and trivial_votes >= MAJORITY:
                keep[i] = False
                reasons[i] = "trivial"
                total_removed_trivial += 1
            elif not args.no_impossible and impossible_votes >= MAJORITY:
                keep[i] = False
                reasons[i] = "impossible"
                total_removed_impossible += 1
            elif not args.no_misaligned and misaligned_votes >= MAJORITY:
                keep[i] = False
                reasons[i] = "misaligned"
                total_removed_misaligned += 1
            elif not args.no_unstable and h in unstable_rubrics and i in unstable_rubrics[h]:
                keep[i] = False
                reasons[i] = "unstable"
                total_removed_unstable += 1

        # Count removals
        n_removed = sum(1 for k in keep if not k)
        total_removed_total += n_removed
        n_kept = sum(1 for k in keep if k)
        total_rubrics_after += n_kept

        if n_removed > 0:
            questions_modified += 1

        # Rebuild filtered checklist into new field, keep original untouched
        filtered_rubrics = [r for r, k in zip(rubrics, keep) if k]
        filtered_indices = [i for i, k in enumerate(keep) if k]
        q_copy = dict(q)
        q_copy["checklist_filtered"] = rebuild_checklist(filtered_rubrics) if filtered_rubrics else ""
        q_copy["checklist_filtered_indices"] = filtered_indices
        q_copy["question_excluded"] = len(filtered_indices) <= args.min_rubrics
        updated_questions.append(q_copy)

    # Report
    print("\n" + "=" * 70)
    print("RUBRIC FILTERING REPORT")
    print("=" * 70)
    print(f"  Judges:                  {len(JUDGES)} ({', '.join(j.split('-2025')[0] for j in JUDGES)})")
    print(f"  Majority threshold:      >= {MAJORITY}/{len(JUDGES)}")
    print(f"  Questions:               {len(questions)}")
    print(f"  Questions modified:      {questions_modified}")
    print(f"")
    print(f"  Rubrics before:          {total_rubrics_before:,}")
    print(f"  Removed (trivial):       {total_removed_trivial:,}  (all models pass)")
    print(f"  Removed (impossible):    {total_removed_impossible:,}  (all models fail)")
    print(f"  Removed (misaligned):    {total_removed_misaligned:,}  (weak > strong)")
    print(f"  Removed (unstable):      {total_removed_unstable:,}  (flips across runs)")
    print(f"  Removed (total):         {total_removed_total:,}  ({total_removed_total/total_rubrics_before:.1%})")
    print(f"  Rubrics after:           {total_rubrics_after:,}  ({total_rubrics_after/total_rubrics_before:.1%} retained)")
    print(f"  Avg rubrics/question:    {total_rubrics_before/len(questions):.1f} -> {total_rubrics_after/len(questions):.1f}")

    # Questions excluded (too few rubrics)
    n_excluded = sum(1 for q in updated_questions if q.get("question_excluded", False))
    print(f"  Questions excluded:     {n_excluded}  (<= {args.min_rubrics} rubrics after filtering)")
    zero_rubric = sum(1 for q in updated_questions if not q.get("checklist_filtered", "").strip())
    if zero_rubric:
        print(f"  WARNING: {zero_rubric} of those have ZERO rubrics!")

    if args.dry_run:
        print(f"\n  [DRY RUN] No changes written to {args.question_file}")
    else:
        print(f"\n  Writing filtered questions to {args.question_file}")
        with open(args.question_file, "w", encoding="utf-8") as f:
            for q in updated_questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"  Done. {total_removed_total:,} rubrics removed from {questions_modified} questions.")

    print("=" * 70)


if __name__ == "__main__":
    main()
