"""Replace checklist field in question.jsonl with rubrics from an RRD checklist file.

Usage:
    python -m prosa.update_checklists \
        --checklist-file data/prosa/checklist/rrd_*.jsonl
"""

import argparse
import json
import os
import shutil


def main():
    parser = argparse.ArgumentParser(
        description="Replace checklists in question.jsonl with rubrics from a checklist file"
    )
    parser.add_argument(
        "--bench-name", type=str, default="prosa",
        help="Benchmark name (default: prosa)",
    )
    parser.add_argument(
        "--checklist-file", type=str, required=True,
        help="Path to the RRD checklist JSONL file",
    )
    args = parser.parse_args()

    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", args.bench_name
    )
    question_file = os.path.join(data_dir, "question.jsonl")

    # Load new checklists keyed by hash
    new_checklists = {}
    with open(args.checklist_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                h = d["conversation_hash"]
                checklist_str = d.get("checklist", "")
                new_checklists[h] = checklist_str

    print(f"Loaded {len(new_checklists)} checklists from {args.checklist_file}")

    # Load questions
    questions = []
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    print(f"Loaded {len(questions)} questions from {question_file}")

    # Replace checklists
    updated = 0
    missing = 0
    for q in questions:
        h = q["conversation_hash"]
        if h in new_checklists:
            q["checklist"] = new_checklists[h]
            updated += 1
        else:
            missing += 1

    if missing:
        print(f"Warning: {missing} questions have no matching checklist in the file")

    # Backup and write
    backup_file = question_file + ".bak"
    shutil.copy2(question_file, backup_file)
    print(f"Backup saved to {backup_file}")

    with open(question_file, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Updated {updated} checklists in {question_file}")


if __name__ == "__main__":
    main()
