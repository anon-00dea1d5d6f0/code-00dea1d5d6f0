"""
Generate rubric-level binary YES/NO verdicts (paper §4.2 Rubric-level binary scoring).

Evaluates a single model's response against all rubrics of the checklist in
ONE judge call, following the RRD paper evaluation prompt (arXiv:2602.05125,
Appendix F.3) adapted for Prosa:

  - Translated to Portuguese (PT-BR)
  - Conversation history included (omitted for single-turn)
  - Batched rubric evaluation (N rubrics judged in a single API call)

The judge receives:
  - Conversation history (if multi-turn)
  - Current user query
  - AI response (single model)
  - Numbered list of rubrics parsed from question["checklist"]

And produces one YES/NO per rubric, wrapped in <EVALUATION_i> tags. The
aggregate score is computed as pass_rate * 10 (0-10 scale, matching the
existing 0-10 score range).

Usage:
python -m prosa.gen_score \
    --model gpt-5.2-2025-12-11 \
    --judge-model gpt-4.1-2025-04-14 \
    --parallel 50
"""

import argparse
import json
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import openai

from prosa.prosa_utils import (
    load_prosa_questions,
    load_prosa_model_answers,
    get_model_response,
)

# ---------------------------------------------------------------------------
# Prompt template — RRD F.3 adapted for Prosa (PT-BR + history + batch)
# ---------------------------------------------------------------------------
#
# Modular build (header + optional history + body) following the same pattern
# used in gen_rubrics.py. The history block is omitted for
# single-turn questions so the judge does not try to invent context.

_PROMPT_HEADER = """\
Você é um juiz, avaliando se uma resposta satisfaz as rúbricas fornecidas. \
Para cada rúbrica, se a resposta satisfaz o critério da rúbrica, produza YES; \
caso contrário, produza NO.

Requisito:
- Você deve seguir cada rúbrica estritamente, e apenas considerar os critérios \
listados na rúbrica.
- Você NÃO deve considerar quaisquer outros fatores, como suas próprias \
opiniões ou conhecimento externo.

"""

_PROMPT_HISTORY = """\
# Conversa entre Usuário e IA

## Histórico
<|begin_of_history|>

{history}

<|end_of_history|>

"""

_PROMPT_BODY = """\
## Consulta Atual do Usuário
<|begin_of_query|>

{user_query}

<|end_of_query|>

Abaixo entre <RESPONSE> e </RESPONSE> está a resposta a ser avaliada:
<RESPONSE>
{response}
</RESPONSE>

{rubrics_intro}
{rubrics}

Produza ESTRITAMENTE no formato abaixo{output_intro}. Nenhum outro texto é permitido:
{output_spec}"""


def build_score_rubric_prompt(
    history: str,
    user_query: str,
    response: str,
    rubrics: list,
) -> str:
    """Build the RRD-style scoring prompt (omits history for single-turn)."""
    parts = [_PROMPT_HEADER]
    if history and history.strip():
        parts.append(_PROMPT_HISTORY.format(history=history))

    n = len(rubrics)
    rubrics_block = "\n".join(
        f"<RUBRIC_{i + 1}> {r} </RUBRIC_{i + 1}>" for i, r in enumerate(rubrics)
    )
    if n == 1:
        rubrics_intro = (
            "Abaixo entre <RUBRIC_1> e </RUBRIC_1> está a rúbrica na qual avaliar:"
        )
        output_intro = ""
        output_spec = "<EVALUATION_1> YES/NO </EVALUATION_1>"
    elif n == 2:
        rubrics_intro = (
            "Abaixo entre <RUBRIC_i> e </RUBRIC_i> estão as 2 rúbricas nas quais "
            "avaliar, numeradas como 1 e 2:"
        )
        output_intro = ", uma avaliação por rúbrica"
        output_spec = (
            "<EVALUATION_1> YES/NO </EVALUATION_1>\n"
            "<EVALUATION_2> YES/NO </EVALUATION_2>"
        )
    else:
        rubrics_intro = (
            f"Abaixo entre <RUBRIC_i> e </RUBRIC_i> estão as {n} rúbricas nas quais "
            f"avaliar, numeradas de 1 até {n}:"
        )
        output_intro = f", uma avaliação por rúbrica (numeradas de 1 até {n})"
        output_spec = (
            "<EVALUATION_1> YES/NO </EVALUATION_1>\n"
            "<EVALUATION_2> YES/NO </EVALUATION_2>\n"
            "...\n"
            f"<EVALUATION_{n}> YES/NO </EVALUATION_{n}>"
        )

    parts.append(
        _PROMPT_BODY.format(
            user_query=user_query,
            response=response,
            rubrics_intro=rubrics_intro,
            rubrics=rubrics_block,
            output_intro=output_intro,
            output_spec=output_spec,
        )
    )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Checklist parser: markdown numbered list -> list of rubric strings
# ---------------------------------------------------------------------------

_RUBRIC_ITEM_RE = re.compile(r"^\s*(\d+)\.\s+(.+)$")


def parse_checklist(checklist_text: str) -> list:
    """Parse a markdown numbered list into a list of rubric strings.

    Each item starts with "N. " on its own line. Continuation lines (not
    starting with a new number) are appended to the current rubric.
    """
    if not checklist_text or not checklist_text.strip():
        return []

    rubrics = []
    current = None
    for line in checklist_text.split("\n"):
        m = _RUBRIC_ITEM_RE.match(line)
        if m:
            if current is not None:
                rubrics.append(current.strip())
            current = m.group(2).strip()
        elif current is not None and line.strip():
            current += " " + line.strip()
    if current is not None:
        rubrics.append(current.strip())
    return rubrics


# ---------------------------------------------------------------------------
# Output parser: <EVALUATION_i> YES/NO </EVALUATION_i>
# ---------------------------------------------------------------------------

_EVAL_TAG_RE = re.compile(
    r"<EVALUATION_(\d+)>\s*(YES|NO|SIM|N[ÃA]O)\s*</EVALUATION_\1>",
    re.IGNORECASE,
)


def parse_evaluations(judge_response: str, n_rubrics: int) -> list:
    """Extract a list of 'YES'/'NO' (length n_rubrics) from the judge response.

    Missing/malformed evaluations are filled with None. Normalizes PT tokens
    (SIM/NAO/NÃO) to YES/NO for robustness.
    """
    results = [None] * n_rubrics
    for m in _EVAL_TAG_RE.finditer(judge_response):
        idx = int(m.group(1)) - 1  # tags are 1-indexed
        if 0 <= idx < n_rubrics:
            token = m.group(2).upper()
            if token in ("YES", "SIM"):
                results[idx] = "YES"
            elif token in ("NO", "NÃO", "NAO"):
                results[idx] = "NO"
    return results


def compute_score(evaluations: list) -> dict:
    """Aggregate a list of 'YES'/'NO'/None evaluations into a 0-10 score."""
    n_total = len(evaluations)
    n_valid = sum(1 for e in evaluations if e in ("YES", "NO"))
    n_passed = sum(1 for e in evaluations if e == "YES")
    pass_rate = n_passed / n_valid if n_valid else None
    score = pass_rate * 10 if pass_rate is not None else None
    return {
        "n_rubrics": n_total,
        "n_valid": n_valid,
        "n_passed": n_passed,
        "pass_rate": pass_rate,
        "score": score,
    }


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

_write_lock = threading.Lock()


def evaluate_single_rrd(
    question,
    answer,
    judge_model,
    output_file,
    max_retries=3,
    reasoning_effort=None,
    api_base=None,
    api_key=None,
    max_tokens=4096,
):
    """Evaluate a single model response against all rubrics in ONE judge call."""
    conversation_hash = question["conversation_hash"]
    history = question.get("history_text", "")
    user_query = question.get("last_query", "")
    checklist_text = question.get("checklist", "")

    rubrics = parse_checklist(checklist_text)
    if not rubrics:
        print(f"Skipping {conversation_hash}: empty or unparseable checklist")
        return None

    response_text = get_model_response(answer)
    prompt = build_score_rubric_prompt(history, user_query, response_text, rubrics)

    content = None

    for attempt in range(max_retries):
        try:
            client_kwargs = {}
            if api_base:
                client_kwargs["base_url"] = api_base
            if api_key:
                client_kwargs["api_key"] = api_key
            client = openai.OpenAI(**client_kwargs)
            messages = [{"role": "user", "content": prompt}]
            is_reasoning_model = (
                judge_model.startswith("o1")
                or judge_model.startswith("o3")
                or judge_model.startswith("gpt-5")
            )
            common_args = {
                "model": judge_model,
                "messages": messages,
                "n": 1,
            }
            if is_reasoning_model:
                pass  # use API default temperature
            else:
                common_args["temperature"] = 0
                common_args["max_tokens"] = max_tokens
            if reasoning_effort:
                common_args["reasoning_effort"] = reasoning_effort
            response = client.chat.completions.create(**common_args)
            content = response.choices[0].message.content
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Error for {conversation_hash} (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(wait_time)
            else:
                print(f"Failed {conversation_hash} after {max_retries} attempts: {e}")

    if content is None:
        return None

    evaluations = parse_evaluations(content, len(rubrics))
    stats = compute_score(evaluations)

    result = {
        "conversation_hash": conversation_hash,
        "evaluations": evaluations,
        **stats,
    }

    with _write_lock:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prosa rubric-level YES/NO evaluation "
                    "(RRD F.3 prompt, batched rubric judging, one call per response)"
    )
    parser.add_argument("--bench-name", type=str, default="prosa")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to evaluate (must have answers in model_answer/)")
    parser.add_argument("--judge-model", type=str, required=True,
                        help="Judge model name")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max output tokens for the judge (scales with #rubrics)")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["low", "medium", "high"])
    parser.add_argument("--question-begin", type=int, default=None)
    parser.add_argument("--question-end", type=int, default=None)
    parser.add_argument("--score-dir", type=str, default=None,
                        help="Custom directory for score output (default: data/{bench}/score_rubric/)")
    parser.add_argument("--api-base", type=str, default=None,
                        help="Custom API base URL (e.g. https://openrouter.ai/api/v1)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Custom API key (overrides OPENAI_API_KEY)")
    parser.add_argument("--judge-id", type=str, default=None,
                        help="Judge identifier for output filename (default: same as --judge-model)")
    args = parser.parse_args()

    # Paths
    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", args.bench_name
    )
    question_file = os.path.join(data_dir, "question.jsonl")
    answer_file = os.path.join(data_dir, "model_answer", f"{args.model}.jsonl")
    score_dir = args.score_dir if args.score_dir else os.path.join(data_dir, "score_rubric")
    judge_id = args.judge_id if args.judge_id else args.judge_model
    output_file = os.path.join(
        score_dir, f"{args.model}_by_{judge_id}.jsonl"
    )

    # Load data
    print(f"Loading questions from {question_file}")
    questions = load_prosa_questions(
        question_file, args.question_begin, args.question_end
    )
    print(f"Loaded {len(questions)} questions")

    print(f"Loading answers from {answer_file}")
    answers = load_prosa_model_answers(answer_file)
    print(f"Loaded {len(answers)} answers")

    # Load existing results to skip
    existing = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing.add(json.loads(line)["conversation_hash"])
        print(f"Found {len(existing)} existing scores, skipping them")

    # Build work items
    work = []
    missing_answers = 0
    missing_checklists = 0
    for q in questions:
        h = q["conversation_hash"]
        if h in existing:
            continue
        if h not in answers:
            missing_answers += 1
            continue
        if not q.get("checklist", "").strip():
            missing_checklists += 1
            continue
        work.append((q, answers[h]))

    if missing_answers:
        print(f"Warning: {missing_answers} questions have no answer for {args.model}")
    if missing_checklists:
        print(f"Warning: {missing_checklists} questions have no checklist")
    print(f"Running {len(work)} RRD evaluations ({args.judge_model} as judge, id={judge_id})")

    # Run
    def run_one(item):
        q, a = item
        return evaluate_single_rrd(
            q, a, args.judge_model, output_file,
            max_retries=args.max_retries,
            reasoning_effort=args.reasoning_effort,
            api_base=args.api_base,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
        )

    failed = 0
    if args.parallel == 1:
        for item in tqdm(work, desc="Scoring (RRD F.3)"):
            r = run_one(item)
            if r is None:
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            results = list(tqdm(
                executor.map(run_one, work),
                total=len(work),
                desc="Scoring (RRD F.3)",
            ))
            failed = sum(1 for r in results if r is None)

    if failed:
        print(f"\nWarning: {failed} evaluations failed. Re-run to retry.")

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
