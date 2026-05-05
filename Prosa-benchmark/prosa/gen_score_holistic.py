"""
Generate holistic 1-10 scores for a candidate model (paper §5.1 baseline).

The judge receives:
  - Conversation history
  - Current user query
  - AI response (single model)
  - Checklist (binary criteria to guide evaluation)

And produces a JSON with: strengths, weaknesses, score (1-10).

Usage:
python -m prosa.gen_score_holistic \
    --bench-name prosa \
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
# Holistic 1-10 score prompt template (PT-BR)
# ---------------------------------------------------------------------------

SCORE_PROMPT_TEMPLATE = """\
# Instrução

Você é um avaliador especialista. Sua tarefa é avaliar a qualidade das respostas geradas por modelos de IA.
Nós forneceremos a você a consulta do usuário e uma respostas geradas por IA.
Você deve primeiro ler a consulta do usuário e o histórico da conversa cuidadosamente para analisar a tarefa, e então avaliar a qualidade das respostas com base nas e regras fornecidas abaixo.

# Conversa entre Usuário e IA

## Histórico
<|begin_of_history|>

{history}

<|end_of_history|>

## Consulta Atual do Usuário
<|begin_of_query|>

{user_query}

<|end_of_query|>

## Resposta da IA
<|begin_of_response|>

{model_output}

<|end_of_response|>


# Avaliação

## Checklist

<|begin_of_checklist|>

{checklist}

<|end_of_checklist|>

Por favor use este checklist para guiar sua avaliação, mas não limite sua avaliação ao checklist.

## Regras

Você deve comparar a resposta acima com base na sua análise das consultas do usuário e do histórico da conversa.
Você deve primeiro escrever sua análise e o checklist que você usou para a avaliação, e então fornecer sua avaliação de acordo com o checklist.
As pontuações estão no intervalo de 1~10, onde 1 significa que a resposta é muito ruim e 10 significa que a resposta é perfeita.
Aqui estão critérios mais detalhados para as pontuações:

- Pontuação 1~2: A resposta é muito ruim e não faz sentido algum.
- Pontuação 3~4: A resposta é ruim e ajuda o usuário a resolver o problema de maneira significativa.
- Pontuação 5~6: A resposta é razoável mas tem alguns problemas (por exemplo, erros factuais, alucinações, falta de informações-chave).
- Pontuação 7~8: A resposta é boa o suficiente mas poderia ser melhorada de algumas maneiras.
- Pontuação 9~10: A resposta é perfeita e fornece informações úteis que podem ajudar o usuário a resolver o problema.

## Formato de Saída
Primeiro, por favor produza sua análise para a resposta do modelo, e então resuma sua avaliação em dois aspectos: "pontos_fortes" e "pontos_fracos"; Por fim, por favor escreva sua nota para a avaliação.

Por favor forneça seus resultados de avaliação no seguinte formato json preenchendo os placeholders em []:
```
{{
    "pontos_fortes": "[análise dos pontos fortes da resposta]",
    "pontos_fracos": "[análise dos pontos fracos da resposta]",
    "pontuacao": "[1~10]"
}}
```"""

# ---------------------------------------------------------------------------
# Score parser
# ---------------------------------------------------------------------------

# Primary: JSON format with quotes around key
SCORE_PATTERN = re.compile(r'"(?:score|pontuacao|pontuação)"\s*:\s*"?(\d+(?:\.\d+)?)"?')
# Fallback: plain text format without quotes (e.g. "pontuacao: 10")
SCORE_PATTERN_FALLBACK = re.compile(r'(?:score|pontuacao|pontuação)\s*:\s*"?(\d+(?:\.\d+)?)"?')


def parse_score(judgment: str):
    """Extract numeric score (1-10) from judge response."""
    m = SCORE_PATTERN.search(judgment)
    if m:
        return float(m.group(1))
    m = SCORE_PATTERN_FALLBACK.search(judgment)
    if m:
        return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

_write_lock = threading.Lock()


def evaluate_single(
    question,
    answer,
    judge_model,
    output_file,
    max_retries=3,
    reasoning_effort=None,
    api_base=None,
    api_key=None,
):
    """Evaluate a single model response and write result to output_file."""
    conversation_hash = question["conversation_hash"]
    history = question.get("history_text", "")
    user_query = question.get("last_query", "")
    checklist = question.get("checklist", "")
    model_output = get_model_response(answer)

    if not history.strip():
        history = "(Sem histórico anterior)"

    prompt = SCORE_PROMPT_TEMPLATE.format(
        history=history,
        user_query=user_query,
        model_output=model_output,
        checklist=checklist,
    )

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
                pass  # don't pass temperature; use API default
            else:
                common_args["temperature"] = 0
                common_args["max_tokens"] = 2048
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

    score = parse_score(content)

    result = {
        "conversation_hash": conversation_hash,
        "score": score,
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
        description="Prosa holistic 1-10 score evaluation"
    )
    parser.add_argument("--bench-name", type=str, default="prosa")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to evaluate (must have answers in model_answer/)")
    parser.add_argument("--judge-model", type=str, required=True,
                        help="Judge model name")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["low", "medium", "high"])
    parser.add_argument("--question-begin", type=int, default=None)
    parser.add_argument("--question-end", type=int, default=None)
    parser.add_argument("--score-dir", type=str, default=None,
                        help="Custom directory for score output (default: data/{bench}/score_holistic/)")
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
    score_dir = args.score_dir if args.score_dir else os.path.join(data_dir, "score_holistic")
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
    for q in questions:
        h = q["conversation_hash"]
        if h in existing:
            continue
        if h not in answers:
            missing_answers += 1
            continue
        work.append((q, answers[h]))

    if missing_answers:
        print(f"Warning: {missing_answers} questions have no answer for {args.model}")
    print(f"Running {len(work)} evaluations ({args.judge_model} as judge, id={judge_id})")

    # Run
    def run_one(item):
        q, a = item
        return evaluate_single(
            q, a, args.judge_model, output_file,
            max_retries=args.max_retries,
            reasoning_effort=args.reasoning_effort,
            api_base=args.api_base,
            api_key=args.api_key,
        )

    failed = 0
    if args.parallel == 1:
        for item in tqdm(work, desc="Scoring"):
            r = run_one(item)
            if r is None:
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            results = list(tqdm(
                executor.map(run_one, work),
                total=len(work),
                desc="Scoring",
            ))
            failed = sum(1 for r in results if r is None)

    if failed:
        print(f"\nWarning: {failed} evaluations failed. Re-run to retry.")

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
