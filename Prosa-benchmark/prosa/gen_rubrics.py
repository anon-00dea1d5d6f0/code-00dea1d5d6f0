"""Generate evaluation rubrics for Prosa (paper §4.1 Rubric generation).

Uses the rubric generation prompt from the RRD paper (arXiv:2602.05125,
Appendix F.1a). Takes responses from three reference models as inputs.

Usage:
    python -m prosa.gen_rubrics \
        --model-a gpt-5.2-2025-12-11 \
        --model-b Qwen3.5-397B-A17B \
        --model-c gemini-3-pro-preview \
        --judge-model gpt-4.1-2025-04-14 \
        --parallel 50
"""

import argparse
import json
import os
import random
import re
import time
import concurrent.futures
import threading
from typing import Optional, Dict, Any

import tqdm

from prosa.prosa_utils import (
    load_prosa_questions,
    load_prosa_model_answers,
    get_model_response,
)

file_lock = threading.Lock()

# ---------------------------------------------------------------------------
# RRD Paper Prompt — F.1a Initial Rubric Generation (arXiv:2602.05125)
# ---------------------------------------------------------------------------
_RRD_SYSTEM = "Você é um designer de rubricas para um sistema de LLM-as-judge."

_RRD_PROMPT = """\
Entradas que você receberá:
- Histórico: o contexto anterior da conversa.
- Consulta: a tarefa/pergunta que a resposta deve responder.
- Respostas: um conjunto de respostas a ser avaliado com base nas rubricas.

Objetivo: Projetar um conjunto abrangente de rubricas para avaliar respostas à \
consulta fornecida dado o histórico. Escreva apenas rubricas nas quais você tenha confiança. Proponha \
apenas as melhores rubricas.

Requisitos:
- Proponha rubricas que, em conjunto, cubram as dimensões mais importantes \
necessárias para julgar se uma resposta satisfaz corretamente e de forma útil a consulta dado o histórico.
- Use o histórico apenas quando ele for necessário para interpretar corretamente a consulta.
- Não crie critérios baseados em detalhes irrelevantes do histórico.
- Cada rubrica deve poder ser julgada de forma consistente em muitas respostas (evite \
formulações vagas como "boa", "legal", "de alta qualidade").
- Cada rubrica deve ser específica a consulta dado o histórico (vinculada ao que o usuário pediu), e não \
um conselho genérico de escrita.
- Cada rubrica deve ser escrita como um único critério, com limites claros e binários \
de aprovação/reprovação. Prefira verificações objetivas.
- A rubrica NÃO DEVE responder diretamente à pergunta.
- A rubrica NÃO DEVE repetir nenhuma das respostas fornecidas.

Dicas para escrever boas rubricas:
i. MECE:
- Mutuamente Exclusivas, Coletivamente Exaustivas.

ii. Completude:
- Considere todos os elementos que você gostaria de incluir para \
criar uma resposta perfeita e coloque-os na rubrica. Isso significa incluir \
não apenas os fatos e afirmações diretamente solicitados pela consulta dado o histórico, mas também \
os detalhes de suporte que fornecem justificativa, raciocínio e lógica para \
sua resposta. Cada um desses elementos deve ter um critério, porque cada \
critério ajuda a desenvolver a resposta à pergunta sob um ângulo ligeiramente \
diferente.

iii. Sem sobreposição:
- o mesmo erro de um modelo não deve ser punido \
múltiplas vezes.

iv. Diversidade:
- Os itens da rubrica devem incluir tipos variados de informação.
- Se todos os critérios forem do tipo 'a resposta menciona A', 'a resposta menciona B', \
então isso não é uma boa rubrica.

v. Quantos itens de rubrica para cada consulta:
- Não existe um padrão-ouro, e o número desejado de rubricas varia conforme \
os tipos de tarefa.
- Escreva rubricas que cubram todos os aspectos de uma resposta ideal.

vi. Quantos itens de rubrica reprovar:
- Uma boa regra prática é que o modelo reprove em 50 por cento dos itens da rubrica.

vii. Atomicidade / Não empilhadas:
- Cada critério da rubrica deve avaliar exatamente um aspecto distinto. Evite \
agrupar múltiplos critérios em uma única rubrica. A maioria dos critérios empilhados com a \
palavra 'e' pode ser quebrada em múltiplas partes.
- RUIM: A resposta identifica George Washington como o primeiro presidente dos EUA e \
menciona que ele serviu por dois mandatos.
- BOM: A resposta identifica George Washington como o primeiro presidente dos EUA.
- BOM: A resposta menciona que George Washington serviu por dois mandatos.

viii. Especificidade:
- Os critérios devem ser binários (verdadeiro ou falso) e objetivos.
- Evite descrições vagas (por exemplo, 'a resposta deve ser precisa' é vago).
- Exemplo: 'A resposta deve listar exatamente três exemplos.'

ix. Autocontida:
- Cada critério deve conter todas as informações necessárias para avaliar uma resposta. \
Por exemplo: 'Menciona a capital do Canadá' -> RUIM; 'Menciona que a capital do \
Canadá é Ottawa' -> BOM.

x. O critério deve ser verificável sem exigir busca externa:
- RUIM: A resposta nomeia qualquer um dos vencedores do Prêmio Nobel de Física em 2023.
- BOM: A resposta nomeia qualquer um dos seguintes vencedores do Prêmio Nobel de Física em \
2023: Pierre Agostini, Ferenc Krausz ou Anne L'Huillier.

Abaixo estão as entradas:

<|comeco_do_historico|>

{history}

<|fim_do_historico|>

<|comeco_da_consulta|>

{user_query}

<|fim_da_consulta|>


<|comeco_da_resposta_referencia1|>

{reference_response1}

<|fim_da_resposta_referencia1|>

<|comeco_da_resposta_referencia2|>

{reference_response2}

<|fim_da_resposta_referencia2|>

<|comeco_da_resposta_referencia3|>

{reference_response3}

<|fim_da_resposta_referencia3|>


Produza a saída ESTRITAMENTE no formato abaixo. Nenhum outro texto é permitido:
<rubrica> Rubrica 1 </rubrica>
<rubrica> Rubrica 2 </rubrica>
..."""

_RRD_PROMPT_SINGLE_TURN = """\
Entradas que você receberá:
- Consulta: a tarefa/pergunta que a resposta deve responder.
- Respostas: um conjunto de respostas a ser avaliado com base nas rubricas.

Objetivo: Projetar um conjunto abrangente de rubricas para avaliar respostas à \
consulta fornecida. Escreva apenas rubricas nas quais você tenha confiança. Proponha \
apenas as melhores rubricas.

Requisitos:
- Proponha rubricas que, em conjunto, cubram as dimensões mais importantes \
necessárias para julgar se uma resposta satisfaz corretamente e de forma útil a consulta.
- Cada rubrica deve poder ser julgada de forma consistente em muitas respostas (evite \
formulações vagas como "boa", "legal", "de alta qualidade").
- Cada rubrica deve ser específica a consulta (vinculada ao que o usuário pediu), e não \
um conselho genérico de escrita.
- Cada rubrica deve ser escrita como um único critério, com limites claros e binários \
de aprovação/reprovação. Prefira verificações objetivas.
- A rubrica NÃO DEVE responder diretamente à pergunta.
- A rubrica NÃO DEVE repetir nenhuma das respostas fornecidas.

Dicas para escrever boas rubricas:
i. MECE:
- Mutuamente Exclusivas, Coletivamente Exaustivas.

ii. Completude:
- Considere todos os elementos que você gostaria de incluir para \
criar uma resposta perfeita e coloque-os na rubrica. Isso significa incluir \
não apenas os fatos e afirmações diretamente solicitados pela consulta, mas também \
os detalhes de suporte que fornecem justificativa, raciocínio e lógica para \
sua resposta. Cada um desses elementos deve ter um critério, porque cada \
critério ajuda a desenvolver a resposta à pergunta sob um ângulo ligeiramente \
diferente.

iii. Sem sobreposição:
- o mesmo erro de um modelo não deve ser punido \
múltiplas vezes.

iv. Diversidade:
- Os itens da rubrica devem incluir tipos variados de informação.
- Se todos os critérios forem do tipo 'a resposta menciona A', 'a resposta menciona B', \
então isso não é uma boa rubrica.

v. Quantos itens de rubrica para cada consulta:
- Não existe um padrão-ouro, e o número desejado de rubricas varia conforme \
os tipos de tarefa.
- Escreva rubricas que cubram todos os aspectos de uma resposta ideal.

vi. Quantos itens de rubrica reprovar:
- Uma boa regra prática é que o modelo reprove em 50 por cento dos itens da rubrica.

vii. Atomicidade / Não empilhadas:
- Cada critério da rubrica deve avaliar exatamente um aspecto distinto. Evite \
agrupar múltiplos critérios em uma única rubrica. A maioria dos critérios empilhados com a \
palavra 'e' pode ser quebrada em múltiplas partes.
- RUIM: A resposta identifica George Washington como o primeiro presidente dos EUA e \
menciona que ele serviu por dois mandatos.
- BOM: A resposta identifica George Washington como o primeiro presidente dos EUA.
- BOM: A resposta menciona que George Washington serviu por dois mandatos.

viii. Especificidade:
- Os critérios devem ser binários (verdadeiro ou falso) e objetivos.
- Evite descrições vagas (por exemplo, 'a resposta deve ser precisa' é vago).
- Exemplo: 'A resposta deve listar exatamente três exemplos.'

ix. Autocontida:
- Cada critério deve conter todas as informações necessárias para avaliar uma resposta. \
Por exemplo: 'Menciona a capital do Canadá' -> RUIM; 'Menciona que a capital do \
Canadá é Ottawa' -> BOM.

x. O critério deve ser verificável sem exigir busca externa:
- RUIM: A resposta nomeia qualquer um dos vencedores do Prêmio Nobel de Física em 2023.
- BOM: A resposta nomeia qualquer um dos seguintes vencedores do Prêmio Nobel de Física em \
2023: Pierre Agostini, Ferenc Krausz ou Anne L'Huillier.

Abaixo estão as entradas:

<|comeco_da_consulta|>

{user_query}

<|fim_da_consulta|>


<|comeco_da_resposta_referencia1|>

{reference_response1}

<|fim_da_resposta_referencia1|>

<|comeco_da_resposta_referencia2|>

{reference_response2}

<|fim_da_resposta_referencia2|>

<|comeco_da_resposta_referencia3|>

{reference_response3}

<|fim_da_resposta_referencia3|>


Produza a saída ESTRITAMENTE no formato abaixo. Nenhum outro texto é permitido:
<rubrica> Rubrica 1 </rubrica>
<rubrica> Rubrica 2 </rubrica>
..."""


def _parse_rubrics(content: str) -> list[str]:
    """Extract rubrics from <rubrica>...</rubrica> tags (case-insensitive)."""
    rubrics = re.findall(r"<rubrica>\s*(.*?)\s*</rubrica>", content, re.DOTALL | re.IGNORECASE)
    cleaned = []
    for r in rubrics:
        # Strip auto-generated prefixes like "Rubrica 1:", "Nova rubrica 2:", etc.
        r = re.sub(r"^(?:Nova?\s+)?[Rr]ubrica\s*\d+\s*[:\.]\s*", "", r.strip())
        if r:
            cleaned.append(r)
    return cleaned


def generate_checklist_rrd(
    question: Dict[str, Any],
    answers_a: Dict[str, Dict],
    answers_b: Dict[str, Dict],
    answers_c: Dict[str, Dict],
    model_a_name: str,
    model_b_name: str,
    model_c_name: str,
    judge_model: str,
    output_file: str,
    max_retries: int = 3,
    reasoning_effort: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Generate a checklist for a single question using the RRD prompt."""
    conversation_hash = question["conversation_hash"]
    history = question["history_text"]
    user_query = question["last_query"]

    # Get responses from all three models
    answer_a = answers_a.get(conversation_hash)
    answer_b = answers_b.get(conversation_hash)
    answer_c = answers_c.get(conversation_hash)

    if not answer_a:
        print(f"Skipping {conversation_hash}: missing answer from {model_a_name}")
        return None
    if not answer_b:
        print(f"Skipping {conversation_hash}: missing answer from {model_b_name}")
        return None
    if not answer_c:
        print(f"Skipping {conversation_hash}: missing answer from {model_c_name}")
        return None

    response_a = get_model_response(answer_a)
    response_b = get_model_response(answer_b)
    response_c = get_model_response(answer_c)

    # Apply pre-computed order for this question (thread-safe)
    responses_list = [
        (model_a_name, response_a),
        (model_b_name, response_b),
        (model_c_name, response_c),
    ]
    order = question.get("_rrd_order")
    if order:
        responses = [responses_list[i] for i in order]
    else:
        random.shuffle(responses_list)
        responses = responses_list
    assignment = {
        "ref1": responses[0][0],
        "ref2": responses[1][0],
        "ref3": responses[2][0],
    }

    # Choose prompt based on whether there is conversation history
    has_history = bool(history and history.strip())
    if has_history:
        user_message = _RRD_PROMPT.format(
            history=history,
            user_query=user_query,
            reference_response1=responses[0][1],
            reference_response2=responses[1][1],
            reference_response3=responses[2][1],
        )
    else:
        user_message = _RRD_PROMPT_SINGLE_TURN.format(
            user_query=user_query,
            reference_response1=responses[0][1],
            reference_response2=responses[1][1],
            reference_response3=responses[2][1],
        )

    # Call judge
    content = None
    for attempt in range(max_retries):
        try:
            import openai
            client_kwargs = {}
            if api_base:
                client_kwargs["base_url"] = api_base
            if api_key:
                client_kwargs["api_key"] = api_key
            client = openai.OpenAI(**client_kwargs)
            messages = [
                {"role": "system", "content": _RRD_SYSTEM},
                {"role": "user", "content": user_message},
            ]
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
                pass  # don't set temperature for reasoning models
            else:
                common_args["temperature"] = 0
                common_args["max_tokens"] = 4096
            if reasoning_effort:
                common_args["reasoning_effort"] = reasoning_effort
            response = client.chat.completions.create(**common_args)
            content = response.choices[0].message.content
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Error for {conversation_hash} (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Failed {conversation_hash} after {max_retries} attempts: {e}")

    if content is None:
        return None

    # Parse rubrics from <rubric> tags
    rubrics = _parse_rubrics(content)

    if not rubrics:
        print(f"Warning: no rubrics parsed for {conversation_hash}, skipping (will retry)")
        return None

    # Build checklist as a compatible string (numbered list, like the original script)
    checklist_str = "\n".join(f"{i}. {r}" for i, r in enumerate(rubrics, 1))

    result = {
        "conversation_hash": conversation_hash,
        "checklist": checklist_str,
    }

    with file_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result


def load_existing(output_file: str) -> set:
    existing = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        existing.add(d["conversation_hash"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return existing


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation checklists for Prosa using the RRD prompt"
    )
    parser.add_argument(
        "--bench-name", type=str, default="prosa",
        help="Name of the benchmark (default: prosa)",
    )
    parser.add_argument(
        "--model-a", type=str, required=True,
        help="First model name (must have answers in model_answer/)",
    )
    parser.add_argument(
        "--model-b", type=str, required=True,
        help="Second model name (must have answers in model_answer/)",
    )
    parser.add_argument(
        "--model-c", type=str, required=True,
        help="Third model name (must have answers in model_answer/)",
    )
    parser.add_argument(
        "--judge-model", type=str, default="gpt-4.1-2025-04-14",
        help="Judge model for rubric generation (default: gpt-4.1-2025-04-14)",
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Number of parallel API calls (default: 1)",
    )
    parser.add_argument(
        "--output-file", type=str, default=None,
        help="Output file path (optional, auto-generated if omitted)",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Maximum retry attempts per question (default: 3)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for response order randomization (default: 42)",
    )
    parser.add_argument(
        "--reasoning-effort", type=str, default=None,
        choices=["low", "medium", "high"],
        help="Reasoning effort level for the judge model (e.g., gpt-5, o1, o3)",
    )
    parser.add_argument(
        "--api-base", type=str, default=None,
        help="API base URL for the judge model (OpenAI-compatible)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key for the judge model",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Paths
    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", args.bench_name,
    )
    question_file = os.path.join(data_dir, "question.jsonl")
    answer_dir = os.path.join(data_dir, "model_answer")

    # Output
    if args.output_file:
        output_file = args.output_file
    else:
        checklist_dir = os.path.join(data_dir, "checklist")
        os.makedirs(checklist_dir, exist_ok=True)
        output_file = os.path.join(
            checklist_dir,
            f"rrd_{args.model_a}_{args.model_b}_{args.model_c}_by_{args.judge_model}.jsonl",
        )

    # Load data
    print(f"Loading questions from {question_file}")
    questions = load_prosa_questions(question_file)
    print(f"Loaded {len(questions)} questions")

    model_names = [args.model_a, args.model_b, args.model_c]
    all_answers = {}
    for model_name in model_names:
        answer_file = os.path.join(answer_dir, f"{model_name}.jsonl")
        print(f"Loading answers for {model_name} from {answer_file}")
        all_answers[model_name] = load_prosa_model_answers(answer_file)
        print(f"Loaded {len(all_answers[model_name])} answers for {model_name}")

    # Resume
    existing = load_existing(output_file)
    if existing:
        print(f"Found {len(existing)} existing checklists (resuming)")

    questions_to_process = [
        q for q in questions
        if q["conversation_hash"] not in existing
        and q["conversation_hash"] in all_answers[args.model_a]
        and q["conversation_hash"] in all_answers[args.model_b]
        and q["conversation_hash"] in all_answers[args.model_c]
    ]
    print(f"Processing {len(questions_to_process)} questions")

    # Pre-compute shuffle orders (thread-safe: uses seed deterministically)
    for q in questions_to_process:
        order = [0, 1, 2]
        random.shuffle(order)
        q["_rrd_order"] = order

    if not questions_to_process:
        print("No new questions to process")
    else:
        print(f"Generating RRD checklists with {args.parallel} parallel workers")
        print(f"Judge: {args.judge_model}")
        print(f"Models: {args.model_a}, {args.model_b}, {args.model_c}")
        print(f"Seed: {args.seed}")
        print(f"Output: {output_file}")

        def run_batch(batch, desc):
            """Run a batch of questions and return the list of failed ones."""
            failed_questions = []
            futures_map = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
                for question in batch:
                    future = executor.submit(
                        generate_checklist_rrd,
                        question,
                        all_answers[args.model_a],
                        all_answers[args.model_b],
                        all_answers[args.model_c],
                        args.model_a,
                        args.model_b,
                        args.model_c,
                        args.judge_model,
                        output_file,
                        args.max_retries,
                        args.reasoning_effort,
                        args.api_base,
                        args.api_key,
                    )
                    futures_map[future] = question

                for future in tqdm.tqdm(
                    concurrent.futures.as_completed(futures_map),
                    total=len(futures_map),
                    desc=desc,
                ):
                    question = futures_map[future]
                    try:
                        result = future.result()
                        if result is None:
                            failed_questions.append(question)
                    except Exception as e:
                        print(f"Error: {e}")
                        failed_questions.append(question)

            return failed_questions

        # First pass
        failed = run_batch(questions_to_process, "Generating RRD checklists")

        # Retry pass for failed questions
        if failed:
            print(f"\n{len(failed)} questions failed. Retrying...")
            time.sleep(5)
            still_failed = run_batch(failed, "Retrying failed")

            if still_failed:
                print(f"\n{'='*70}")
                print(f"WARNING: {len(still_failed)} questions failed after all retries:")
                for q in still_failed:
                    print(f"  - {q['conversation_hash']}")
                print("Run the same command again to retry these questions.")
                print(f"{'='*70}")

    # Count total
    total = len(load_existing(output_file))
    print(f"Total: {total} checklists in {output_file}")
    print("Done!")


if __name__ == "__main__":
    main()
