"""Generate model answers for Prosa evaluation.

This script generates responses for the 1,000 Prosa benchmark questions.
The model receives the conversation history as context and produces a
single response to the last user turn.

Usage:
python -m prosa.gen_answer \
    --bench-name prosa \
    --model gpt-4.1-2025-04-14 \
    --parallel 50
"""

import argparse
import json
import os
import time
import concurrent.futures
from typing import Optional, Dict, Any, List
import threading

import tqdm

from prosa.common import (
    API_ERROR_OUTPUT,
    chat_completion_openai,
)
from prosa.prosa_utils import load_prosa_questions
from prosa.conversation import get_conv_template


# Thread-safe file writing
file_lock = threading.Lock()


def get_answer(
    question: Dict[str, Any],
    model: str,
    model_id: str,
    max_tokens: Optional[int],
    temperature: Optional[float],
    answer_file: str,
    verbose: bool,
    api_base: Optional[str],
    api_key: Optional[str],
    max_retries: int = 3,
    reasoning_effort: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Generate an answer for a Prosa question.

    Args:
        question: Question dict with conversation_input and last_query
        model: Model name for API calls
        model_id: Identifier for the model in output
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        answer_file: Path to output file
        verbose: Print debug information
        api_base: Optional API base URL
        api_key: Optional API key
        max_retries: Maximum number of retry attempts (default: 3)
        reasoning_effort: Reasoning effort level (e.g., low, medium, high)

    Returns:
        Answer dict with conversation_hash and output, or None if failed
    """
    conversation_hash = question["conversation_hash"]
    conversation_input = question["conversation_input"]

    # Build conversation context: all messages (last message is always from user)
    context_messages = conversation_input

    # Create conversation template
    conv = get_conv_template("zero_shot")

    # Add all context messages to the conversation
    for msg in context_messages:
        if msg["role"] == "user":
            conv.append_message(conv.roles[0], msg["content"])
        elif msg["role"] == "assistant":
            conv.append_message(conv.roles[1], msg["content"])

    if verbose:
        print(100 * "-")
        print(f"Conversation Hash: {conversation_hash}")
        print(f"Category: {question.get('category', 'unknown')}")
        print(f"Num Turns: {question.get('num_turns', 1)}")
        print(f"Last Query: {question['last_query'][:100]}...")
        print(f"Context Messages: {len(context_messages)}")

    # Generate response with retry logic
    content = None
    last_error = None

    for attempt in range(max_retries):
        try:
            content = chat_completion_openai(
                model, conv, temperature, max_tokens,
                api_base=api_base, api_key=api_key,
                reasoning_effort=reasoning_effort,
            )
            # Success - break out of retry loop
            break
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"Error generating answer for {conversation_hash} (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Failed to generate answer for {conversation_hash} after {max_retries} attempts: {e}")

    # If all retries failed or API returned error sentinel, return None
    if content is None or content == API_ERROR_OUTPUT:
        return None

    if verbose:
        print(f"Response: {content[:200]}...")
        print(100 * "-")

    # Build the answer record
    answer = {
        "conversation_hash": conversation_hash,
        "output": [content],
    }

    # Write to file (thread-safe)
    with file_lock:
        with open(answer_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(answer, ensure_ascii=False) + "\n")

    return answer


def load_existing_answers(answer_file: str) -> set:
    """Load existing answers to avoid regenerating."""
    existing = set()
    if os.path.exists(answer_file):
        with open(answer_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        ans = json.loads(line)
                        existing.add(ans["conversation_hash"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return existing


def count_answers(answer_file: str) -> int:
    """Count the number of answers in a JSONL file."""
    count = 0
    if os.path.exists(answer_file):
        with open(answer_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Generate model answers for Prosa evaluation"
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="prosa",
        help="Name of the benchmark (default: prosa)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name for API calls (e.g., gpt-4.1-2025-04-14)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Model identifier for output file (defaults to --model)",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        default=None,
        help="Start question index (for debugging)",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        default=None,
        help="End question index (for debugging)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate (default: None, uses model default)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: None, uses model default)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel API calls (default: 1)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base URL for OpenAI-compatible APIs",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (or use environment variable OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--answer-file",
        type=str,
        default=None,
        help="Output answer file path (optional)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug information",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from existing answers (regenerate all)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts per question (default: 3)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        choices=["none", "low", "medium", "high"],
        help="Reasoning effort level for reasoning models (e.g., gpt-5, o1, o3)",
    )
    args = parser.parse_args()

    # Set model_id
    model_id = args.model_id or args.model

    # Set up paths
    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        args.bench_name,
    )
    question_file = os.path.join(data_dir, "question.jsonl")
    answer_dir = os.path.join(data_dir, "model_answer")

    # Output file
    if args.answer_file:
        answer_file = args.answer_file
    else:
        os.makedirs(answer_dir, exist_ok=True)
        answer_file = os.path.join(answer_dir, f"{model_id}.jsonl")

    # Load questions
    print(f"Loading questions from {question_file}")
    questions = load_prosa_questions(
        question_file, args.question_begin, args.question_end
    )
    print(f"Loaded {len(questions)} questions")

    # Load existing answers (resume by default)
    existing_hashes = set()
    if not args.no_resume:
        existing_hashes = load_existing_answers(answer_file)
        if existing_hashes:
            print(f"Found {len(existing_hashes)} existing answers (use --no-resume to regenerate)")

    # Filter questions to process
    questions_to_process = [
        q for q in questions
        if q["conversation_hash"] not in existing_hashes
    ]
    print(f"Processing {len(questions_to_process)} questions")

    if not questions_to_process:
        print("No new questions to process")
    else:
        # Process questions
        print(f"Generating answers with {args.parallel} parallel workers")
        print(f"Output: {answer_file}")

        def run_batch(batch, desc):
            """Run a batch of questions and return the list of failed ones."""
            failed_questions = []
            futures_map = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
                for question in batch:
                    future = executor.submit(
                        get_answer,
                        question,
                        args.model,
                        model_id,
                        args.max_tokens,
                        args.temperature,
                        answer_file,
                        args.verbose,
                        args.api_base,
                        args.api_key,
                        args.max_retries,
                        args.reasoning_effort,
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
        failed = run_batch(questions_to_process, "Generating answers")

        # Retry pass for failed questions
        if failed:
            print(f"\n{len(failed)} questions failed. Retrying...")
            time.sleep(5)
            still_failed = run_batch(failed, "Retrying failed")

            if still_failed:
                # Save as $ERROR$ and warn
                print(f"\n{'='*70}")
                print(f"WARNING: {len(still_failed)} questions failed after all retries.")
                print(f"Saving as $ERROR$:")
                for q in still_failed:
                    h = q["conversation_hash"]
                    print(f"  - {h}")
                    error_answer = {
                        "conversation_hash": h,
                        "output": [API_ERROR_OUTPUT],
                    }
                    with file_lock:
                        with open(answer_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(error_answer, ensure_ascii=False) + "\n")
                print(f"{'='*70}")

    num_answers = count_answers(answer_file)
    print(f"Total: {num_answers} answers in {answer_file}")
    print("Done!")


if __name__ == "__main__":
    main()
