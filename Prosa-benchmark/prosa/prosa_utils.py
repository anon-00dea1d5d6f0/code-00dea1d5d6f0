"""
Prosa specific utilities for evaluation.

This module provides functions for evaluating models using the Prosa methodology,
which focuses on evaluating only the last turn of multi-turn conversations while using
previous turns as context.
"""

import json
from typing import Optional, Dict, Any




def load_prosa_questions(question_file: str, begin: Optional[int] = None, end: Optional[int] = None):
    """Load Prosa questions from a file.

    The Prosa format includes:
    - conversation_hash: Unique identifier for the conversation
    - conversation_input: Full conversation in OpenAI format
    - last_query: The last user query to be evaluated
    - history_text: Plain text history for judge (USER: ... ASSISTANT: ...)
    - num_turns: Number of user turns in the conversation
    - difficulty: Dict with 'intent', 'knowledge', and 'level' fields
    """
    questions = []
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    if begin is not None or end is not None:
        questions = questions[begin:end]
    return questions


def load_prosa_model_answers(answer_file: str) -> Dict[str, Dict]:
    """Load model answers from a JSONL file.

    Returns:
        Dict mapping conversation_hash to answer dict
    """
    answers = {}
    with open(answer_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                answers[item["conversation_hash"]] = item
    return answers


def get_model_response(answer: Dict[str, Any]) -> str:
    """Extract model response from an answer dict."""
    if "output" in answer:
        output = answer["output"]
        return output[0] if isinstance(output, list) else output
    else:
        raise ValueError(f"Unknown answer format: {list(answer.keys())}")


