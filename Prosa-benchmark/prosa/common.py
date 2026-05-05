"""OpenAI-compatible chat-completion helper for Prosa."""

import time

import openai


API_MAX_RETRY = 3
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"


def chat_completion_openai(model, conv, temperature, max_tokens=None, api_base=None, api_key=None, reasoning_effort=None):
    client_kwargs = {}
    if api_base:
        client_kwargs["base_url"] = api_base
    if api_key:
        client_kwargs["api_key"] = api_key

    client = openai.OpenAI(**client_kwargs)

    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()

            common_args = {
                "model": model,
                "messages": messages,
                "n": 1,
            }

            if temperature is not None:
                common_args["temperature"] = temperature

            if reasoning_effort:
                common_args["reasoning_effort"] = reasoning_effort

            # Use 'max_completion_tokens' for reasoning models
            if model.startswith("o1") or model.startswith("o3") or model.startswith("gpt-5"):
                if max_tokens is not None:
                    common_args["max_completion_tokens"] = max_tokens
            else:
                if max_tokens is not None:
                    common_args["max_tokens"] = max_tokens

            response = client.chat.completions.create(**common_args)
            return response.choices[0].message.content
        except openai.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return API_ERROR_OUTPUT
