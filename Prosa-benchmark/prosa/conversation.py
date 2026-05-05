"""Minimal conversation template for OpenAI-compatible API message formatting."""

import dataclasses
from typing import Union, Tuple, List


@dataclasses.dataclass
class Conversation:
    """Manages prompt templates and conversation history."""

    name: str
    system_template: str = "{system_message}"
    system_message: str = ""
    roles: Tuple[str] = ("USER", "ASSISTANT")
    messages: List[List[str]] = ()
    offset: int = 0
    sep: str = "\n"
    sep2: str = None
    stop_str: Union[str, List[str]] = None
    stop_token_ids: List[int] = None

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        if self.system_message == "":
            ret = []
        else:
            ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )


# Default conversation template (used by all Prosa scripts)
_default_conv = Conversation(
    name="zero_shot",
    system_message=(
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    ),
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep="\n",
)


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template by name.

    For Prosa, all templates use the same simple format
    since we only need to_openai_api_messages().
    """
    return _default_conv.copy()


def get_conversation_template(model_path: str) -> Conversation:
    """Get the default conversation template for a model.

    In standalone mode, this returns the same default template
    since Prosa only uses to_openai_api_messages().
    """
    return _default_conv.copy()
