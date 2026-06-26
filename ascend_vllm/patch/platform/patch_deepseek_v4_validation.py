from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from transformers import PreTrainedTokenizerFast
from vllm.tokenizers import deepseek_v4 as deepseek_v4_tokenizer
from vllm_ascend.patch.platform import patch_deepseek_v4_thinking as ascend_patch

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionMessageParam

_VALIDATION_PATCH_APPLIED = False


def _patched_get_deepseek_v4_tokenizer(tokenizer: deepseek_v4_tokenizer.HfTokenizer):
    dsv4_tokenizer = copy.copy(tokenizer)

    added_vocab = tokenizer.get_added_vocab()
    added_vocab_size = len(added_vocab)
    tokenizer_vocab_size = tokenizer.vocab_size

    class _DeepseekV4Tokenizer(tokenizer.__class__):
        @classmethod
        def validate_messages(cls, messages):
            try:
                if len(messages) > 0 and messages[-1].get("role") == "system":
                    return (
                        False,
                        (
                            f"Invalid system message at message index {len(messages) - 1}, "
                            "system message can only be the first message in the conversation"
                        ),
                    )

                if len(messages) > 0 and messages[-1].get("role") == "assistant":
                    if isinstance(messages[-1].get("prefix"), bool) and messages[-1].get("prefix"):
                        return True, ""
                    if messages[-1].get("tool_calls") is not None:
                        return True, ""
                    return False, f"Invalid consecutive assistant message at message index {len(messages) - 1}"

                for index in range(len(messages)):
                    msg = messages[index]
                    role = msg.get("role")
                    if role not in ["system", "developer", "user", "tool", "assistant"]:
                        return False, f"Unkown role: {role}. Role just can be [system, developer, user, tool,assistant]"
                return True, ""
            except Exception:
                return True, ""

        @classmethod
        def validate_sufficient_tools(cls, messages):
            try:
                for index in range(len(messages)):
                    msg = messages[index]
                    role = msg.get("role")
                    if role == "assistant":
                        tool_calls_id = []
                        message_tool_id = []
                        tool_calls = msg.get("tool_calls")
                        if isinstance(tool_calls, list) and len(tool_calls) == 0:
                            return (
                                False,
                                (
                                    f"Invalid message at message[{index}].tool_calls: empty error. "
                                    "Expected an array with minimum length 1, but got an empty array instead."
                                ),
                            )
                        if tool_calls and isinstance(tool_calls, list):
                            tool_counter = 0
                            for tool_call in tool_calls:
                                tool_counter += 1
                                if (
                                    index + tool_counter >= len(messages)
                                    or messages[index + tool_counter].get("role") != "tool"
                                ):
                                    return (
                                        False,
                                        "An assistant message with tool calls must be followed by a tool messages "
                                        "responding to each tool call",
                                    )
                                else:
                                    tool_calls_id.append(tool_call.get("id"))
                                    message_tool_id.append(messages[index + tool_counter].get("tool_call_id"))
                            if sorted(tool_calls_id) != sorted(message_tool_id):
                                return (
                                    False,
                                    "An assistant message with tool calls must be followed by a tool messages "
                                    "responding to each tool call",
                                )
                            index += tool_counter
                return True, ""
            except Exception:
                return True, ""

        def apply_chat_template(
            self,
            messages: list[ChatCompletionMessageParam],
            tools: list[dict[str, Any]] | None = None,
            **kwargs,
        ) -> str | list[int]:
            conversation = kwargs.get("conversation", messages)
            msgs_to_validte = conversation.copy()

            valid, err = self.validate_messages(msgs_to_validte)
            if not valid:
                raise ValueError(err)

            valid, err = self.validate_sufficient_tools(msgs_to_validte)
            if not valid:
                raise ValueError(err)

            thinking = kwargs.get("thinking", False)
            enable_thinking = kwargs.get("enable_thinking", False)
            thinking = thinking or enable_thinking
            thinking_mode = "thinking" if thinking else "chat"

            messages = conversation.copy()
            if tools is not None and len(tools) > 0:
                messages.insert(0, {"role": "system"})
                messages[0]["tools"] = tools

            reasoning_effort = kwargs.get("reasoning_effort")
            if not isinstance(reasoning_effort, str):
                reasoning_effort = None
            elif reasoning_effort == "none":
                thinking_mode = "chat"
                reasoning_effort = None
            elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
            else:
                reasoning_effort = "high"

            prompt_str = deepseek_v4_tokenizer.encode_messages(
                messages,
                thinking_mode=thinking_mode,
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=reasoning_effort,
            )

            if kwargs.get("tokenize", True):
                tokenizer_kwargs = {k: kwargs[k] for k in ("truncation", "max_length") if k in kwargs}
                return self.encode(
                    prompt_str,
                    add_special_tokens=False,
                    **tokenizer_kwargs,
                )

            return prompt_str

        def num_special_tokens_to_add(self) -> int:
            return len(self.encode(""))

        def __len__(self) -> int:
            return tokenizer_vocab_size + added_vocab_size

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

        def __reduce__(self):
            return _patched_get_deepseek_v4_tokenizer, (tokenizer,)

    _DeepseekV4Tokenizer.__name__ = f"DSV4{tokenizer.__class__.__name__}"

    dsv4_tokenizer.__class__ = _DeepseekV4Tokenizer
    return dsv4_tokenizer


def _patched_deepseek_v4_from_pretrained(cls, *args, **kwargs):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(*args, **kwargs)
    return deepseek_v4_tokenizer.get_cached_tokenizer(_patched_get_deepseek_v4_tokenizer(tokenizer))


def apply_patch() -> None:
    global _VALIDATION_PATCH_APPLIED

    if _VALIDATION_PATCH_APPLIED:
        return

    deepseek_v4_tokenizer.get_deepseek_v4_tokenizer = _patched_get_deepseek_v4_tokenizer
    deepseek_v4_tokenizer.DeepseekV4Tokenizer.from_pretrained = classmethod(_patched_deepseek_v4_from_pretrained)
    ascend_patch._patched_get_deepseek_v4_tokenizer = _patched_get_deepseek_v4_tokenizer

    _VALIDATION_PATCH_APPLIED = True


apply_patch()
