"""
llm/providers.py — LLM Provider 抽象层

支持:
  - Anthropic Claude（Prompt Caching + Tool Use）
  - OpenAI GPT / Codex（Function Calling）
  - 运行时模型切换
"""
import json
from abc import ABC, abstractmethod
from typing import Optional

from core import config


class ToolCall:
    """统一的工具调用表示"""
    def __init__(self, call_id: str, name: str, arguments: dict):
        self.call_id = call_id
        self.name = name
        self.arguments = arguments


class ChatResult:
    """统一的 LLM 响应"""
    def __init__(
        self,
        text: str = "",
        tool_calls: list[ToolCall] = None,
        stats: dict = None,
        stop_reason: str = "end",
    ):
        self.text = text
        self.tool_calls = tool_calls or []
        self.stats = stats or {}
        self.stop_reason = stop_reason  # "end" | "tool_use"

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMProvider(ABC):
    @abstractmethod
    def chat(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict] = None,
        max_tokens: int = 4096,
    ) -> ChatResult:
        ...

    @abstractmethod
    def provider_name(self) -> str:
        ...

    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def set_model(self, model: str):
        ...

    @abstractmethod
    def format_tool_result(self, call_id: str, result: dict) -> dict:
        """将工具执行结果格式化为该 Provider 的消息格式"""
        ...


class ClaudeProvider(LLMProvider):
    def __init__(self, model: str = None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self._model = model or config.CLAUDE_MODEL

    def chat(self, system_prompt: str, messages: list[dict], tools: list[dict] = None, max_tokens: int = 4096) -> ChatResult:
        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
        }
        if tools:
            from core.tools import get_claude_tools
            kwargs["tools"] = get_claude_tools()

        response = self.client.messages.create(**kwargs)
        usage = response.usage
        stats = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_tokens": getattr(usage, "cache_read_input_tokens", 0),
        }

        text = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    call_id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        stop = "tool_use" if response.stop_reason == "tool_use" else "end"
        return ChatResult(text=text, tool_calls=tool_calls, stats=stats, stop_reason=stop)

    def format_tool_result(self, call_id: str, result: dict) -> dict:
        return {
            "type": "tool_result",
            "tool_use_id": call_id,
            "content": json.dumps(result, ensure_ascii=False),
        }

    def provider_name(self) -> str:
        return "claude"

    def model_name(self) -> str:
        return self._model

    def set_model(self, model: str):
        self._model = model


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = None):
        from openai import OpenAI
        kwargs = {"api_key": config.OPENAI_API_KEY}
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        self.client = OpenAI(**kwargs)
        self._model = model or config.OPENAI_MODEL

    def chat(self, system_prompt: str, messages: list[dict], tools: list[dict] = None, max_tokens: int = 4096) -> ChatResult:
        openai_messages = [{"role": "system", "content": system_prompt}]
        openai_messages.extend(messages)

        kwargs = {
            "model": self._model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
        }
        if tools:
            from core.tools import get_openai_tools
            kwargs["tools"] = get_openai_tools()

        response = self.client.chat.completions.create(**kwargs)
        usage = response.usage
        stats = {
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "cached_tokens": 0,
        }

        choice = response.choices[0]
        text = choice.message.content or ""
        tool_calls = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(
                    call_id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        stop = "tool_use" if choice.finish_reason == "tool_calls" or tool_calls else "end"
        return ChatResult(text=text, tool_calls=tool_calls, stats=stats, stop_reason=stop)

    def format_tool_result(self, call_id: str, result: dict) -> dict:
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(result, ensure_ascii=False),
        }

    def provider_name(self) -> str:
        return "openai"

    def model_name(self) -> str:
        return self._model

    def set_model(self, model: str):
        self._model = model


def create_provider(provider_type: str = None, model: str = None) -> LLMProvider:
    provider = (provider_type or config.LLM_PROVIDER).lower()
    if provider == "claude":
        return ClaudeProvider(model=model)
    elif provider == "openai":
        return OpenAIProvider(model=model)
    else:
        raise ValueError(f"不支持的 Provider: {provider}，可选 claude / openai")


def provider_for_model(model_id: str) -> tuple[str, LLMProvider]:
    from core.conversations import AVAILABLE_MODELS
    for provider_type, models in AVAILABLE_MODELS.items():
        for m in models:
            if m["id"] == model_id:
                return provider_type, create_provider(provider_type, model=model_id)
    if model_id.startswith("claude"):
        return "claude", create_provider("claude", model=model_id)
    else:
        return "openai", create_provider("openai", model=model_id)
