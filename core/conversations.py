"""
core/conversations.py — 对话持久化管理

功能：
  - 多对话框（类似 ChatGPT / Claude 网页）
  - 按工程隔离存储
  - 自动保存 / 加载 / 删除 / 重命名
  - 记录使用的模型
"""
import os
import json
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

from core import config


CONVERSATIONS_ROOT = str(Path(config._project_root) / "conversations")


@dataclass
class ConversationMeta:
    """对话元信息"""
    conv_id: str
    title: str
    model: str               # 创建时使用的模型
    project_id: str           # 关联的工程 ID
    created_at: float = 0.0
    updated_at: float = 0.0
    message_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


@dataclass
class Conversation:
    """完整对话（元信息 + 消息）"""
    meta: ConversationMeta
    messages: list[dict]      # [{"role": "user"/"assistant", "content": "..."}]

    def to_dict(self) -> dict:
        return {
            "meta": asdict(self.meta),
            "messages": self.messages,
        }

    @staticmethod
    def from_dict(d: dict) -> "Conversation":
        meta = ConversationMeta(**d["meta"])
        return Conversation(meta=meta, messages=d.get("messages", []))


class ConversationManager:
    """对话管理器"""

    def __init__(self, project_id: str = "_global"):
        self._project_id = project_id
        self._dir = os.path.join(CONVERSATIONS_ROOT, project_id)
        os.makedirs(self._dir, exist_ok=True)

    def _conv_path(self, conv_id: str) -> str:
        return os.path.join(self._dir, f"{conv_id}.json")

    # ── CRUD ──────────────────────────────────

    def create(self, model: str, title: str = "") -> Conversation:
        conv_id = uuid.uuid4().hex[:10]
        now = time.time()
        meta = ConversationMeta(
            conv_id=conv_id,
            title=title or "新对话",
            model=model,
            project_id=self._project_id,
            created_at=now,
            updated_at=now,
        )
        conv = Conversation(meta=meta, messages=[])
        self._save(conv)
        return conv

    def load(self, conv_id: str) -> Optional[Conversation]:
        path = self._conv_path(conv_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return Conversation.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError):
            return None

    def save(self, conv: Conversation):
        conv.meta.updated_at = time.time()
        conv.meta.message_count = len(conv.messages)
        self._save(conv)

    def _save(self, conv: Conversation):
        path = self._conv_path(conv.meta.conv_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(conv.to_dict(), f, ensure_ascii=False, indent=2)

    def delete(self, conv_id: str) -> bool:
        path = self._conv_path(conv_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def rename(self, conv_id: str, new_title: str) -> bool:
        conv = self.load(conv_id)
        if not conv:
            return False
        conv.meta.title = new_title
        self.save(conv)
        return True

    def list_conversations(self) -> list[ConversationMeta]:
        """列出所有对话，按更新时间降序"""
        result = []
        if not os.path.isdir(self._dir):
            return result
        for fname in os.listdir(self._dir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(self._dir, fname), encoding="utf-8") as f:
                    data = json.load(f)
                result.append(ConversationMeta(**data["meta"]))
            except (json.JSONDecodeError, KeyError):
                continue
        return sorted(result, key=lambda c: c.updated_at, reverse=True)

    def auto_title(self, conv: Conversation, first_message: str):
        """用第一条消息自动生成标题"""
        title = first_message.strip()[:40]
        if len(first_message.strip()) > 40:
            title += "..."
        conv.meta.title = title

    def update_token_stats(self, conv: Conversation, input_tokens: int, output_tokens: int):
        conv.meta.total_input_tokens += input_tokens
        conv.meta.total_output_tokens += output_tokens


# ── 可用模型列表 ──────────────────────────────────

AVAILABLE_MODELS = {
    "claude": [
        {"id": "claude-opus-4-6", "name": "Claude Opus 4.6", "desc": "最强推理"},
        {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "desc": "性价比"},
        {"id": "claude-haiku-4-5", "name": "Claude Haiku 4.5", "desc": "最快"},
    ],
    "openai": [
        {"id": "gpt-5.4", "name": "GPT-5.4", "desc": "最新"},
        {"id": "gpt-5.3-codex", "name": "GPT-5.3 Codex", "desc": "代码专用(新)"},
        {"id": "gpt-5.1-codex", "name": "GPT-5.1 Codex", "desc": "代码专用"},
        {"id": "gpt-4o", "name": "GPT-4o", "desc": "通用"},
        {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "desc": "轻量"},
    ],
}


def get_models_for_provider(provider: str = None) -> list[dict]:
    p = (provider or config.LLM_PROVIDER).lower()
    return AVAILABLE_MODELS.get(p, [])


def get_all_models() -> list[dict]:
    result = []
    for provider, models in AVAILABLE_MODELS.items():
        for m in models:
            result.append({**m, "provider": provider})
    return result
