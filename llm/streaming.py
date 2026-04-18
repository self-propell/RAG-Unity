"""
Streaming utilities for LangGraph agent
"""
from dataclasses import dataclass
from typing import AsyncIterator, Literal


@dataclass
class StreamEvent:
    """Unified streaming event"""
    type: Literal["token", "tool_start", "tool_end", "node_start", "node_end", "error"]
    content: str = ""
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "content": self.content,
            "metadata": self.metadata
        }


@dataclass
class ChatChunk:
    """Chunk from LLM streaming"""
    text: str = ""
    tool_call: dict = None
    finish_reason: str = None

    def __post_init__(self):
        if self.tool_call is None:
            self.tool_call = {}
