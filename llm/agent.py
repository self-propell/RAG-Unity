"""
llm/agent.py — RAG Agent 核心

功能：
  - 多轮对话 + RAG 检索
  - Tool Use 循环（LLM 可调用本地工具操作文件）
  - 对话持久化
  - 运行时模型切换
"""
import json
from typing import Optional

from core import config
from core.projects import ProjectInfo
from core.conversations import ConversationManager, Conversation
from core.tools import execute_tool, TOOL_DEFINITIONS
from core.skills import load_all_skills, get_skill_by_name, format_skills_for_prompt
from rag.retriever import Retriever, SearchResult
from llm.providers import LLMProvider, ChatResult, create_provider, provider_for_model


# ────────────────────────────────────────────────
# System Prompt
# ────────────────────────────────────────────────

def build_system_prompt(
    retriever: Retriever,
    project: Optional[ProjectInfo] = None,
    extra_context: str = "",
) -> str:
    all_classes = retriever.list_all_classes()
    scenes = retriever.list_scenes()

    class_list = "\n".join(
        f"  - {c['class_name']} ({c['relative_path']})"
        for c in all_classes[:50]
    ) or "  （尚未索引）"

    scene_list = "\n".join(
        f"  - {s['relative_path']}" for s in scenes
    ) or "  （无场景文件）"

    unity_version = project.unity_version if project else config.UNITY_VERSION
    render_pipeline = project.render_pipeline if project else config.RENDER_PIPELINE
    project_path_str = project.path if project else "(未指定)"

    # 加载已安装 skill
    skills = load_all_skills(project.path if project else "")
    skills_text = format_skills_for_prompt(skills) if skills else ""

    return f"""你是这个 Unity 工程的专属 AI 编程助手，深度了解此工程的代码结构。
你可以通过工具直接读写工程中的文件、搜索代码和执行命令。

## 工程基本信息
- Unity 版本：{unity_version}
- 渲染管线：{render_pipeline}
- 工程路径：{project_path_str}

## 工程类列表（共 {len(all_classes)} 个）
{class_list}

## 场景列表
{scene_list}

## 你的职责
1. **代码问答**：基于上下文和工具回答问题
2. **代码编写**：使用 write_file 工具直接创建或修改代码文件
3. **Bug 分析**：使用 read_file/search_code 深入查看代码，分析问题
4. **架构建议**：使用 list_files 了解项目结构，给出合理建议

## 工具使用规范
- 所有文件路径相对于工程根目录
- 优先使用 search_code 定位代码，再用 read_file 查看详情
- 修改文件前先 read_file 查看当前内容
- 生成代码时保持与工程现有风格一致

{extra_context}

{skills_text}""".strip()


# ────────────────────────────────────────────────
# Agent
# ────────────────────────────────────────────────

MAX_TOOL_ROUNDS = 10  # 最多工具调用轮数，防止死循环


class UnityAgent:
    def __init__(
        self,
        project: Optional[ProjectInfo] = None,
        provider: Optional[LLMProvider] = None,
        retriever: Optional[Retriever] = None,
    ):
        self.project = project
        self.provider = provider or create_provider()
        self.retriever = retriever or Retriever(
            db_path=project.db_path if project else config.CHROMA_DB_ROOT,
            collection_name=project.collection_name if project else "unity_codebase",
        )
        self.messages: list[dict] = []
        self._system_prompt: Optional[str] = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._cached_tokens = 0
        self._tool_calls_log: list[dict] = []  # 本次 chat 的工具调用记录

        project_id = project.project_id if project else "_global"
        self._conv_manager = ConversationManager(project_id)
        self._current_conv: Optional[Conversation] = None

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = build_system_prompt(self.retriever, self.project)
        return self._system_prompt

    @property
    def project_root(self) -> str:
        return self.project.path if self.project else ""

    @property
    def tools_enabled(self) -> bool:
        return bool(self.project_root)

    @property
    def _rag_context(self) -> dict:
        if self.project:
            return {
                "db_path": self.project.db_path,
                "collection_name": self.project.collection_name,
                "project_id": self.project.project_id,
            }
        return {}

    def chat(
        self,
        user_message: str,
        top_k: int | None = None,
        filter_path: str | None = None,
        filter_type: str | None = None,
        filter_domain: str | None = None,
    ) -> tuple[str, list[SearchResult], list[dict]]:
        """
        发送消息并返回 (回复文本, 检索结果, 工具调用日志)
        """
        self._tool_calls_log = []

        # 1. RAG 检索
        search_results = self.retriever.search(
            query=user_message, top_k=top_k,
            filter_path=filter_path, filter_type=filter_type,
            filter_domain=filter_domain,
        )

        # 2. 构建消息
        context_block = self._build_context_block(search_results)
        full_user_message = f"{context_block}\n\n## 问题\n\n{user_message}"
        self.messages.append({"role": "user", "content": full_user_message})

        # 3. Tool Use 循环
        tools = TOOL_DEFINITIONS if self.tools_enabled else None
        final_text = ""

        for _ in range(MAX_TOOL_ROUNDS):
            result = self.provider.chat(
                system_prompt=self.system_prompt,
                messages=self.messages,
                tools=tools,
            )
            self._accumulate_stats(result.stats)

            if not result.has_tool_calls:
                final_text = result.text
                break

            # 记录 assistant 的工具调用到消息历史
            self._append_assistant_tool_calls(result)

            # 执行每个工具调用
            for tc in result.tool_calls:
                tool_result = execute_tool(
                    self.project_root, tc.name, tc.arguments,
                    rag_context=self._rag_context,
                )
                # RAG 索引变更后刷新 retriever
                if tc.name in ("rag_update", "rag_rebuild") and not tool_result.get("error"):
                    self.retriever.reset_collection()
                    self._system_prompt = None

                self._tool_calls_log.append({
                    "tool": tc.name,
                    "arguments": tc.arguments,
                    "result": tool_result,
                })
                # 追加工具结果到消息历史
                self._append_tool_result(tc, tool_result)

            # 如果也有文本，先记录
            if result.text:
                final_text = result.text
        else:
            # 超出最大轮数
            if not final_text:
                final_text = "(工具调用轮数已达上限)"

        # 4. 最终回复加入历史
        self.messages.append({"role": "assistant", "content": final_text})

        # 5. 自动保存对话
        self._auto_save(user_message)

        return final_text, search_results, self._tool_calls_log

    # ── 内部: Tool Use 消息构建 ──────────────────

    def _append_assistant_tool_calls(self, result: ChatResult):
        """将 assistant 的工具调用追加到消息历史"""
        provider = self.provider.provider_name()
        if provider == "claude":
            content = []
            if result.text:
                content.append({"type": "text", "text": result.text})
            for tc in result.tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tc.call_id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            self.messages.append({"role": "assistant", "content": content})
        else:
            # OpenAI 格式
            msg = {"role": "assistant", "content": result.text or None}
            msg["tool_calls"] = [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                }
                for tc in result.tool_calls
            ]
            self.messages.append(msg)

    def _append_tool_result(self, tc, tool_result: dict):
        """将工具执行结果追加到消息历史"""
        provider = self.provider.provider_name()
        if provider == "claude":
            self.messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tc.call_id,
                    "content": json.dumps(tool_result, ensure_ascii=False)[:8000],
                }],
            })
        else:
            self.messages.append({
                "role": "tool",
                "tool_call_id": tc.call_id,
                "content": json.dumps(tool_result, ensure_ascii=False)[:8000],
            })

    def _accumulate_stats(self, stats: dict):
        self._total_input_tokens += stats.get("input_tokens", 0)
        self._total_output_tokens += stats.get("output_tokens", 0)
        self._cached_tokens += stats.get("cached_tokens", 0)

    # ── 对话管理 ──────────────────────────────────

    def _auto_save(self, user_message: str):
        input_tok = self._total_input_tokens
        output_tok = self._total_output_tokens

        if self._current_conv is None:
            self._current_conv = self._conv_manager.create(model=self.provider.model_name())
        if not self._current_conv.messages:
            self._conv_manager.auto_title(self._current_conv, user_message)

        # 保存时只存 text 格式的消息（过滤掉 tool 中间步骤）
        save_messages = []
        for m in self.messages:
            if m["role"] in ("user", "assistant") and isinstance(m.get("content"), str):
                save_messages.append(m)
        self._current_conv.messages = save_messages
        self._current_conv.meta.model = self.provider.model_name()
        self._current_conv.meta.total_input_tokens = input_tok
        self._current_conv.meta.total_output_tokens = output_tok
        self._conv_manager.save(self._current_conv)

    def new_conversation(self):
        self.messages = []
        self._system_prompt = None
        self._current_conv = None
        self._tool_calls_log = []

    def start_new_conversation(self) -> str:
        self.messages = []
        self._system_prompt = None
        self._tool_calls_log = []
        self._current_conv = self._conv_manager.create(model=self.provider.model_name())
        return self._current_conv.meta.conv_id

    def reset_conversation(self):
        self.new_conversation()

    def load_conversation(self, conv_id: str) -> bool:
        conv = self._conv_manager.load(conv_id)
        if not conv:
            return False
        self._current_conv = conv
        self.messages = conv.messages.copy()
        self._system_prompt = None
        if conv.meta.model != self.provider.model_name():
            self.switch_model(conv.meta.model)
        return True

    def delete_conversation(self, conv_id: str) -> bool:
        ok = self._conv_manager.delete(conv_id)
        if ok and self._current_conv and self._current_conv.meta.conv_id == conv_id:
            self.new_conversation()
        return ok

    def rename_conversation(self, conv_id: str, new_title: str) -> bool:
        return self._conv_manager.rename(conv_id, new_title)

    def list_conversations(self):
        return self._conv_manager.list_conversations()

    def current_conv_id(self) -> Optional[str]:
        return self._current_conv.meta.conv_id if self._current_conv else None

    def current_conv_title(self) -> str:
        return self._current_conv.meta.title if self._current_conv else "(新对话)"

    # ── 模型切换 ──────────────────────────────────

    def switch_model(self, model_id: str):
        provider_type, new_provider = provider_for_model(model_id)
        self.provider = new_provider
        self._system_prompt = None

    # ── 统计 ──────────────────────────────────

    def get_token_stats(self) -> dict:
        return {
            "total_input": self._total_input_tokens,
            "total_output": self._total_output_tokens,
            "cached_tokens": self._cached_tokens,
            "conversation_turns": len([m for m in self.messages if m["role"] == "user" and isinstance(m.get("content"), str)]),
            "provider": self.provider.provider_name(),
            "model": self.provider.model_name(),
            "conv_id": self.current_conv_id(),
            "conv_title": self.current_conv_title(),
        }

    @staticmethod
    def _build_context_block(results: list[SearchResult]) -> str:
        if not results:
            return "## 相关代码上下文\n\n（未检索到相关代码片段）"
        parts = ["## 相关代码上下文（按相关度排列）\n"]
        for i, r in enumerate(results, 1):
            parts.append(f"### [{i}] {r.format_for_prompt()}")
        return "\n\n".join(parts)
