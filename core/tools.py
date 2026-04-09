"""
core/tools.py — 沙盒化工具系统

提供 LLM 可调用的本地操作工具，所有文件操作限制在工程目录内。

工具列表：
  - read_file     读取文件内容
  - write_file    写入/创建文件
  - list_files    列出目录文件
  - search_code   在文件中搜索文本
  - run_command   执行 shell 命令（限定工作目录）
"""
import os
import re
import subprocess
import difflib
from pathlib import Path
from typing import Optional


# ────────────────────────────────────────────────
# 沙盒路径校验
# ────────────────────────────────────────────────

class SandboxError(Exception):
    """路径越权异常"""
    pass


def _display_path(path: str, project_root: str) -> str:
    try:
        return os.path.relpath(path, project_root).replace("\\", "/")
    except ValueError:
        return path.replace("\\", "/")


def _resolve_safe(project_root: str, relative_path: str) -> str:
    """将相对路径解析为绝对路径，确保在工程目录内"""
    root = Path(project_root).resolve()
    # 支持绝对路径（但必须在 root 内）
    target = Path(relative_path)
    if target.is_absolute():
        resolved = target.resolve()
    else:
        resolved = (root / relative_path).resolve()

    # 检查是否在工程目录内
    try:
        resolved.relative_to(root)
    except ValueError:
        raise SandboxError(f"路径越权：{relative_path} 不在工程目录 {root} 内")

    return str(resolved)


# ────────────────────────────────────────────────
# 工具实现
# ────────────────────────────────────────────────

def tool_read_file(project_root: str, path: str, max_lines: int = 500) -> dict:
    """读取文件内容"""
    try:
        safe_path = _resolve_safe(project_root, path)
    except SandboxError as e:
        return {"error": str(e)}

    if not os.path.isfile(safe_path):
        return {"error": f"文件不存在: {path}"}

    try:
        with open(safe_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total = len(lines)
        content = "".join(lines[:max_lines])
        truncated = total > max_lines

        return {
            "content": content,
            "total_lines": total,
            "truncated": truncated,
            "path": _display_path(safe_path, project_root),
        }
    except Exception as e:
        return {"error": f"读取失败: {e}"}


def tool_write_file(project_root: str, path: str, content: str, create_dirs: bool = True) -> dict:
    """写入文件（创建或覆盖）"""
    try:
        safe_path = _resolve_safe(project_root, path)
    except SandboxError as e:
        return {"error": str(e)}

    # 禁止写入敏感文件
    basename = os.path.basename(safe_path).lower()
    if basename in (".env", "credentials.json", ".git"):
        return {"error": f"禁止写入敏感文件: {basename}"}

    try:
        if create_dirs:
            os.makedirs(os.path.dirname(safe_path), exist_ok=True)

        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content)

        return {
            "success": True,
            "path": _display_path(safe_path, project_root),
            "bytes_written": len(content.encode("utf-8")),
        }
    except Exception as e:
        return {"error": f"写入失败: {e}"}


def tool_replace_in_file(
    project_root: str,
    path: str,
    search_text: str,
    replace_text: str,
    count: int = 1,
) -> dict:
    """在文件中替换指定文本"""
    try:
        safe_path = _resolve_safe(project_root, path)
    except SandboxError as e:
        return {"error": str(e)}

    if not os.path.isfile(safe_path):
        return {"error": f"文件不存在: {path}"}
    if not search_text:
        return {"error": "search_text 不能为空"}

    try:
        with open(safe_path, encoding="utf-8", errors="replace") as f:
            original = f.read()

        if search_text not in original:
            return {"error": f"未找到指定文本: {path}"}

        replace_count = original.count(search_text) if count <= 0 else count
        updated = original.replace(search_text, replace_text, replace_count)

        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(updated)

        return {
            "success": True,
            "path": _display_path(safe_path, project_root),
            "replacements": replace_count,
        }
    except Exception as e:
        return {"error": f"替换失败: {e}"}


def tool_diff_file(
    project_root: str,
    path: str,
    proposed_content: str,
    context_lines: int = 3,
) -> dict:
    """生成当前文件与提议内容的 unified diff"""
    try:
        safe_path = _resolve_safe(project_root, path)
    except SandboxError as e:
        return {"error": str(e)}

    if not os.path.isfile(safe_path):
        return {"error": f"文件不存在: {path}"}

    try:
        with open(safe_path, encoding="utf-8", errors="replace") as f:
            original = f.read()

        diff = "".join(difflib.unified_diff(
            original.splitlines(keepends=True),
            proposed_content.splitlines(keepends=True),
            fromfile=path,
            tofile=f"{path} (proposed)",
            n=context_lines,
        ))

        return {
            "path": _display_path(safe_path, project_root),
            "diff": diff,
            "changed": original != proposed_content,
        }
    except Exception as e:
        return {"error": f"生成 diff 失败: {e}"}


def tool_list_files(project_root: str, path: str = ".", pattern: str = "*", max_items: int = 200) -> dict:
    """列出目录中的文件"""
    try:
        safe_path = _resolve_safe(project_root, path)
    except SandboxError as e:
        return {"error": str(e)}

    if not os.path.isdir(safe_path):
        return {"error": f"目录不存在: {path}"}

    IGNORED = {"Library", "Temp", "Logs", "obj", "Build", ".git", ".vs", "__pycache__", "node_modules"}
    items = []
    try:
        for entry in sorted(os.scandir(safe_path), key=lambda e: (not e.is_dir(), e.name.lower())):
            if entry.name in IGNORED:
                continue
            if pattern != "*" and not re.match(pattern.replace("*", ".*"), entry.name, re.IGNORECASE):
                continue
            rel = os.path.relpath(entry.path, project_root).replace("\\", "/")
            items.append({
                "name": entry.name,
                "path": rel,
                "type": "dir" if entry.is_dir() else "file",
                "size": entry.stat().st_size if entry.is_file() else 0,
            })
            if len(items) >= max_items:
                break

        return {"items": items, "directory": _display_path(safe_path, project_root)}
    except Exception as e:
        return {"error": f"列出失败: {e}"}


def tool_search_code(project_root: str, query: str, path: str = ".", file_pattern: str = "*.cs", max_results: int = 30) -> dict:
    """在文件中搜索文本（类似 grep）"""
    try:
        safe_path = _resolve_safe(project_root, path)
    except SandboxError as e:
        return {"error": str(e)}

    IGNORED = {"Library", "Temp", "Logs", "obj", "Build", ".git", ".vs", "__pycache__"}
    results = []
    ext_filter = file_pattern.replace("*", "")  # e.g. "*.cs" → ".cs"

    try:
        for root, dirs, files in os.walk(safe_path):
            dirs[:] = [d for d in dirs if d not in IGNORED]
            for fname in files:
                if ext_filter and not fname.lower().endswith(ext_filter.lower()):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, encoding="utf-8", errors="ignore") as f:
                        for lineno, line in enumerate(f, 1):
                            if query.lower() in line.lower():
                                rel = os.path.relpath(fpath, project_root).replace("\\", "/")
                                results.append({
                                    "file": rel,
                                    "line": lineno,
                                    "content": line.rstrip()[:200],
                                })
                                if len(results) >= max_results:
                                    return {"results": results, "truncated": True}
                except Exception:
                    continue

        return {"results": results, "truncated": False}
    except Exception as e:
        return {"error": f"搜索失败: {e}"}


def tool_run_command(project_root: str, command: str, timeout: int = 30) -> dict:
    """执行 shell 命令（工作目录限定为工程目录）"""
    # 禁止危险命令
    dangerous = ["rm -rf /", "del /s /q c:", "format ", "mkfs", "dd if="]
    cmd_lower = command.lower()
    for d in dangerous:
        if d in cmd_lower:
            return {"error": f"禁止执行危险命令: {command}"}

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout
        stderr = result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr

        return {
            "stdout": output,
            "stderr": stderr,
            "return_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"命令超时 ({timeout}s): {command}"}
    except Exception as e:
        return {"error": f"执行失败: {e}"}


# ────────────────────────────────────────────────
# RAG 工具
# ────────────────────────────────────────────────

def tool_rag_search(db_path: str, collection_name: str, query: str, top_k: int = 8) -> dict:
    """语义搜索 RAG 向量库"""
    try:
        from rag.retriever import Retriever
        retriever = Retriever(db_path=db_path, collection_name=collection_name)
        results = retriever.search(query, top_k=top_k)
        return {
            "results": [
                {
                    "file": r.relative_path,
                    "class": r.class_name,
                    "method": r.method_name,
                    "type": r.chunk_type,
                    "score": round(r.score, 3),
                    "snippet": r.text[:500],
                }
                for r in results
            ],
            "count": len(results),
        }
    except Exception as e:
        return {"error": f"RAG 搜索失败: {e}"}


def tool_rag_update(project_path: str, db_path: str, collection_name: str, project_id: str = None) -> dict:
    """增量更新 RAG 索引"""
    try:
        from rag.indexer import update_index
        update_index(project_path, db_path, collection_name, project_id)
        from rag.retriever import Retriever
        stats = Retriever(db_path=db_path, collection_name=collection_name).get_stats()
        return {"success": True, "total_chunks": stats["total_chunks"], "file_count": stats["file_count"]}
    except Exception as e:
        return {"error": f"索引更新失败: {e}"}


def tool_rag_rebuild(project_path: str, db_path: str, collection_name: str, project_id: str = None) -> dict:
    """全量重建 RAG 索引"""
    try:
        from rag.indexer import build_index
        chunks, files = build_index(project_path, db_path, collection_name, project_id)
        return {"success": True, "chunks": chunks, "files": files}
    except Exception as e:
        return {"error": f"索引重建失败: {e}"}


def tool_rag_stats(db_path: str, collection_name: str) -> dict:
    """查看 RAG 索引统计"""
    try:
        from rag.retriever import Retriever
        return Retriever(db_path=db_path, collection_name=collection_name).get_stats()
    except Exception as e:
        return {"error": f"统计获取失败: {e}"}


# ────────────────────────────────────────────────
# 工具注册表 & Schema（供 LLM Tool Use）
# ────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "读取工程目录内的文件内容。路径相对于工程根目录。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径（相对于工程根目录）"},
                "max_lines": {"type": "integer", "description": "最多读取行数，默认500", "default": 500},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "在工程目录内创建或覆盖文件。路径相对于工程根目录。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径（相对于工程根目录）"},
                "content": {"type": "string", "description": "要写入的完整文件内容"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "replace_in_file",
        "description": "在单个文件中按文本精确替换，适合小范围修改。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径（相对于工程根目录）"},
                "search_text": {"type": "string", "description": "要替换的原文本"},
                "replace_text": {"type": "string", "description": "替换后的文本"},
                "count": {"type": "integer", "description": "替换次数，0 表示全部", "default": 1}
            },
            "required": ["path", "search_text", "replace_text"]
        }
    },
    {
        "name": "diff_file",
        "description": "生成当前文件与提议内容的 unified diff，用于修改预览。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径（相对于工程根目录）"},
                "proposed_content": {"type": "string", "description": "建议的完整文件内容"},
                "context_lines": {"type": "integer", "description": "diff 上下文行数", "default": 3}
            },
            "required": ["path", "proposed_content"]
        }
    },
    {
        "name": "list_files",
        "description": "列出工程目录中的文件和子目录。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "目录路径（相对于工程根目录），默认为根目录", "default": "."},
                "pattern": {"type": "string", "description": "文件名匹配模式，如 *.cs", "default": "*"},
            },
            "required": [],
        },
    },
    {
        "name": "search_code",
        "description": "在工程文件中搜索文本（类似 grep），返回匹配的行。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要搜索的文本"},
                "path": {"type": "string", "description": "搜索起始目录，默认为工程根目录", "default": "."},
                "file_pattern": {"type": "string", "description": "文件类型过滤，如 *.cs、*.unity", "default": "*.cs"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "run_command",
        "description": "在工程目录中执行 shell 命令。工作目录限定为工程根目录。",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的命令"},
                "timeout": {"type": "integer", "description": "超时秒数，默认30", "default": 30},
            },
            "required": ["command"],
        },
    },
    {
        "name": "rag_search",
        "description": "语义搜索 RAG 向量库，找到与查询最相关的代码片段。比 search_code 的文本搜索更智能，能理解语义。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "自然语言查询，如「玩家跳跃逻辑」「UI 血量更新」"},
                "top_k": {"type": "integer", "description": "返回结果数量，默认8", "default": 8},
            },
            "required": ["query"],
        },
    },
    {
        "name": "rag_update",
        "description": "增量更新 RAG 索引。在修改或新建文件后调用，让 RAG 数据库同步最新代码。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "rag_rebuild",
        "description": "全量重建 RAG 索引。当大量文件变动或索引损坏时使用，耗时较长。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "rag_stats",
        "description": "查看当前工程的 RAG 索引统计信息，包括总 chunk 数、文件数、类型分布。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "use_skill",
        "description": "调用已安装的 Claude Code / Codex skill。使用前可先通过 list_skills 查看可用 skill 列表。",
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "Skill 名称"},
                "arguments": {"type": "string", "description": "传递给 skill 的参数", "default": ""},
            },
            "required": ["skill_name"],
        },
    },
    {
        "name": "list_skills",
        "description": "列出所有已安装的 Claude Code / Codex skill，包括名称、来源和描述。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


def tool_list_skills(project_root: str) -> dict:
    """列出所有可用 skill"""
    try:
        from core.skills import load_all_skills
        skills = load_all_skills(project_root)
        return {
            "skills": [
                {
                    "name": s.name,
                    "description": s.description[:200],
                    "source": s.source,
                    "user_invocable": s.user_invocable,
                    "has_scripts": len(s.scripts) > 0,
                }
                for s in skills
            ],
            "count": len(skills),
        }
    except Exception as e:
        return {"error": f"加载 skill 失败: {e}"}


def tool_use_skill(project_root: str, skill_name: str, arguments: str = "") -> dict:
    """调用一个 skill，返回其指令内容（供 LLM 遵循执行）"""
    try:
        from core.skills import get_skill_by_name
        skill = get_skill_by_name(skill_name, project_root)
        if not skill:
            return {"error": f"未找到 skill: {skill_name}"}

        result = {
            "name": skill.name,
            "description": skill.description,
            "instructions": skill.body,
            "arguments_received": arguments,
            "source": skill.source,
        }

        # 附带引用文档
        if skill.references:
            result["references"] = {
                name: content[:3000] for name, content in list(skill.references.items())[:5]
            }

        # 附带脚本路径
        if skill.scripts:
            result["available_scripts"] = skill.scripts

        return result
    except Exception as e:
        return {"error": f"调用 skill 失败: {e}"}


def execute_tool(
    project_root: str,
    tool_name: str,
    arguments: dict,
    rag_context: dict = None,
) -> dict:
    """
    统一执行入口

    Args:
        project_root: 工程根目录
        tool_name: 工具名称
        arguments: LLM 传入的参数
        rag_context: RAG 工具所需上下文 {"db_path", "collection_name", "project_id"}
    """
    rc = rag_context or {}

    dispatch = {
        "read_file": lambda args: tool_read_file(project_root, **args),
        "write_file": lambda args: tool_write_file(project_root, **args),
        "replace_in_file": lambda args: tool_replace_in_file(project_root, **args),
        "diff_file": lambda args: tool_diff_file(project_root, **args),
        "list_files": lambda args: tool_list_files(project_root, **args),
        "search_code": lambda args: tool_search_code(project_root, **args),
        "run_command": lambda args: tool_run_command(project_root, **args),
        "rag_search": lambda args: tool_rag_search(rc.get("db_path", ""), rc.get("collection_name", ""), **args),
        "rag_update": lambda args: tool_rag_update(project_root, rc.get("db_path", ""), rc.get("collection_name", ""), rc.get("project_id")),
        "rag_rebuild": lambda args: tool_rag_rebuild(project_root, rc.get("db_path", ""), rc.get("collection_name", ""), rc.get("project_id")),
        "rag_stats": lambda args: tool_rag_stats(rc.get("db_path", ""), rc.get("collection_name", "")),
        "list_skills": lambda args: tool_list_skills(project_root),
        "use_skill": lambda args: tool_use_skill(project_root, **args),
    }
    fn = dispatch.get(tool_name)
    if not fn:
        return {"error": f"未知工具: {tool_name}"}
    try:
        return fn(arguments)
    except Exception as e:
        return {"error": f"工具执行异常: {e}"}


def get_openai_tools() -> list[dict]:
    """转换为 OpenAI function calling 格式"""
    return [
        {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
        for t in TOOL_DEFINITIONS
    ]


def get_claude_tools() -> list[dict]:
    """转换为 Claude tool_use 格式"""
    return [
        {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
        for t in TOOL_DEFINITIONS
    ]
