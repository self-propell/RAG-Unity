"""
core/config.py — 全局配置，从 .env 读取

支持:
  - 多 LLM Provider (Claude / OpenAI)
  - 多工程管理
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 确保从项目根目录加载 .env
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

# 静默 HuggingFace / transformers 无关警告
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ────────────────────────────────────────────────
# LLM Provider 配置
# ────────────────────────────────────────────────

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "claude")  # "claude" | "openai"

# Claude (Anthropic)
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

# OpenAI / Codex
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")

# ────────────────────────────────────────────────
# 多工程管理
# ────────────────────────────────────────────────

PROJECTS_DB_PATH: str = os.getenv("PROJECTS_DB_PATH", str(_project_root / "projects.json"))
CHROMA_DB_ROOT: str = os.getenv("CHROMA_DB_ROOT", str(_project_root / "rag_databases"))

# ────────────────────────────────────────────────
# RAG 参数
# ────────────────────────────────────────────────

RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "6"))
CHUNK_MAX_TOKENS: int = int(os.getenv("CHUNK_MAX_TOKENS", "600"))

# 工程元信息（注入 system prompt）
UNITY_VERSION: str = os.getenv("UNITY_VERSION", "2022.3 LTS")
RENDER_PIPELINE: str = os.getenv("RENDER_PIPELINE", "URP")

# ────────────────────────────────────────────────
# 索引配置
# ────────────────────────────────────────────────

IGNORED_DIRS = {
    "Library", "Temp", "Logs", "obj", "Build", "Builds",
    ".git", ".vs", "node_modules", "__pycache__"
}

INDEXED_EXTENSIONS = {
    ".cs": "script",
    ".unity": "scene",
    ".prefab": "prefab",
    ".asset": "asset",
    ".asmdef": "assembly",
    ".spriteatlas": "image",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tga": "image",
    ".psd": "image",
    ".exr": "image",
    ".wwu": "audio",
    ".bnk": "audio",
    ".wem": "audio",
    ".wav": "audio",
    ".ogg": "audio",
    ".mp3": "audio",
}

# ────────────────────────────────────────────────
# Dashboard 配置
# ────────────────────────────────────────────────

DASHBOARD_HOST: str = os.getenv("DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT: int = int(os.getenv("DASHBOARD_PORT", "5000"))


def validate():
    """启动时校验 LLM 配置"""
    errors = []
    provider = LLM_PROVIDER.lower()
    if provider == "claude":
        if not ANTHROPIC_API_KEY:
            errors.append("ANTHROPIC_API_KEY 未设置，请在 .env 中配置")
    elif provider == "openai":
        if not OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY 未设置，请在 .env 中配置")
    else:
        errors.append(f"不支持的 LLM_PROVIDER: {LLM_PROVIDER}，可选 claude / openai")
    return errors


def validate_project(project_path: str):
    """校验工程路径"""
    errors = []
    if not project_path:
        errors.append("工程路径未设置")
    elif not os.path.isdir(project_path):
        errors.append(f"工程路径不存在: {project_path}")
    return errors
