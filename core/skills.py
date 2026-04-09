"""
core/skills.py — Skill 加载器

读取已安装的 Claude Code / Codex 的 skill，解析 SKILL.md 格式，
将其注入 agent 的 system prompt 或作为可调用工具。

扫描路径：
  1. ~/.claude/skills/           （用户自定义 skill）
  2. ~/.claude/plugins/.../skills/ （插件 skill）
  3. 项目 .claude/skills/         （项目级 skill）
"""
import os
import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Skill:
    """解析后的 Skill"""
    name: str
    description: str
    source: str                  # "claude" | "codex" | "plugin"
    source_path: str             # SKILL.md 所在目录
    body: str                    # SKILL.md 正文（指令部分）
    user_invocable: bool = False
    argument_hint: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    version: str = ""
    references: dict[str, str] = field(default_factory=dict)  # name -> content
    scripts: list[str] = field(default_factory=list)           # 脚本文件路径

    def summary(self) -> str:
        inv = " [command]" if self.user_invocable else ""
        return f"{self.name}{inv} — {self.description}"


def _parse_skill_md(skill_dir: str) -> Optional[Skill]:
    """解析一个 SKILL.md 文件"""
    skill_path = os.path.join(skill_dir, "SKILL.md")
    if not os.path.isfile(skill_path):
        return None

    try:
        with open(skill_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return None

    # 解析 frontmatter
    frontmatter = {}
    body = content
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
    if fm_match:
        try:
            frontmatter = yaml.safe_load(fm_match.group(1)) or {}
        except Exception:
            frontmatter = {}
        body = fm_match.group(2).strip()

    name = frontmatter.get("name", os.path.basename(skill_dir))
    description = frontmatter.get("description", "")
    user_invocable = frontmatter.get("user-invocable", False)
    argument_hint = frontmatter.get("argument-hint", "")
    version = frontmatter.get("version", "")

    allowed_tools = frontmatter.get("allowed-tools", [])
    if isinstance(allowed_tools, str):
        allowed_tools = [t.strip() for t in allowed_tools.strip("[]").split(",")]

    # 加载 references/
    references = {}
    ref_dir = os.path.join(skill_dir, "references")
    if os.path.isdir(ref_dir):
        for fname in os.listdir(ref_dir):
            if fname.endswith(".md"):
                try:
                    with open(os.path.join(ref_dir, fname), encoding="utf-8", errors="replace") as f:
                        references[fname] = f.read()
                except Exception:
                    pass

    # 收集 scripts/
    scripts = []
    script_dir = os.path.join(skill_dir, "scripts")
    if os.path.isdir(script_dir):
        for fname in os.listdir(script_dir):
            scripts.append(os.path.join(script_dir, fname))

    return Skill(
        name=name,
        description=description,
        source="",  # 调用方设置
        source_path=skill_dir,
        body=body,
        user_invocable=user_invocable,
        argument_hint=argument_hint,
        allowed_tools=allowed_tools,
        version=version,
        references=references,
        scripts=scripts,
    )


def _scan_skills_dir(base_dir: str, source: str) -> list[Skill]:
    """扫描一个 skills 目录下的所有 skill"""
    results = []
    if not os.path.isdir(base_dir):
        return results

    for entry in os.scandir(base_dir):
        if entry.is_dir():
            skill = _parse_skill_md(entry.path)
            if skill:
                skill.source = source
                results.append(skill)

    return results


def _scan_plugins_dir(plugins_dir: str) -> list[Skill]:
    """扫描 Claude Code plugins 目录"""
    results = []
    if not os.path.isdir(plugins_dir):
        return results

    for root, dirs, files in os.walk(plugins_dir):
        if "SKILL.md" in files and os.path.basename(os.path.dirname(root)) == "skills":
            skill = _parse_skill_md(root)
            if skill:
                skill.source = "plugin"
                results.append(skill)

    return results


def load_all_skills(project_path: str = "") -> list[Skill]:
    """
    加载所有可用 skill

    扫描顺序（后加载的同名 skill 覆盖先加载的）：
      1. Claude Code 插件 skill
      2. Claude Code 用户 skill
      3. 项目级 skill
    """
    home = Path.home()
    claude_dir = home / ".claude"

    skills_map: dict[str, Skill] = {}

    # 1. 插件 skill
    plugins_dir = claude_dir / "plugins"
    for skill in _scan_plugins_dir(str(plugins_dir)):
        skills_map[skill.name] = skill

    # 2. 用户 skill (~/.claude/skills/)
    user_skills_dir = claude_dir / "skills"
    for skill in _scan_skills_dir(str(user_skills_dir), "claude"):
        skills_map[skill.name] = skill

    # 3. 项目级 skill (<project>/.claude/skills/)
    if project_path:
        project_skills_dir = Path(project_path) / ".claude" / "skills"
        for skill in _scan_skills_dir(str(project_skills_dir), "project"):
            skills_map[skill.name] = skill

    return sorted(skills_map.values(), key=lambda s: s.name)


def get_skill_by_name(name: str, project_path: str = "") -> Optional[Skill]:
    """按名称查找 skill"""
    for skill in load_all_skills(project_path):
        if skill.name == name:
            return skill
    return None


def format_skills_for_prompt(skills: list[Skill], max_chars: int = 5000) -> str:
    """将 skill 列表格式化为可注入 system prompt 的文本"""
    if not skills:
        return ""

    lines = ["## 可用 Skills\n"]
    total = 0
    for s in skills:
        entry = f"- **{s.name}** ({s.source}): {s.description}"
        if s.user_invocable:
            entry += f" [可调用: /{s.name}]"
        if total + len(entry) > max_chars:
            lines.append(f"- ... 共 {len(skills)} 个 skill")
            break
        lines.append(entry)
        total += len(entry)

    return "\n".join(lines)
