"""
Interactive CLI for the Unity RAG Agent.

Focus:
- Project and conversation management
- Domain-aware retrieval shortcuts
- Direct developer utilities for reading/listing/grepping project files
"""
import shlex
from datetime import datetime

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from core import config
from core.conversations import get_all_models
from core.projects import ProjectInfo, get_project_manager
from core.skills import get_skill_by_name, load_all_skills
from core.tools import tool_list_files, tool_read_file, tool_run_command, tool_search_code
from llm.agent import UnityAgent
from llm.providers import create_provider
from rag.retriever import Retriever

console = Console()

DOMAIN_LABELS = {
    "code": "代码",
    "scene": "场景",
    "prefab": "预制体",
    "audio": "音频",
    "image": "图片",
    "asset": "资产",
}

_MODEL_SHORTCUTS = {
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
    "gpt-5.4": "gpt-5.4",
    "gpt5.4": "gpt-5.4",
    "codex": "gpt-5.3-codex",
    "gpt-5.3-codex": "gpt-5.3-codex",
    "gpt-5.1-codex": "gpt-5.1-codex",
    "gpt-4o": "gpt-4o",
    "gpt4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "mini": "gpt-4o-mini",
}


def resolve_model(name: str) -> str:
    normalized = name.lower().strip()
    if normalized in _MODEL_SHORTCUTS:
        return _MODEL_SHORTCUTS[normalized]
    for shortcut, model_id in _MODEL_SHORTCUTS.items():
        if normalized in shortcut or normalized in model_id:
            return model_id
    return name


def _get_shortcut(model_id: str) -> str:
    for shortcut, value in _MODEL_SHORTCUTS.items():
        if value == model_id:
            return shortcut
    return ""


class SessionState:
    def __init__(self, project: ProjectInfo = None, provider_type: str = None, model: str = None):
        self.project = project
        self.provider_type = provider_type
        self.model = model
        self.top_k = config.RAG_TOP_K
        self.agent: UnityAgent | None = None
        self._init_agent()

    def _init_agent(self):
        provider = create_provider(self.provider_type, model=self.model)
        retriever = None
        if self.project:
            retriever = Retriever(
                db_path=self.project.db_path,
                collection_name=self.project.collection_name,
            )
        self.agent = UnityAgent(
            project=self.project,
            provider=provider,
            retriever=retriever,
        )

    def switch_project(self, project: ProjectInfo):
        self.project = project
        self._init_agent()

    def switch_provider(self, provider_type: str):
        self.provider_type = provider_type
        self.model = None
        self._init_agent()


def print_welcome(project: ProjectInfo = None, provider_name: str = "", model_name: str = ""):
    project_display = project.name if project else "(未配置工程)"
    provider_display = f"{provider_name} / {model_name}" if provider_name else config.CLAUDE_MODEL
    console.print(Panel(
        "[bold cyan]Unity RAG Agent[/bold cyan]\n"
        f"工程: [green]{project_display}[/green]\n"
        f"模型: [yellow]{provider_display}[/yellow]\n\n"
        "[dim]输入 /help 查看命令，Ctrl+C 退出[/dim]",
        border_style="cyan",
        box=box.ROUNDED,
    ))


def print_help():
    table = Table(title="CLI 命令", box=box.SIMPLE)
    table.add_column("命令", style="cyan", width=28)
    table.add_column("说明")

    commands = [
        ("/help", "显示帮助"),
        ("/status 或 /stats", "查看当前工程、模型、索引和 token 状态"),
        ("/topk <N>", "设置检索返回数量"),
        ("", ""),
        ("会话", ""),
        ("/new", "新建对话"),
        ("/conversations", "列出对话"),
        ("/recent", "显示最近 5 个对话"),
        ("/load <ID>", "加载对话"),
        ("/delete <ID>", "删除对话"),
        ("/rename <ID> <标题>", "重命名对话"),
        ("/title <标题>", "重命名当前对话"),
        ("/reset", "清空当前上下文并开始新对话"),
        ("", ""),
        ("工程", ""),
        ("/projects", "列出工程"),
        ("/switch <名称或ID>", "切换工程"),
        ("/add <路径> [名称]", "添加工程"),
        ("/remove <名称或ID>", "移除工程"),
        ("/index", "全量重建索引"),
        ("/update", "增量更新索引"),
        ("", ""),
        ("模型", ""),
        ("/model", "查看可用模型"),
        ("/model <名称>", "切换模型"),
        ("/provider [claude|openai]", "查看或切换 provider"),
        ("", ""),
        ("检索", ""),
        ("/search <查询> [--domain code] [--path Assets/UI] [--topk 12]", "直接检索，不调用 LLM"),
        ("/code <查询>", "仅检索代码域"),
        ("/scene <查询>", "仅检索场景域"),
        ("/prefab <查询>", "仅检索预制体域"),
        ("/audio <查询>", "仅检索音频域"),
        ("/image <查询>", "仅检索图片域"),
        ("/asset <查询>", "仅检索通用资产域"),
        ("/classes", "列出类"),
        ("/class <类名>", "查看某个类的块"),
        ("/scenes", "列出场景"),
        ("/ui <查询>", "限定 UI 路径检索/问答"),
        ("", ""),
        ("文件工具", ""),
        ("/read <路径> [max_lines]", "读取文件"),
        ("/ls [路径] [pattern]", "列目录"),
        ("/grep <query> [path] [pattern]", "文本搜索"),
        ("/refs <symbol>", "在 C# 中查引用"),
        ("/run <command>", "在工程目录执行命令"),
        ("", ""),
        ("技能", ""),
        ("/skills", "列出已安装技能"),
        ("/skill <名称>", "查看技能详情"),
        ("", ""),
        ("/quit 或 /exit", "退出"),
    ]

    for command, description in commands:
        if not command and not description:
            table.add_row("", "")
        elif description == "":
            table.add_row(f"[bold]{command}[/bold]", "")
        else:
            table.add_row(command, description)

    console.print(table)


def _tokenize(arg: str) -> list[str]:
    try:
        return shlex.split(arg)
    except ValueError:
        return arg.split()


def _parse_search_args(arg: str) -> tuple[str, dict]:
    tokens = _tokenize(arg)
    options = {"domain": None, "path": None, "top_k": None}
    query_parts = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--domain" and index + 1 < len(tokens):
            options["domain"] = tokens[index + 1]
            index += 2
        elif token == "--path" and index + 1 < len(tokens):
            options["path"] = tokens[index + 1]
            index += 2
        elif token == "--topk" and index + 1 < len(tokens):
            try:
                options["top_k"] = int(tokens[index + 1])
            except ValueError:
                pass
            index += 2
        else:
            query_parts.append(token)
            index += 1
    return " ".join(query_parts).strip(), options


def _print_sources(results):
    if not results:
        return

    console.print("\n[dim]来源[/dim]")
    for idx, result in enumerate(results, 1):
        location = result.relative_path
        if result.class_name:
            location += f" > {result.class_name}"
        if result.method_name:
            location += f".{result.method_name}"
        domain = f" [{result.collection_domain}]" if getattr(result, "collection_domain", "") else ""
        console.print(f"  [{idx}] [cyan]{location}[/cyan]{domain} [dim]{result.score:.3f}[/dim]")


def _render_search_results(results):
    if not results:
        console.print("[yellow]未找到结果[/yellow]")
        return

    for result in results:
        title = f"{result.relative_path} [{result.chunk_type}] score={result.score:.3f}"
        if getattr(result, "collection_domain", ""):
            title = f"{title} domain={result.collection_domain}"
        console.print(Panel(
            result.text[:700] + ("..." if len(result.text) > 700 else ""),
            title=title,
            border_style="dim",
        ))


def _show_status(state: SessionState):
    stats = state.agent.get_token_stats()
    index_stats = state.agent.retriever.get_stats()

    table = Table(title="当前状态", box=box.SIMPLE)
    table.add_column("项目", style="cyan")
    table.add_column("值", style="green")
    table.add_row("工程", state.project.name if state.project else "(未配置)")
    table.add_row("Provider", stats["provider"])
    table.add_row("模型", stats["model"])
    table.add_row("对话", stats["conv_title"])
    table.add_row("轮次", str(stats["conversation_turns"]))
    table.add_row("Top-K", str(state.top_k))
    table.add_row("输入 Tokens", f"{stats['total_input']:,}")
    table.add_row("输出 Tokens", f"{stats['total_output']:,}")
    table.add_row("总 Chunks", str(index_stats["total_chunks"]))
    table.add_row("文件数", str(index_stats["file_count"]))
    console.print(table)

    domains = index_stats.get("domain_distribution") or {}
    if domains:
        domain_table = Table(title="索引域分布", box=box.SIMPLE)
        domain_table.add_column("域", style="cyan")
        domain_table.add_column("Chunks", style="green")
        for domain, count in sorted(domains.items(), key=lambda item: -item[1]):
            domain_table.add_row(DOMAIN_LABELS.get(domain, domain), str(count))
        console.print(domain_table)


def _show_conversations(state: SessionState, limit: int | None = None):
    conversations = state.agent.list_conversations()
    if not conversations:
        console.print("[yellow]暂无保存的对话[/yellow]")
        return

    current_id = state.agent.current_conv_id()
    if limit:
        conversations = conversations[:limit]

    table = Table(title=f"对话列表（{len(conversations)}）", box=box.SIMPLE)
    table.add_column("", width=2)
    table.add_column("ID", style="dim", width=12)
    table.add_column("标题", style="cyan")
    table.add_column("模型", style="yellow")
    table.add_column("轮次", justify="right")
    table.add_column("更新时间", style="dim")
    for conv in conversations:
        marker = "→" if conv.conv_id == current_id else ""
        updated = datetime.fromtimestamp(conv.updated_at).strftime("%m-%d %H:%M")
        turns = str(conv.message_count // 2)
        table.add_row(marker, conv.conv_id, conv.title, conv.model, turns, updated)
    console.print(table)


def _show_models(state: SessionState):
    current = state.agent.provider.model_name()
    table = Table(title="可用模型", box=box.SIMPLE)
    table.add_column("快捷名", style="cyan", width=12)
    table.add_column("模型 ID", style="green")
    table.add_column("Provider")
    table.add_column("说明", style="dim")
    for item in get_all_models():
        marker = " ←" if item["id"] == current else ""
        table.add_row(_get_shortcut(item["id"]) + marker, item["id"], item["provider"], item["desc"])
    console.print(table)


def _show_projects():
    pm = get_project_manager()
    projects = pm.list_projects()
    if not projects:
        console.print("[yellow]暂无注册工程，使用 /add <路径> 注册[/yellow]")
        return

    active = pm.get_active()
    table = Table(title="已注册工程", box=box.SIMPLE)
    table.add_column("", width=2)
    table.add_column("名称", style="cyan")
    table.add_column("ID", style="dim")
    table.add_column("路径")
    table.add_column("Chunks", style="green")
    table.add_column("文件数")
    for project in projects:
        marker = "→" if active and project.project_id == active.project_id else ""
        table.add_row(marker, project.name, project.project_id, project.path, str(project.chunk_count), str(project.file_count))
    console.print(table)


def _show_classes(state: SessionState):
    classes = state.agent.retriever.list_all_classes()
    table = Table(title=f"类列表（{len(classes)}）", box=box.SIMPLE)
    table.add_column("类名", style="cyan")
    table.add_column("程序集", style="yellow")
    table.add_column("命名空间", style="dim")
    table.add_column("路径")
    for item in classes:
        table.add_row(item["class_name"], item.get("assembly_name", ""), item["namespace"], item["relative_path"])
    console.print(table)


def _run_search(state: SessionState, query: str, *, domain: str | None = None, path: str | None = None, top_k: int | None = None):
    if not query:
        console.print("[yellow]请提供查询内容[/yellow]")
        return
    results = state.agent.retriever.search(
        query,
        top_k=top_k or state.top_k,
        filter_path=path,
        filter_domain=domain,
    )
    _render_search_results(results)


def _show_file_read(state: SessionState, path: str, max_lines: int = 200):
    if not state.project:
        console.print("[yellow]当前没有活跃工程[/yellow]")
        return
    result = tool_read_file(state.project.path, path, max_lines=max_lines)
    if result.get("error"):
        console.print(f"[red]{result['error']}[/red]")
        return
    title = f"{result['path']} ({result['total_lines']} lines)"
    console.print(Panel(result["content"], title=title, border_style="blue"))


def _show_list_files(state: SessionState, path: str = ".", pattern: str = "*"):
    if not state.project:
        console.print("[yellow]当前没有活跃工程[/yellow]")
        return
    result = tool_list_files(state.project.path, path=path, pattern=pattern)
    if result.get("error"):
        console.print(f"[red]{result['error']}[/red]")
        return

    table = Table(title=f"目录: {result['directory']}", box=box.SIMPLE)
    table.add_column("类型", width=6)
    table.add_column("名称", style="cyan")
    table.add_column("路径")
    table.add_column("大小", justify="right")
    for item in result["items"]:
        table.add_row(item["type"], item["name"], item["path"], str(item["size"]))
    console.print(table)


def _show_grep(state: SessionState, query: str, path: str = ".", pattern: str = "*.cs"):
    if not state.project:
        console.print("[yellow]当前没有活跃工程[/yellow]")
        return
    result = tool_search_code(state.project.path, query=query, path=path, file_pattern=pattern)
    if result.get("error"):
        console.print(f"[red]{result['error']}[/red]")
        return
    rows = result.get("results", [])
    if not rows:
        console.print("[yellow]没有匹配结果[/yellow]")
        return

    table = Table(title=f"搜索: {query}", box=box.SIMPLE)
    table.add_column("文件", style="cyan")
    table.add_column("行号", justify="right")
    table.add_column("内容")
    for row in rows:
        table.add_row(row["file"], str(row["line"]), row["content"])
    console.print(table)


def _run_shell(state: SessionState, command: str):
    if not state.project:
        console.print("[yellow]当前没有活跃工程[/yellow]")
        return
    result = tool_run_command(state.project.path, command)
    if result.get("error"):
        console.print(f"[red]{result['error']}[/red]")
        return

    if result.get("stdout"):
        console.print(Panel(result["stdout"], title="stdout", border_style="green"))
    if result.get("stderr"):
        console.print(Panel(result["stderr"], title="stderr", border_style="yellow"))
    console.print(f"[dim]return_code={result['return_code']}[/dim]")


def _show_skills(state: SessionState):
    project_path = state.project.path if state.project else ""
    skills = load_all_skills(project_path)
    if not skills:
        console.print("[yellow]未找到已安装技能[/yellow]")
        return

    table = Table(title=f"Skills（{len(skills)}）", box=box.SIMPLE)
    table.add_column("名称", style="cyan")
    table.add_column("来源", style="dim", width=8)
    table.add_column("可调用", width=8)
    table.add_column("说明")
    for skill in skills:
        description = skill.description[:70] + ("..." if len(skill.description) > 70 else "")
        table.add_row(skill.name, skill.source, "yes" if skill.user_invocable else "", description)
    console.print(table)


def _show_skill_detail(state: SessionState, skill_name: str):
    project_path = state.project.path if state.project else ""
    skill = get_skill_by_name(skill_name, project_path)
    if not skill:
        console.print(f"[red]未找到 skill: {skill_name}[/red]")
        return
    console.print(Panel(
        f"[bold]{skill.name}[/bold] ({skill.source})\n"
        f"{skill.description}\n\n"
        f"版本: {skill.version or 'N/A'}\n"
        f"可调用: {'yes' if skill.user_invocable else 'no'}\n"
        f"参数提示: {skill.argument_hint or 'N/A'}\n"
        f"引用文档: {len(skill.references)}\n"
        f"脚本: {len(skill.scripts)}\n"
        f"路径: {skill.source_path}",
        title=f"Skill: {skill.name}",
        border_style="cyan",
    ))


def _rename_current_conversation(state: SessionState, new_title: str):
    conv_id = state.agent.current_conv_id()
    if not conv_id:
        console.print("[yellow]当前没有持久化对话，先发送一条消息或使用 /new[/yellow]")
        return
    if state.agent.rename_conversation(conv_id, new_title):
        console.print(f"[green]当前对话已重命名为: {new_title}[/green]")
    else:
        console.print("[red]重命名失败[/red]")


def handle_command(cmd: str, state: SessionState) -> bool:
    raw_parts = cmd.strip().split(None, 1)
    command = raw_parts[0].lower()
    arg = raw_parts[1].strip() if len(raw_parts) > 1 else ""

    if command in ("/quit", "/exit", "/q"):
        return False

    if command == "/help":
        print_help()
        return True

    if command in ("/stats", "/status"):
        _show_status(state)
        return True

    if command == "/topk":
        try:
            state.top_k = int(arg)
            console.print(f"[green]检索数量已设为 {state.top_k}[/green]")
        except ValueError:
            console.print("[yellow]请提供数字[/yellow]")
        return True

    if command == "/new":
        state.agent.new_conversation()
        console.print("[green]已新建对话[/green]")
        return True

    if command in ("/conversations", "/convs", "/chats"):
        _show_conversations(state)
        return True

    if command == "/recent":
        _show_conversations(state, limit=5)
        return True

    if command == "/load":
        if not arg:
            console.print("[yellow]请提供对话 ID[/yellow]")
        elif state.agent.load_conversation(arg):
            console.print(f"[green]已加载对话: {state.agent.current_conv_title()}[/green]")
        else:
            console.print(f"[red]找不到对话: {arg}[/red]")
        return True

    if command == "/delete":
        if not arg:
            console.print("[yellow]请提供对话 ID[/yellow]")
        elif state.agent.delete_conversation(arg):
            console.print(f"[yellow]已删除对话: {arg}[/yellow]")
        else:
            console.print(f"[red]找不到对话: {arg}[/red]")
        return True

    if command == "/rename":
        parts = _tokenize(arg)
        if len(parts) < 2:
            console.print("[yellow]用法: /rename <ID> <新标题>[/yellow]")
        else:
            conv_id = parts[0]
            new_title = " ".join(parts[1:])
            if state.agent.rename_conversation(conv_id, new_title):
                console.print(f"[green]已重命名: {new_title}[/green]")
            else:
                console.print(f"[red]找不到对话: {conv_id}[/red]")
        return True

    if command == "/title":
        if not arg:
            console.print("[yellow]用法: /title <新标题>[/yellow]")
        else:
            _rename_current_conversation(state, arg)
        return True

    if command == "/model":
        if not arg:
            _show_models(state)
        else:
            model_id = resolve_model(arg)
            try:
                state.agent.switch_model(model_id)
                console.print(f"[green]已切换模型: {model_id}[/green]")
            except Exception as exc:
                console.print(f"[red]切换失败: {exc}[/red]")
        return True

    if command == "/provider":
        if not arg:
            console.print(f"当前: [cyan]{state.agent.provider.provider_name()}[/cyan] ({state.agent.provider.model_name()})")
        elif arg.lower() in ("claude", "openai"):
            try:
                state.switch_provider(arg.lower())
                console.print(f"[green]已切换到 {state.agent.provider.provider_name()} ({state.agent.provider.model_name()})[/green]")
            except Exception as exc:
                console.print(f"[red]切换失败: {exc}[/red]")
        else:
            console.print("[yellow]可选: claude, openai[/yellow]")
        return True

    if command == "/projects":
        _show_projects()
        return True

    if command == "/switch":
        if not arg:
            console.print("[yellow]请提供工程名称或 ID[/yellow]")
        else:
            pm = get_project_manager()
            project = pm.find_by_name(arg)
            if not project:
                console.print(f"[red]找不到工程: {arg}[/red]")
            else:
                pm.set_active(project.project_id)
                state.switch_project(project)
                console.print(f"[green]已切换到工程: {project.name}[/green]")
        return True

    if command == "/add":
        parts = _tokenize(arg)
        if not parts:
            console.print("[yellow]用法: /add <路径> [名称][/yellow]")
        else:
            path = parts[0]
            name = " ".join(parts[1:]) if len(parts) > 1 else ""
            try:
                pm = get_project_manager()
                project = pm.add_project(path, name=name)
                pm.set_active(project.project_id)
                state.switch_project(project)
                console.print(f"[green]已注册工程: {project.name}[/green] (ID: {project.project_id})")
                console.print("[dim]运行 /index 建立索引[/dim]")
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
        return True

    if command == "/remove":
        if not arg:
            console.print("[yellow]请提供工程名称或 ID[/yellow]")
        else:
            pm = get_project_manager()
            project = pm.find_by_name(arg)
            if not project:
                console.print(f"[red]找不到工程: {arg}[/red]")
            else:
                pm.remove_project(project.project_id)
                console.print(f"[yellow]已移除工程: {project.name}[/yellow]")
                active = pm.get_active()
                if active:
                    state.switch_project(active)
        return True

    if command == "/index":
        if not state.project:
            console.print("[yellow]当前无活跃工程，请先 /add 注册[/yellow]")
        else:
            from rag.indexer import build_index

            build_index(state.project.path, state.project.db_path, state.project.collection_name, state.project.project_id)
            state.agent.retriever.reset_collection()
            state.agent.new_conversation()
        return True

    if command == "/update":
        if not state.project:
            console.print("[yellow]当前无活跃工程[/yellow]")
        else:
            from rag.indexer import update_index

            update_index(state.project.path, state.project.db_path, state.project.collection_name, state.project.project_id)
            state.agent.retriever.reset_collection()
        return True

    if command == "/skills":
        _show_skills(state)
        return True

    if command == "/skill":
        if not arg:
            console.print("[yellow]请提供 skill 名称[/yellow]")
        else:
            _show_skill_detail(state, _tokenize(arg)[0])
        return True

    if command == "/classes":
        _show_classes(state)
        return True

    if command == "/class":
        if not arg:
            console.print("[yellow]请提供类名[/yellow]")
        else:
            results = state.agent.retriever.search_by_class(arg)
            _render_search_results(results)
        return True

    if command == "/scenes":
        scenes = state.agent.retriever.list_scenes()
        if not scenes:
            console.print("[yellow]未找到场景[/yellow]")
        else:
            for item in scenes:
                console.print(f"[cyan]{item['relative_path']}[/cyan]")
        return True

    if command == "/search":
        query, options = _parse_search_args(arg)
        _run_search(state, query, domain=options["domain"], path=options["path"], top_k=options["top_k"])
        return True

    if command in ("/code", "/scene", "/prefab", "/audio", "/image", "/asset"):
        domain = command[1:]
        _run_search(state, arg, domain=domain)
        return True

    if command == "/ui":
        if not arg:
            console.print("[yellow]请在 /ui 后输入问题[/yellow]")
        else:
            _do_chat(state, arg, filter_path="UI")
        return True

    if command == "/read":
        parts = _tokenize(arg)
        if not parts:
            console.print("[yellow]用法: /read <路径> [max_lines][/yellow]")
        else:
            max_lines = 200
            if len(parts) > 1:
                try:
                    max_lines = int(parts[1])
                except ValueError:
                    pass
            _show_file_read(state, parts[0], max_lines=max_lines)
        return True

    if command == "/ls":
        parts = _tokenize(arg)
        path = parts[0] if parts else "."
        pattern = parts[1] if len(parts) > 1 else "*"
        _show_list_files(state, path=path, pattern=pattern)
        return True

    if command in ("/grep", "/refs"):
        parts = _tokenize(arg)
        if not parts:
            console.print("[yellow]用法: /grep <query> [path] [pattern][/yellow]")
        else:
            query = parts[0]
            path = parts[1] if len(parts) > 1 else "."
            pattern = parts[2] if len(parts) > 2 else "*.cs"
            _show_grep(state, query=query, path=path, pattern=pattern)
        return True

    if command == "/run":
        if not arg:
            console.print("[yellow]用法: /run <command>[/yellow]")
        else:
            _run_shell(state, arg)
        return True

    if command == "/reset":
        state.agent.new_conversation()
        console.print("[yellow]已重置当前对话[/yellow]")
        return True

    console.print(f"[yellow]未知命令: {command}，输入 /help 查看[/yellow]")
    return True


def _do_chat(
    state: SessionState,
    question: str,
    filter_path: str | None = None,
    filter_type: str | None = None,
    filter_domain: str | None = None,
):
    console.print()
    try:
        reply, results, tool_log = state.agent.chat(
            question,
            top_k=state.top_k,
            filter_path=filter_path,
            filter_type=filter_type,
            filter_domain=filter_domain,
        )
        if tool_log:
            for tool_call in tool_log:
                error = tool_call["result"].get("error")
                rendered_args = ", ".join(f"{k}={v!r}" for k, v in tool_call["arguments"].items())
                if error:
                    console.print(f"  [red]Tool {tool_call['tool']}[/red]({rendered_args}): {error}")
                else:
                    console.print(f"  [dim]Tool {tool_call['tool']}({rendered_args})[/dim]")

        model = state.agent.provider.model_name()
        console.print(Panel(
            Markdown(reply),
            title=f"[bold green]{model}[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))
        _print_sources(results)
    except Exception as exc:
        console.print(f"[red]调用失败: {exc}[/red]")


def run_interactive(state: SessionState):
    print_welcome(
        state.project,
        state.agent.provider.provider_name(),
        state.agent.provider.model_name(),
    )

    while True:
        try:
            conv_title = state.agent.current_conv_title()
            model = state.agent.provider.model_name()
            project_name = state.project.name if state.project else "未配置工程"
            prompt_text = f"\n[dim]{project_name} | {conv_title} | {model} | topk={state.top_k}[/dim]\n[bold blue]你[/bold blue]"
            user_input = Prompt.ask(prompt_text).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]再见[/dim]")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            if not handle_command(user_input, state):
                console.print("[dim]再见[/dim]")
                break
        else:
            _do_chat(state, user_input)

    stats = state.agent.get_token_stats()
    if stats["conversation_turns"] > 0:
        console.print(
            f"\n[dim]本次会话: {stats['conversation_turns']} 轮, 输入 {stats['total_input']:,} tokens[/dim]"
        )

