#!/usr/bin/env python3
"""
main.py - Unity RAG Agent unified entrypoint.

Usage:
  python main.py                       # Launch dashboard (default)
  python main.py dashboard             # Launch dashboard
  python main.py cli                   # Start CLI in the current console
  python main.py cli -q "question"     # Single query
  python main.py index                 # Full index rebuild
  python main.py index --update        # Incremental index update
  python main.py index --add <path>    # Register and index a new project
  python main.py index --list          # List registered projects
  python main.py index --stats         # Show index stats
  python main.py watch                 # Start file watcher
  python main.py setup                 # Setup wizard
"""

import argparse
import os
import sys


def cmd_cli(args):
    from core import config
    from core.projects import get_project_manager
    from rich.console import Console
    from rich.panel import Panel
    from ui.cli import SessionState, _do_chat, run_interactive

    console = Console()

    provider_type = args.provider or config.LLM_PROVIDER
    errors = config.validate()
    if errors:
        for error in errors:
            console.print(f"[red]Configuration error: {error}[/red]")
        sys.exit(1)

    pm = get_project_manager()
    if args.project:
        project = pm.find_by_name(args.project)
        if not project:
            console.print(f"[red]Project not found: {args.project}[/red]")
            sys.exit(1)
    else:
        project = pm.get_active()

    state = SessionState(project=project, provider_type=provider_type)
    state.top_k = args.topk

    count = state.agent.retriever.get_index_count()
    if count == 0:
        name = project.name if project else "(default)"
        console.print(f"[yellow]Warning: project {name} has no index yet. Run indexing first.[/yellow]")

    if args.search:
        results = state.agent.retriever.search(args.search, top_k=args.topk)
        for result in results:
            console.print(Panel(
                result.text[:600],
                title=f"{result.relative_path} [{result.chunk_type}] score={result.score:.3f}",
                border_style="dim",
            ))
        return

    if args.question:
        _do_chat(state, args.question)
        stats = state.agent.get_token_stats()
        console.print(f"\n[dim]Input: {stats['total_input']} | Output: {stats['total_output']} tokens[/dim]")
        return

    run_interactive(state)


def cmd_dashboard(args):
    from ui.dashboard import run_dashboard

    run_dashboard(host=args.host, port=args.port, debug=args.debug)


def cmd_index(args):
    from core.projects import get_project_manager
    from rag.indexer import build_index, show_stats, update_index
    from rich.console import Console
    from rich.table import Table

    console = Console()
    pm = get_project_manager()

    if args.list:
        projects = pm.list_projects()
        if not projects:
            console.print("[yellow]No registered projects. Use --add <path> to register one.[/yellow]")
        else:
            active = pm.get_active()
            table = Table(title="Registered Projects")
            table.add_column("", width=2)
            table.add_column("Name", style="cyan")
            table.add_column("ID", style="dim")
            table.add_column("Path")
            table.add_column("Chunks", style="green")
            for project in projects:
                marker = "->" if active and project.project_id == active.project_id else ""
                table.add_row(marker, project.name, project.project_id, project.path, str(project.chunk_count))
            console.print(table)
        return

    if args.add:
        info = pm.add_project(args.add, name=args.name or "")
        console.print(f"[green]Registered project {info.name}[/green] (ID: {info.project_id})")
        pm.set_active(info.project_id)
        project = info
    elif args.project:
        project = pm.find_by_name(args.project)
        if not project:
            if os.path.isdir(args.project):
                console.print(f"[yellow]Path is not registered yet, use --add to register it: {args.project}[/yellow]")
                return
            console.print(f"[red]Project not found: {args.project}[/red]")
            return
    else:
        project = pm.get_active()

    if args.stats:
        if project:
            show_stats(project.db_path, project.collection_name)
        else:
            console.print("[yellow]No project selected.[/yellow]")
        return

    if not project:
        console.print("[red]No project selected. Use --project or --add.[/red]")
        return

    if not os.path.isdir(project.path):
        console.print(f"[red]Project path does not exist: {project.path}[/red]")
        return

    if args.update:
        update_index(project.path, project.db_path, project.collection_name, project.project_id)
    else:
        build_index(project.path, project.db_path, project.collection_name, project.project_id)


def cmd_watch(args):
    from core.projects import get_project_manager
    from rag.watcher import start_watcher
    from rich.console import Console

    console = Console()
    pm = get_project_manager()

    if args.project:
        info = pm.find_by_name(args.project)
        if not info:
            console.print(f"[red]Project not found: {args.project}[/red]")
            sys.exit(1)
    else:
        info = pm.get_active()

    if not info:
        console.print("[red]No project selected.[/red]")
        sys.exit(1)

    if not os.path.isdir(info.path):
        console.print(f"[red]Project path does not exist: {info.path}[/red]")
        sys.exit(1)

    start_watcher(info.path, info.db_path, info.collection_name, info.name)


def cmd_setup(args):
    from pathlib import Path
    import subprocess

    from core import config
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt

    console = Console()
    project_root = Path(__file__).resolve().parent

    console.print(Panel(
        "[bold cyan]Unity RAG Agent - Setup Wizard[/bold cyan]\n"
        "[dim]Supports Claude + OpenAI | Multi-project management | Web dashboard[/dim]",
        border_style="cyan",
    ))

    if sys.version_info < (3, 10):
        console.print(f"[red]Python 3.10+ is required. Current version: {sys.version}[/red]")
        sys.exit(1)
    console.print(f"[green]OK[/green] Python {sys.version.split()[0]}")

    console.print("\n[cyan]Installing dependencies...[/cyan]")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(project_root / "requirements.txt"), "-q"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print(f"[red]Dependency installation failed:\n{result.stderr}[/red]")
        sys.exit(1)
    console.print("[green]OK[/green] Dependencies installed")

    env_path = project_root / ".env"
    if env_path.exists() and not Confirm.ask(".env already exists. Overwrite it?", default=False):
        console.print("[dim]Keeping existing configuration[/dim]")
    else:
        provider = Prompt.ask("Choose LLM provider", default="claude", choices=["claude", "openai"])
        if provider == "claude":
            anthropic_key = Prompt.ask("Anthropic API key (sk-ant-...)")
            claude_model = Prompt.ask(
                "Claude model",
                default="claude-sonnet-4-6",
                choices=["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5"],
            )
            openai_key = ""
            openai_model = "gpt-4o"
            openai_base = ""
        else:
            anthropic_key = ""
            claude_model = "claude-sonnet-4-6"
            openai_key = Prompt.ask("OpenAI API key (sk-...)")
            openai_model = Prompt.ask("OpenAI model", default="gpt-4o")
            openai_base = Prompt.ask("Custom API base URL (leave empty for default)", default="")

        env_content = f"""# Unity RAG Agent configuration
LLM_PROVIDER={provider}
ANTHROPIC_API_KEY={anthropic_key}
CLAUDE_MODEL={claude_model}
OPENAI_API_KEY={openai_key}
OPENAI_MODEL={openai_model}
OPENAI_BASE_URL={openai_base}
PROJECTS_DB_PATH=./projects.json
CHROMA_DB_ROOT=./rag_databases
RAG_TOP_K=6
CHUNK_MAX_TOKENS=600
UNITY_VERSION=2022.3 LTS
RENDER_PIPELINE=URP
DASHBOARD_HOST=127.0.0.1
DASHBOARD_PORT=5000
"""
        env_path.write_text(env_content, encoding="utf-8")
        console.print("[green]OK[/green] Configuration saved")

    project_path = Prompt.ask("Unity project root", default="D:/YourUnityProject")
    if os.path.isdir(project_path):
        from core.projects import get_project_manager
        from rag.indexer import build_index

        unity_version = Prompt.ask("Unity version", default=config.UNITY_VERSION)
        render_pipeline = Prompt.ask("Render pipeline", default=config.RENDER_PIPELINE, choices=["URP", "HDRP", "Built-in"])
        pm = get_project_manager()
        info = pm.add_project(project_path, unity_version=unity_version, render_pipeline=render_pipeline)
        console.print(f"[green]OK[/green] Registered project {info.name}")

        if Confirm.ask("Build the index now? (about 1-5 minutes)", default=True):
            build_index(info.path, info.db_path, info.collection_name, info.project_id)
    else:
        console.print(f"[yellow]Path does not exist, skipping project registration: {project_path}[/yellow]")

    console.print(Panel(
        "[bold green]Setup complete[/bold green]\n\n"
        "Getting started:\n"
        "  [cyan]python main.py[/cyan]                     # Web dashboard\n"
        "  [cyan]python main.py dashboard[/cyan]            # Web dashboard\n"
        "  [cyan]python main.py cli[/cyan]                  # CLI in current console\n"
        "  [cyan]python main.py cli -q 'question'[/cyan]    # Single query\n"
        "  [cyan]python main.py index --list[/cyan]         # List registered projects\n"
        "  [cyan]python main.py index --add <path>[/cyan]   # Register a new project\n"
        "  [cyan]python main.py watch[/cyan]                # File watcher",
        border_style="green",
    ))


def main():
    parser = argparse.ArgumentParser(
        description="Unity RAG Agent unified entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="subcommands")

    p_cli = subparsers.add_parser("cli", help="interactive CLI")
    p_cli.add_argument("-q", "--question", type=str, help="single question, then exit")
    p_cli.add_argument("--search", type=str, help="vector search only")
    p_cli.add_argument("--topk", type=int, default=6, help="number of retrieved chunks")
    p_cli.add_argument("--project", type=str, help="project name or id")
    p_cli.add_argument("--provider", type=str, choices=["claude", "openai"], help="LLM provider")

    p_dash = subparsers.add_parser("dashboard", help="web dashboard")
    p_dash.add_argument("--host", default=None, help="bind host")
    p_dash.add_argument("--port", type=int, default=None, help="bind port")
    p_dash.add_argument("--debug", action="store_true", help="debug mode")

    p_idx = subparsers.add_parser("index", help="index management")
    p_idx.add_argument("--update", action="store_true", help="incremental update")
    p_idx.add_argument("--stats", action="store_true", help="show index stats")
    p_idx.add_argument("--project", type=str, help="project name or id")
    p_idx.add_argument("--add", type=str, help="register a project path")
    p_idx.add_argument("--name", type=str, help="display name used with --add")
    p_idx.add_argument("--list", action="store_true", help="list all projects")

    p_watch = subparsers.add_parser("watch", help="watch project file changes")
    p_watch.add_argument("--project", type=str, help="project name or id")

    subparsers.add_parser("setup", help="setup wizard")

    args = parser.parse_args()

    if args.command is None:
        args.command = "dashboard"
        args.host = None
        args.port = None
        args.debug = False

    dispatch = {
        "cli": cmd_cli,
        "dashboard": cmd_dashboard,
        "index": cmd_index,
        "watch": cmd_watch,
        "setup": cmd_setup,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
