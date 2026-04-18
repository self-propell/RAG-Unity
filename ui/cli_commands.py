"""
CLI commands for LangGraph agent

Additional commands:
  /rollback [steps]  - Roll back conversation by N steps (default 1)
  /history          - Show checkpoint history
  /approval         - View/edit approval configuration
  /yolo             - Toggle YOLO mode (auto-approve all tools)
"""
from rich.console import Console
from rich.table import Table

console = Console()


def cmd_rollback(agent, steps: int = 1):
    """Roll back conversation by N steps"""
    if not hasattr(agent, 'rollback'):
        console.print("[red]Rollback not supported (using old agent)[/red]")
        return

    success = agent.rollback(steps)
    if success:
        console.print(f"[green]Rolled back {steps} step(s)[/green]")
    else:
        console.print(f"[red]Cannot rollback {steps} step(s)[/red]")


def cmd_history(agent):
    """Show checkpoint history"""
    if not hasattr(agent, 'get_checkpoint_history'):
        console.print("[red]Checkpoint history not supported (using old agent)[/red]")
        return

    history = agent.get_checkpoint_history()

    if not history:
        console.print("[yellow]No checkpoint history available[/yellow]")
        return

    table = Table(title="Checkpoint History")
    table.add_column("Step", style="cyan")
    table.add_column("Timestamp", style="green")
    table.add_column("Iteration", style="yellow")
    table.add_column("Messages", style="blue")

    for i, checkpoint in enumerate(reversed(history)):
        from datetime import datetime
        ts = datetime.fromtimestamp(checkpoint.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M:%S")
        table.add_row(
            str(i),
            ts,
            str(checkpoint.get("iteration", 0)),
            str(checkpoint.get("message_count", 0))
        )

    console.print(table)


def cmd_approval(agent):
    """View/edit approval configuration"""
    console.print("\n[bold]Approval Configuration[/bold]")
    console.print("\nEdit approval_config in projects.json to customize.")
    console.print("\nDefault dangerous tools:")
    console.print("  - run_command")
    console.print("  - rag_rebuild")
    console.print("\nModes:")
    console.print("  auto  - No approval needed")
    console.print("  batch - Approve batches of tool calls")
    console.print("  yolo  - Approve once, then auto-approve all")


def cmd_yolo(state):
    """Toggle YOLO mode"""
    console.print("\n[yellow]YOLO mode is session-based and activated during approval prompts.[/yellow]")
    console.print("When prompted for approval, choose 'yolo' to enable it.")
