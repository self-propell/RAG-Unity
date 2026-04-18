"""
CLI commands for Planning Agent

Usage:
  /plan <task>          - Create and execute a plan
  /plan-show           - Show current plan
  /plan-history        - Show planning history
"""
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


def cmd_plan(state, task: str):
    """Create and execute a plan for a task"""
    if not task:
        console.print("[red]Please provide a task description[/red]")
        return

    console.print(f"\n[cyan]Planning:[/cyan] {task}\n")

    # Create planning agent
    from llm.planning import PlanningAgent

    planning_agent = PlanningAgent(
        project=state.project,
        provider=state.agent.provider
    )

    # Execute plan
    with console.status("[bold green]Creating plan..."):
        result = planning_agent.plan_and_execute(task, max_replans=2)

    # Display plan
    console.print("\n[bold]📋 Plan:[/bold]")
    plan_table = Table(show_header=True, header_style="bold cyan")
    plan_table.add_column("Step", style="cyan", width=6)
    plan_table.add_column("Description", style="white")
    plan_table.add_column("Tool", style="yellow", width=15)
    plan_table.add_column("Status", width=10)

    for step in result["plan"]:
        status = step.get("status", "pending")
        status_icon = {
            "completed": "✓",
            "failed": "✗",
            "in_progress": "⋯",
            "pending": "○"
        }.get(status, "?")

        status_color = {
            "completed": "green",
            "failed": "red",
            "in_progress": "yellow",
            "pending": "dim"
        }.get(status, "white")

        plan_table.add_row(
            str(step["step_id"]),
            step["description"],
            step["tool"],
            f"[{status_color}]{status_icon} {status}[/{status_color}]"
        )

    console.print(plan_table)

    # Display execution results
    console.print("\n[bold]🔧 Execution Results:[/bold]")
    for exec_result in result["execution_results"]:
        status = "✓" if exec_result["success"] else "✗"
        color = "green" if exec_result["success"] else "red"
        console.print(f"  [{color}]{status}[/{color}] Step {exec_result['step_id']}: {exec_result['tool']}")

    # Display reflection
    if result["reflection"]:
        console.print("\n[bold]💭 Reflection:[/bold]")
        console.print(Panel(result["reflection"], border_style="blue"))

    # Display replan count
    if result["replan_count"] > 0:
        console.print(f"\n[yellow]Replanned {result['replan_count']} time(s)[/yellow]")

    # Success/failure
    if result["success"]:
        console.print("\n[bold green]✓ Task completed successfully![/bold green]")
    else:
        console.print("\n[bold red]✗ Task failed or incomplete[/bold red]")


def cmd_plan_show(state):
    """Show current plan (if any)"""
    console.print("[yellow]Current plan display not yet implemented[/yellow]")
    console.print("Use /plan <task> to create a new plan")


def cmd_plan_history(state):
    """Show planning history"""
    console.print("[yellow]Planning history not yet implemented[/yellow]")
    console.print("Future feature: view past plans and their outcomes")
