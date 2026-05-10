#!/usr/bin/env python3
"""Typer CLI for the WhatsApp Voice Clone system."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import settings
from logger import setup_logging
from profiler.analyzer import VoiceProfile, analyze_chat
from profiler.profile_store import ProfileStore

app = typer.Typer(name="whatsapp-voice-clone", help="WhatsApp Voice Clone — train, serve & test AI voice profiles.", add_completion=False)
console = Console()

# Ensure data directories exist
DATA_EXPORTS = Path("data/exports")
DATA_PROFILES = Path("data/profiles")
DATA_EXPORTS.mkdir(parents=True, exist_ok=True)
DATA_PROFILES.mkdir(parents=True, exist_ok=True)


@app.command()
def train(
    export: str = typer.Option(..., "--export", "-e", help="Path to exported WhatsApp chat file (.txt or .json)"),
    name: str = typer.Option(..., "--name", "-n", help="Profile name (e.g. 'Mario Pizzeria')"),
) -> None:
    """Train a new voice profile from an exported WhatsApp chat."""
    setup_logging(settings.log_level)
    export_path = Path(export)

    if not export_path.exists():
        console.print(f"[red]✗ Export file not found: {export_path}[/red]")
        raise typer.Exit(1)

    console.print(Panel(f"[bold yellow]🔊 Training voice profile:[/bold yellow] [cyan]{name}[/cyan]", width=70))
    console.print(f"   📄 Export: {export_path.resolve()}")

    with console.status("[yellow]Analyzing chat history...[/yellow]", spinner="dots"):
        profile = analyze_chat(str(export_path))
        profile.name = name

    # Show profile summary
    formality_label = profile.formality.get("label", "sconosciuto")
    vocab_count = len(profile.vocabulary.get("top_words", []))
    emoji_count = len(profile.emoji.get("preferred", []))
    greeting_count = sum(profile.greetings.get("patterns", {}).values())
    top_greeting = profile.greetings.get("most_common", "N/A")
    busiest_hour = profile.time_patterns.get("busiest_hour", "N/A")

    console.print(f"\n[bold green]✅ Profile analyzed![/bold green]")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Dimension", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Formality", f"{formality_label} ({profile.formality.get('score', 0):.2f})")
    table.add_row("Top Words", f"{vocab_count} unique")
    table.add_row("Preferred Emojis", f"{emoji_count} found")
    table.add_row("Greetings", f"{greeting_count} total — most common: {top_greeting}")
    table.add_row("Avg Message", f"{profile.response_style.get('avg_length_chars', 0):.0f} chars")
    table.add_row("Questions", f"{profile.response_style.get('question_pct', 0):.1f}% of messages")
    table.add_row("Busiest Hour", f"{busiest_hour}:00")
    console.print(table)

    # Save to ChromaDB
    store = ProfileStore(persist_dir=str(settings.chroma_path))
    with console.status("[yellow]Saving profile to database...[/yellow]"):
        profile_id = store.save(profile)

    console.print(f"\n[bold green]✓[/bold green] Profile saved! [dim]ID: {profile_id}[/dim]")
    console.print(f"   [bold]Run:[/bold] python main.py serve --name \"{name}\"")


@app.command()
def serve(
    name: str = typer.Option(..., "--name", "-n", help="Profile name to use"),
    port: int = typer.Option(8080, "--port", "-p", help="HTTP port"),
) -> None:
    """Start the FastAPI webhook server."""
    setup_logging(settings.log_level)

    # Verify profile exists
    store = ProfileStore(persist_dir=str(settings.chroma_path))
    profile = store.load(name)
    if profile is None:
        console.print(f"[red]✗ Profile '{name}' not found. Train it first:[/red]")
        console.print(f"   python main.py train --export ... --name \"{name}\"")
        raise typer.Exit(1)

    console.print(Panel(f"[bold green]🚀 Serving profile:[/bold green] [cyan]{name}[/cyan] on :{port}", width=60))
    console.print(f"   📡 Webhook: POST http://0.0.0.0:{port}/webhook/twilio")
    console.print(f"   💚 Health:   GET  http://0.0.0.0:{port}/health")
    console.print(f"   📋 Profiles: GET  http://0.0.0.0:{port}/profiles")
    console.print("\n[dim]Set this webhook URL as your Twilio WhatsApp webhook.[/dim]")

    # Override settings
    settings.port = port
    settings.profile_name = name

    # Start server
    import uvicorn
    uvicorn.run(
        "server.webhook:app",
        host="0.0.0.0",
        port=port,
        log_level=settings.log_level.lower(),
    )


@app.command(name="test-prompt")
def test_prompt(
    profile_name: str = typer.Option(..., "--profile", "-p", help="Profile name"),
    message: str = typer.Option(..., "--message", "-m", help="Customer message to test"),
) -> None:
    """Test a customer message against a voice profile without serving."""
    setup_logging(settings.log_level)

    store = ProfileStore(persist_dir=str(settings.chroma_path))
    profile = store.load(profile_name)
    if profile is None:
        console.print(f"[red]✗ Profile '{profile_name}' not found.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold yellow]🧪 Testing prompt for:[/bold yellow] [cyan]{profile_name}[/cyan]")
    console.print(f"[dim]Customer:[/dim] {message}\n")

    # Run through full pipeline (dry run without LLM call if --dry flag)
    from responder.guardrails import Guardrails
    from responder.prompt_builder import PromptBuilder

    guardrails = Guardrails()
    result = guardrails.check(message)

    builder = PromptBuilder(profile)
    system_prompt = builder.build_system_prompt()
    user_prompt = builder.build_user_prompt(message)

    console.print(Panel(system_prompt[:2000], title="[bold]System Prompt[/bold]", border_style="blue"))
    console.print(f"\n[bold]User Prompt:[/bold] [dim]{user_prompt[:300]}[/dim]")

    if result.should_escalate:
        console.print(f"\n[red]🚨 GUARDRAIL TRIGGERED:[/red] {result.reason}")
        console.print(f"   [red]→ This message would be ESCALATED to the owner.[/red]")
    else:
        console.print(f"\n[green]✓ Guardrail passed (confidence: {result.confidence:.2f})[/green]")

    # Show profile summary
    console.print(f"\n[bold]Profile Summary:[/bold]")
    console.print(builder.get_profile_summary())


@app.command(name="list-profiles")
def list_profiles() -> None:
    """List all stored voice profiles."""
    setup_logging(settings.log_level)
    store = ProfileStore(persist_dir=str(settings.chroma_path))
    profiles = store.list_profiles()

    if not profiles:
        console.print("[yellow]No profiles found. Train one first:[/yellow]")
        console.print("   python main.py train --export data/exports/chat.txt --name \"My Profile\"")
        return

    table = Table(title="Stored Voice Profiles", title_style="bold yellow")
    table.add_column("Name", style="cyan")
    table.add_column("Created", style="white")
    table.add_column("ID", style="dim")

    for p in profiles:
        table.add_row(p.get("name", "?"), p.get("created_at", "?"), p.get("id", "?")[:12] + "...")

    console.print(table)


if __name__ == "__main__":
    app()
