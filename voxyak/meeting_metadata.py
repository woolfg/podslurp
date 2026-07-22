"""Optional free-text context collected after an interactive recording."""

from __future__ import annotations

from typing import Any


def _ask(console: Any, prompt: str) -> str | None:
    try:
        value = console.input(prompt).strip()
    except (EOFError, OSError):
        return None
    return value or None


def collect_meeting_metadata(console: Any) -> dict[str, str]:
    """Ask optional questions after recording; every answer remains free text."""
    console.print("\n[bold]Optional meeting context[/bold] [dim](Enter to skip)[/dim]")
    values = {
        "participants": _ask(console, "Speakers / participants: "),
        "participant_count": _ask(console, "Number of speakers: "),
        "context": _ask(console, "Meeting context (one line): "),
    }
    return {key: value for key, value in values.items() if value}
