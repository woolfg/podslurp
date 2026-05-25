"""
__main__.py — interactive CLI entry point for podslurp.
"""
from __future__ import annotations

import sys
from datetime import datetime
import json
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from .api import get_episodes, search_podcasts
from .config import load_config
from .downloader import download_audio
from .output import build_stem, write_outputs
from .transcriber import Segment, TranscriptResult, _assign_speakers, transcribe

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: Optional[int]) -> str:
    if not seconds:
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def _prompt(text: str, default: str = "") -> str:
    try:
        value = console.input(f"[bold cyan]{text}[/bold cyan] ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Bye![/yellow]")
        sys.exit(0)
    return value or default


def _pick_number(prompt_text: str, max_val: int, back_key: str = "") -> Optional[int]:
    """Prompt user to pick 1–max_val. Returns None if they typed *back_key*."""
    while True:
        raw = _prompt(prompt_text)
        if back_key and raw.lower() == back_key:
            return None
        if raw.isdigit() and 1 <= int(raw) <= max_val:
            return int(raw)
        console.print(
            f"[red]Please enter a number between 1 and {max_val}"
            + (f", or '{back_key}' to go back" if back_key else "")
            + ".[/red]"
        )


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------

def _show_podcast_table(feeds: list[dict]) -> None:
    table = Table(box=box.SIMPLE_HEAD, show_lines=False, highlight=True)
    table.add_column("#", style="bold", width=4, justify="right")
    table.add_column("Podcast", style="bold white", min_width=30)
    table.add_column("Author", style="dim", min_width=20)
    table.add_column("Episodes", justify="right", width=9)
    table.add_column("Lang", width=6)
    for i, feed in enumerate(feeds, 1):
        table.add_row(
            str(i),
            feed.get("title", ""),
            feed.get("author", ""),
            str(feed.get("episodeCount", "?")),
            feed.get("language", ""),
        )
    console.print(table)


def _show_episode_table(episodes: list[dict]) -> None:
    table = Table(box=box.SIMPLE_HEAD, show_lines=False, highlight=True)
    table.add_column("#", style="bold", width=4, justify="right")
    table.add_column("Episode", style="bold white", min_width=40)
    table.add_column("Date", width=12)
    table.add_column("Duration", width=10, justify="right")
    for i, ep in enumerate(episodes, 1):
        table.add_row(
            str(i),
            ep.get("title", ""),
            ep.get("datePublishedPretty", "")[:10],
            _fmt_duration(ep.get("duration")),
        )
    console.print(table)


def _show_episode_detail(ep: dict, feed: dict) -> None:
    desc = (ep.get("description") or "").strip()
    if len(desc) > 300:
        desc = desc[:297] + "…"
    lines = [
        f"[bold]{ep.get('title', '')}[/bold]",
        f"Podcast:  {feed.get('title', '')}",
        f"Date:     {ep.get('datePublishedPretty', '')}",
        f"Duration: {_fmt_duration(ep.get('duration'))}",
        f"URL:      [dim]{ep.get('enclosureUrl', '')}[/dim]",
    ]
    if desc:
        lines += ["", desc]
    console.print(Panel("\n".join(lines), title="Episode", border_style="cyan"))


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def _run_pipeline(ep: dict, feed: dict, config) -> None:
    """Download + transcribe + write outputs for the chosen episode."""
    podcast_title: str = feed.get("title", "podcast")
    episode_title: str = ep.get("title", "episode")
    enclosure_url: str = ep.get("enclosureUrl", "")
    
    raw_lang = ep.get("feedLanguage") or feed.get("language")
    feed_language: Optional[str] = raw_lang.split("-")[0].lower() if raw_lang else None

    if not enclosure_url:
        console.print("[red]No audio URL found for this episode.[/red]")
        return

    # Derive a safe filename for the downloaded audio
    stem = build_stem(podcast_title, episode_title, ep.get("datePublished", 0))
    suffix = ".mp3"
    if "m4a" in (ep.get("enclosureType") or ""):
        suffix = ".m4a"
    audio_filename = stem + suffix

    # 1. Download
    audio_path = config.download_dir / audio_filename
    if audio_path.exists() and audio_path.stat().st_size > 0:
        console.print(f"\n[bold]Audio already downloaded:[/bold] {audio_path}")
    else:
        console.print(f"\n[bold]Downloading:[/bold] {audio_filename}")
        audio_path = download_audio(enclosure_url, config.download_dir, audio_filename)
        console.print(f"[green]Saved to:[/green] {audio_path}")

    # 2. Transcribe
    console.print(
        f"\n[bold]Transcribing[/bold] with [cyan]{config.whisper_model}[/cyan]"
        + (f" (language hint: {feed_language})" if feed_language else " (auto-detect language)")
        + " — this may take a while…"
    )
    result = transcribe(audio_path, feed_language, config)
    console.print(
        f"[green]Done.[/green] Detected language: [cyan]{result.detected_language}[/cyan]"
        f" ({result.detected_language_probability:.0%})"
        f"  |  Duration: {result.duration:.0f}s"
    )

    # 3. Write outputs
    txt_path, json_path = write_outputs(
        result,
        podcast_title=podcast_title,
        episode_title=episode_title,
        episode_url=enclosure_url,
        feed_url=feed.get("url", ""),
        date_published=ep.get("datePublished", 0),
        feed_language=feed_language,
        config=config,
        download_path=audio_path,
    )
    console.print("\n[bold green]Transcription saved:[/bold green]")
    console.print(f"  Text: [link]{txt_path}[/link]")
    console.print(f"  JSON: [link]{json_path}[/link]")


def main() -> None:
    config = load_config()

    if len(sys.argv) >= 3 and sys.argv[1] == "--transcribe":
        audio_path = Path(sys.argv[2])
        feed_language = None
        if len(sys.argv) >= 5 and sys.argv[3] == "--lang":
            feed_language = sys.argv[4]

        if not audio_path.exists() or not audio_path.is_file():
            console.print(f"[red]Error:[/red] File not found or is not a file: {audio_path}")
            sys.exit(1)

        lang_str = f" (language hint: {feed_language})" if feed_language else " (auto-detect language)"
        console.print(
            f"\n[bold]Transcribing[/bold] with [cyan]{config.whisper_model}[/cyan]"
            f"{lang_str} — this may take a while…"
        )
        result = transcribe(audio_path, feed_language, config)
        console.print(
            f"[green]Done.[/green] Detected language: [cyan]{result.detected_language}[/cyan]"
            f" ({result.detected_language_probability:.0%})"
            f"  |  Duration: {result.duration:.0f}s"
        )
        txt_path, json_path = write_outputs(
            result,
            podcast_title="Local Audio",
            episode_title=audio_path.stem,
            episode_url="",
            feed_url="",
            date_published=int(audio_path.stat().st_mtime),
            feed_language=feed_language,
            config=config,
            download_path=audio_path,
        )
        console.print("\n[bold green]Transcription saved:[/bold green]")
        console.print(f"  Text: [link]{txt_path}[/link]")
        console.print(f"  JSON: [link]{json_path}[/link]")
        sys.exit(0)

    if len(sys.argv) >= 3 and sys.argv[1] == "--diarize":
        json_path = Path(sys.argv[2])
        audio_path = Path(sys.argv[4]) if len(sys.argv) >= 5 and sys.argv[3] == "--audio" else None

        if not json_path.exists() or not json_path.is_file():
            console.print(f"[red]Error:[/red] JSON file not found: {json_path}")
            sys.exit(1)

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        meta = payload.get("metadata", {})

        if audio_path is None:
            # Try to find the audio from the download_path stored in the JSON
            stored = meta.get("download_path", "")
            if stored:
                audio_path = Path(stored)

        if not audio_path or not audio_path.exists():
            console.print(
                "[red]Error:[/red] Could not locate the audio file.\n"
                "Pass it explicitly: make diarize json=<file.json> audio=<file.mp3>"
            )
            sys.exit(1)

        if not config.diarize:
            console.print(
                "[red]Error:[/red] PODSLURP_DIARIZE is not enabled in your .env.\n"
                "Set PODSLURP_DIARIZE=true and provide a PYANNOTE_TOKEN."
            )
            sys.exit(1)

        segments = [
            Segment(
                start=s["start"],
                end=s["end"],
                text=s["text"],
                avg_logprob=s["avg_logprob"],
                no_speech_prob=s["no_speech_prob"],
                speaker=s.get("speaker"),
            )
            for s in payload.get("segments", [])
        ]
        result = TranscriptResult(
            segments=segments,
            detected_language=meta.get("detected_language", ""),
            detected_language_probability=meta.get("detected_language_probability", 0.0),
            duration=meta.get("duration_seconds", 0.0),
        )

        console.print("[bold]Running speaker diarization on existing transcript…[/bold]")
        _assign_speakers(
            result.segments,
            audio_path,
            config.pyannote_token,
            num_speakers=config.diarize_num_speakers,
            min_speakers=config.diarize_min_speakers,
            max_speakers=config.diarize_max_speakers,
        )
        # Rebuild full_text with speaker labels
        result.__post_init__()

        import calendar
        date_published = int(calendar.timegm(
            datetime.fromisoformat(meta["date_published_iso"]).timetuple()
        )) if meta.get("date_published_iso") else 0

        txt_path, json_path = write_outputs(
            result,
            podcast_title=meta.get("podcast_title", ""),
            episode_title=meta.get("episode_title", ""),
            episode_url=meta.get("episode_url", ""),
            feed_url=meta.get("feed_url", ""),
            date_published=date_published,
            feed_language=meta.get("feed_language"),
            config=config,
            download_path=audio_path,
        )
        console.print("\n[bold green]Diarized transcript saved:[/bold green]")
        console.print(f"  Text: [link]{txt_path}[/link]")
        console.print(f"  JSON: [link]{json_path}[/link]")
        sys.exit(0)

    console.print(
        Panel(
            "[bold cyan]podslurp[/bold cyan]  —  Podcast Search · Download · Transcribe",
            border_style="cyan",
            expand=False,
        )
    )

    while True:
        # --- Search for a podcast ---
        query = _prompt("Search for a podcast:")
        if not query:
            continue

        console.print(f"[dim]Searching PodcastIndex for '[bold]{query}[/bold]'…[/dim]")
        try:
            feeds = search_podcasts(query, config)
        except Exception as exc:
            console.print(f"[red]API error:[/red] {exc}")
            continue

        if not feeds:
            console.print("[yellow]No podcasts found. Try a different query.[/yellow]")
            continue

        _show_podcast_table(feeds)

        feed_idx = _pick_number(
            "Select podcast number (or 's' to search again):",
            len(feeds),
            back_key="s",
        )
        if feed_idx is None:
            continue

        feed = feeds[feed_idx - 1]
        feed_id: int = feed["id"]

        # --- Search within the selected podcast ---
        while True:
            keyword = _prompt("Episode keyword filter (Enter = 10 most recent):")

            console.print("[dim]Fetching episodes…[/dim]")
            try:
                all_episodes = get_episodes(feed_id, config)
            except Exception as exc:
                console.print(f"[red]API error:[/red] {exc}")
                break

            if keyword:
                episodes = [
                    ep for ep in all_episodes
                    if keyword.lower() in (ep.get("title") or "").lower()
                ]
            else:
                episodes = all_episodes[:10]

            if not episodes:
                console.print(
                    "[yellow]No episodes matched. Try a different keyword.[/yellow]"
                )
                continue

            _show_episode_table(episodes)

            ep_idx = _pick_number(
                "Select episode number (or 'b' to go back, 's' to search podcasts again):",
                len(episodes),
                back_key="b",
            )
            if ep_idx is None:
                break  # back to podcast search

            ep = episodes[ep_idx - 1]
            _show_episode_detail(ep, feed)

            confirm = _prompt("Download and transcribe? [Y/n]:").lower()
            if confirm in ("", "y", "yes"):
                try:
                    _run_pipeline(ep, feed, config)
                except Exception as exc:
                    console.print(f"[red]Error:[/red] {exc}")

            again = _prompt("Transcribe another episode? [Y/n]:").lower()
            if again not in ("", "y", "yes"):
                console.print("[cyan]Goodbye![/cyan]")
                sys.exit(0)


if __name__ == "__main__":
    main()
