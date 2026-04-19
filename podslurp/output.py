"""
output.py — file naming helpers and writers for .txt / .json transcription output.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .transcriber import TranscriptResult


def _sanitize(text: str, max_len: int = 80) -> str:
    """Lower-case, replace non-alphanumeric chars with underscores, collapse runs."""
    slug = re.sub(r"[^\w]", "_", text.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:max_len]


def build_stem(
    podcast_title: str,
    episode_title: str,
    date_published: int,
) -> str:
    """Return the base filename (no extension) for output files."""
    date_str = datetime.fromtimestamp(date_published, tz=timezone.utc).strftime(
        "%Y-%m-%d"
    )
    podcast_slug = _sanitize(podcast_title)
    episode_slug = _sanitize(episode_title)
    return f"{podcast_slug}_{date_str}_{episode_slug}"


def write_outputs(
    result: TranscriptResult,
    *,
    podcast_title: str,
    episode_title: str,
    episode_url: str,
    feed_url: str,
    date_published: int,
    feed_language: Optional[str],
    config,
    download_path: Path,
) -> tuple[Path, Path]:
    """Write .txt and .json transcription files.

    Returns ``(txt_path, json_path)``.
    """
    output_dir: Path = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = build_stem(podcast_title, episode_title, date_published)
    txt_path = output_dir / f"{stem}.txt"
    json_path = output_dir / f"{stem}.json"

    date_pretty = datetime.fromtimestamp(date_published, tz=timezone.utc).strftime(
        "%Y-%m-%d"
    )

    # --- .txt ------------------------------------------------------------------
    header_lines = [
        f"Podcast:           {podcast_title}",
        f"Episode:           {episode_title}",
        f"Published:         {date_pretty}",
        f"Language (feed):   {feed_language or 'unknown'}",
        f"Detected language: {result.detected_language}"
        f" ({result.detected_language_probability:.0%})",
        f"Whisper model:     {config.whisper_model}",
        f"Duration:          {result.duration:.0f}s",
        "",
        "--- TRANSCRIPT ---",
        "",
    ]
    txt_path.write_text(
        "\n".join(header_lines) + result.full_text,
        encoding="utf-8",
    )

    # --- .json -----------------------------------------------------------------
    payload = {
        "metadata": {
            "podcast_title": podcast_title,
            "episode_title": episode_title,
            "episode_url": episode_url,
            "feed_url": feed_url,
            "date_published_iso": datetime.fromtimestamp(
                date_published, tz=timezone.utc
            ).isoformat(),
            "duration_seconds": result.duration,
            "feed_language": feed_language,
            "detected_language": result.detected_language,
            "detected_language_probability": result.detected_language_probability,
            "whisper_model": config.whisper_model,
            "download_path": str(download_path),
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
        },
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "avg_logprob": s.avg_logprob,
                "no_speech_prob": s.no_speech_prob,
            }
            for s in result.segments
        ],
        "full_text": result.full_text,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return txt_path, json_path
