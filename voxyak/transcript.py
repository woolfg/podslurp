"""VoxYak's versioned transcript document and rendering helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    text: str
    speaker: str | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


class TranscriptSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module: str
    title: str
    audio_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TranscriptionDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module: str
    model: str
    language: str
    language_probability: float
    duration_seconds: float
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class TranscriptDocument(BaseModel):
    """The only transcript wire format supported by VoxYak 0.2."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    source: TranscriptSource
    transcription: TranscriptionDetails
    segments: list[TranscriptSegment]
    full_text: str


def build_full_text(segments: list[TranscriptSegment]) -> str:
    parts: list[str] = []
    current_speaker: str | None = None
    for segment in segments:
        if segment.speaker and segment.speaker != current_speaker:
            current_speaker = segment.speaker
            parts.append(f"\n[{current_speaker}]")
        parts.append(segment.text.strip())
    return " ".join(parts).strip()


def load_transcript(path: Path) -> TranscriptDocument:
    """Load only the VoxYak schema; legacy transcript layouts are rejected."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid transcript JSON: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Not a VoxYak transcript (schema_version must be 1).")
    return TranscriptDocument.model_validate(raw)


def render_transcript_text(document: TranscriptDocument) -> str:
    source = document.source
    details = document.transcription
    header = [
        f"Title:             {source.title}",
        f"Input module:      {source.module}",
        f"Transcriber:       {details.module}",
        f"Detected language: {details.language} ({details.language_probability:.0%})",
        f"Model:             {details.model}",
        f"Duration:          {details.duration_seconds:.1f}s",
        "",
        "--- TRANSCRIPT ---",
        "",
    ]
    return "\n".join(header) + document.full_text + "\n"


def write_transcript(
    document: TranscriptDocument,
    directory: Path,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "transcript.json"
    text_path = directory / "transcript.txt"
    json_path.write_text(
        document.model_dump_json(indent=2),
        encoding="utf-8",
    )
    text_path.write_text(render_transcript_text(document), encoding="utf-8")
    return json_path, text_path
