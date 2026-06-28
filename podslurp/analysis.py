"""
analysis.py - helpers for summarizing saved transcript JSON files.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


UNKNOWN_SPEAKER = "UNKNOWN"


@dataclass(frozen=True)
class SpeakerTime:
    speaker: str
    seconds: float
    segments: int
    percentage: float


@dataclass(frozen=True)
class SpeakingTimeAnalysis:
    speakers: list[SpeakerTime]
    total_seconds: float
    segment_count: int
    unlabeled_segment_count: int
    duration_seconds: Optional[float]
    metadata: dict[str, Any]


def _as_float(value: Any, field_name: str, segment_number: int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Segment {segment_number} has a non-numeric {field_name!r} value."
        ) from exc


def _optional_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_duration(seconds: float) -> str:
    """Format seconds as a compact duration for terminal output."""
    seconds = round(max(0.0, seconds), 1)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remainder = seconds % 60

    if hours:
        return f"{hours}h {minutes:02d}m {remainder:04.1f}s"
    if minutes:
        return f"{minutes}m {remainder:04.1f}s"
    return f"{remainder:.1f}s"


def analyze_transcript(payload: dict[str, Any]) -> SpeakingTimeAnalysis:
    """Calculate speaking time per speaker from a transcript payload."""
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("Transcript JSON must contain a 'segments' list.")

    totals: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, int] = defaultdict(int)
    unlabeled_segment_count = 0

    for index, segment in enumerate(raw_segments, start=1):
        if not isinstance(segment, dict):
            raise ValueError(f"Segment {index} must be a JSON object.")

        start = _as_float(segment.get("start"), "start", index)
        end = _as_float(segment.get("end"), "end", index)
        if end < start:
            raise ValueError(f"Segment {index} ends before it starts.")

        speaker = segment.get("speaker")
        if not speaker:
            speaker = UNKNOWN_SPEAKER
            unlabeled_segment_count += 1

        speaker_name = str(speaker)
        totals[speaker_name] += end - start
        counts[speaker_name] += 1

    total_seconds = sum(totals.values())
    speakers = [
        SpeakerTime(
            speaker=speaker,
            seconds=seconds,
            segments=counts[speaker],
            percentage=(seconds / total_seconds * 100) if total_seconds else 0.0,
        )
        for speaker, seconds in totals.items()
    ]
    speakers.sort(key=lambda item: (-item.seconds, item.speaker))

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    return SpeakingTimeAnalysis(
        speakers=speakers,
        total_seconds=total_seconds,
        segment_count=len(raw_segments),
        unlabeled_segment_count=unlabeled_segment_count,
        duration_seconds=_optional_float(metadata.get("duration_seconds")),
        metadata=metadata,
    )


def analyze_transcript_file(json_path: Path) -> SpeakingTimeAnalysis:
    """Load a transcript JSON file and calculate speaking time per speaker."""
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Transcript JSON root must be an object.")

    return analyze_transcript(payload)
