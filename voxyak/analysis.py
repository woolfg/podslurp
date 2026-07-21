"""Speaker-time analysis for VoxYak transcript documents."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .transcript import TranscriptDocument


UNKNOWN_SPEAKER = "UNKNOWN"


@dataclass(frozen=True)
class SpeakerTime:
    speaker: str
    seconds: float
    segments: int
    percentage: float


@dataclass(frozen=True)
class SpeakerAnalysis:
    speakers: list[SpeakerTime]
    total_seconds: float
    segment_count: int
    unlabeled_segment_count: int
    duration_seconds: float


def analyze_transcript(document: TranscriptDocument) -> SpeakerAnalysis:
    totals: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, int] = defaultdict(int)
    unlabeled = 0
    for number, segment in enumerate(document.segments, start=1):
        if segment.end < segment.start:
            raise ValueError(f"Segment {number} ends before it starts.")
        speaker = segment.speaker or UNKNOWN_SPEAKER
        if segment.speaker is None:
            unlabeled += 1
        totals[speaker] += segment.end - segment.start
        counts[speaker] += 1
    total = sum(totals.values())
    speakers = [
        SpeakerTime(
            speaker=speaker,
            seconds=seconds,
            segments=counts[speaker],
            percentage=(seconds / total * 100) if total else 0.0,
        )
        for speaker, seconds in totals.items()
    ]
    speakers.sort(key=lambda item: (-item.seconds, item.speaker))
    return SpeakerAnalysis(
        speakers=speakers,
        total_seconds=total,
        segment_count=len(document.segments),
        unlabeled_segment_count=unlabeled,
        duration_seconds=document.transcription.duration_seconds,
    )


def format_duration(seconds: float) -> str:
    seconds = round(max(0.0, seconds), 1)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remainder = seconds % 60
    if hours:
        return f"{hours}h {minutes:02d}m {remainder:04.1f}s"
    if minutes:
        return f"{minutes}m {remainder:04.1f}s"
    return f"{remainder:.1f}s"
