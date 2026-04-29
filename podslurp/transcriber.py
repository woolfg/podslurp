"""
transcriber.py — faster-whisper wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .config import Config


@dataclass
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float


@dataclass
class TranscriptResult:
    segments: list[Segment]
    detected_language: str
    detected_language_probability: float
    duration: float
    full_text: str = field(init=False)

    def __post_init__(self) -> None:
        self.full_text = " ".join(s.text.strip() for s in self.segments)


def transcribe(
    audio_path: Path,
    feed_language: Optional[str],
    config: Config,
) -> TranscriptResult:
    """Transcribe *audio_path* with faster-whisper.

    *feed_language* (e.g. ``"en"``, ``"de"``) is passed to Whisper as a hint
    so it skips the auto-detection pass and starts transcribing immediately.
    The detected language is still recorded in the returned result.
    """
    # Import here so the rest of the module loads fast (model init is slow)
    from faster_whisper import WhisperModel  # type: ignore[import]

    model = WhisperModel(
        config.whisper_model,
        device=config.whisper_device,
        compute_type=config.whisper_compute_type,
    )

    raw_segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        language=feed_language or None,
        vad_filter=True,
        word_timestamps=False,
    )

    # raw_segments is a lazy generator; update progress as decoding advances.
    segments: list[Segment] = []
    progress_seconds = 0.0
    with tqdm(
        total=info.duration,
        unit="s",
        unit_scale=True,
        desc="Transcribing",
        leave=True,
    ) as bar:
        for s in raw_segments:
            segments.append(
                Segment(
                    start=s.start,
                    end=s.end,
                    text=s.text,
                    avg_logprob=s.avg_logprob,
                    no_speech_prob=s.no_speech_prob,
                )
            )
            next_progress = min(s.end, info.duration)
            bar.update(max(0.0, next_progress - progress_seconds))
            progress_seconds = next_progress

        if progress_seconds < info.duration:
            bar.update(info.duration - progress_seconds)

    return TranscriptResult(
        segments=segments,
        detected_language=info.language,
        detected_language_probability=info.language_probability,
        duration=info.duration,
    )
