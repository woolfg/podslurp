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
    speaker: Optional[str] = None


@dataclass
class TranscriptResult:
    segments: list[Segment]
    detected_language: str
    detected_language_probability: float
    duration: float
    full_text: str = field(init=False)

    def __post_init__(self) -> None:
        parts: list[str] = []
        current_speaker: Optional[str] = None
        for s in self.segments:
            if s.speaker and s.speaker != current_speaker:
                current_speaker = s.speaker
                parts.append(f"\n[{current_speaker}]")
            parts.append(s.text.strip())
        self.full_text = " ".join(parts).strip()


def _assign_speakers(
    segments: list[Segment],
    audio_path: Path,
    pyannote_token: str,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> None:
    """Run pyannote diarization and tag each segment with the dominant speaker in-place."""
    try:
        from pyannote.audio import Pipeline  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pyannote.audio is required for diarization. "
            "Install it with: uv sync --extra diarize"
        ) from exc

    print("Running speaker diarization …")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=pyannote_token,
    )
    # Load the full waveform upfront via torchaudio (already a pyannote dependency).
    # Passing a pre-loaded tensor avoids MP3 duration estimation errors that cause
    # pyannote's crop() to raise a ValueError when a chunk slightly overshoots EOF.
    import torch
    import torchaudio  # type: ignore[import]
    waveform, sample_rate = torchaudio.load(str(audio_path.resolve()))
    audio_file = {"waveform": waveform, "sample_rate": sample_rate, "uri": audio_path.stem}
    diarization = pipeline(
        audio_file,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    # Build a list of (start, end, speaker) tuples from the diarization result.
    # The community pipeline returns a DiarizeOutput dataclass; extract the
    # Annotation via .speaker_diarization. Fall back to the object itself for
    # older / standard pipelines that return an Annotation directly.
    annotation = (
        diarization.speaker_diarization
        if hasattr(diarization, "speaker_diarization")
        else diarization
    )
    dia_turns: list[tuple[float, float, str]] = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]

    for seg in segments:
        # Find the diarization turn with the greatest overlap with this segment
        best_speaker: Optional[str] = None
        best_overlap = 0.0
        for d_start, d_end, speaker in dia_turns:
            overlap = max(0.0, min(seg.end, d_end) - max(seg.start, d_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        seg.speaker = best_speaker


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

    if config.diarize:
        _assign_speakers(
            segments,
            audio_path,
            config.pyannote_token,
            num_speakers=config.diarize_num_speakers,
            min_speakers=config.diarize_min_speakers,
            max_speakers=config.diarize_max_speakers,
        )

    return TranscriptResult(
        segments=segments,
        detected_language=info.language,
        detected_language_probability=info.language_probability,
        duration=info.duration,
    )
