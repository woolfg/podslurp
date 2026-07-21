"""Local faster-whisper transcription with optional speaker diarization."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from tqdm import tqdm

from ...sdk import (
    AudioArtifact,
    ModuleConfig,
    RunContext,
    TranscriptArtifact,
    TranscriptionModule,
)
from ...transcript import (
    TranscriptDocument,
    TranscriptSegment,
    TranscriptSource,
    TranscriptionDetails,
    build_full_text,
    load_transcript,
    write_transcript,
)


class DiarizationConfig(ModuleConfig):
    enabled: bool = False
    num_speakers: int | None = Field(default=None, ge=1)
    min_speakers: int | None = Field(default=None, ge=1)
    max_speakers: int | None = Field(default=None, ge=1)
    device: str | None = None
    batch_size: int = Field(default=32, ge=1)

    @model_validator(mode="after")
    def validate_speaker_bounds(self) -> "DiarizationConfig":
        if self.num_speakers is not None and (
            self.min_speakers is not None or self.max_speakers is not None
        ):
            raise ValueError("num_speakers cannot be combined with min/max_speakers.")
        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.min_speakers > self.max_speakers
        ):
            raise ValueError("min_speakers cannot exceed max_speakers.")
        return self


class FasterWhisperConfig(ModuleConfig):
    model: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "auto"
    beam_size: int = Field(default=5, ge=1)
    vad_filter: bool = True
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)


def _require_dependency(module: str, install_hint: str) -> None:
    if importlib.util.find_spec(module) is None:
        raise ValueError(f"Missing dependency {module!r}. Install with: {install_hint}")


def _language_hint(settings: FasterWhisperConfig, audio: AudioArtifact) -> str | None:
    if settings.language == "auto":
        return None
    if settings.language == "source":
        value = audio.metadata.get("language_hint")
        return str(value) if value else None
    return settings.language.split("-")[0].lower()


def _assign_speakers(
    segments: list[TranscriptSegment],
    audio_path: Path,
    settings: DiarizationConfig,
) -> None:
    from pyannote.audio import Pipeline  # type: ignore[import-not-found]
    import torch
    import torchaudio  # type: ignore[import-not-found]

    token = os.getenv("PYANNOTE_TOKEN", "").strip()
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token,
    )
    device = settings.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(torch.device(device))
    waveform, sample_rate = torchaudio.load(str(audio_path.resolve()))
    result = pipeline(
        {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "uri": audio_path.stem,
        },
        num_speakers=settings.num_speakers,
        min_speakers=settings.min_speakers,
        max_speakers=settings.max_speakers,
        batch_size=settings.batch_size,
    )
    annotation = (
        result.speaker_diarization
        if hasattr(result, "speaker_diarization")
        else result
    )
    turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    for segment in segments:
        best_speaker: str | None = None
        best_overlap = 0.0
        for start, end, speaker in turns:
            overlap = max(0.0, min(segment.end, end) - max(segment.start, start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        segment.speaker = best_speaker


def _validate_diarization(settings: DiarizationConfig) -> None:
    if not settings.enabled:
        return
    if not os.getenv("PYANNOTE_TOKEN", "").strip():
        raise ValueError("PYANNOTE_TOKEN is required when diarization is enabled.")
    _require_dependency("pyannote.audio", "uv sync --extra diarize")


class FasterWhisperTranscriber(TranscriptionModule):
    config_model = FasterWhisperConfig

    def preflight(self, config: BaseModel) -> None:
        settings = FasterWhisperConfig.model_validate(config)
        _require_dependency("faster_whisper", "uv sync")
        _validate_diarization(settings.diarization)

    def run(
        self,
        context: RunContext,
        audio: AudioArtifact,
        config: BaseModel,
    ) -> TranscriptArtifact:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]

        settings = FasterWhisperConfig.model_validate(config)
        hint = _language_hint(settings, audio)
        context.console.print(
            f"[bold]Transcribing[/bold] with [cyan]{settings.model}[/cyan]"
            + (f" (language: {hint})" if hint else " (automatic language detection)")
        )
        model = WhisperModel(
            settings.model,
            device=settings.device,
            compute_type=settings.compute_type,
        )
        raw_segments, info = model.transcribe(
            str(audio.path),
            beam_size=settings.beam_size,
            language=hint,
            vad_filter=settings.vad_filter,
            word_timestamps=False,
        )
        segments: list[TranscriptSegment] = []
        progress_seconds = 0.0
        with tqdm(
            total=info.duration,
            unit="s",
            unit_scale=True,
            desc="Transcribing",
        ) as progress:
            for raw in raw_segments:
                segments.append(
                    TranscriptSegment(
                        start=raw.start,
                        end=raw.end,
                        text=raw.text,
                        avg_logprob=raw.avg_logprob,
                        no_speech_prob=raw.no_speech_prob,
                    )
                )
                next_progress = min(raw.end, info.duration)
                progress.update(max(0.0, next_progress - progress_seconds))
                progress_seconds = next_progress
            progress.update(max(0.0, info.duration - progress_seconds))

        if settings.diarization.enabled:
            context.console.print("[bold]Running speaker diarization…[/bold]")
            _assign_speakers(segments, audio.path, settings.diarization)

        document = TranscriptDocument(
            source=TranscriptSource(
                module=str(audio.metadata.get("input_module") or "unknown"),
                title=str(audio.metadata.get("title") or audio.path.stem),
                audio_path=str(audio.path),
                metadata={
                    key: value
                    for key, value in audio.metadata.items()
                    if key not in {"input_module", "title"}
                },
            ),
            transcription=TranscriptionDetails(
                module="faster-whisper",
                model=settings.model,
                language=info.language,
                language_probability=info.language_probability,
                duration_seconds=info.duration,
            ),
            segments=segments,
            full_text=build_full_text(segments),
        )
        json_path, text_path = write_transcript(document, context.work_dir)
        return TranscriptArtifact(
            path=json_path.resolve(),
            text_path=text_path.resolve(),
            language=info.language,
            duration_seconds=info.duration,
            metadata={"title": document.source.title},
        )


def diarize_transcript_file(
    transcript_path: Path,
    audio_path: Path | None,
    settings: DiarizationConfig,
) -> tuple[Path, Path]:
    settings = settings.model_copy(update={"enabled": True})
    _validate_diarization(settings)
    document = load_transcript(transcript_path)
    resolved_audio = audio_path or Path(document.source.audio_path)
    if not resolved_audio.is_file():
        raise ValueError(f"Audio file not found: {resolved_audio}")
    _assign_speakers(document.segments, resolved_audio, settings)
    document.full_text = build_full_text(document.segments)
    return write_transcript(document, transcript_path.parent)
