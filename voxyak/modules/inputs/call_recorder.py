"""Linux PulseAudio/PipeWire call recording through pactl and ffmpeg."""

from __future__ import annotations

import re
import shutil
import signal
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ...meeting_metadata import collect_meeting_metadata
from ...metadata import PipelineMetadata
from ...sdk import AudioArtifact, InputModule, ModuleConfig, RunContext


class CallRecorderConfig(ModuleConfig):
    microphone_pattern: str
    system_pattern: str
    format: Literal["mp3", "wav"] = "mp3"
    mp3_quality: int = Field(default=2, ge=0, le=9)
    keep_separate_tracks: bool = True

    @field_validator("microphone_pattern", "system_pattern")
    @classmethod
    def valid_regex(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"Invalid source regex {value!r}: {exc}") from exc
        return value


def _require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise ValueError(f"Required command not found on PATH: {name}")


def _sources() -> list[str]:
    completed = subprocess.run(
        ["pactl", "list", "short", "sources"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        fields[1]
        for line in completed.stdout.splitlines()
        if len(fields := line.split()) >= 2
    ]


def _match_source(sources: list[str], pattern: str, label: str) -> str:
    try:
        matcher = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid {label} source regex {pattern!r}: {exc}") from exc
    matches = [source for source in sources if matcher.search(source)]
    if not matches:
        raise ValueError(f"No {label} source matches {pattern!r}.")
    if len(matches) > 1:
        formatted = "\n  ".join(matches)
        raise ValueError(
            f"Multiple {label} sources match {pattern!r}; narrow the pattern:\n  "
            f"{formatted}"
        )
    return matches[0]


def _codec_args(settings: CallRecorderConfig) -> list[str]:
    if settings.format == "wav":
        return ["-c:a", "pcm_s16le"]
    return ["-c:a", "libmp3lame", "-q:a", str(settings.mp3_quality)]


class CallRecorderInput(InputModule):
    config_model = CallRecorderConfig

    def preflight(self, config: BaseModel) -> None:
        CallRecorderConfig.model_validate(config)
        _require_command("pactl")
        _require_command("ffmpeg")

    def run(self, context: RunContext, config: BaseModel) -> AudioArtifact:
        settings = CallRecorderConfig.model_validate(config)
        sources = _sources()
        system_source = _match_source(sources, settings.system_pattern, "system")
        microphone_source = _match_source(
            sources, settings.microphone_pattern, "microphone"
        )
        if system_source == microphone_source:
            raise ValueError("System and microphone patterns resolved to the same source.")

        context.work_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        base = context.work_dir / f"call-{timestamp}"
        microphone_path = Path(f"{base}-microphone.{settings.format}")
        system_path = Path(f"{base}-system.{settings.format}")
        merged_path = Path(f"{base}-merged.{settings.format}")
        codec = _codec_args(settings)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "info",
            "-y",
            "-f",
            "pulse",
            "-i",
            system_source,
            "-f",
            "pulse",
            "-i",
            microphone_source,
            "-map",
            "0:a",
            *codec,
            str(system_path),
            "-map",
            "1:a",
            *codec,
            str(microphone_path),
        ]
        context.console.print(f"[bold]System source:[/bold] {system_source}")
        context.console.print(f"[bold]Microphone source:[/bold] {microphone_source}")
        context.console.print("[green]Recording. Press Enter or Ctrl-C to stop.[/green]")
        process = subprocess.Popen(command, start_new_session=True)
        try:
            context.console.input("")
        except (KeyboardInterrupt, EOFError, OSError):
            pass
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=5)

        for track in (microphone_path, system_path):
            if not track.is_file() or track.stat().st_size == 0:
                raise ValueError(f"Recording did not produce a usable track: {track}")

        merge_command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "info",
            "-y",
            "-i",
            str(microphone_path),
            "-i",
            str(system_path),
            "-filter_complex",
            "amix=inputs=2:duration=longest:dropout_transition=0",
            *_codec_args(settings),
            str(merged_path),
        ]
        subprocess.run(merge_command, check=True)
        if not merged_path.is_file() or merged_path.stat().st_size == 0:
            raise ValueError("ffmpeg merge did not produce a usable audio file.")

        context_metadata = collect_meeting_metadata(context.console)
        tracks = {
            "primary": merged_path.resolve(),
            "microphone": microphone_path.resolve(),
            "system": system_path.resolve(),
        }
        if not settings.keep_separate_tracks:
            microphone_path.unlink()
            system_path.unlink()
            tracks = {"primary": merged_path.resolve()}

        media_type = "audio/wav" if settings.format == "wav" else "audio/mpeg"
        return AudioArtifact(
            path=merged_path.resolve(),
            media_type=media_type,
            tracks=tracks,
            metadata=PipelineMetadata(
                source_module="call-recorder",
                title=base.name,
                **context_metadata,
                attributes={
                    "microphone_source": microphone_source,
                    "system_source": system_source,
                },
            ),
        )
