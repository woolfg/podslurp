"""Input module for an existing local audio file."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from pydantic import BaseModel

from ...sdk import AudioArtifact, InputModule, ModuleConfig, RunContext


class FileInputConfig(ModuleConfig):
    path: Path
    title: str | None = None
    language: str | None = None


class FileInput(InputModule):
    config_model = FileInputConfig

    def preflight(self, config: BaseModel) -> None:
        settings = FileInputConfig.model_validate(config)
        if not settings.path.is_file():
            raise ValueError(f"Audio file not found: {settings.path}")
        if settings.path.stat().st_size == 0:
            raise ValueError(f"Audio file is empty: {settings.path}")

    def run(self, context: RunContext, config: BaseModel) -> AudioArtifact:
        settings = FileInputConfig.model_validate(config)
        path = settings.path.resolve()
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return AudioArtifact(
            path=path,
            media_type=media_type,
            tracks={"primary": path},
            metadata={
                "input_module": "file",
                "title": settings.title or path.stem,
                "language_hint": settings.language,
            },
        )
