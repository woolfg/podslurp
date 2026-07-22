"""Stable contracts for VoxYak pipeline modules and artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .metadata import PipelineMetadata


PLUGIN_API_VERSION = 1
ModuleKind = Literal["input", "transcription", "processor"]


class AudioArtifact(BaseModel):
    """Audio produced or selected by an input module."""

    model_config = ConfigDict(extra="forbid")

    artifact_type: Literal["audio"] = "audio"
    path: Path
    media_type: str = "audio/mpeg"
    tracks: dict[str, Path] = Field(default_factory=dict)
    metadata: PipelineMetadata = Field(default_factory=PipelineMetadata)


class TranscriptArtifact(BaseModel):
    """Paths and essential metadata for a VoxYak transcript."""

    model_config = ConfigDict(extra="forbid")

    artifact_type: Literal["transcript"] = "transcript"
    path: Path
    text_path: Path
    language: str
    duration_seconds: float
    metadata: PipelineMetadata = Field(default_factory=PipelineMetadata)


class OutputArtifact(BaseModel):
    """A file emitted by a processing module."""

    model_config = ConfigDict(extra="forbid")

    artifact_type: Literal["output"] = "output"
    path: Path
    media_type: str
    metadata: PipelineMetadata = Field(default_factory=PipelineMetadata)


Artifact = AudioArtifact | TranscriptArtifact | OutputArtifact
_ARTIFACT_ADAPTER = TypeAdapter(Artifact)


def artifact_from_dict(value: dict[str, Any]) -> Artifact:
    """Rebuild an artifact stored in a run manifest."""
    return _ARTIFACT_ADAPTER.validate_python(value)


@dataclass(frozen=True)
class RunContext:
    """Filesystem and presentation context passed to a module invocation."""

    run_id: str
    pipeline_name: str
    run_dir: Path
    stage_id: str
    work_dir: Path
    console: Any


class ModuleConfig(BaseModel):
    """Base for strict module configuration schemas."""

    model_config = ConfigDict(extra="forbid")


class BaseModule(ABC):
    """Common metadata and preflight contract for all module kinds."""

    api_version: ClassVar[int] = PLUGIN_API_VERSION
    module_version: ClassVar[str] = "1"
    kind: ClassVar[ModuleKind]
    config_model: ClassVar[type[BaseModel]]

    def preflight(self, config: BaseModel) -> None:
        """Check runtime prerequisites before any pipeline stage starts."""


class InputModule(BaseModule):
    kind: ClassVar[ModuleKind] = "input"

    @abstractmethod
    def run(self, context: RunContext, config: BaseModel) -> AudioArtifact:
        """Acquire or select audio for the pipeline."""


class TranscriptionModule(BaseModule):
    kind: ClassVar[ModuleKind] = "transcription"

    @abstractmethod
    def run(
        self,
        context: RunContext,
        audio: AudioArtifact,
        config: BaseModel,
    ) -> TranscriptArtifact:
        """Turn an audio artifact into a transcript artifact."""


class ProcessorModule(BaseModule):
    kind: ClassVar[ModuleKind] = "processor"

    @abstractmethod
    def run(
        self,
        context: RunContext,
        transcript: TranscriptArtifact,
        prior_outputs: list[OutputArtifact],
        config: BaseModel,
    ) -> list[OutputArtifact]:
        """Process a transcript and return generated files."""
