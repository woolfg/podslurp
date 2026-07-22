"""YAML configuration models and secret-safe serialization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModuleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    uses: str
    options: dict[str, Any] = Field(default_factory=dict, alias="with")


class ProcessorSpec(ModuleSpec):
    id: str


class PipelineDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: ModuleSpec
    transcription: ModuleSpec
    processing: list[ProcessorSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_processor_ids(self) -> "PipelineDefinition":
        identifiers = [processor.id for processor in self.processing]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Processor ids must be unique within a pipeline.")
        return self


class PipelineFile(PipelineDefinition):
    version: Literal[1]


class VoxYakConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipelines: dict[str, PipelineDefinition]


def default_config_path() -> Path:
    return Path(os.getenv("VOXYAK_CONFIG", "pipelines"))


def default_runs_dir() -> Path:
    return Path(os.getenv("VOXYAK_RUNS_DIR", "runs"))


def load_pipeline(path: Path) -> PipelineFile:
    if not path.is_file():
        raise ValueError(f"Pipeline file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be an object: {path}")
    return PipelineFile.model_validate(raw)


def load_config(path: Path) -> VoxYakConfig:
    """Load one pipeline file or every pipeline file in a directory."""
    if path.is_file():
        paths = [path]
    elif path.is_dir():
        paths = sorted((*path.glob("*.yaml"), *path.glob("*.yml")))
    else:
        raise ValueError(f"Pipeline path not found: {path}")

    pipelines = {
        pipeline_path.stem: load_pipeline(pipeline_path) for pipeline_path in paths
    }
    return VoxYakConfig(pipelines=pipelines)


_SECRET_MARKERS = ("api_key", "secret", "token", "password", "credential")


def redact_secrets(value: Any) -> Any:
    """Recursively redact likely credentials before snapshotting configuration."""
    if isinstance(value, dict):
        return {
            key: (
                "<redacted>"
                if any(marker in str(key).lower() for marker in _SECRET_MARKERS)
                else redact_secrets(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def config_snapshot(pipeline: PipelineDefinition) -> dict[str, Any]:
    raw = {
        "version": 1,
        **pipeline.model_dump(mode="json", by_alias=True, exclude={"version"}),
    }
    return redact_secrets(raw)
