"""Sequential VoxYak pipeline execution, manifests, and resume support."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .config import PipelineDefinition, VoxYakConfig, config_snapshot, load_pipeline
from .registry import ModuleRegistry
from .sdk import (
    Artifact,
    AudioArtifact,
    BaseModule,
    ModuleKind,
    OutputArtifact,
    RunContext,
    TranscriptArtifact,
    artifact_from_dict,
)


StageStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]
RunStatus = Literal["running", "succeeded", "failed", "cancelled"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: ModuleKind
    module: str
    module_version: str
    status: StageStatus = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str
    pipeline_name: str
    status: RunStatus = "running"
    created_at: str
    updated_at: str
    stages: list[StageRecord]


@dataclass(frozen=True)
class ResolvedModule:
    id: str
    kind: ModuleKind
    name: str
    module: BaseModule
    config: BaseModel


@dataclass(frozen=True)
class ResolvedPipeline:
    input: ResolvedModule
    transcription: ResolvedModule
    processing: list[ResolvedModule]

    def all_modules(self) -> list[ResolvedModule]:
        return [self.input, self.transcription, *self.processing]


class PipelineRunError(RuntimeError):
    def __init__(self, message: str, *, run_id: str | None = None) -> None:
        super().__init__(message)
        self.run_id = run_id


class PipelineRunner:
    def __init__(self, registry: ModuleRegistry, runs_dir: Path, console) -> None:
        self.registry = registry
        self.runs_dir = runs_dir
        self.console = console

    def resolve(self, pipeline: PipelineDefinition) -> ResolvedPipeline:
        def resolve_module(
            identifier: str,
            kind: ModuleKind,
            name: str,
            options: dict[str, Any],
        ) -> ResolvedModule:
            descriptor = self.registry.descriptor(kind, name)
            module = descriptor.module_class()
            validated = descriptor.module_class.config_model.model_validate(options)
            return ResolvedModule(
                id=identifier,
                kind=kind,
                name=name,
                module=module,
                config=validated,
            )

        input_module = resolve_module(
            "input",
            "input",
            pipeline.input.uses,
            pipeline.input.options,
        )
        transcription = resolve_module(
            "transcription",
            "transcription",
            pipeline.transcription.uses,
            pipeline.transcription.options,
        )
        processors = [
            resolve_module(
                f"processing.{processor.id}",
                "processor",
                processor.uses,
                processor.options,
            )
            for processor in pipeline.processing
        ]
        return ResolvedPipeline(input_module, transcription, processors)

    def validate_configuration(self, config: VoxYakConfig) -> None:
        for pipeline in config.pipelines.values():
            self.resolve(pipeline)

    def run_named(self, config: VoxYakConfig, name: str) -> RunManifest:
        try:
            pipeline = config.pipelines[name]
        except KeyError as exc:
            available = ", ".join(sorted(config.pipelines)) or "none"
            raise ValueError(
                f"Unknown pipeline {name!r}. Available: {available}."
            ) from exc
        return self.run_pipeline(name, pipeline)

    def run_pipeline(
        self,
        name: str,
        pipeline: PipelineDefinition,
    ) -> RunManifest:
        resolved = self.resolve(pipeline)
        self._preflight(resolved.all_modules())
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{uuid.uuid4().hex[:8]}"
        run_dir = (self.runs_dir / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        snapshot = config_snapshot(pipeline)
        (run_dir / "pipeline.yaml").write_text(
            yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        manifest = RunManifest(
            run_id=run_id,
            pipeline_name=name,
            created_at=_now(),
            updated_at=_now(),
            stages=[
                StageRecord(
                    id=item.id,
                    kind=item.kind,
                    module=item.name,
                    module_version=item.module.module_version,
                )
                for item in resolved.all_modules()
            ],
        )
        self._save_manifest(run_dir, manifest)
        return self._execute(run_dir, manifest, resolved)

    def resume(self, run_reference: str | Path) -> RunManifest:
        supplied = Path(run_reference)
        run_dir = supplied if supplied.is_dir() else self.runs_dir / supplied
        run_dir = run_dir.resolve()
        manifest = self._load_manifest(run_dir)
        pipeline = load_pipeline(run_dir / "pipeline.yaml")
        resolved = self.resolve(pipeline)
        records = {record.id: record for record in manifest.stages}
        expected_ids = [item.id for item in resolved.all_modules()]
        if list(records) != expected_ids:
            raise PipelineRunError(
                "Saved manifest stages do not match the pipeline snapshot.",
                run_id=manifest.run_id,
            )
        for item in resolved.all_modules():
            record = records[item.id]
            if record.module != item.name or record.module_version != item.module.module_version:
                raise PipelineRunError(
                    f"Module identity changed for saved stage {item.id!r}; "
                    "start a new run instead of resuming.",
                    run_id=manifest.run_id,
                )
            if record.status == "succeeded":
                for raw_artifact in record.artifacts:
                    self._validate_artifact_file(artifact_from_dict(raw_artifact))
        pending = [
            item
            for item in resolved.all_modules()
            if records[item.id].status != "succeeded"
        ]
        self._preflight(pending)
        manifest.status = "running"
        manifest.updated_at = _now()
        self._save_manifest(run_dir, manifest)
        return self._execute(run_dir, manifest, resolved)

    @staticmethod
    def _preflight(modules: list[ResolvedModule]) -> None:
        for item in modules:
            item.module.preflight(item.config)

    def _execute(
        self,
        run_dir: Path,
        manifest: RunManifest,
        resolved: ResolvedPipeline,
    ) -> RunManifest:
        records = {record.id: record for record in manifest.stages}
        audio: AudioArtifact
        transcript: TranscriptArtifact
        prior_outputs: list[OutputArtifact] = []

        try:
            input_record = records[resolved.input.id]
            if input_record.status == "succeeded":
                audio = self._single_artifact(input_record, AudioArtifact)
            else:
                artifacts = self._run_stage(
                    run_dir,
                    manifest,
                    resolved.input,
                    lambda context: [
                        resolved.input.module.run(context, resolved.input.config)
                    ],
                )
                audio = self._expect_artifact(artifacts[0], AudioArtifact)

            transcription_record = records[resolved.transcription.id]
            if transcription_record.status == "succeeded":
                transcript = self._single_artifact(
                    transcription_record, TranscriptArtifact
                )
            else:
                artifacts = self._run_stage(
                    run_dir,
                    manifest,
                    resolved.transcription,
                    lambda context: [
                        resolved.transcription.module.run(
                            context,
                            audio,
                            resolved.transcription.config,
                        )
                    ],
                )
                transcript = self._expect_artifact(
                    artifacts[0], TranscriptArtifact
                )

            for processor in resolved.processing:
                record = records[processor.id]
                if record.status == "succeeded":
                    prior_outputs.extend(
                        self._expect_artifact(
                            artifact_from_dict(raw), OutputArtifact
                        )
                        for raw in record.artifacts
                    )
                    continue
                artifacts = self._run_stage(
                    run_dir,
                    manifest,
                    processor,
                    lambda context, item=processor: item.module.run(
                        context,
                        transcript,
                        list(prior_outputs),
                        item.config,
                    ),
                )
                prior_outputs.extend(
                    self._expect_artifact(artifact, OutputArtifact)
                    for artifact in artifacts
                )
        except PipelineRunError:
            raise
        except KeyboardInterrupt as exc:
            manifest.status = "cancelled"
            manifest.updated_at = _now()
            self._save_manifest(run_dir, manifest)
            raise PipelineRunError(
                "Pipeline cancelled.", run_id=manifest.run_id
            ) from exc

        manifest.status = "succeeded"
        manifest.updated_at = _now()
        self._save_manifest(run_dir, manifest)
        self.console.print(
            f"[bold green]Pipeline complete:[/bold green] {manifest.run_id}"
        )
        self.console.print(f"[dim]{run_dir}[/dim]")
        return manifest

    def _run_stage(
        self,
        run_dir: Path,
        manifest: RunManifest,
        resolved: ResolvedModule,
        callback: Callable[[RunContext], list[Artifact]],
    ) -> list[Artifact]:
        record = next(item for item in manifest.stages if item.id == resolved.id)
        record.status = "running"
        record.started_at = _now()
        record.finished_at = None
        record.error = None
        record.artifacts = []
        manifest.status = "running"
        manifest.updated_at = _now()
        self._save_manifest(run_dir, manifest)
        work_dir = self._work_dir(run_dir, resolved.id)
        work_dir.mkdir(parents=True, exist_ok=True)
        context = RunContext(
            run_id=manifest.run_id,
            pipeline_name=manifest.pipeline_name,
            run_dir=run_dir,
            stage_id=resolved.id,
            work_dir=work_dir,
            console=self.console,
        )
        self.console.print(
            f"\n[bold cyan]{resolved.id}[/bold cyan] · {resolved.name}"
        )
        try:
            artifacts = callback(context)
            if not isinstance(artifacts, list) or not artifacts:
                raise ValueError("Module returned no artifacts.")
            expected_type: type[Artifact]
            if resolved.kind == "input":
                expected_type = AudioArtifact
            elif resolved.kind == "transcription":
                expected_type = TranscriptArtifact
            else:
                expected_type = OutputArtifact
            if resolved.kind != "processor" and len(artifacts) != 1:
                raise ValueError(
                    f"{resolved.kind} modules must return exactly one artifact."
                )
            for artifact in artifacts:
                if not isinstance(artifact, expected_type):
                    raise ValueError(
                        f"Module returned {type(artifact).__name__}; "
                        f"expected {expected_type.__name__}."
                    )
                self._validate_artifact_file(artifact)
        except KeyboardInterrupt:
            record.status = "cancelled"
            record.finished_at = _now()
            record.error = "Cancelled by user."
            manifest.status = "cancelled"
            manifest.updated_at = _now()
            self._save_manifest(run_dir, manifest)
            raise
        except Exception as exc:
            record.status = "failed"
            record.finished_at = _now()
            record.error = f"{type(exc).__name__}: {exc}"
            manifest.status = "failed"
            manifest.updated_at = _now()
            self._save_manifest(run_dir, manifest)
            raise PipelineRunError(
                f"Stage {resolved.id!r} failed: {exc}",
                run_id=manifest.run_id,
            ) from exc
        record.artifacts = [
            artifact.model_dump(mode="json") for artifact in artifacts
        ]
        record.status = "succeeded"
        record.finished_at = _now()
        manifest.updated_at = _now()
        self._save_manifest(run_dir, manifest)
        return artifacts

    @staticmethod
    def _work_dir(run_dir: Path, stage_id: str) -> Path:
        if stage_id.startswith("processing."):
            return run_dir / "processing" / stage_id.split(".", 1)[1]
        return run_dir / stage_id

    @staticmethod
    def _expect_artifact(artifact: Artifact, expected: type):
        if not isinstance(artifact, expected):
            raise PipelineRunError(
                f"Module returned {type(artifact).__name__}; "
                f"expected {expected.__name__}."
            )
        return artifact

    @staticmethod
    def _validate_artifact_file(artifact: Artifact) -> None:
        if not artifact.path.is_file():
            raise PipelineRunError(f"Artifact file does not exist: {artifact.path}")
        if isinstance(artifact, AudioArtifact):
            for label, track in artifact.tracks.items():
                if not track.is_file():
                    raise PipelineRunError(
                        f"Audio track {label!r} does not exist: {track}"
                    )
        if isinstance(artifact, TranscriptArtifact) and not artifact.text_path.is_file():
            raise PipelineRunError(
                f"Transcript text file does not exist: {artifact.text_path}"
            )

    def _single_artifact(self, record: StageRecord, expected: type):
        if len(record.artifacts) != 1:
            raise PipelineRunError(
                f"Completed stage {record.id!r} has an invalid artifact count."
            )
        return self._expect_artifact(
            artifact_from_dict(record.artifacts[0]), expected
        )

    @staticmethod
    def _manifest_path(run_dir: Path) -> Path:
        return run_dir / "manifest.json"

    def _save_manifest(self, run_dir: Path, manifest: RunManifest) -> None:
        path = self._manifest_path(run_dir)
        temporary = run_dir / ".manifest.json.tmp"
        temporary.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)

    def _load_manifest(self, run_dir: Path) -> RunManifest:
        path = self._manifest_path(run_dir)
        if not path.is_file():
            raise PipelineRunError(f"Run manifest not found: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PipelineRunError(f"Invalid run manifest: {exc}") from exc
        return RunManifest.model_validate(raw)
