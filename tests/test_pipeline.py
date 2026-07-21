from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from voxyak.config import ModuleSpec, PipelineDefinition, ProcessorSpec
from voxyak.pipeline import PipelineRunError, PipelineRunner
from voxyak.registry import ModuleRegistry
from voxyak.sdk import (
    AudioArtifact,
    InputModule,
    ModuleConfig,
    OutputArtifact,
    ProcessorModule,
    RunContext,
    TranscriptArtifact,
    TranscriptionModule,
)
from voxyak.transcript import (
    TranscriptDocument,
    TranscriptSegment,
    TranscriptSource,
    TranscriptionDetails,
    write_transcript,
)


class EmptyConfig(ModuleConfig):
    pass


class FakeInput(InputModule):
    config_model = EmptyConfig
    calls = 0

    def run(self, context: RunContext, config: EmptyConfig) -> AudioArtifact:
        type(self).calls += 1
        context.work_dir.mkdir(parents=True, exist_ok=True)
        path = context.work_dir / "audio.mp3"
        path.write_bytes(b"audio")
        return AudioArtifact(
            path=path,
            tracks={"primary": path},
            metadata={"input_module": "fake", "title": "Test"},
        )


class FakeTranscriber(TranscriptionModule):
    config_model = EmptyConfig
    calls = 0

    def run(
        self,
        context: RunContext,
        audio: AudioArtifact,
        config: EmptyConfig,
    ) -> TranscriptArtifact:
        type(self).calls += 1
        document = TranscriptDocument(
            source=TranscriptSource(
                module="fake", title="Test", audio_path=str(audio.path)
            ),
            transcription=TranscriptionDetails(
                module="fake-transcriber",
                model="test",
                language="en",
                language_probability=1,
                duration_seconds=1,
            ),
            segments=[TranscriptSegment(start=0, end=1, text="hello")],
            full_text="hello",
        )
        json_path, text_path = write_transcript(document, context.work_dir)
        return TranscriptArtifact(
            path=json_path,
            text_path=text_path,
            language="en",
            duration_seconds=1,
        )


class FlakyProcessor(ProcessorModule):
    config_model = EmptyConfig
    calls = 0
    should_fail = True

    def run(
        self,
        context: RunContext,
        transcript: TranscriptArtifact,
        prior_outputs: list[OutputArtifact],
        config: EmptyConfig,
    ) -> list[OutputArtifact]:
        type(self).calls += 1
        if type(self).should_fail:
            raise RuntimeError("temporary failure")
        context.work_dir.mkdir(parents=True, exist_ok=True)
        path = context.work_dir / "result.txt"
        path.write_text("done", encoding="utf-8")
        return [OutputArtifact(path=path, media_type="text/plain")]


@pytest.fixture
def runner(tmp_path: Path) -> PipelineRunner:
    FakeInput.calls = 0
    FakeTranscriber.calls = 0
    FlakyProcessor.calls = 0
    FlakyProcessor.should_fail = True
    registry = ModuleRegistry()
    registry.register("fake-input", "input", FakeInput, source="test")
    registry.register(
        "fake-transcriber", "transcription", FakeTranscriber, source="test"
    )
    registry.register("flaky", "processor", FlakyProcessor, source="test")
    return PipelineRunner(
        registry,
        tmp_path / "runs",
        Console(file=StringIO(), force_terminal=False),
    )


def _pipeline() -> PipelineDefinition:
    return PipelineDefinition(
        input=ModuleSpec(uses="fake-input"),
        transcription=ModuleSpec(uses="fake-transcriber"),
        processing=[ProcessorSpec(id="result", uses="flaky")],
    )


def test_failed_processor_can_resume_without_repeating_prior_stages(
    runner: PipelineRunner,
) -> None:
    with pytest.raises(PipelineRunError, match="temporary failure") as failure:
        runner.run_pipeline("test", _pipeline())
    run_id = failure.value.run_id
    assert run_id
    assert FakeInput.calls == 1
    assert FakeTranscriber.calls == 1
    assert FlakyProcessor.calls == 1

    FlakyProcessor.should_fail = False
    manifest = runner.resume(run_id)
    assert manifest.status == "succeeded"
    assert [stage.status for stage in manifest.stages] == [
        "succeeded",
        "succeeded",
        "succeeded",
    ]
    assert FakeInput.calls == 1
    assert FakeTranscriber.calls == 1
    assert FlakyProcessor.calls == 2


def test_manifest_is_written_after_failure(runner: PipelineRunner) -> None:
    with pytest.raises(PipelineRunError) as failure:
        runner.run_pipeline("test", _pipeline())
    run_dir = runner.runs_dir / str(failure.value.run_id)
    manifest = runner._load_manifest(run_dir)
    assert manifest.status == "failed"
    assert manifest.stages[-1].status == "failed"
    assert "temporary failure" in str(manifest.stages[-1].error)
    assert (run_dir / "pipeline.yaml").is_file()
