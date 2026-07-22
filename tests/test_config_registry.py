from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from voxyak.config import PipelineFile, load_config, redact_secrets
from voxyak.registry import ModuleRegistry, RegistryError
from voxyak.sdk import AudioArtifact, InputModule, ModuleConfig, RunContext


class ExampleConfig(ModuleConfig):
    value: str = "ok"


class ExampleInput(InputModule):
    config_model = ExampleConfig

    def run(self, context: RunContext, config: ExampleConfig) -> AudioArtifact:
        raise NotImplementedError


def test_shipped_configuration_parses() -> None:
    config = load_config(Path("pipelines"))
    assert set(config.pipelines) == {"call-summary", "podcast"}
    assert config.pipelines["call-summary"].processing[0].uses == "openai-summary"


def test_duplicate_processor_ids_are_rejected() -> None:
    raw = {
        "version": 1,
        "input": {"uses": "file", "with": {"path": "audio.mp3"}},
        "transcription": {"uses": "faster-whisper"},
        "processing": [
            {"id": "same", "uses": "speaker-time"},
            {"id": "same", "uses": "speaker-time"},
        ],
    }
    with pytest.raises(ValidationError, match="Processor ids must be unique"):
        PipelineFile.model_validate(raw)


def test_module_options_are_strict() -> None:
    registry = ModuleRegistry()
    registry.register("example", "input", ExampleInput, source="test")
    descriptor = registry.descriptor("input", "example")
    with pytest.raises(ValidationError):
        descriptor.module_class.config_model.model_validate({"unexpected": True})


def test_registry_rejects_duplicate_names() -> None:
    registry = ModuleRegistry()
    registry.register("example", "input", ExampleInput, source="first")
    with pytest.raises(RegistryError, match="Duplicate input module"):
        registry.register("example", "input", ExampleInput, source="second")


def test_registry_rejects_wrong_plugin_api() -> None:
    class IncompatibleInput(ExampleInput):
        api_version = 99

    registry = ModuleRegistry()
    with pytest.raises(RegistryError, match="plugin API 99"):
        registry.register("bad", "input", IncompatibleInput, source="test")


def test_external_entry_point_is_discovered(monkeypatch: pytest.MonkeyPatch) -> None:
    class EntryPoint:
        name = "external"

        @staticmethod
        def load():
            return ExampleInput

    class EntryPoints:
        @staticmethod
        def select(*, group: str):
            return [EntryPoint()] if group == "voxyak.inputs" else []

    monkeypatch.setattr("voxyak.registry.entry_points", lambda: EntryPoints())
    registry = ModuleRegistry()
    registry.discover_entry_points()
    descriptor = registry.descriptor("input", "external")
    assert descriptor.source == "entry point voxyak.inputs"


def test_secret_redaction_is_recursive() -> None:
    value = {
        "api_key": "secret",
        "nested": {"access_token": "token", "normal": "kept"},
        "items": [{"password": "pw"}],
    }
    assert redact_secrets(value) == {
        "api_key": "<redacted>",
        "nested": {"access_token": "<redacted>", "normal": "kept"},
        "items": [{"password": "<redacted>"}],
    }


def test_wrong_pipeline_version_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineFile.model_validate(
            {
                "version": 2,
                "input": {"uses": "file"},
                "transcription": {"uses": "faster-whisper"},
            }
        )
