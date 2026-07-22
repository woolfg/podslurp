"""Built-in and package-entry-point module discovery."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Iterable

from pydantic import BaseModel

from .sdk import (
    PLUGIN_API_VERSION,
    BaseModule,
    InputModule,
    ModuleKind,
    ProcessorModule,
    TranscriptionModule,
)


ENTRY_POINT_GROUPS: dict[ModuleKind, str] = {
    "input": "voxyak.inputs",
    "transcription": "voxyak.transcribers",
    "processor": "voxyak.processors",
}
MODULE_BASES: dict[ModuleKind, type[BaseModule]] = {
    "input": InputModule,
    "transcription": TranscriptionModule,
    "processor": ProcessorModule,
}


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class ModuleDescriptor:
    name: str
    kind: ModuleKind
    module_class: type[BaseModule]
    version: str
    source: str


class ModuleRegistry:
    def __init__(self) -> None:
        self._modules: dict[ModuleKind, dict[str, ModuleDescriptor]] = {
            kind: {} for kind in ENTRY_POINT_GROUPS
        }

    def register(
        self,
        name: str,
        kind: ModuleKind,
        module_class: type[BaseModule],
        *,
        source: str,
    ) -> None:
        if name in self._modules[kind]:
            existing = self._modules[kind][name]
            raise RegistryError(
                f"Duplicate {kind} module {name!r} from {source}; "
                f"already registered by {existing.source}."
            )
        expected_base = MODULE_BASES[kind]
        if not isinstance(module_class, type) or not issubclass(
            module_class, expected_base
        ):
            raise RegistryError(
                f"{source} entry {name!r} must subclass {expected_base.__name__}."
            )
        if module_class.api_version != PLUGIN_API_VERSION:
            raise RegistryError(
                f"{source} entry {name!r} uses plugin API "
                f"{module_class.api_version}; VoxYak requires {PLUGIN_API_VERSION}."
            )
        if module_class.kind != kind:
            raise RegistryError(
                f"{source} entry {name!r} declares kind {module_class.kind!r}, "
                f"expected {kind!r}."
            )
        config_model = getattr(module_class, "config_model", None)
        if not isinstance(config_model, type) or not issubclass(config_model, BaseModel):
            raise RegistryError(
                f"{source} entry {name!r} must expose a Pydantic config_model."
            )
        self._modules[kind][name] = ModuleDescriptor(
            name=name,
            kind=kind,
            module_class=module_class,
            version=module_class.module_version,
            source=source,
        )

    def discover_entry_points(self) -> None:
        available = entry_points()
        for kind, group in ENTRY_POINT_GROUPS.items():
            selected: Iterable = available.select(group=group)
            for entry_point in selected:
                self.register(
                    entry_point.name,
                    kind,
                    entry_point.load(),
                    source=f"entry point {group}",
                )

    def descriptor(self, kind: ModuleKind, name: str) -> ModuleDescriptor:
        try:
            return self._modules[kind][name]
        except KeyError as exc:
            available = ", ".join(sorted(self._modules[kind])) or "none"
            raise RegistryError(
                f"Unknown {kind} module {name!r}. Available: {available}."
            ) from exc

    def create(self, kind: ModuleKind, name: str) -> BaseModule:
        return self.descriptor(kind, name).module_class()

    def list(self) -> list[ModuleDescriptor]:
        return sorted(
            (
                descriptor
                for modules in self._modules.values()
                for descriptor in modules.values()
            ),
            key=lambda item: (item.kind, item.name),
        )


def default_registry(*, discover_external: bool = True) -> ModuleRegistry:
    from .modules.inputs.call_recorder import CallRecorderInput
    from .modules.inputs.file import FileInput
    from .modules.inputs.podcast import PodcastIndexInput
    from .modules.processors.openai_summary import OpenAISummaryProcessor
    from .modules.processors.speaker_time import SpeakerTimeProcessor
    from .modules.transcribers.faster_whisper import FasterWhisperTranscriber

    registry = ModuleRegistry()
    builtins: list[tuple[str, ModuleKind, type[BaseModule]]] = [
        ("file", "input", FileInput),
        ("podcast-index", "input", PodcastIndexInput),
        ("call-recorder", "input", CallRecorderInput),
        ("faster-whisper", "transcription", FasterWhisperTranscriber),
        ("speaker-time", "processor", SpeakerTimeProcessor),
        ("openai-summary", "processor", OpenAISummaryProcessor),
    ]
    for name, kind, module_class in builtins:
        registry.register(name, kind, module_class, source="VoxYak built-in")
    if discover_external:
        registry.discover_entry_points()
    return registry
