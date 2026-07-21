from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from pydantic import ValidationError

from voxyak.modules.inputs import call_recorder
from voxyak.modules.inputs.call_recorder import (
    CallRecorderConfig,
    CallRecorderInput,
    _match_source,
)
from voxyak.sdk import RunContext


def test_source_matching_rejects_missing_and_ambiguous_sources() -> None:
    sources = ["mic-one", "mic-two", "speaker.monitor"]
    with pytest.raises(ValueError, match="No microphone source"):
        _match_source(sources, "absent", "microphone")
    with pytest.raises(ValueError, match="Multiple microphone sources"):
        _match_source(sources, "mic", "microphone")
    assert _match_source(sources, "speaker", "system") == "speaker.monitor"


def test_invalid_source_regex_is_rejected_during_validation() -> None:
    with pytest.raises(ValidationError, match="Invalid source regex"):
        CallRecorderConfig(microphone_pattern="[", system_pattern="monitor")


def test_recorder_keeps_tracks_and_creates_merged_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command, **kwargs):
        if command[:3] == ["pactl", "list", "short"]:
            return SimpleNamespace(
                stdout="1\tbluez_output.headset.monitor\n2\tQ2U_microphone\n"
            )
        Path(command[-1]).write_bytes(b"merged")
        return SimpleNamespace(returncode=0)

    class FakeProcess:
        returncode = None

        def __init__(self, command, **kwargs):
            self.command = command
            for value in command:
                if str(value).endswith(("-microphone.mp3", "-system.mp3")):
                    Path(value).write_bytes(b"track")

        def poll(self):
            return self.returncode

        def send_signal(self, sent_signal):
            self.returncode = 255

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(call_recorder.subprocess, "run", fake_run)
    monkeypatch.setattr(call_recorder.subprocess, "Popen", FakeProcess)
    context = RunContext(
        run_id="run",
        pipeline_name="call",
        run_dir=tmp_path,
        stage_id="input",
        work_dir=tmp_path / "input",
        console=Console(file=StringIO(), force_terminal=False),
    )
    artifact = CallRecorderInput().run(
        context,
        CallRecorderConfig(
            microphone_pattern="Q2U",
            system_pattern="bluez_output.*monitor",
        ),
    )
    assert artifact.path.read_bytes() == b"merged"
    assert set(artifact.tracks) == {"primary", "microphone", "system"}
