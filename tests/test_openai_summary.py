from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import openai
import pytest
from rich.console import Console

from voxyak.cli import main
from voxyak.modules.processors import openai_summary
from voxyak.modules.processors.openai_summary import (
    ActionItem,
    MeetingBrief,
    OpenAISummaryConfig,
    OpenAISummaryProcessor,
)
from voxyak.sdk import RunContext, TranscriptArtifact
from voxyak.transcript import (
    TranscriptDocument,
    TranscriptSegment,
    TranscriptSource,
    TranscriptionDetails,
    write_transcript,
)


def _artifact(tmp_path: Path) -> TranscriptArtifact:
    document = TranscriptDocument(
        source=TranscriptSource(
            module="file",
            title="Team call",
            audio_path="call.mp3",
            metadata={
                "participants": "Sam and Alex",
                "participant_count": "two",
                "context": "Weekly delivery review",
                "collection_title": "Delivery Talks",
                "creator": "Example Host",
                "description": "A weekly review of delivery risks.",
            },
        ),
        transcription=TranscriptionDetails(
            module="faster-whisper",
            model="small",
            language="en",
            language_probability=1,
            duration_seconds=2,
        ),
        segments=[TranscriptSegment(start=0, end=2, text="Sam will send the report.")],
        full_text="Sam will send the report.",
    )
    json_path, text_path = write_transcript(document, tmp_path / "transcription")
    return TranscriptArtifact(
        path=json_path,
        text_path=text_path,
        language="en",
        duration_seconds=2,
        metadata=document.source.metadata.model_copy(deep=True),
    )


def test_summary_uses_structured_responses_without_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    brief = MeetingBrief(
        overview="The report was assigned.",
        key_points=["A report is needed."],
        decisions=[],
        action_items=[ActionItem(task="Send the report", owner="Sam")],
        open_questions=[],
    )

    class Responses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_parsed=brief)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(responses=Responses()),
    )
    context = RunContext(
        run_id="run",
        pipeline_name="summary",
        run_dir=tmp_path,
        stage_id="processing.summary",
        work_dir=tmp_path / "processing" / "summary",
        console=Console(file=StringIO(), force_terminal=False),
    )
    outputs = OpenAISummaryProcessor().run(
        context,
        _artifact(tmp_path),
        [],
        OpenAISummaryConfig(prompt="Focus especially on delivery risks."),
    )
    assert len(calls) == 1
    assert calls[0]["store"] is False
    assert calls[0]["reasoning"] == {"effort": "low"}
    assert calls[0]["model"] == "gpt-5.6-terra"
    assert "Focus especially on delivery risks." in calls[0]["input"][0]["content"]
    assert "Weekly delivery review" in calls[0]["input"][1]["content"]
    assert "Delivery Talks" in calls[0]["input"][1]["content"]
    payload = json.loads(outputs[0].path.read_text(encoding="utf-8"))
    assert payload["brief"]["action_items"][0]["owner"] == "Sam"
    assert outputs[0].metadata.context == "Weekly delivery review"
    assert outputs[0].metadata.producer_module == "openai-summary"
    assert "## Action items" in outputs[1].path.read_text(encoding="utf-8")


def test_refusal_does_not_write_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refusal = SimpleNamespace(type="refusal", refusal="cannot summarize", parsed=None)
    message = SimpleNamespace(type="message", content=[refusal])

    class Responses:
        def parse(self, **kwargs):
            return SimpleNamespace(output_parsed=None, output=[message], status="completed")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(responses=Responses()),
    )
    context = RunContext(
        run_id="run",
        pipeline_name="summary",
        run_dir=tmp_path,
        stage_id="processing.summary",
        work_dir=tmp_path / "processing" / "summary",
        console=Console(file=StringIO(), force_terminal=False),
    )
    with pytest.raises(ValueError, match="refused"):
        OpenAISummaryProcessor().run(
            context,
            _artifact(tmp_path),
            [],
            OpenAISummaryConfig(),
        )
    assert not (context.work_dir / "summary.json").exists()


def test_long_transcript_uses_map_reduce(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    brief = MeetingBrief(
        overview="Combined",
        key_points=[],
        decisions=[],
        action_items=[],
        open_questions=[],
    )

    class Responses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_parsed=brief)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(responses=Responses()),
    )
    monkeypatch.setattr(
        openai_summary,
        "_chunk_transcript",
        lambda document, model, maximum_tokens: ["chunk one", "chunk two"],
    )
    context = RunContext(
        run_id="run",
        pipeline_name="summary",
        run_dir=tmp_path,
        stage_id="processing.summary",
        work_dir=tmp_path / "processing" / "summary",
        console=Console(file=StringIO(), force_terminal=False),
    )
    OpenAISummaryProcessor().run(
        context,
        _artifact(tmp_path),
        [],
        OpenAISummaryConfig(),
    )
    assert len(calls) == 3
    assert "partial briefs" in calls[-1]["input"][0]["content"]


def test_summarize_command_reuses_pipeline_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    brief = MeetingBrief(
        overview="Standalone summary",
        key_points=[],
        decisions=[],
        action_items=[],
        open_questions=[],
    )

    class Responses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_parsed=brief)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(responses=Responses()),
    )
    output_dir = tmp_path / "summary"

    result = main(
        [
            "--config",
            "pipelines",
            "summarize",
            str(_artifact(tmp_path).path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "summary.md").is_file()
    assert "Prioritize the main topics" in calls[0]["input"][0]["content"]
