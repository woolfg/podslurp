"""Speaker-time processor."""

from __future__ import annotations

import json

from pydantic import BaseModel

from ...analysis import analyze_transcript, format_duration
from ...sdk import (
    ModuleConfig,
    OutputArtifact,
    ProcessorModule,
    RunContext,
    TranscriptArtifact,
)
from ...transcript import load_transcript


class SpeakerTimeConfig(ModuleConfig):
    pass


class SpeakerTimeProcessor(ProcessorModule):
    config_model = SpeakerTimeConfig

    def run(
        self,
        context: RunContext,
        transcript: TranscriptArtifact,
        prior_outputs: list[OutputArtifact],
        config: BaseModel,
    ) -> list[OutputArtifact]:
        SpeakerTimeConfig.model_validate(config)
        document = load_transcript(transcript.path)
        analysis = analyze_transcript(document)
        context.work_dir.mkdir(parents=True, exist_ok=True)
        json_path = context.work_dir / "speaker-time.json"
        markdown_path = context.work_dir / "speaker-time.md"
        payload = {
            "schema_version": 1,
            "total_seconds": analysis.total_seconds,
            "segment_count": analysis.segment_count,
            "unlabeled_segment_count": analysis.unlabeled_segment_count,
            "duration_seconds": analysis.duration_seconds,
            "speakers": [
                {
                    "speaker": item.speaker,
                    "seconds": item.seconds,
                    "segments": item.segments,
                    "percentage": item.percentage,
                }
                for item in analysis.speakers
            ],
        }
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        rows = [
            "# Speaker time",
            "",
            "| Speaker | Segments | Time | Share |",
            "| --- | ---: | ---: | ---: |",
        ]
        rows.extend(
            f"| {item.speaker} | {item.segments} | "
            f"{format_duration(item.seconds)} | {item.percentage:.1f}% |"
            for item in analysis.speakers
        )
        rows.extend(
            [
                "",
                f"Total speaking time: {format_duration(analysis.total_seconds)}.",
            ]
        )
        markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return [
            OutputArtifact(
                path=json_path.resolve(),
                media_type="application/json",
                metadata={"processor": "speaker-time"},
            ),
            OutputArtifact(
                path=markdown_path.resolve(),
                media_type="text/markdown",
                metadata={"processor": "speaker-time"},
            ),
        ]
