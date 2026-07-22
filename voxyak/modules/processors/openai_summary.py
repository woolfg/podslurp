"""Structured meeting summaries produced with the OpenAI Responses API."""

from __future__ import annotations

import importlib.util
import json
import os
from typing import Literal

from pydantic import BaseModel, Field

from ...sdk import (
    ModuleConfig,
    OutputArtifact,
    ProcessorModule,
    RunContext,
    TranscriptArtifact,
)
from ...transcript import TranscriptDocument, TranscriptSegment, load_transcript


class ActionItem(BaseModel):
    task: str
    owner: str | None = None
    due_date: str | None = None


class MeetingBrief(BaseModel):
    overview: str
    key_points: list[str]
    decisions: list[str]
    action_items: list[ActionItem]
    open_questions: list[str]


_DEFAULT_SUMMARY_PROMPT = (
    "Create a concise meeting brief covering the important points, decisions, "
    "action items, and unresolved questions."
)


class OpenAISummaryConfig(ModuleConfig):
    model: str = "gpt-5.6-terra"
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = (
        "low"
    )
    output_language: str = "auto"
    prompt: str = Field(default=_DEFAULT_SUMMARY_PROMPT, min_length=1)
    max_input_tokens: int = Field(default=200_000, ge=1_000)
    max_output_tokens: int = Field(default=8_000, ge=256)


def _require_openai() -> None:
    if importlib.util.find_spec("openai") is None:
        raise ValueError("Missing dependency 'openai'. Install with: uv sync")
    if importlib.util.find_spec("tiktoken") is None:
        raise ValueError("Missing dependency 'tiktoken'. Install with: uv sync")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError("OPENAI_API_KEY is required for the openai-summary processor.")


def _format_segment(segment: TranscriptSegment) -> str:
    speaker = f"[{segment.speaker}] " if segment.speaker else ""
    return f"[{segment.start:.2f}-{segment.end:.2f}] {speaker}{segment.text.strip()}"


def _encoding(model: str):
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return _CodepointEncoding()


class _CodepointEncoding:
    """Conservative offline fallback when tiktoken data is not cached."""

    @staticmethod
    def encode(value: str) -> list[int]:
        return [ord(character) for character in value]

    @staticmethod
    def decode(value: list[int]) -> str:
        return "".join(chr(character) for character in value)


def _chunk_transcript(
    document: TranscriptDocument,
    model: str,
    maximum_tokens: int,
) -> list[str]:
    encoding = _encoding(model)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for segment in document.segments:
        line = _format_segment(segment)
        line_tokens = len(encoding.encode(line))
        if current and current_tokens + line_tokens > maximum_tokens:
            chunks.append("\n".join(current))
            current = []
            current_tokens = 0
        if line_tokens > maximum_tokens:
            encoded = encoding.encode(line)
            for offset in range(0, len(encoded), maximum_tokens):
                if current:
                    chunks.append("\n".join(current))
                    current = []
                    current_tokens = 0
                chunks.append(encoding.decode(encoded[offset : offset + maximum_tokens]))
            continue
        current.append(line)
        current_tokens += line_tokens
    if current:
        chunks.append("\n".join(current))
    return chunks or [document.full_text]


def _extract_brief(response) -> MeetingBrief:
    parsed = getattr(response, "output_parsed", None)
    if parsed is not None:
        return MeetingBrief.model_validate(parsed)
    refusals: list[str] = []
    for output in getattr(response, "output", []):
        if getattr(output, "type", None) != "message":
            continue
        for content in getattr(output, "content", []):
            if getattr(content, "type", None) == "refusal":
                refusals.append(str(getattr(content, "refusal", "Request refused")))
            item = getattr(content, "parsed", None)
            if item is not None:
                return MeetingBrief.model_validate(item)
    if refusals:
        raise ValueError(f"OpenAI refused to summarize the transcript: {refusals[0]}")
    status = getattr(response, "status", "unknown")
    raise ValueError(f"OpenAI returned no structured meeting brief (status: {status}).")


def _instructions(settings: OpenAISummaryConfig, *, synthesis: bool) -> str:
    language = (
        "Use the same language as the supplied transcript."
        if settings.output_language == "auto"
        else f"Write the result in {settings.output_language}."
    )
    source_description = (
        "The input contains partial briefs from consecutive transcript chunks. "
        "Combine them without duplicating information."
        if synthesis
        else "The input is a timestamped transcript or transcript chunk."
    )
    return (
        f"{settings.prompt.strip()} "
        "Create an accurate meeting brief grounded only in the supplied input. "
        f"{source_description} {language} "
        "Include the overview, important points, explicit decisions, explicit action "
        "items, and unresolved questions. Do not invent owners, deadlines, decisions, "
        "or commitments; use null for an action item's unknown owner or due date. "
        "Return empty lists when a category has no evidence."
    )


def _request_brief(client, text: str, settings: OpenAISummaryConfig, *, synthesis: bool):
    response = client.responses.parse(
        model=settings.model,
        reasoning={"effort": settings.reasoning_effort},
        store=False,
        max_output_tokens=settings.max_output_tokens,
        input=[
            {
                "role": "developer",
                "content": _instructions(settings, synthesis=synthesis),
            },
            {"role": "user", "content": text},
        ],
        text_format=MeetingBrief,
    )
    return _extract_brief(response)


def _render_markdown(title: str, brief: MeetingBrief) -> str:
    lines = [f"# {title}", "", "## Overview", "", brief.overview]
    sections: list[tuple[str, list[str]]] = [
        ("Key points", brief.key_points),
        ("Decisions", brief.decisions),
        ("Open questions", brief.open_questions),
    ]
    for heading, values in sections:
        lines.extend(["", f"## {heading}", ""])
        lines.extend(f"- {value}" for value in values)
        if not values:
            lines.append("- None identified.")
    lines.extend(["", "## Action items", ""])
    if brief.action_items:
        for item in brief.action_items:
            qualifiers = [value for value in (item.owner, item.due_date) if value]
            suffix = f" ({' · '.join(qualifiers)})" if qualifiers else ""
            lines.append(f"- {item.task}{suffix}")
    else:
        lines.append("- None identified.")
    return "\n".join(lines) + "\n"


class OpenAISummaryProcessor(ProcessorModule):
    config_model = OpenAISummaryConfig

    def preflight(self, config: BaseModel) -> None:
        OpenAISummaryConfig.model_validate(config)
        _require_openai()

    def run(
        self,
        context: RunContext,
        transcript: TranscriptArtifact,
        prior_outputs: list[OutputArtifact],
        config: BaseModel,
    ) -> list[OutputArtifact]:
        from openai import OpenAI

        settings = OpenAISummaryConfig.model_validate(config)
        document = load_transcript(transcript.path)
        chunks = _chunk_transcript(
            document,
            settings.model,
            settings.max_input_tokens,
        )
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        if len(chunks) == 1:
            brief = _request_brief(client, chunks[0], settings, synthesis=False)
        else:
            partials = [
                _request_brief(client, chunk, settings, synthesis=False)
                for chunk in chunks
            ]
            brief = _request_brief(
                client,
                json.dumps(
                    [partial.model_dump(mode="json") for partial in partials],
                    ensure_ascii=False,
                ),
                settings,
                synthesis=True,
            )

        context.work_dir.mkdir(parents=True, exist_ok=True)
        json_path = context.work_dir / "summary.json"
        markdown_path = context.work_dir / "summary.md"
        payload = {
            "schema_version": 1,
            "model": settings.model,
            "source_transcript": str(transcript.path),
            "brief": brief.model_dump(mode="json"),
        }
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        title = str(document.source.title or "Meeting summary")
        markdown_path.write_text(
            _render_markdown(title, brief),
            encoding="utf-8",
        )
        return [
            OutputArtifact(
                path=json_path.resolve(),
                media_type="application/json",
                metadata={"processor": "openai-summary", "model": settings.model},
            ),
            OutputArtifact(
                path=markdown_path.resolve(),
                media_type="text/markdown",
                metadata={"processor": "openai-summary", "model": settings.model},
            ),
        ]
