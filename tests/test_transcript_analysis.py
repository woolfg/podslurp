from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxyak.analysis import analyze_transcript
from voxyak.transcript import (
    TranscriptDocument,
    TranscriptSegment,
    TranscriptSource,
    TranscriptionDetails,
    build_full_text,
    load_transcript,
    write_transcript,
)


def _document() -> TranscriptDocument:
    segments = [
        TranscriptSegment(start=0, end=2, text="Hello", speaker="A"),
        TranscriptSegment(start=2, end=3, text="Hi", speaker="B"),
        TranscriptSegment(start=3, end=4, text="Unknown"),
    ]
    return TranscriptDocument(
        source=TranscriptSource(
            module="file",
            title="Example",
            audio_path="audio.mp3",
            metadata={"title": "Example", "context": "Customer interview"},
        ),
        transcription=TranscriptionDetails(
            module="faster-whisper",
            model="small",
            language="en",
            language_probability=1,
            duration_seconds=4,
        ),
        segments=segments,
        full_text=build_full_text(segments),
    )


def test_transcript_round_trip_and_speaker_analysis(tmp_path: Path) -> None:
    json_path, text_path = write_transcript(_document(), tmp_path)
    loaded = load_transcript(json_path)
    analysis = analyze_transcript(loaded)
    assert text_path.is_file()
    assert "Context: Customer interview" in text_path.read_text(encoding="utf-8")
    assert analysis.total_seconds == 4
    assert analysis.unlabeled_segment_count == 1
    assert analysis.speakers[0].speaker == "A"
    assert analysis.speakers[0].percentage == 50


def test_legacy_shaped_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps({"metadata": {}, "segments": [], "full_text": ""}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Not a VoxYak transcript"):
        load_transcript(path)
