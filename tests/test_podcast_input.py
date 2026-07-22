from __future__ import annotations

from pathlib import Path

import pytest

from voxyak.modules.inputs import podcast
from voxyak.modules.inputs.podcast import (
    PodcastIndexConfig,
    PodcastIndexInput,
    _tag_audio_file,
)
from voxyak.metadata import PipelineMetadata
from voxyak.sdk import RunContext


class FakeConsole:
    def __init__(self, answers: list[str]) -> None:
        self.answers = iter(answers)

    def input(self, prompt: str) -> str:
        return next(self.answers)

    def print(self, *args, **kwargs) -> None:
        pass


def test_audio_metadata_is_embedded_without_reencoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "episode.mp3"
    audio_path.write_bytes(b"original")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"tagged")

    monkeypatch.setattr(podcast.subprocess, "run", fake_run)
    _tag_audio_file(
        audio_path,
        PipelineMetadata(title="Episode One", collection_title="Example Show"),
    )

    assert audio_path.read_bytes() == b"tagged"
    assert "copy" in commands[0]
    assert "title=Episode One" in commands[0]
    assert "genre=Podcast" in commands[0]


def test_podcast_search_selection_and_download_are_adapted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_request(endpoint, params, settings):
        if endpoint == "/search/byterm":
            return {
                "feeds": [
                    {
                        "id": 42,
                        "title": "Example Show",
                        "author": "Host",
                        "language": "de-DE",
                        "url": "https://example.test/feed.xml",
                    }
                ]
            }
        return {
            "items": [
                {
                    "title": "Episode One",
                    "datePublished": 1_700_000_000,
                    "datePublishedPretty": "2023-11-14",
                    "enclosureUrl": "https://example.test/episode.mp3",
                    "enclosureType": "audio/mpeg",
                    "feedLanguage": "de-DE",
                    "description": "A discussion about the first topic.",
                    "guid": "episode-1",
                    "link": "https://example.test/episodes/one",
                }
            ]
        }

    def fake_download(url, destination, timeout_seconds):
        destination.write_bytes(b"audio")

    tagged = []

    def fake_tag_audio_file(path, metadata):
        tagged.append(metadata)

    monkeypatch.setattr(podcast, "_request", fake_request)
    monkeypatch.setattr(podcast, "_download", fake_download)
    monkeypatch.setattr(podcast, "_tag_audio_file", fake_tag_audio_file)
    context = RunContext(
        run_id="run",
        pipeline_name="podcast",
        run_dir=tmp_path,
        stage_id="input",
        work_dir=tmp_path / "input",
        console=FakeConsole(["example", "1", "", "1"]),
    )
    artifact = PodcastIndexInput().run(context, PodcastIndexConfig())
    assert artifact.path.read_bytes() == b"audio"
    assert artifact.metadata.title == "Episode One"
    assert artifact.metadata.language == "de"
    assert artifact.metadata.collection_title == "Example Show"
    assert artifact.metadata.creator == "Host"
    assert artifact.metadata.attributes["episode_guid"] == "episode-1"
    assert tagged == [artifact.metadata]
