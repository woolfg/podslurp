from __future__ import annotations

from pathlib import Path

import pytest

from voxyak.modules.inputs import podcast
from voxyak.modules.inputs.podcast import PodcastIndexConfig, PodcastIndexInput
from voxyak.sdk import RunContext


class FakeConsole:
    def __init__(self, answers: list[str]) -> None:
        self.answers = iter(answers)

    def input(self, prompt: str) -> str:
        return next(self.answers)

    def print(self, *args, **kwargs) -> None:
        pass


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
                }
            ]
        }

    def fake_download(url, destination, timeout_seconds):
        destination.write_bytes(b"audio")

    monkeypatch.setattr(podcast, "_request", fake_request)
    monkeypatch.setattr(podcast, "_download", fake_download)
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
    assert artifact.metadata["title"] == "Episode One"
    assert artifact.metadata["language_hint"] == "de"
    assert artifact.metadata["podcast_title"] == "Example Show"
