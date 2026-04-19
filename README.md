# podslurp

Search for a podcast episode, download the audio, and get a full text transcription — all from the command line.

## What it does

1. **Search** — type a podcast name; podslurp queries [PodcastIndex](https://podcastindex.org/) and shows matching feeds.
2. **Browse** — pick the podcast, then search or scroll through its most recent episodes.
3. **Confirm** — review the episode details before anything is downloaded.
4. **Download** — the audio file is streamed to `downloads/` with a live progress bar.
5. **Transcribe** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper) runs locally and produces a full transcript. The language is read from the RSS feed and passed to the model as a hint (faster than auto-detection).
6. **Save** — two files land in `transcriptions/` named after the podcast, date and episode title:
   - `{podcast}_{YYYY-MM-DD}_{episode}.txt` — human-readable transcript with a metadata header
   - `{podcast}_{YYYY-MM-DD}_{episode}.json` — structured output with per-segment timestamps, log-probabilities and full metadata

## Requirements

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (`pip install uv` or `brew install uv`)
- A free [PodcastIndex API key](https://api.podcastindex.org/) (takes ~30 seconds to sign up)
- A CPU or CUDA GPU (CPU works fine with the `int8` compute type)

## Installation

```bash
git clone https://github.com/your-username/podslurp.git
cd podslurp
make install          # creates .venv, installs deps, copies .env.example → .env
```

Then open `.env` and fill in your credentials:

```
PODCASTINDEX_API_KEY=your_key_here
PODCASTINDEX_API_SECRET=your_secret_here
```

## Usage

```bash
make run
# or
uv run podslurp
```

The CLI is fully interactive:

```
podslurp  —  Podcast Search · Download · Transcribe

Search for a podcast: lex fridman

  #   Podcast                    Author           Episodes   Lang
  1   Lex Fridman Podcast        Lex Fridman         461     en
  2   ...

Select podcast number (or 's' to search again): 1

Episode keyword filter (Enter = 10 most recent): elon musk

  #   Episode                                   Date         Duration
  1   Elon Musk: War, AI & the Future …         2024-03-10    8h 27m
  2   ...

Select episode number (or 'b' to go back): 1

╭── Episode ────────────────────────────────────────────────────────╮
│ Elon Musk: War, AI & the Future of Humanity                       │
│ Podcast:  Lex Fridman Podcast                                     │
│ Date:     March 10, 2024                                          │
│ Duration: 8h 27m                                                  │
╰───────────────────────────────────────────────────────────────────╯

Download and transcribe? [Y/n]: y

Downloading: lex_fridman_podcast_2024-03-10_elon_musk…mp3
100%|████████████████████| 462M/462M [01:23<00:00, 5.5MB/s]

Transcribing with large-v3 (language hint: en) — this may take a while…
Done. Detected language: en (99%)  |  Duration: 30447s

Transcription saved:
  Text: transcriptions/lex_fridman_podcast_2024-03-10_elon_musk_war_ai_the_future_of_humanity.txt
  JSON: transcriptions/lex_fridman_podcast_2024-03-10_elon_musk_war_ai_the_future_of_humanity.json
```

## Configuration

All settings live in `.env`:

| Variable | Default | Description |
|---|---|---|
| `PODCASTINDEX_API_KEY` | — | **Required.** Your PodcastIndex API key |
| `PODCASTINDEX_API_SECRET` | — | **Required.** Your PodcastIndex API secret |
| `WHISPER_MODEL` | `large-v3` | Model size: `tiny`, `base`, `small`, `medium`, `large-v3`, `turbo` |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8` (CPU), `float16` (GPU), `int8_float16` (GPU, lower VRAM) |
| `PODSLURP_OUTPUT_DIR` | `./transcriptions` | Where transcript files are written |
| `PODSLURP_DOWNLOAD_DIR` | `./downloads` | Where audio files are saved |

### Choosing a Whisper model

| Model | Speed (CPU) | Accuracy | Notes |
|---|---|---|---|
| `tiny` / `base` | Very fast | Lower | Good for quick drafts |
| `small` | Fast | Good | Solid for most podcasts |
| `large-v3` | Slow | Best | Recommended default |
| `turbo` | Fast | Near large-v3 | Best speed/accuracy tradeoff (GPU recommended) |

### GPU usage

Install the matching CUDA runtime, then set:

```
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
```

## Makefile targets

| Command | Description |
|---|---|
| `make install` | Set up the virtual env and install all dependencies |
| `make run` | Launch the interactive CLI |
| `make lint` | Run `ruff` over the source |
| `make clean` | Remove `.venv`, `downloads/`, `transcriptions/`, caches |

## Output files

### `.txt`
```
Podcast:           Lex Fridman Podcast
Episode:           Elon Musk: War, AI & the Future of Humanity
Published:         2024-03-10
Language (feed):   en
Detected language: en (99%)
Whisper model:     large-v3
Duration:          30447s
--- TRANSCRIPT ---

Joe Rogan: Welcome back. Today my guest is...
```

### `.json`
```json
{
  "metadata": {
    "podcast_title": "Lex Fridman Podcast",
    "episode_title": "Elon Musk: War, AI & the Future of Humanity",
    "date_published_iso": "2024-03-10T00:00:00+00:00",
    "duration_seconds": 30447.0,
    "feed_language": "en",
    "detected_language": "en",
    "detected_language_probability": 0.99,
    "whisper_model": "large-v3",
    ...
  },
  "segments": [
    { "start": 0.0, "end": 4.2, "text": "Welcome back.", "avg_logprob": -0.21, "no_speech_prob": 0.01 },
    ...
  ],
  "full_text": "Welcome back. Today my guest is..."
}
```

## License

MIT
