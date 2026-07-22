# VoxYak

VoxYak is a local, modular audio pipeline: acquire audio, transcribe it, then run
one or more processors such as meeting summarization or speaker-time analysis.

## Pipeline model

Every YAML pipeline has three typed stages:

1. one input module;
2. one transcription module;
3. zero or more ordered processor modules.

VoxYak includes these modules:

| Type | Module | Purpose |
| --- | --- | --- |
| Input | `file` | Use an existing local audio file |
| Input | `podcast-index` | Search PodcastIndex and download an episode |
| Input | `call-recorder` | Record microphone and system audio on Linux |
| Transcription | `faster-whisper` | Transcribe locally, with optional diarization |
| Processor | `speaker-time` | Calculate speaking time per speaker |
| Processor | `openai-summary` | Create a structured meeting brief |

Installed Python packages can contribute modules through the `voxyak.inputs`,
`voxyak.transcribers`, and `voxyak.processors` entry-point groups. Plugin classes
subclass the contracts in `voxyak.sdk` and expose a Pydantic `config_model`.

## Requirements

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` and `pactl` for call recording
- PodcastIndex credentials for the podcast input
- An OpenAI API key for cloud summaries

## Installation

```bash
make install
```

Fill in `.env`, then validate the shipped configuration:

```bash
make validate
```

Optional speaker diarization requires the additional dependency and access to
the gated pyannote model:

```bash
make install-diarize
```

## Usage

```bash
# Inspect configuration and installed modules
voxyak pipelines list
voxyak pipelines validate
voxyak modules

# Existing local audio
voxyak transcribe recordings/interview.mp3 --language de

# Interactive podcast selection and local transcription
voxyak run podcast

# Record until Enter or Ctrl-C, then transcribe and summarize
voxyak run call-summary

# Continue after a failed cloud processor without recording again
voxyak resume 20260721-143000-a1b2c3d4

# Standalone transcript operations
voxyak diarize runs/<run-id>/transcription/transcript.json
voxyak analyze runs/<run-id>/transcription/transcript.json
```

The summary processor sends transcript text—not audio—to the OpenAI Responses
API with API storage disabled. Selecting a pipeline containing this processor is
explicit authorization to send that transcript text to the configured service.

## Configuration

Each YAML file in `pipelines/` defines one pipeline. Its filename (without the
extension) is the pipeline name; for example, `pipelines/meeting.yaml` defines
the `meeting` pipeline. Module options are validated before a run starts.
Credentials stay in environment variables and are never written to saved
pipeline snapshots or manifests.

```yaml
version: 1
input:
  uses: file
  with:
    path: ./meeting.mp3
transcription:
  uses: faster-whisper
  with:
    model: small
    language: auto
processing:
  - id: summary
    uses: openai-summary
    with:
      model: gpt-5.6-terra
      reasoning_effort: low
      prompt: |
        Summarize the main conclusions, decisions, action items, and open questions.
```

Generated inputs, transcripts, processor output, a secret-redacted pipeline
snapshot, and an atomic status manifest are stored under `runs/<run-id>/`.
Completed stages are reused by `voxyak resume`.

## Transcript format

VoxYak writes `transcription/transcript.json` with `schema_version`, generic
source metadata, transcription metadata, timestamped segments, and full text.
The adjacent text file is a human-readable rendering of the same document.

## Development

```bash
make lint
make test
```
