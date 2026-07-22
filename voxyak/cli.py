"""VoxYak command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError
from rich import box
from rich.console import Console
from rich.table import Table

from .analysis import analyze_transcript, format_duration
from .config import (
    ModuleSpec,
    PipelineDefinition,
    default_config_path,
    default_runs_dir,
    load_config,
)
from .modules.transcribers.faster_whisper import (
    DiarizationConfig,
    diarize_transcript_file,
)
from .pipeline import PipelineRunError, PipelineRunner
from .registry import RegistryError, default_registry
from .sdk import RunContext, TranscriptArtifact
from .transcript import load_transcript


console = Console()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxyak",
        description="Modular audio transcription and processing pipelines.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="Pipeline YAML file or directory (default: pipelines or VOXYAK_CONFIG).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=default_runs_dir(),
        help="Run artifact directory (default: runs or VOXYAK_RUNS_DIR).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    pipelines = commands.add_parser("pipelines", help="Inspect pipeline definitions.")
    pipeline_commands = pipelines.add_subparsers(
        dest="pipelines_command", required=True
    )
    pipeline_commands.add_parser("list", help="List configured pipelines.")
    validate = pipeline_commands.add_parser(
        "validate", help="Validate YAML and module configurations."
    )
    validate.add_argument("pipeline", nargs="?", help="Validate one pipeline only.")

    commands.add_parser("modules", help="List installed modules.")

    run = commands.add_parser("run", help="Run a configured pipeline.")
    run.add_argument("pipeline")

    resume = commands.add_parser("resume", help="Resume an incomplete run.")
    resume.add_argument("run_id", help="Run id or path to a run directory.")

    transcribe = commands.add_parser(
        "transcribe", help="Transcribe an existing local audio file."
    )
    transcribe.add_argument("file", type=Path)
    transcribe.add_argument("--title")
    transcribe.add_argument("--language", default="auto")
    transcribe.add_argument("--model", default="small")
    transcribe.add_argument("--device", default="cpu")
    transcribe.add_argument("--compute-type", default="int8")
    transcribe.add_argument("--diarize", action="store_true")
    transcribe.add_argument("--num-speakers", type=int)
    transcribe.add_argument("--min-speakers", type=int)
    transcribe.add_argument("--max-speakers", type=int)

    diarize = commands.add_parser(
        "diarize", help="Add speaker labels to a VoxYak transcript."
    )
    diarize.add_argument("transcript", type=Path)
    diarize.add_argument("--audio", type=Path)
    diarize.add_argument("--num-speakers", type=int)
    diarize.add_argument("--min-speakers", type=int)
    diarize.add_argument("--max-speakers", type=int)
    diarize.add_argument("--device")

    summarize = commands.add_parser(
        "summarize", help="Summarize an existing VoxYak transcript."
    )
    summarize.add_argument("transcript", type=Path)
    summarize.add_argument(
        "--pipeline",
        default="call-summary",
        help="Pipeline providing the openai-summary settings (default: call-summary).",
    )
    summarize.add_argument(
        "--output-dir",
        type=Path,
        help="Summary output directory (default: <transcript directory>/summary).",
    )

    analyze = commands.add_parser(
        "analyze", help="Calculate speaker time from a VoxYak transcript."
    )
    analyze.add_argument("transcript", type=Path)
    return parser


def _runner(args) -> PipelineRunner:
    return PipelineRunner(
        default_registry(),
        args.runs_dir,
        console,
    )


def _list_pipelines(args) -> None:
    config = load_config(args.config)
    table = Table(box=box.SIMPLE_HEAD)
    table.add_column("Pipeline", style="bold")
    table.add_column("Input")
    table.add_column("Transcription")
    table.add_column("Processing")
    for name, pipeline in config.pipelines.items():
        table.add_row(
            name,
            pipeline.input.uses,
            pipeline.transcription.uses,
            ", ".join(item.uses for item in pipeline.processing) or "—",
        )
    console.print(table)


def _validate_pipelines(args) -> None:
    config = load_config(args.config)
    runner = _runner(args)
    if args.pipeline:
        if args.pipeline not in config.pipelines:
            raise ValueError(f"Unknown pipeline: {args.pipeline}")
        runner.resolve(config.pipelines[args.pipeline])
        console.print(f"[green]Valid:[/green] {args.pipeline}")
    else:
        runner.validate_configuration(config)
        console.print(
            f"[green]Valid:[/green] {len(config.pipelines)} pipeline(s) in {args.config}"
        )


def _list_modules(args) -> None:
    registry = default_registry()
    table = Table(box=box.SIMPLE_HEAD)
    table.add_column("Kind")
    table.add_column("Name", style="bold")
    table.add_column("Version")
    table.add_column("Source", style="dim")
    for item in registry.list():
        table.add_row(item.kind, item.name, item.version, item.source)
    console.print(table)


def _transcribe(args) -> None:
    input_options = {
        "path": args.file,
        "title": args.title,
        "language": None if args.language in {"auto", "source"} else args.language,
    }
    transcription_options = {
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
        "language": args.language,
        "diarization": {
            "enabled": args.diarize,
            "num_speakers": args.num_speakers,
            "min_speakers": args.min_speakers,
            "max_speakers": args.max_speakers,
        },
    }
    pipeline = PipelineDefinition(
        input=ModuleSpec(uses="file", options=input_options),
        transcription=ModuleSpec(
            uses="faster-whisper", options=transcription_options
        ),
        processing=[],
    )
    _runner(args).run_pipeline("local-transcription", pipeline)


def _diarize(args) -> None:
    settings = DiarizationConfig(
        enabled=True,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        device=args.device,
    )
    json_path, text_path = diarize_transcript_file(
        args.transcript,
        args.audio,
        settings,
    )
    console.print("[bold green]Diarization complete.[/bold green]")
    console.print(f"JSON: {json_path}")
    console.print(f"Text: {text_path}")


def _analyze(args) -> None:
    document = load_transcript(args.transcript)
    analysis = analyze_transcript(document)
    table = Table(box=box.SIMPLE_HEAD)
    table.add_column("Speaker", style="bold")
    table.add_column("Segments", justify="right")
    table.add_column("Speaking time", justify="right")
    table.add_column("Share", justify="right")
    for item in analysis.speakers:
        table.add_row(
            item.speaker,
            str(item.segments),
            format_duration(item.seconds),
            f"{item.percentage:.1f}%",
        )
    console.print(table)
    console.print(f"Total: {format_duration(analysis.total_seconds)}")


def _summarize(args) -> None:
    document = load_transcript(args.transcript)
    config = load_config(args.config)
    try:
        pipeline = config.pipelines[args.pipeline]
    except KeyError as exc:
        available = ", ".join(sorted(config.pipelines)) or "none"
        raise ValueError(
            f"Unknown pipeline {args.pipeline!r}. Available: {available}."
        ) from exc

    processors = [
        item for item in pipeline.processing if item.uses == "openai-summary"
    ]
    if not processors:
        raise ValueError(
            f"Pipeline {args.pipeline!r} has no openai-summary processor."
        )
    if len(processors) > 1:
        raise ValueError(
            f"Pipeline {args.pipeline!r} has multiple openai-summary processors."
        )

    processor_spec = processors[0]
    descriptor = default_registry().descriptor("processor", "openai-summary")
    processor = descriptor.module_class()
    settings = descriptor.module_class.config_model.model_validate(
        processor_spec.options
    )
    processor.preflight(settings)

    transcript_path = args.transcript.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else transcript_path.parent / "summary"
    )
    context = RunContext(
        run_id=f"standalone-summary-{transcript_path.stem}",
        pipeline_name=args.pipeline,
        run_dir=output_dir.parent,
        stage_id=f"processing.{processor_spec.id}",
        work_dir=output_dir,
        console=console,
    )
    artifact = TranscriptArtifact(
        path=transcript_path,
        text_path=transcript_path.with_suffix(".txt"),
        language=document.transcription.language,
        duration_seconds=document.transcription.duration_seconds,
        metadata=document.source.metadata.model_copy(deep=True),
    )
    outputs = processor.run(context, artifact, [], settings)
    console.print("[bold green]Summary complete.[/bold green]")
    for output in outputs:
        console.print(output.path)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "pipelines":
            if args.pipelines_command == "list":
                _list_pipelines(args)
            else:
                _validate_pipelines(args)
        elif args.command == "modules":
            _list_modules(args)
        elif args.command == "run":
            _runner(args).run_named(load_config(args.config), args.pipeline)
        elif args.command == "resume":
            _runner(args).resume(args.run_id)
        elif args.command == "transcribe":
            _transcribe(args)
        elif args.command == "diarize":
            _diarize(args)
        elif args.command == "summarize":
            _summarize(args)
        elif args.command == "analyze":
            _analyze(args)
        return 0
    except (ValueError, ValidationError, RegistryError, PipelineRunError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        if isinstance(exc, PipelineRunError) and exc.run_id:
            console.print(f"Resume later with: voxyak resume {exc.run_id}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
