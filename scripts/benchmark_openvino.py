#!/usr/bin/env python3
"""Compare Voxtype transcription engines on AC power.

The default preset uses the OpenVINO benchmark artifacts under
~/.cache/voxtype-openvino-test and binaries under
~/.cache/voxtype-bench-targets. Start it detached, close other applications,
and leave the machine alone:

    ./scripts/benchmark_openvino.py --detach

Use --dry-run to validate inputs without waiting or transcribing. Custom cases
use LABEL,MODEL,BINARY,CONFIG:

    --case 'openvino-igpu,base.en,/path/to/voxtype,/path/to/config.toml'

Each process receives one separate warm-up clip followed by the complete
corpus. The warm-up is discarded; no corpus clip is removed from WER or timing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


CACHE_HOME = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
DEFAULT_ROOT = CACHE_HOME / "voxtype-openvino-test"
DEFAULT_TARGETS = CACHE_HOME / "voxtype-bench-targets"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TIME_RE = re.compile(r"^Transcribed in ([0-9.]+)s$")
SEP_RE = re.compile(r"^=== (.+) ===$")
LOG_RE = re.compile(
    r"^\s*\d{4}-\d{2}-\d{2}T[\d:.+-]+Z?\s+"
    r"(INFO|WARN|ERROR|DEBUG|TRACE)\b"
)


class BenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class Case:
    label: str
    model: str
    binary: Path
    config: Path
    repeats: int = 1


@dataclass(frozen=True)
class Clip:
    clip_id: str
    duration_s: float
    reference: str
    wav: Path


@dataclass(frozen=True)
class Transcription:
    clip_id: str
    seconds: float
    hypothesis: str


def default_cases(root: Path = DEFAULT_ROOT) -> list[Case]:
    configs = root / "bench-cfg"
    return [
        Case(
            "openvino-igpu",
            "base.en",
            DEFAULT_TARGETS / "openvino/release/voxtype",
            configs / "openvino-base.en.toml",
        ),
        Case(
            "whisper-vulkan",
            "base.en",
            DEFAULT_TARGETS / "vulkan/release/voxtype",
            configs / "whisper-base.en.toml",
        ),
        Case(
            "openvino-npu",
            "base.en",
            DEFAULT_TARGETS / "openvino/release/voxtype",
            configs / "openvino-base.en-npu.toml",
        ),
        Case(
            "whisper-cpu",
            "base.en",
            DEFAULT_TARGETS / "cpu/release/voxtype",
            configs / "whisper-base.en.toml",
        ),
        Case(
            "openvino-igpu",
            "large-v3-turbo",
            DEFAULT_TARGETS / "openvino/release/voxtype",
            configs / "openvino-large-v3-turbo.toml",
        ),
        Case(
            "whisper-vulkan",
            "large-v3-turbo",
            DEFAULT_TARGETS / "vulkan/release/voxtype",
            configs / "whisper-large-v3-turbo.toml",
        ),
    ]


def parse_case(spec: str, with_repeats: bool = False) -> Case:
    expected = 5 if with_repeats else 4
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != expected:
        suffix = ",REPEATS" if with_repeats else ""
        raise argparse.ArgumentTypeError(
            f"case must be LABEL,MODEL,BINARY,CONFIG{suffix}: {spec!r}"
        )
    label, model, binary, config = parts[:4]
    if not all((label, model, binary, config)):
        raise argparse.ArgumentTypeError(f"case contains an empty field: {spec!r}")
    repeats = 1
    if with_repeats:
        try:
            repeats = int(parts[4])
        except ValueError as error:
            raise argparse.ArgumentTypeError("REPEATS must be an integer") from error
        if repeats < 1:
            raise argparse.ArgumentTypeError("REPEATS must be at least 1")
    return Case(label, model, Path(binary).expanduser(), Path(config).expanduser(), repeats)


def load_manifest(path: Path) -> list[Clip]:
    if not path.is_file():
        raise BenchmarkError(f"manifest not found: {path}")
    clips: list[Clip] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                clip_id = row["id"].strip()
                duration_s = float(row["duration_s"])
                reference = row["reference"].strip()
            except (KeyError, TypeError, ValueError) as error:
                raise BenchmarkError(f"invalid manifest row: {row}") from error
            wav = path.parent / f"{clip_id}.wav"
            if duration_s <= 0 or not reference:
                raise BenchmarkError(f"invalid manifest row for {clip_id}")
            if not wav.is_file():
                raise BenchmarkError(f"audio file not found: {wav}")
            clips.append(Clip(clip_id, duration_s, reference, wav))
    if len(clips) < 2:
        raise BenchmarkError("the manifest needs at least two clips")
    if len({clip.clip_id for clip in clips}) != len(clips):
        raise BenchmarkError("manifest clip IDs must be unique")
    return clips


def validate_cases(cases: Sequence[Case]) -> None:
    if not cases:
        raise BenchmarkError("at least one benchmark case is required")
    identities: set[tuple[str, str]] = set()
    for case in cases:
        identity = (case.label, case.model)
        if identity in identities:
            raise BenchmarkError(f"duplicate case: {case.label}/{case.model}")
        identities.add(identity)
        if not case.binary.is_file():
            raise BenchmarkError(f"binary not found: {case.binary}")
        if not os.access(case.binary, os.X_OK):
            raise BenchmarkError(f"binary is not executable: {case.binary}")
        if not case.config.is_file():
            raise BenchmarkError(f"config not found: {case.config}")


def normalize_words(text: str) -> list[str]:
    text = text.casefold().replace("\u2019", "'")
    cleaned = "".join(character if character.isalnum() or character in "' " else " " for character in text)
    return [word for token in cleaned.split() if (word := token.strip("'"))]


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, ref_word in enumerate(reference, start=1):
        current = [row]
        for column, hyp_word in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_word != hyp_word),
                )
            )
        previous = current
    return previous[-1]


def corpus_wer(clips: Sequence[Clip], transcriptions: Sequence[Transcription]) -> float:
    if len(clips) != len(transcriptions):
        raise BenchmarkError("cannot calculate WER for mismatched corpus lengths")
    errors = 0
    words = 0
    for clip, transcription in zip(clips, transcriptions, strict=True):
        reference = normalize_words(clip.reference)
        hypothesis = normalize_words(transcription.hypothesis)
        errors += edit_distance(reference, hypothesis)
        words += len(reference)
    if words == 0:
        raise BenchmarkError("manifest references contain no words")
    return errors / words * 100.0


def parse_transcriptions(stdout: str) -> list[Transcription]:
    parsed: list[Transcription] = []
    clip_id: str | None = None
    seconds: float | None = None
    hypothesis: list[str] = []

    def flush() -> None:
        nonlocal clip_id, seconds, hypothesis
        if clip_id is None:
            return
        if seconds is None:
            raise BenchmarkError(f"missing timing for {clip_id}")
        parsed.append(Transcription(clip_id, seconds, " ".join(hypothesis)))

    for raw_line in stdout.splitlines():
        line = ANSI_RE.sub("", raw_line).strip()
        separator = SEP_RE.match(line)
        if separator:
            flush()
            clip_id = Path(separator.group(1)).stem
            seconds = None
            hypothesis = []
            continue
        timing = TIME_RE.match(line)
        if timing and clip_id is not None:
            seconds = float(timing.group(1))
            continue
        if seconds is not None and line and not LOG_RE.match(line):
            hypothesis.append(line)
    flush()
    return parsed


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise BenchmarkError("cannot calculate a percentile without values")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def read_cpu_ticks() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text(encoding="ascii").splitlines()[0].split()[1:]
    ticks = [int(field) for field in fields]
    total = sum(ticks)
    idle = ticks[3] + (ticks[4] if len(ticks) > 4 else 0)
    return total, idle


def cpu_busy_percent(seconds: float) -> float:
    total_before, idle_before = read_cpu_ticks()
    time.sleep(seconds)
    total_after, idle_after = read_cpu_ticks()
    total_delta = total_after - total_before
    if total_delta <= 0:
        raise BenchmarkError("CPU counters did not advance")
    return (1.0 - (idle_after - idle_before) / total_delta) * 100.0


def wait_for_cpu_idle(max_percent: float, check_seconds: float, timeout: float) -> float:
    deadline = time.monotonic() + timeout
    while True:
        busy = cpu_busy_percent(check_seconds)
        print(f"Idle check: aggregate CPU busy {busy:.1f}%", flush=True)
        if busy <= max_percent:
            return busy
        if time.monotonic() >= deadline:
            raise BenchmarkError(
                f"CPU did not fall below {max_percent:.1f}% within {timeout:.0f}s"
            )
        time.sleep(min(5.0, max(0.0, deadline - time.monotonic())))


def mains_online() -> bool:
    supplies = Path("/sys/class/power_supply")
    for supply in supplies.glob("*"):
        try:
            if (supply / "type").read_text().strip() != "Mains":
                continue
            if (supply / "online").read_text().strip() == "1":
                return True
        except OSError:
            continue
    return False


def child_environment(library_dir: Path | None) -> dict[str, str]:
    environment = os.environ.copy()
    environment["NO_COLOR"] = "1"
    if library_dir and library_dir.is_dir():
        old_path = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = (
            f"{library_dir}:{old_path}" if old_path else str(library_dir)
        )
    return environment


def machine_metadata() -> dict[str, Any]:
    profile_path = Path("/sys/firmware/acpi/platform_profile")
    profile = profile_path.read_text().strip() if profile_path.is_file() else None
    governors = set()
    for path in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor"):
        try:
            governors.add(path.read_text().strip())
        except OSError:
            pass
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "kernel": platform.release(),
        "platform_profile": profile,
        "cpu_governors": sorted(governors),
        "mains_online": mains_online(),
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_case(
    case: Case,
    clips: Sequence[Clip],
    environment: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    evaluation_clips = list(clips) * case.repeats
    command_clips = [clips[0], *evaluation_clips]
    command = [
        str(case.binary),
        "--config",
        str(case.config),
        "transcribe",
        *(str(clip.wav) for clip in command_clips),
    ]
    print(
        f"Running {case.label}/{case.model}: one warm-up + "
        f"{len(evaluation_clips)} measured clips",
        flush=True,
    )
    started = time.perf_counter()
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise BenchmarkError(f"{case.label}/{case.model} timed out") from error
    wall_s = time.perf_counter() - started
    if process.returncode != 0:
        detail = process.stderr[-2000:].strip()
        raise BenchmarkError(
            f"{case.label}/{case.model} exited with {process.returncode}:\n{detail}"
        )

    parsed = parse_transcriptions(process.stdout)
    expected_ids = [clip.clip_id for clip in command_clips]
    actual_ids = [item.clip_id for item in parsed]
    if actual_ids != expected_ids:
        raise BenchmarkError(
            f"{case.label}/{case.model} output mismatch: expected "
            f"{len(expected_ids)} ordered clips, parsed {len(actual_ids)}"
        )
    measured = parsed[1:]
    inference_times = [item.seconds for item in measured]
    audio_s = sum(clip.duration_s for clip in evaluation_clips)
    wer_percent = corpus_wer(evaluation_clips, measured)
    median_s = statistics.median(inference_times)
    word_accuracy = max(0.0, 1.0 - wer_percent / 100.0)
    result = {
        "label": case.label,
        "model": case.model,
        "valid": True,
        "measured_clips": len(measured),
        "audio_s": audio_s,
        "warmup_inference_s": parsed[0].seconds,
        "median_inference_s": median_s,
        "p90_inference_s": percentile(inference_times, 0.90),
        "total_inference_s": sum(inference_times),
        "rtf": sum(inference_times) / audio_s,
        "wer_percent": wer_percent,
        "quality_adjusted_speed": word_accuracy / median_s,
        "wall_s": wall_s,
        "binary": str(case.binary),
        "config": str(case.config),
    }
    print(
        f"  median={median_s * 1000:.0f}ms "
        f"p90={result['p90_inference_s'] * 1000:.0f}ms "
        f"RTF={result['rtf']:.4f} WER={wer_percent:.2f}%",
        flush=True,
    )
    return result


def aggregate_performance_results(
    runs: Sequence[dict[str, Any]], cases: Sequence[Case], rounds: int
) -> list[dict[str, Any]]:
    metrics = (
        "median_inference_s",
        "p90_inference_s",
        "rtf",
        "wer_percent",
        "quality_adjusted_speed",
    )
    aggregated: list[dict[str, Any]] = []
    for case in cases:
        matching = [
            result
            for result in runs
            if result["label"] == case.label and result["model"] == case.model
        ]
        invalid = [result for result in matching if not result.get("valid")]
        if len(matching) != rounds or invalid:
            aggregated.append(
                {
                    "label": case.label,
                    "model": case.model,
                    "valid": False,
                    "invalid_reasons": [
                        result.get("error", "invalid benchmark round") for result in invalid
                    ]
                    or [f"completed {len(matching)} of {rounds} rounds"],
                }
            )
            continue
        result: dict[str, Any] = {
            "label": case.label,
            "model": case.model,
            "valid": True,
            "rounds": rounds,
        }
        for metric in metrics:
            values = [float(item[metric]) for item in matching]
            result[metric] = statistics.median(values)
            result[f"{metric}_min"] = min(values)
            result[f"{metric}_max"] = max(values)
        aggregated.append(result)
    return aggregated


def decisive_minimum(group: Sequence[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    ordered = sorted(group, key=lambda result: result[metric])
    if len(ordered) < 2:
        return None
    best, runner_up = ordered[:2]
    if best.get(f"{metric}_max", best[metric]) < runner_up.get(
        f"{metric}_min", runner_up[metric]
    ):
        return best
    return None


def decisive_maximum(group: Sequence[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    ordered = sorted(group, key=lambda result: result[metric], reverse=True)
    if len(ordered) < 2:
        return None
    best, runner_up = ordered[:2]
    if best.get(f"{metric}_min", best[metric]) > runner_up.get(
        f"{metric}_max", runner_up[metric]
    ):
        return best
    return None


def print_winners(
    results: Sequence[dict[str, Any]],
    expected_cases: Sequence[Case],
    energy: bool = False,
    minimum_energy_margin: float = 0.0,
    energy_key: str = "steady_energy_per_audio_s",
) -> None:
    models = sorted({case.model for case in expected_cases})
    print("\nWinners", flush=True)
    for model in models:
        expected = [case for case in expected_cases if case.model == model]
        group = [
            result
            for result in results
            if result.get("valid") and result["model"] == model
        ]
        if len(group) != len(expected) or len(group) < 2:
            print(
                f"  {model}: no winner; {len(group)}/{len(expected)} requested cases are valid",
                flush=True,
            )
            continue
        if energy:
            ordered_energy = sorted(
                group,
                key=lambda result: result.get(energy_key, result["joules_per_audio_s"]),
            )
            energy_winner, energy_runner_up = ordered_energy[:2]
            winner_value = energy_winner.get(energy_key, energy_winner["joules_per_audio_s"])
            runner_value = energy_runner_up.get(
                energy_key, energy_runner_up["joules_per_audio_s"]
            )
            margin = runner_value / winner_value - 1.0
            energy_label = (
                energy_winner["label"]
                if margin >= minimum_energy_margin
                else f"tie (<{minimum_energy_margin:.0%} margin)"
            )
            adjusted = max(
                group, key=lambda result: result["quality_adjusted_energy_efficiency"]
            )
            fastest = min(group, key=lambda result: result["median_inference_s"])
            most_accurate = min(group, key=lambda result: result["wer_percent"])
            basis = "battery" if energy_key.startswith("gross") else "steady"
            print(
                f"  {model}: lowest-energy={energy_label} "
                f"({winner_value:.4f} J/audio-s {basis}, margin={margin:.1%}), "
                f"quality-adjusted={adjusted['label']}, "
                f"fastest={fastest['label']}, lowest-WER={most_accurate['label']}",
                flush=True,
            )
        else:
            fastest = decisive_minimum(group, "median_inference_s")
            most_accurate = decisive_minimum(group, "wer_percent")
            adjusted = decisive_maximum(group, "quality_adjusted_speed")
            print(
                f"  {model}: adjusted-speed={adjusted['label'] if adjusted else 'tie'}, "
                f"fastest={fastest['label'] if fastest else 'tie'}, "
                f"lowest-WER={most_accurate['label'] if most_accurate else 'tie'}",
                flush=True,
            )


def detach(kind: str, log_path: Path | None) -> int:
    if log_path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = DEFAULT_ROOT / "benchmark-runs" / f"{kind}-{stamp}.log"
    log_path = log_path.expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_arguments = [argument for argument in sys.argv[1:] if argument != "--detach"]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(sys.argv[0]).resolve()), *child_arguments],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=environment,
        )
    print(f"Started detached benchmark PID {process.pid}")
    print(f"Log: {log_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark warm Voxtype transcription speed and WER on AC power.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        help="LABEL,MODEL,BINARY,CONFIG; repeat for each engine",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_ROOT / "bench-dictation/manifest.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="result JSON path; defaults to a timestamped file under the benchmark cache",
    )
    parser.add_argument("--start-delay", type=float, default=120.0)
    parser.add_argument("--max-cpu-percent", type=float, default=5.0)
    parser.add_argument("--idle-check-seconds", type=float, default=10.0)
    parser.add_argument("--idle-timeout", type=float, default=600.0)
    parser.add_argument("--between-case-delay", type=float, default=60.0)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-wer", type=float, default=20.0)
    parser.add_argument("--case-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=Path("~/.local/lib/voxtype-openvino").expanduser(),
    )
    parser.add_argument("--allow-battery", action="store_true")
    parser.add_argument("--skip-idle-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--log", type=Path, help="detached stdout/stderr log path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cases = args.case or default_cases()
    manifest = args.manifest.expanduser()
    clips = load_manifest(manifest)
    validate_cases(cases)
    if (
        args.start_delay < 0
        or args.between_case_delay < 0
        or args.case_timeout <= 0
        or args.idle_check_seconds <= 0
        or args.idle_timeout <= 0
        or not 0 <= args.max_cpu_percent <= 100
        or not 0 <= args.max_wer <= 100
        or args.rounds < 2
    ):
        parser.error("invalid delay, timeout, CPU, WER, or rounds setting")
    print(f"Validated {len(cases)} cases and {len(clips)} corpus clips.", flush=True)
    for case in cases:
        print(f"  {case.label}/{case.model}: {case.binary}", flush=True)
    if args.dry_run:
        return 0
    if args.detach:
        return detach("performance", args.log)
    if not args.allow_battery and not mains_online():
        raise BenchmarkError("AC power is required; use --allow-battery only intentionally")
    if args.start_delay:
        print(
            f"Starting in {args.start_delay:.0f}s. Close browsers, agents, media, and other apps.",
            flush=True,
        )
        time.sleep(args.start_delay)
    if not args.allow_battery and not mains_online():
        raise BenchmarkError("AC power was disconnected during the start delay")

    metadata = machine_metadata()
    environment = child_environment(args.library_dir.expanduser())
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = (
        args.output.expanduser()
        if args.output
        else DEFAULT_ROOT / "benchmark-runs" / f"performance-{stamp}.json"
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "ac-performance",
        "metadata": metadata,
        "manifest": str(manifest),
        "rounds": args.rounds,
        "runs": [],
        "results": [],
    }
    run_index = 0
    for round_number in range(1, args.rounds + 1):
        round_cases = list(cases) if round_number % 2 else list(reversed(cases))
        print(f"\nRound {round_number}/{args.rounds}", flush=True)
        for case in round_cases:
            if run_index and args.between_case_delay:
                print(
                    f"Cooling down for {args.between_case_delay:.0f}s before the next case.",
                    flush=True,
                )
                time.sleep(args.between_case_delay)
            run_index += 1
            if not args.allow_battery and not mains_online():
                raise BenchmarkError("AC power was disconnected during the benchmark")
            if not args.skip_idle_check:
                wait_for_cpu_idle(
                    args.max_cpu_percent, args.idle_check_seconds, args.idle_timeout
                )
            try:
                result = run_case(case, clips, environment, args.case_timeout)
                if result["wer_percent"] > args.max_wer:
                    result["valid"] = False
                    result["error"] = (
                        f"WER {result['wer_percent']:.2f}% exceeds {args.max_wer:.2f}%"
                    )
            except (BenchmarkError, OSError, UnicodeError, ValueError) as error:
                result = {
                    "label": case.label,
                    "model": case.model,
                    "valid": False,
                    "error": str(error),
                }
                print(f"  INVALID: {error}", flush=True)
            result["round"] = round_number
            payload["runs"].append(result)
            payload["results"] = aggregate_performance_results(
                payload["runs"], cases, args.rounds
            )
            atomic_write_json(output, payload)

    print_winners(payload["results"], cases)
    print(f"Results: {output}", flush=True)
    return 0 if all(result.get("valid") for result in payload["results"]) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
