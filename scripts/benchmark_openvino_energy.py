#!/usr/bin/env python3
"""Compare Voxtype engine speed, WER, and battery energy use.

This script is deliberately strict. It waits for low, stable idle power,
measures idle before and after every engine, records every battery sample, and
refuses to name an energy winner when quality checks fail.

Charge the laptop, unplug AC, then start the benchmark detached and close all
other applications during the countdown:

    ./scripts/benchmark_openvino_energy.py --detach

The default preset uses artifacts under ~/.cache/voxtype-openvino-test and
~/.cache/voxtype-bench-targets. Use --dry-run to validate them. Custom cases use
LABEL,MODEL,BINARY,CONFIG,REPEATS.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, TextIO

try:
    import benchmark_openvino as benchmark
except ModuleNotFoundError:
    from . import benchmark_openvino as benchmark


SAMPLE_FIELDS = [
    "timestamp_utc",
    "phase",
    "phase_elapsed_s",
    "voltage_v",
    "current_a",
    "power_w",
    "charge_ah",
    "capacity_percent",
    "status",
    "mains_online",
]


def parse_energy_case(spec: str) -> benchmark.Case:
    return benchmark.parse_case(spec, with_repeats=True)


def default_energy_cases() -> list[benchmark.Case]:
    repeats = {
        ("openvino-igpu", "base.en"): 13,
        ("whisper-vulkan", "base.en"): 5,
        ("openvino-npu", "base.en"): 12,
        ("whisper-cpu", "base.en"): 3,
        ("openvino-igpu", "large-v3-turbo"): 4,
        ("whisper-vulkan", "large-v3-turbo"): 1,
    }
    return [
        benchmark.Case(
            case.label,
            case.model,
            case.binary,
            case.config,
            repeats[(case.label, case.model)],
        )
        for case in benchmark.default_cases()
    ]


class Battery:
    def __init__(self, path: Path):
        self.path = path

    REQUIRED = ("current_now", "voltage_now", "charge_now", "capacity", "status")

    @classmethod
    def discover(cls) -> "Battery":
        seen: list[str] = []
        for path in sorted(Path("/sys/class/power_supply").glob("*")):
            try:
                if (path / "type").read_text().strip() != "Battery":
                    continue
                missing = [name for name in cls.REQUIRED if not (path / name).is_file()]
                if not missing:
                    return cls(path)
                seen.append(f"{path.name} is missing {', '.join(missing)}")
            except OSError as error:
                seen.append(f"{path.name} could not be read: {error}")
        # Some laptops expose energy_now and power_now instead of the charge and
        # current pair this script integrates. Naming what was actually missing
        # saves the next person from guessing whether the battery or the glob
        # was at fault.
        detail = "; ".join(seen) if seen else "no battery devices were found"
        raise benchmark.BenchmarkError(
            f"no battery exposes {', '.join(cls.REQUIRED)}: {detail}"
        )

    def read_int(self, attribute: str) -> int:
        return int((self.path / attribute).read_text().strip())

    def status(self) -> str:
        return (self.path / "status").read_text().strip()

    def sample(self, phase: str, phase_elapsed_s: float) -> dict[str, Any]:
        voltage_v = self.read_int("voltage_now") / 1_000_000
        current_a = abs(self.read_int("current_now")) / 1_000_000
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "phase_elapsed_s": phase_elapsed_s,
            "voltage_v": voltage_v,
            "current_a": current_a,
            "power_w": voltage_v * current_a,
            "charge_ah": self.read_int("charge_now") / 1_000_000,
            "capacity_percent": self.read_int("capacity"),
            "status": self.status(),
            "mains_online": benchmark.mains_online(),
        }


class PowerSampler(threading.Thread):
    def __init__(
        self,
        battery: Battery,
        phase: str,
        interval: float,
        writer: csv.DictWriter,
        output: TextIO,
    ):
        super().__init__(daemon=True)
        self.battery = battery
        self.phase = phase
        self.interval = interval
        self.writer = writer
        self.output = output
        self.started_at = 0.0
        self.samples: list[dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.error: Exception | None = None

    def capture(self) -> None:
        sample = self.battery.sample(self.phase, time.monotonic() - self.started_at)
        self.samples.append(sample)
        self.writer.writerow(sample)
        self.output.flush()

    def run(self) -> None:
        self.started_at = time.monotonic()
        try:
            while True:
                self.capture()
                if self.stop_event.wait(self.interval):
                    break
        except Exception as error:  # Propagate sampler failures to the main thread.
            self.error = error

    def finish(self) -> list[dict[str, Any]]:
        self.stop_event.set()
        self.join()
        if self.error:
            raise benchmark.BenchmarkError(f"battery sampling failed: {self.error}")
        try:
            self.capture()
        except OSError as error:
            raise benchmark.BenchmarkError(f"final battery sample failed: {error}") from error
        return self.samples


def measure_phase(
    battery: Battery,
    phase: str,
    interval: float,
    writer: csv.DictWriter,
    output: TextIO,
    duration: float | None = None,
    command: Sequence[str] | None = None,
    environment: dict[str, str] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    if (duration is None) == (command is None):
        raise benchmark.BenchmarkError("a phase needs exactly one of duration or command")
    charge_before = battery.read_int("charge_now") / 1_000_000
    total_before, idle_before = benchmark.read_cpu_ticks()
    sampler = PowerSampler(battery, phase, interval, writer, output)
    sampler.start()
    started = time.monotonic()
    process: subprocess.CompletedProcess[str] | None = None
    try:
        if duration is not None:
            time.sleep(duration)
        else:
            assert command is not None
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=environment,
                timeout=timeout,
                check=False,
            )
    except subprocess.TimeoutExpired as error:
        raise benchmark.BenchmarkError(f"phase {phase} timed out") from error
    finally:
        samples = sampler.finish()
        elapsed_s = time.monotonic() - started
    total_after, idle_after = benchmark.read_cpu_ticks()
    charge_after = battery.read_int("charge_now") / 1_000_000
    total_delta = total_after - total_before
    cpu_busy_percent = (
        (1.0 - (idle_after - idle_before) / total_delta) * 100.0
        if total_delta > 0
        else float("nan")
    )
    return {
        "phase": phase,
        "elapsed_s": elapsed_s,
        "samples": samples,
        "charge_before_ah": charge_before,
        "charge_after_ah": charge_after,
        "cpu_busy_percent": cpu_busy_percent,
        "process": process,
    }


def power_stats(phase: dict[str, Any], evaluation_start_s: float) -> dict[str, float]:
    powers = [
        sample["power_w"]
        for sample in phase["samples"]
        if sample["phase_elapsed_s"] >= evaluation_start_s
    ]
    if len(powers) < 3:
        raise benchmark.BenchmarkError(
            f"{phase['phase']} has only {len(powers)} samples after settling"
        )
    return {
        "median_w": statistics.median(powers),
        "mean_w": statistics.mean(powers),
        "stdev_w": statistics.pstdev(powers),
        "p10_w": benchmark.percentile(powers, 0.10),
        "p90_w": benchmark.percentile(powers, 0.90),
        "range80_w": benchmark.percentile(powers, 0.90)
        - benchmark.percentile(powers, 0.10),
        "samples": len(powers),
    }


def integrate_power(phase: dict[str, Any]) -> float:
    samples = phase["samples"]
    if len(samples) < 2:
        raise benchmark.BenchmarkError(f"not enough samples to integrate {phase['phase']}")
    elapsed_s = phase["elapsed_s"]
    joules = samples[0]["power_w"] * samples[0]["phase_elapsed_s"]
    for previous, current in zip(samples, samples[1:]):
        delta_s = current["phase_elapsed_s"] - previous["phase_elapsed_s"]
        joules += (previous["power_w"] + current["power_w"]) / 2 * delta_s
    final_gap = max(0.0, elapsed_s - samples[-1]["phase_elapsed_s"])
    return joules + samples[-1]["power_w"] * final_gap


def wait_for_quiet(
    battery: Battery,
    args: argparse.Namespace,
    writer: csv.DictWriter,
    output: TextIO,
    attempt_prefix: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + args.idle_timeout
    attempt = 0
    while True:
        attempt += 1
        phase = measure_phase(
            battery,
            f"{attempt_prefix}-quiet-{attempt}",
            args.sample_interval,
            writer,
            output,
            duration=args.quiet_seconds,
        )
        stats = power_stats(phase, 0.0)
        quiet = (
            stats["median_w"] <= args.max_idle_w
            and stats["range80_w"] <= args.max_idle_range_w
            and phase["cpu_busy_percent"] <= args.max_cpu_percent
            and not any(sample["mains_online"] for sample in phase["samples"])
        )
        print(
            f"Quiet check {attempt}: median={stats['median_w']:.2f}W "
            f"p10-p90={stats['range80_w']:.2f}W "
            f"CPU={phase['cpu_busy_percent']:.1f}% "
            f"({'pass' if quiet else 'wait'})",
            flush=True,
        )
        if quiet:
            return phase
        if time.monotonic() >= deadline:
            raise benchmark.BenchmarkError(
                "machine never reached the requested idle power and CPU limits"
            )


def validate_idle(
    name: str,
    stats: dict[str, float],
    args: argparse.Namespace,
    reasons: list[str],
) -> None:
    if stats["median_w"] > args.max_idle_w:
        reasons.append(
            f"{name} median {stats['median_w']:.2f}W exceeds {args.max_idle_w:.2f}W"
        )
    if stats["range80_w"] > args.max_idle_range_w:
        reasons.append(
            f"{name} p10-p90 range {stats['range80_w']:.2f}W exceeds "
            f"{args.max_idle_range_w:.2f}W"
        )


def active_command(case: benchmark.Case, clips: Sequence[benchmark.Clip]) -> tuple[list[str], list[benchmark.Clip]]:
    measured_clips = list(clips) * case.repeats
    command_clips = [clips[0], *measured_clips]
    command = [
        str(case.binary),
        "--config",
        str(case.config),
        "transcribe",
        *(str(clip.wav) for clip in command_clips),
    ]
    return command, command_clips


def evaluate_energy_case(
    case: benchmark.Case,
    clips: Sequence[benchmark.Clip],
    pre: dict[str, Any],
    active: dict[str, Any],
    post: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    settle = args.sensor_settle_seconds
    pre_stats = power_stats(pre, max(0.0, pre["elapsed_s"] - args.idle_evaluation_seconds))
    post_stats = power_stats(post, max(0.0, post["elapsed_s"] - args.idle_evaluation_seconds))
    active_stats = power_stats(active, settle)
    baseline_w = (pre_stats["median_w"] + post_stats["median_w"]) / 2
    net_power_w = active_stats["median_w"] - baseline_w
    drift_w = post_stats["median_w"] - pre_stats["median_w"]

    process = active["process"]
    if process is None:
        raise benchmark.BenchmarkError("active phase did not run a process")
    reasons: list[str] = []
    validate_idle("idle_pre", pre_stats, args, reasons)
    validate_idle("idle_post", post_stats, args, reasons)
    if process.returncode != 0:
        reasons.append(f"transcriber exited with {process.returncode}: {process.stderr[-1000:]}")
    if active["elapsed_s"] < settle + args.min_active_evaluation_seconds:
        reasons.append(
            f"active phase {active['elapsed_s']:.1f}s is too short for "
            f"{settle:.1f}s sensor settling"
        )
    if net_power_w <= 0:
        reasons.append(f"net power is not positive ({net_power_w:.2f}W)")
    if abs(drift_w) > args.max_idle_drift_w:
        reasons.append(
            f"idle drift {drift_w:+.2f}W exceeds {args.max_idle_drift_w:.2f}W"
        )
    if net_power_w > 0 and abs(drift_w) / net_power_w > args.max_drift_signal_fraction:
        reasons.append(
            f"idle drift is {abs(drift_w) / net_power_w:.0%} of the engine signal"
        )
    if any(sample["mains_online"] for sample in active["samples"] + post["samples"]):
        reasons.append("AC power appeared during the measurement")
    if any(
        sample["status"] != "Discharging"
        for sample in active["samples"] + post["samples"]
    ):
        reasons.append("battery status changed during the measurement")

    expected_clips = [clips[0], *(list(clips) * case.repeats)]
    try:
        parsed = benchmark.parse_transcriptions(process.stdout)
    except benchmark.BenchmarkError as error:
        parsed = []
        reasons.append(str(error))
    if [item.clip_id for item in parsed] != [clip.clip_id for clip in expected_clips]:
        reasons.append(
            f"output mismatch: expected {len(expected_clips)} ordered clips, parsed {len(parsed)}"
        )

    measured = parsed[1:] if len(parsed) == len(expected_clips) else []
    measured_clips = expected_clips[1:]
    audio_s = sum(clip.duration_s for clip in measured_clips)
    inference_times = [item.seconds for item in measured]
    wer_percent = (
        benchmark.corpus_wer(measured_clips, measured) if measured else float("nan")
    )

    gross_integral_j = integrate_power(active)
    average_voltage_v = statistics.mean(sample["voltage_v"] for sample in active["samples"])
    charge_delta_ah = active["charge_before_ah"] - active["charge_after_ah"]
    gross_charge_j = charge_delta_ah * 3600 * average_voltage_v
    baseline_j = baseline_w * active["elapsed_s"]
    net_integral_j = gross_integral_j - baseline_j
    net_charge_j = gross_charge_j - baseline_j
    meter_divergence = abs(gross_integral_j - gross_charge_j) / max(
        gross_integral_j, gross_charge_j, 1e-9
    )
    net_meter_divergence = abs(net_integral_j - net_charge_j) / max(
        net_integral_j, net_charge_j, 1e-9
    )
    if charge_delta_ah <= 0:
        reasons.append("charge_now did not decrease during the active phase")
    if net_integral_j <= 0:
        reasons.append(f"baseline-adjusted current integral is {net_integral_j:.0f}J")
    if net_charge_j <= 0:
        reasons.append(f"baseline-adjusted charge energy is {net_charge_j:.0f}J")
    if meter_divergence > args.max_meter_divergence:
        reasons.append(
            f"battery meters diverged by {meter_divergence:.0%}, above "
            f"{args.max_meter_divergence:.0%}"
        )
    if net_meter_divergence > args.max_net_meter_divergence:
        reasons.append(
            f"baseline-adjusted meters diverged by {net_meter_divergence:.0%}, above "
            f"{args.max_net_meter_divergence:.0%}"
        )

    valid = not reasons
    median_inference_s = statistics.median(inference_times) if inference_times else float("nan")
    # Steady-state energy per audio second: settled net watts times the time
    # spent per audio second (RTF). Excludes model load and the discarded
    # warm-up clip, so repeat counts do not change the ranking.
    total_inference_s = sum(inference_times) if inference_times else 0.0
    rtf_value = total_inference_s / audio_s if inference_times and audio_s > 0 else float("nan")
    steady_energy_per_audio_s = (
        net_power_w * rtf_value
        if valid and math.isfinite(rtf_value) and rtf_value > 0
        else float("nan")
    )
    # Full-run energy from the charge meter, including load and warm-up. Kept
    # for reporting and cross-checking, not for the winner ranking.
    joules_per_audio_s = net_charge_j / audio_s if valid and audio_s > 0 else float("nan")
    word_accuracy = max(0.0, 1.0 - wer_percent / 100.0) if valid else float("nan")

    def json_number(value: float) -> float | None:
        return value if math.isfinite(value) else None

    return {
        "label": case.label,
        "model": case.model,
        "valid": valid,
        "invalid_reasons": reasons,
        "repeats": case.repeats,
        "measured_clips": len(measured),
        "audio_s": audio_s,
        "active_wall_s": active["elapsed_s"],
        "median_inference_s": json_number(median_inference_s),
        "p90_inference_s": (
            benchmark.percentile(inference_times, 0.90) if inference_times else None
        ),
        "rtf": sum(inference_times) / audio_s if inference_times else None,
        "wer_percent": json_number(wer_percent),
        "idle_pre": pre_stats,
        "active": active_stats,
        "idle_post": post_stats,
        "baseline_w": baseline_w,
        "net_power_w": net_power_w,
        "idle_drift_w": drift_w,
        "gross_integral_j": gross_integral_j,
        "gross_charge_j": gross_charge_j,
        "baseline_energy_j": baseline_j,
        "net_integral_j": net_integral_j,
        "net_charge_j": net_charge_j,
        "meter_divergence_percent": meter_divergence * 100,
        "net_meter_divergence_percent": net_meter_divergence * 100,
        "joules_per_audio_s": json_number(joules_per_audio_s),
        "steady_energy_per_audio_s": json_number(steady_energy_per_audio_s),
        "quality_adjusted_energy_efficiency": (
            word_accuracy * audio_s / net_charge_j if valid else None
        ),
        "capacity_start_percent": active["samples"][0]["capacity_percent"],
        "capacity_end_percent": active["samples"][-1]["capacity_percent"],
        "binary": str(case.binary),
        "config": str(case.config),
        "energy_ranking_method": "baseline-adjusted charge_now energy",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Voxtype speed, WER, and battery energy use.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_energy_case,
        help="LABEL,MODEL,BINARY,CONFIG,REPEATS; repeat for each engine",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=benchmark.DEFAULT_ROOT / "bench-dictation/manifest.tsv",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start-delay", type=float, default=180.0)
    parser.add_argument("--quiet-seconds", type=float, default=30.0)
    parser.add_argument("--idle-timeout", type=float, default=900.0)
    parser.add_argument("--pre-idle-seconds", type=float, default=120.0)
    parser.add_argument("--post-idle-seconds", type=float, default=120.0)
    parser.add_argument("--idle-evaluation-seconds", type=float, default=60.0)
    parser.add_argument("--sensor-settle-seconds", type=float, default=60.0)
    parser.add_argument("--min-active-evaluation-seconds", type=float, default=30.0)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--max-idle-w", type=float, default=6.0)
    parser.add_argument("--max-idle-range-w", type=float, default=1.5)
    parser.add_argument("--max-idle-drift-w", type=float, default=1.0)
    parser.add_argument("--max-drift-signal-fraction", type=float, default=0.20)
    parser.add_argument("--max-meter-divergence", type=float, default=0.30)
    parser.add_argument("--max-net-meter-divergence", type=float, default=0.30)
    parser.add_argument("--max-cpu-percent", type=float, default=5.0)
    parser.add_argument("--minimum-start-capacity", type=int, default=75)
    parser.add_argument("--stop-capacity", type=int, default=20)
    parser.add_argument("--case-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=Path("~/.local/lib/voxtype-openvino").expanduser(),
    )
    parser.add_argument("--continue-on-invalid", action="store_true")
    parser.add_argument(
        "--minimum-energy-margin",
        type=float,
        default=0.10,
        help="fraction by which the lowest-energy case must beat the runner-up",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--log", type=Path, help="detached stdout/stderr log path")
    return parser


def validate_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    positive = (
        "quiet_seconds",
        "idle_timeout",
        "pre_idle_seconds",
        "post_idle_seconds",
        "idle_evaluation_seconds",
        "sensor_settle_seconds",
        "min_active_evaluation_seconds",
        "sample_interval",
        "max_idle_w",
        "max_idle_range_w",
        "max_idle_drift_w",
        "max_meter_divergence",
        "max_net_meter_divergence",
        "max_cpu_percent",
        "case_timeout",
    )
    if any(getattr(args, name) <= 0 for name in positive) or args.start_delay < 0:
        parser.error("durations, limits, and timeouts must be positive")
    if args.minimum_energy_margin < 0:
        parser.error("--minimum-energy-margin must be non-negative")
    if args.idle_evaluation_seconds > min(args.pre_idle_seconds, args.post_idle_seconds):
        parser.error("idle evaluation window cannot exceed an idle phase")
    if not 0 < args.max_drift_signal_fraction <= 1:
        parser.error("--max-drift-signal-fraction must be between 0 and 1")
    if not 0 < args.max_meter_divergence <= 1:
        parser.error("--max-meter-divergence must be between 0 and 1")
    if not 0 < args.max_net_meter_divergence <= 1:
        parser.error("--max-net-meter-divergence must be between 0 and 1")
    if not 0 <= args.stop_capacity < args.minimum_start_capacity <= 100:
        parser.error("battery capacity limits are inconsistent")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_arguments(args, parser)
    if args.detach:
        return benchmark.detach("energy", args.log)

    cases = args.case or default_energy_cases()
    manifest = args.manifest.expanduser()
    clips = benchmark.load_manifest(manifest)
    benchmark.validate_cases(cases)
    print(f"Validated {len(cases)} cases and {len(clips)} corpus clips.", flush=True)
    for case in cases:
        print(
            f"  {case.label}/{case.model}: repeats={case.repeats} {case.binary}",
            flush=True,
        )
    if args.dry_run:
        return 0

    battery = Battery.discover()
    if benchmark.mains_online() or battery.status() != "Discharging":
        raise benchmark.BenchmarkError("unplug AC power before starting the energy benchmark")
    capacity = battery.read_int("capacity")
    if capacity < args.minimum_start_capacity:
        raise benchmark.BenchmarkError(
            f"battery is at {capacity}%; charge to at least {args.minimum_start_capacity}%"
        )
    if args.start_delay:
        print(
            f"Starting in {args.start_delay:.0f}s. Close browsers, agents, media, and other apps now.",
            flush=True,
        )
        time.sleep(args.start_delay)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir
        else benchmark.DEFAULT_ROOT / "benchmark-runs" / f"energy-{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "results.json"
    sample_path = output_dir / "samples.csv"
    environment = benchmark.child_environment(args.library_dir.expanduser())
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "battery-energy",
        "metadata": benchmark.machine_metadata(),
        "battery": str(battery.path),
        "manifest": str(manifest),
        "quality_limits": {
            "max_idle_w": args.max_idle_w,
            "max_idle_range_w": args.max_idle_range_w,
            "max_idle_drift_w": args.max_idle_drift_w,
            "max_drift_signal_fraction": args.max_drift_signal_fraction,
            "max_meter_divergence": args.max_meter_divergence,
            "max_net_meter_divergence": args.max_net_meter_divergence,
        },
        "results": [],
    }

    with sample_path.open("w", newline="", encoding="utf-8", buffering=1) as sample_file:
        writer = csv.DictWriter(sample_file, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        wait_for_quiet(battery, args, writer, sample_file, "initial")

        for index, case in enumerate(cases, start=1):
            if battery.read_int("capacity") <= args.stop_capacity:
                raise benchmark.BenchmarkError(
                    f"battery reached the {args.stop_capacity}% safety cutoff"
                )
            print(
                f"\n[{index}/{len(cases)}] {case.label}/{case.model}", flush=True
            )
            try:
                wait_for_quiet(battery, args, writer, sample_file, f"case-{index}")
                pre = measure_phase(
                    battery,
                    f"case-{index}-idle-pre",
                    args.sample_interval,
                    writer,
                    sample_file,
                    duration=args.pre_idle_seconds,
                )
                pre_stats = power_stats(
                    pre,
                    max(0.0, pre["elapsed_s"] - args.idle_evaluation_seconds),
                )
                pre_reasons: list[str] = []
                validate_idle("idle_pre", pre_stats, args, pre_reasons)
                if pre_reasons:
                    raise benchmark.BenchmarkError("; ".join(pre_reasons))
                command, _ = active_command(case, clips)
                active = measure_phase(
                    battery,
                    f"case-{index}-active",
                    args.sample_interval,
                    writer,
                    sample_file,
                    command=command,
                    environment=environment,
                    timeout=args.case_timeout,
                )
                post = measure_phase(
                    battery,
                    f"case-{index}-idle-post",
                    args.sample_interval,
                    writer,
                    sample_file,
                    duration=args.post_idle_seconds,
                )
                result = evaluate_energy_case(case, clips, pre, active, post, args)
            except benchmark.BenchmarkError as error:
                result = {
                    "label": case.label,
                    "model": case.model,
                    "valid": False,
                    "invalid_reasons": [str(error)],
                }

            payload["results"].append(result)
            benchmark.atomic_write_json(result_path, payload)
            if result["valid"]:
                steady = result.get("steady_energy_per_audio_s")
                full = result.get("joules_per_audio_s")
                steady_text = f"{steady:.4f}" if steady is not None else "n/a"
                full_text = f"{full:.4f}" if full is not None else "n/a"
                print(
                    f"  valid: median={result['median_inference_s'] * 1000:.0f}ms "
                    f"WER={result['wer_percent']:.2f}% "
                    f"energy steady={steady_text} J/audio-s "
                    f"(full-run={full_text}) "
                    f"meters differ={result['meter_divergence_percent']:.0f}%",
                    flush=True,
                )
            else:
                print("  INVALID:", flush=True)
                for reason in result["invalid_reasons"]:
                    print(f"    - {reason}", flush=True)
                if not args.continue_on_invalid:
                    print("Stopping after invalid result. Use --continue-on-invalid to override.")
                    break

    benchmark.print_winners(
        payload["results"],
        cases,
        energy=True,
        minimum_energy_margin=args.minimum_energy_margin,
    )
    print(f"Results: {result_path}", flush=True)
    print(f"Raw samples: {sample_path}", flush=True)
    return 0 if payload["results"] and all(
        result.get("valid") for result in payload["results"]
    ) and len(payload["results"]) == len(cases) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except benchmark.BenchmarkError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
