#!/usr/bin/env python3
"""Summarize two-round VO2 fresh A/B/B0 profile runs.

The expected layout is:

    ROOT/
      round1/{release3,superlinear_off,superlinear_zero,superlinear_mix}/
      round2/{release3,superlinear_off,superlinear_zero,superlinear_mix}/

Each variant directory must contain metadata.txt, time.txt, train.log, and
trace.csv.  The script is intentionally strict: missing fields, abnormal log
tokens, failed exit codes, and inconsistent perf counters are fatal errors.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_ROUNDS = ("round1", "round2")
DEFAULT_VARIANTS = (
    "release3",
    "superlinear_off",
    "superlinear_zero",
    "superlinear_mix",
)

REQUIRED_FILES = ("metadata.txt", "time.txt", "train.log", "trace.csv")
REQUIRED_METADATA_FIELDS = ("variant", "profile_iters", "mpi_ranks", "command")
REQUIRED_TIME_FIELDS = ("runtime_sec", "exit_code")
REQUIRED_PERF_FIELDS = (
    "optimizer",
    "line_search",
    "objective_value_calls",
    "objective_gradient_calls",
    "objective_total_calls",
    "line_search_trial_evaluations",
    "rejected_trials",
    "accepted_steps",
)
REQUIRED_TRACE_FIELDS = (
    "step",
    "total_loss",
    "efs_loss",
    "energy_mae_mev_atom",
    "force_mae_mev_a",
    "stress_mae_ev",
)

ABNORMAL_PATTERNS = (
    ("NaN", re.compile(r"(?i)(^|[^A-Za-z0-9_])nan([^A-Za-z0-9_]|$)")),
    ("CUDA", re.compile(r"(?i)\bcuda\b")),
    ("traceback", re.compile(r"(?i)traceback")),
    ("segfault", re.compile(r"(?i)\b(segmentation fault|segfault)\b")),
    ("core dumped", re.compile(r"(?i)core dumped")),
    ("bus error", re.compile(r"(?i)\bbus error\b")),
    ("floating point exception", re.compile(r"(?i)floating point exception")),
    ("non-finite", re.compile(r"(?i)non[- ]finite")),
    ("abort", re.compile(r"(?i)\babort(?:ed)?\b")),
    ("exception", re.compile(r"(?i)\b(exception|MlipException)\b")),
    ("bad_alloc", re.compile(r"(?i)bad_alloc")),
)


class ProfileError(RuntimeError):
    """Raised for invalid or incomplete profile artifacts."""


@dataclass(frozen=True)
class RunRecord:
    round_name: str
    variant: str
    path: Path
    timestamp: datetime
    runtime_sec: float
    exit_code: int
    profile_iters: int
    mpi_ranks: int
    optimizer: str
    line_search: str
    objective_value_calls: int
    objective_gradient_calls: int
    objective_total_calls: int
    line_search_trial_evaluations: int
    rejected_trials: int
    accepted_steps: int
    trace_rows: int

    @property
    def sec_per_objective_total_call(self) -> float:
        return self.runtime_sec / checked_positive(self.objective_total_calls, "objective_total_calls")

    @property
    def sec_per_gradient_call(self) -> float:
        return self.runtime_sec / checked_positive(self.objective_gradient_calls, "objective_gradient_calls")

    @property
    def sec_per_accepted_step(self) -> float:
        return self.runtime_sec / checked_positive(self.accepted_steps, "accepted_steps")


def checked_positive(value: int, name: str) -> int:
    if value <= 0:
        raise ProfileError(f"{name} must be positive, got {value}")
    return value


def parse_iso_timestamp(line: str, path: Path) -> datetime:
    value = line.strip()
    if not value:
        raise ProfileError(f"{path}: metadata first line is empty; expected ISO timestamp")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ProfileError(f"{path}: metadata first line is not an ISO timestamp: {line!r}") from exc
    return parsed


def parse_key_values(path: Path) -> tuple[datetime, dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ProfileError(f"{path}: file is empty")
    timestamp = parse_iso_timestamp(lines[0], path)
    values: dict[str, str] = {}
    for line in lines[1:]:
        if not line.strip() or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return timestamp, values


def require_fields(path: Path, values: dict[str, str], fields: Iterable[str]) -> None:
    missing = [field for field in fields if field not in values or values[field] == ""]
    if missing:
        raise ProfileError(f"{path}: missing required field(s): {', '.join(missing)}")


def parse_int_field(path: Path, values: dict[str, str], key: str) -> int:
    try:
        return int(values[key])
    except ValueError as exc:
        raise ProfileError(f"{path}: field {key} is not an integer: {values[key]!r}") from exc


def parse_float_field(path: Path, values: dict[str, str], key: str) -> float:
    try:
        result = float(values[key])
    except ValueError as exc:
        raise ProfileError(f"{path}: field {key} is not a float: {values[key]!r}") from exc
    if not math.isfinite(result):
        raise ProfileError(f"{path}: field {key} is non-finite: {values[key]!r}")
    return result


def scan_log(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    text_for_abnormal_scan = text
    if re.search(r"(?m)^center_std weight:\s*0\s*$", text):
        text_for_abnormal_scan = re.sub(
            r"(?m)(center_std\^2:)-?nan(\s)",
            r"\1disabled\2",
            text,
        )
    for label, pattern in ABNORMAL_PATTERNS:
        match = pattern.search(text_for_abnormal_scan)
        if match:
            line_no = text_for_abnormal_scan.count("\n", 0, match.start()) + 1
            raise ProfileError(f"{path}: abnormal log token {label!r} at line {line_no}")

    perf_lines = [
        line for line in text.splitlines()
        if "SUS2_MTPR_OPTIMIZER_PERF_COUNTERS" in line
    ]
    if len(perf_lines) != 1:
        raise ProfileError(
            f"{path}: expected exactly one SUS2_MTPR_OPTIMIZER_PERF_COUNTERS line, "
            f"found {len(perf_lines)}"
        )
    fields = dict(re.findall(r"([A-Za-z0-9_]+)=([^ \t\r\n]+)", perf_lines[0]))
    missing = [field for field in REQUIRED_PERF_FIELDS if field not in fields]
    if missing:
        raise ProfileError(f"{path}: perf counter line missing field(s): {', '.join(missing)}")
    return fields


def parse_trace(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ProfileError(f"{path}: trace.csv is empty")
        missing = [field for field in REQUIRED_TRACE_FIELDS if field not in reader.fieldnames]
        if missing:
            raise ProfileError(f"{path}: trace.csv missing column(s): {', '.join(missing)}")
        rows = 0
        for row_no, row in enumerate(reader, start=2):
            rows += 1
            for field in REQUIRED_TRACE_FIELDS:
                value = row.get(field)
                if value is None or value == "":
                    raise ProfileError(f"{path}: row {row_no} missing value for {field}")
                try:
                    parsed = float(value)
                except ValueError as exc:
                    raise ProfileError(f"{path}: row {row_no} field {field} is not numeric: {value!r}") from exc
                if not math.isfinite(parsed):
                    raise ProfileError(f"{path}: row {row_no} field {field} is non-finite: {value!r}")
    if rows == 0:
        raise ProfileError(f"{path}: trace.csv has no accepted-step rows")
    return rows


def parse_run(root: Path, round_name: str, variant: str) -> RunRecord:
    run_dir = root / round_name / variant
    if not run_dir.is_dir():
        raise ProfileError(f"missing variant directory: {run_dir}")
    for filename in REQUIRED_FILES:
        path = run_dir / filename
        if not path.is_file():
            raise ProfileError(f"{run_dir}: missing required file {filename}")
        if path.stat().st_size == 0:
            raise ProfileError(f"{path}: required file is empty")

    metadata_path = run_dir / "metadata.txt"
    timestamp, metadata = parse_key_values(metadata_path)
    require_fields(metadata_path, metadata, REQUIRED_METADATA_FIELDS)
    if metadata["variant"] != variant:
        raise ProfileError(
            f"{metadata_path}: variant field {metadata['variant']!r} does not match directory {variant!r}"
        )

    time_path = run_dir / "time.txt"
    _, time_values = parse_key_values_with_optional_timestamp(time_path)
    require_fields(time_path, time_values, REQUIRED_TIME_FIELDS)
    runtime_sec = parse_float_field(time_path, time_values, "runtime_sec")
    exit_code = parse_int_field(time_path, time_values, "exit_code")
    if exit_code != 0:
        raise ProfileError(f"{time_path}: exit_code is {exit_code}, expected 0")

    perf = scan_log(run_dir / "train.log")
    objective_value_calls = int_perf(run_dir / "train.log", perf, "objective_value_calls")
    objective_gradient_calls = int_perf(run_dir / "train.log", perf, "objective_gradient_calls")
    objective_total_calls = int_perf(run_dir / "train.log", perf, "objective_total_calls")
    if objective_total_calls != objective_value_calls + objective_gradient_calls:
        raise ProfileError(
            f"{run_dir / 'train.log'}: objective_total_calls={objective_total_calls} "
            f"but value+gradient={objective_value_calls + objective_gradient_calls}"
        )

    trace_rows = parse_trace(run_dir / "trace.csv")
    accepted_steps = int_perf(run_dir / "train.log", perf, "accepted_steps")
    if trace_rows != accepted_steps:
        raise ProfileError(
            f"{run_dir}: trace rows {trace_rows} != accepted_steps {accepted_steps}"
        )

    return RunRecord(
        round_name=round_name,
        variant=variant,
        path=run_dir,
        timestamp=timestamp,
        runtime_sec=runtime_sec,
        exit_code=exit_code,
        profile_iters=parse_int_field(metadata_path, metadata, "profile_iters"),
        mpi_ranks=parse_int_field(metadata_path, metadata, "mpi_ranks"),
        optimizer=perf["optimizer"],
        line_search=perf["line_search"],
        objective_value_calls=objective_value_calls,
        objective_gradient_calls=objective_gradient_calls,
        objective_total_calls=objective_total_calls,
        line_search_trial_evaluations=int_perf(run_dir / "train.log", perf, "line_search_trial_evaluations"),
        rejected_trials=int_perf(run_dir / "train.log", perf, "rejected_trials"),
        accepted_steps=accepted_steps,
        trace_rows=trace_rows,
    )


def parse_key_values_with_optional_timestamp(path: Path) -> tuple[datetime | None, dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ProfileError(f"{path}: file is empty")
    timestamp: datetime | None = None
    values: dict[str, str] = {}
    start = 0
    if "=" not in lines[0]:
        timestamp = parse_iso_timestamp(lines[0], path)
        start = 1
    for line in lines[start:]:
        if not line.strip() or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return timestamp, values


def int_perf(path: Path, values: dict[str, str], key: str) -> int:
    try:
        result = int(values[key])
    except ValueError as exc:
        raise ProfileError(f"{path}: perf field {key} is not an integer: {values[key]!r}") from exc
    if result < 0:
        raise ProfileError(f"{path}: perf field {key} is negative: {result}")
    return result


def fmt_float(value: float) -> str:
    return f"{value:.6g}"


def print_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    widths = [len(header) for header in headers]
    rendered_rows: list[tuple[str, ...]] = []
    for row in rows:
        rendered = tuple(str(item) for item in row)
        rendered_rows.append(rendered)
        for i, item in enumerate(rendered):
            widths[i] = max(widths[i], len(item))
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rendered_rows:
        print("  ".join(item.ljust(widths[i]) for i, item in enumerate(row)))


def summarize(records: list[RunRecord], rounds: tuple[str, str], variants: tuple[str, ...]) -> None:
    for field in ("profile_iters", "mpi_ranks", "optimizer", "line_search"):
        values = {getattr(record, field) for record in records}
        if len(values) != 1:
            rendered = ", ".join(repr(value) for value in sorted(values, key=str))
            raise ProfileError(f"inconsistent {field} across runs: {rendered}")

    by_round: dict[str, list[RunRecord]] = {}
    for round_name in rounds:
        round_records = [record for record in records if record.round_name == round_name]
        timestamps = [record.timestamp for record in round_records]
        if len(set(timestamps)) != len(timestamps):
            raise ProfileError(f"{round_name}: duplicate metadata timestamps; cannot determine run order")
        by_round[round_name] = sorted(round_records, key=lambda record: record.timestamp)

    order1 = [record.variant for record in by_round[rounds[0]]]
    order2 = [record.variant for record in by_round[rounds[1]]]
    print(f"round_order[{rounds[0]}]=" + " -> ".join(order1))
    print(f"round_order[{rounds[1]}]=" + " -> ".join(order2))
    print("round_order_delta=" + ("same" if order1 == order2 else "different"))
    print()

    order_index = {
        (record.round_name, record.variant): index + 1
        for round_name, ordered in by_round.items()
        for index, record in enumerate(ordered)
    }
    run_rows: list[tuple[object, ...]] = []
    for record in sorted(records, key=lambda item: (item.round_name, order_index[(item.round_name, item.variant)])):
        run_rows.append((
            record.round_name,
            order_index[(record.round_name, record.variant)],
            record.variant,
            fmt_float(record.runtime_sec),
            record.objective_total_calls,
            record.objective_gradient_calls,
            record.accepted_steps,
            fmt_float(record.sec_per_objective_total_call),
            fmt_float(record.sec_per_gradient_call),
            fmt_float(record.sec_per_accepted_step),
            record.line_search_trial_evaluations,
            record.rejected_trials,
        ))
    print_table(
        (
            "round",
            "order",
            "variant",
            "runtime_s",
            "objective_total_calls",
            "gradient_calls",
            "accepted_steps",
            "s/objective",
            "s/gradient",
            "s/accepted",
            "line_search_trials",
            "rejected_trials",
        ),
        run_rows,
    )
    print()

    variant_rows: list[tuple[object, ...]] = []
    for variant in variants:
        variant_records = [record for record in records if record.variant == variant]
        if len(variant_records) != 2:
            raise ProfileError(f"{variant}: expected two records, found {len(variant_records)}")
        r1 = next(record for record in variant_records if record.round_name == rounds[0])
        r2 = next(record for record in variant_records if record.round_name == rounds[1])
        runtime_avg = 0.5 * (r1.runtime_sec + r2.runtime_sec)
        delta_pct = 100.0 * (r2.runtime_sec - r1.runtime_sec) / r1.runtime_sec
        variant_rows.append((
            variant,
            fmt_float(r1.runtime_sec),
            fmt_float(r2.runtime_sec),
            fmt_float(runtime_avg),
            fmt_float(delta_pct),
            fmt_float(0.5 * (r1.sec_per_objective_total_call + r2.sec_per_objective_total_call)),
            fmt_float(0.5 * (r1.sec_per_gradient_call + r2.sec_per_gradient_call)),
            fmt_float(0.5 * (r1.sec_per_accepted_step + r2.sec_per_accepted_step)),
        ))
    print_table(
        (
            "variant",
            f"{rounds[0]}_runtime_s",
            f"{rounds[1]}_runtime_s",
            "avg_runtime_s",
            "round2_vs_round1_%",
            "avg_s/objective",
            "avg_s/gradient",
            "avg_s/accepted",
        ),
        variant_rows,
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize strict two-round VO2 fresh profile artifacts."
    )
    parser.add_argument("root", type=Path, help="profile root containing round1/ and round2/")
    parser.add_argument(
        "--rounds",
        default=",".join(DEFAULT_ROUNDS),
        help="comma-separated round directory names; exactly two required",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="comma-separated variant directory names; default is VO2 fresh A/B/B0 set",
    )
    args = parser.parse_args(argv)

    rounds = tuple(item.strip() for item in args.rounds.split(",") if item.strip())
    if len(rounds) != 2:
        raise ProfileError("--rounds must name exactly two round directories")
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    if not variants:
        raise ProfileError("--variants must name at least one variant")

    records = [
        parse_run(args.root, round_name, variant)
        for round_name in rounds
        for variant in variants
    ]
    summarize(records, rounds, variants)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except ProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
