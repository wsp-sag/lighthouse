#!/usr/bin/env python3
"""Benchmark full Boston ActivitySim runs with external memory monitoring.

The script first synchronizes the locked uv environment, then runs:

1. A 5,000-household, single-process Sharrow cache warm-up.
2. A full-population, multiprocess run with Sharrow required.
3. A full-population, multiprocess run without Sharrow.

The two full runs are monitored by this parent process.  Results are written to
a self-contained HTML report and CSV/JSON data files under ``model/output``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import warnings
import webbrowser
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ENV_READY = "LIGHTHOUSE_DIAGNOSTIC_UV_READY"
BYTES_PER_GIB = 1024**3
DEFAULT_SAMPLE_INTERVAL = 1.0


@dataclass
class RunResult:
    """Measurements and output locations for one child model run."""

    key: str
    label: str
    sharrow: str | bool
    multiprocess: bool
    household_sample_size: int
    output_dir: str
    log_file: str
    started_at: str
    duration_seconds: float
    return_code: int
    samples: list[dict[str, float | int]] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.return_code == 0

    @property
    def peak_rss_bytes(self) -> int:
        return max((int(x["rss_bytes"]) for x in self.samples), default=0)

    @property
    def peak_uss_bytes(self) -> int:
        return max((int(x["uss_bytes"]) for x in self.samples), default=0)

    @property
    def uss_available(self) -> bool:
        populated = [x for x in self.samples if int(x["process_count"]) > 0]
        return bool(populated) and all(
            int(x.get("uss_process_count", 0)) == int(x["process_count"])
            for x in populated
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run cache warm-up, Sharrow, and non-Sharrow Boston model diagnostics "
            "and create an HTML runtime/memory report."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=(
            "directory containing full-scale skims and population inputs; "
            "defaults to the repository's smaller model/data subarea"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "new directory for model outputs and the report; defaults to a "
            "timestamped directory under model/output/diagnostics"
        ),
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=DEFAULT_SAMPLE_INTERVAL,
        metavar="SECONDS",
        help="external memory sampling interval (default: %(default)s second)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress messages and do not open the HTML report",
    )
    parser.add_argument("--_worker-spec", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def venv_python(repo_root: Path) -> Path:
    if os.name == "nt":
        return repo_root / ".venv" / "Scripts" / "python.exe"
    return repo_root / ".venv" / "bin" / "python"


def sync_locked_environment(repo_root: Path, quiet: bool) -> None:
    """Run uv sync once, then replace this process with the locked interpreter."""

    if os.environ.get(ENV_READY) == "1":
        return

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required but was not found on PATH")

    if not quiet:
        print("Synchronizing the locked uv environment...", flush=True)
    subprocess.run([uv, "sync", "--locked"], cwd=repo_root, check=True)

    python = venv_python(repo_root)
    if not python.is_file():
        raise RuntimeError(f"uv did not create the expected interpreter: {python}")

    env = os.environ.copy()
    env[ENV_READY] = "1"
    os.execve(
        str(python),
        [str(python), str(Path(__file__).resolve()), *sys.argv[1:]],
        env,
    )


def validate_data_dir(data_dir: Path) -> None:
    if not data_dir.is_dir():
        raise NotADirectoryError(f"data directory does not exist: {data_dir}")

    missing: list[str] = []
    for table in ("households", "persons", "land_use"):
        if not any(data_dir.glob(f"{table}.*")):
            missing.append(f"{table}.csv or {table}.parquet")
    for skim_prefix in ("hwy", "ta", "tw", "nm"):
        if not any(data_dir.glob(f"{skim_prefix}*.omx")):
            missing.append(f"{skim_prefix}*.omx")

    if missing:
        raise FileNotFoundError(
            f"data directory {data_dir} is missing required inputs: "
            + ", ".join(missing)
        )


def make_results_dir(repo_root: Path, requested: Path | None) -> Path:
    if requested is None:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        result = repo_root / "model" / "output" / "diagnostics" / stamp
    else:
        result = requested.expanduser().resolve()

    if result.exists() and any(result.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {result}")
    result.mkdir(parents=True, exist_ok=True)
    return result


def diagnostic_settings() -> dict[str, Any]:
    """Settings that keep ActivitySim's internal diagnostics out of benchmarks."""

    return {
        "instrument": False,
        "memory_profile": False,
        "expression_profile": False,
        "check_for_variability": False,
        "log_alt_losers": False,
        "want_dest_choice_sample_tables": False,
        "trace_hh_id": None,
        "trace_od": None,
        "keep_chunk_logs": False,
        "keep_mem_logs": False,
    }


@contextmanager
def synchronous_sharrow_skim_loading():
    """Force Dask's synchronous scheduler while loading Sharrow skim data."""

    from sharrow.shared_memory import SharedMemDatasetAccessor

    original = SharedMemDatasetAccessor.to_shared_memory

    def to_shared_memory(self, *args, **kwargs):
        kwargs["dask_scheduler"] = "synchronous"
        return original(self, *args, **kwargs)

    SharedMemDatasetAccessor.to_shared_memory = to_shared_memory
    try:
        yield
    finally:
        SharedMemDatasetAccessor.to_shared_memory = original


def run_model_worker(spec_path: Path) -> int:
    """Internal child-process entry point; invoked by the benchmark harness."""

    for name in (
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[name] = "1"

    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    import activitysim
    from activitysim.core.workflow import State

    model_dir = Path(spec["model_dir"])
    configs: tuple[Path, ...]
    if spec["multiprocess"]:
        configs = (model_dir / "configs_mp", model_dir / "configs")
    else:
        configs = (model_dir / "configs",)

    settings = diagnostic_settings()
    settings.update(
        {
            "households_sample_size": int(spec["household_sample_size"]),
            "multiprocess": bool(spec["multiprocess"]),
            "sharrow": spec["sharrow"],
        }
    )

    state = State.make_default(
        working_dir=model_dir,
        configs_dir=configs,
        data_dir=Path(spec["data_dir"]),
        output_dir=Path(spec["output_dir"]),
        cache_dir=Path(spec["output_dir"]) / "cache",
        settings=settings,
    )
    # The ActivitySim CLI normally supplies these values.  State.run.all()
    # forwards them when it creates multiprocessing workers.
    state.set("imported_extensions", ())
    state.set("run_timestamp", dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # track_skim_usage is an internal diagnostic step, not a model component.
    state.settings.models = [
        model for model in state.settings.models if model != "track_skim_usage"
    ]
    state.settings.sharrow = spec["sharrow"]

    if spec["sharrow"]:
        state.filesystem.persist_sharrow_cache()

    print(f"ActivitySim version: {activitysim.__version__}", flush=True)
    print(
        "Diagnostic settings: instrument=False, memory_profile=False, "
        "expression_profile=False, track_skim_usage omitted",
        flush=True,
    )

    skim_loading = (
        synchronous_sharrow_skim_loading() if spec["sharrow"] else nullcontext()
    )
    with skim_loading:
        state.run.all()

    if not spec["multiprocess"]:
        if state.settings.cleanup_pipeline_after_run:
            state.checkpoint.cleanup()
        else:
            state.checkpoint.close_store()
    return 0


def write_worker_spec(
    result_dir: Path,
    *,
    model_dir: Path,
    data_dir: Path,
    output_dir: Path,
    household_sample_size: int,
    sharrow: str | bool,
    multiprocess: bool,
) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = result_dir / "run-spec.json"
    spec = {
        "model_dir": str(model_dir),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "household_sample_size": household_sample_size,
        "sharrow": sharrow,
        "multiprocess": multiprocess,
    }
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return spec_path


def process_tree_memory(pid: int) -> dict[str, int]:
    """Return externally measured aggregate RSS and USS for pid and descendants."""

    import psutil

    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {
            "rss_bytes": 0,
            "uss_bytes": 0,
            "process_count": 0,
            "uss_process_count": 0,
        }

    rss = 0
    uss = 0
    count = 0
    uss_count = 0
    seen: set[int] = set()
    for process in processes:
        if process.pid in seen:
            continue
        seen.add(process.pid)
        try:
            info = process.memory_full_info()
            rss += int(info.rss)
            if hasattr(info, "uss"):
                uss += int(info.uss)
                uss_count += 1
            count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, RuntimeError):
            # macOS commonly permits external RSS access while denying the
            # task inspection needed for USS.  Preserve the available metric.
            try:
                info = process.memory_info()
                rss += int(info.rss)
                count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, RuntimeError):
                continue
    return {
        "rss_bytes": rss,
        "uss_bytes": uss,
        "process_count": count,
        "uss_process_count": uss_count,
    }


def launch_run(
    *,
    key: str,
    label: str,
    sharrow: str | bool,
    multiprocess: bool,
    household_sample_size: int,
    model_dir: Path,
    data_dir: Path,
    result_root: Path,
    sample_interval: float,
    monitor_memory: bool,
    quiet: bool,
) -> RunResult:
    run_dir = result_root / key
    model_output_dir = run_dir / "model-output"
    spec_path = write_worker_spec(
        run_dir,
        model_dir=model_dir,
        data_dir=data_dir,
        output_dir=model_output_dir,
        household_sample_size=household_sample_size,
        sharrow=sharrow,
        multiprocess=multiprocess,
    )
    log_path = run_dir / "console.log"
    command = [
        str(venv_python(repository_root())),
        str(Path(__file__).resolve()),
        "--_worker-spec",
        str(spec_path),
        "--quiet",
    ]
    env = os.environ.copy()
    env[ENV_READY] = "1"
    for name in (
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        env[name] = "1"

    if not quiet:
        scope = "all households" if household_sample_size == 0 else f"{household_sample_size:,} households"
        process_mode = "multiprocess" if multiprocess else "single process"
        print(f"Starting {label}: {process_mode}, {scope}...", flush=True)

    started_wall = dt.datetime.now(dt.timezone.utc)
    started = time.perf_counter()
    samples: list[dict[str, float | int]] = []
    last_progress = started

    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            command,
            cwd=repository_root(),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        while True:
            now = time.perf_counter()
            return_code = process.poll()
            if return_code is not None:
                break

            if monitor_memory:
                sample = process_tree_memory(process.pid)
                sample["elapsed_seconds"] = round(now - started, 3)
                samples.append(sample)

            if not quiet and monitor_memory and now - last_progress >= 60:
                latest = samples[-1]
                uss_text = (
                    format_gib(int(latest["uss_bytes"]))
                    if latest["uss_process_count"] == latest["process_count"]
                    else "unavailable"
                )
                print(
                    f"  {label}: {format_duration(now - started)}, "
                    f"tree RSS {format_gib(int(latest['rss_bytes']))}, "
                    f"USS {uss_text}, "
                    f"{latest['process_count']} processes",
                    flush=True,
                )
                last_progress = now
            try:
                return_code = process.wait(timeout=sample_interval)
                break
            except subprocess.TimeoutExpired:
                continue

    duration = time.perf_counter() - started
    result = RunResult(
        key=key,
        label=label,
        sharrow=sharrow,
        multiprocess=multiprocess,
        household_sample_size=household_sample_size,
        output_dir=str(model_output_dir),
        log_file=str(log_path),
        started_at=started_wall.isoformat(),
        duration_seconds=duration,
        return_code=return_code,
        samples=samples,
    )

    if monitor_memory:
        write_samples_csv(run_dir / "memory-samples.csv", result.samples)

    if not quiet:
        status = "completed" if result.succeeded else f"failed (exit {return_code})"
        memory = f", peak RSS {format_gib(result.peak_rss_bytes)}" if monitor_memory else ""
        print(f"Finished {label}: {status} in {format_duration(duration)}{memory}", flush=True)
    return result


def write_samples_csv(path: Path, samples: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "elapsed_seconds",
                "rss_bytes",
                "uss_bytes",
                "process_count",
                "uss_process_count",
            ),
        )
        writer.writeheader()
        writer.writerows(samples)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{whole_seconds:02d}"
    return f"{minutes:d}:{whole_seconds:02d}"


def format_gib(size: int) -> str:
    return f"{size / BYTES_PER_GIB:.2f} GiB"


def svg_memory_chart(results: list[RunResult]) -> str:
    measured = [result for result in results if result.samples]
    if not measured:
        return '<p class="empty">No memory samples were recorded.</p>'

    width, height = 1120, 500
    left, right, top, bottom = 76, 30, 34, 62
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_x = max(float(s["elapsed_seconds"]) for r in measured for s in r.samples)
    max_y_bytes = max(int(s["rss_bytes"]) for r in measured for s in r.samples)
    max_x = max(max_x, 1.0)
    max_y_gib = max(max_y_bytes / BYTES_PER_GIB, 0.1)
    y_axis_max = math.ceil(max_y_gib / 5.0) * 5.0 if max_y_gib > 5 else math.ceil(max_y_gib)
    y_axis_max = max(y_axis_max, 1.0)

    def x_pos(seconds: float) -> float:
        return left + seconds / max_x * plot_width

    def y_pos(size: int) -> float:
        return top + (1.0 - (size / BYTES_PER_GIB) / y_axis_max) * plot_height

    def path_for(result: RunResult, field_name: str) -> str:
        points = [
            f"{x_pos(float(sample['elapsed_seconds'])):.1f},{y_pos(int(sample[field_name])):.1f}"
            for sample in result.samples
        ]
        return "M " + " L ".join(points)

    colors = {"full-sharrow": "#2563eb", "full-no-sharrow": "#ea580c"}
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Process-tree memory over elapsed runtime">',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
        'fill="#ffffff" stroke="#d7dee8"/>',
    ]

    for tick in range(6):
        y_value = y_axis_max * tick / 5
        y = y_pos(int(y_value * BYTES_PER_GIB))
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" '
            'stroke="#e8edf3"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end">{y_value:.1f}</text>'
        )

    for tick in range(6):
        seconds = max_x * tick / 5
        x = x_pos(seconds)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" '
            'stroke="#eef2f6"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 25}" text-anchor="middle">'
            f'{seconds / 60:.1f}</text>'
        )

    for result in measured:
        color = colors.get(result.key, "#475569")
        parts.append(
            f'<path d="{path_for(result, "rss_bytes")}" fill="none" stroke="{color}" '
            'stroke-width="2.5" vector-effect="non-scaling-stroke"/>'
        )
        if result.uss_available:
            parts.append(
                f'<path d="{path_for(result, "uss_bytes")}" fill="none" stroke="{color}" '
                'stroke-width="1.8" stroke-dasharray="7 5" opacity="0.75" '
                'vector-effect="non-scaling-stroke"/>'
            )

    parts.extend(
        [
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 12}" text-anchor="middle" '
            'class="axis-title">Elapsed runtime (minutes)</text>',
            f'<text x="18" y="{top + plot_height / 2:.1f}" text-anchor="middle" '
            'transform="rotate(-90 18 '
            f'{top + plot_height / 2:.1f})" class="axis-title">Memory (GiB)</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def runtime_table(results: list[RunResult], report_dir: Path) -> str:
    measured = [result for result in results if result.multiprocess]
    successful = [result.duration_seconds for result in measured if result.succeeded]
    fastest = min(successful, default=0.0)
    rows: list[str] = []
    for result in measured:
        relative = result.duration_seconds / fastest if fastest and result.succeeded else None
        difference = result.duration_seconds - fastest if fastest and result.succeeded else None
        log_link = os.path.relpath(result.log_file, report_dir)
        relative_cell = f"<td>{relative:.3f}×</td>" if relative is not None else "<td>—</td>"
        difference_cell = (
            f"<td>{difference:+.1f} s</td>" if difference is not None else "<td>—</td>"
        )
        uss_cell = (
            f"<td>{format_gib(result.peak_uss_bytes)}</td>"
            if result.uss_available
            else "<td>Unavailable</td>"
        )
        row = (
            "<tr>"
            f"<td>{html.escape(result.label)}</td>"
            f"<td>{'require' if result.sharrow else 'False'}</td>"
            f"<td>{'Completed' if result.succeeded else f'Failed ({result.return_code})'}</td>"
            f"<td>{format_duration(result.duration_seconds)}</td>"
            f"{relative_cell}"
            f"{difference_cell}"
            f"<td>{format_gib(result.peak_rss_bytes)}</td>"
            f"{uss_cell}"
            f'<td><a href="{html.escape(log_link)}">console log</a></td>'
            "</tr>"
        )
        rows.append(row)

    return (
        "<table><thead><tr>"
        "<th>Run</th><th>Sharrow</th><th>Status</th><th>Runtime</th>"
        "<th>Relative runtime</th><th>Difference</th><th>Peak tree RSS</th>"
        "<th>Peak tree USS</th><th>Log</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def write_html_report(
    report_path: Path,
    *,
    results: list[RunResult],
    data_dir: Path,
    using_subarea: bool,
    sample_interval: float,
    activitysim_version: str,
    process_count: int | None,
) -> None:
    warmup = next((result for result in results if result.key == "cache-warmup"), None)
    scope = "Repository subarea data" if using_subarea else "Full-scale data"
    scope_class = "warning" if using_subarea else "ok"
    process_text = str(process_count) if process_count else "from configs_mp"
    generated = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    warmup_text = (
        f"{'Completed' if warmup.succeeded else 'Failed'} in {format_duration(warmup.duration_seconds)}"
        if warmup
        else "Not run"
    )

    legend_items = [
        '<span><i style="background:#2563eb"></i>Sharrow RSS</span>',
        '<span><i style="background:#ea580c"></i>No-Sharrow RSS</span>',
    ]
    if any(result.uss_available for result in results):
        legend_items.extend(
            [
                '<span><i class="dash" style="border-color:#2563eb"></i>Sharrow USS</span>',
                '<span><i class="dash" style="border-color:#ea580c"></i>No-Sharrow USS</span>',
            ]
        )
    legend = "".join(legend_items)
    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Boston model performance diagnostics</title>
<style>
:root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; color:#172033; background:#f5f7fa; }}
body {{ margin:0; padding:32px; }}
main {{ max-width:1180px; margin:auto; }}
h1 {{ margin:0 0 8px; font-size:30px; }}
h2 {{ margin-top:34px; font-size:20px; }}
.subtitle, .note {{ color:#536174; line-height:1.55; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; margin:24px 0; }}
.card, .chart {{ background:white; border:1px solid #dce3eb; border-radius:12px; padding:18px; box-shadow:0 2px 8px #1520330a; }}
.card .name {{ color:#64748b; font-size:12px; text-transform:uppercase; letter-spacing:.06em; }}
.card .value {{ margin-top:7px; font-size:18px; font-weight:650; overflow-wrap:anywhere; }}
.warning {{ color:#9a3412; }} .ok {{ color:#166534; }}
table {{ width:100%; border-collapse:collapse; background:white; border:1px solid #dce3eb; border-radius:12px; overflow:hidden; display:block; overflow-x:auto; }}
th, td {{ padding:11px 13px; border-bottom:1px solid #e5eaf0; text-align:right; white-space:nowrap; }}
th {{ background:#f8fafc; color:#475569; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) {{ text-align:left; }}
a {{ color:#1d4ed8; }}
.chart svg {{ width:100%; height:auto; }}
.chart text {{ fill:#667085; font-size:12px; }} .chart .axis-title {{ font-size:13px; font-weight:600; }}
.legend {{ display:flex; flex-wrap:wrap; gap:18px; margin:0 0 12px; color:#526071; font-size:13px; }}
.legend span {{ display:flex; align-items:center; gap:7px; }} .legend i {{ width:22px; height:3px; display:inline-block; }}
.legend i.dash {{ height:0; background:none !important; border-top:2px dashed; }}
.empty {{ padding:40px; text-align:center; color:#64748b; }}
code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.92em; }}
</style>
</head>
<body><main>
<h1>Boston model performance diagnostics</h1>
<p class="subtitle">Generated {html.escape(generated)}. Runtime is measured around each external child process; memory is sampled from the child and its complete descendant process tree.</p>
<section class="cards">
  <div class="card"><div class="name">Data scope</div><div class="value {scope_class}">{html.escape(scope)}</div></div>
  <div class="card"><div class="name">ActivitySim</div><div class="value">{html.escape(activitysim_version)}</div></div>
  <div class="card"><div class="name">Multiprocess workers</div><div class="value">{html.escape(process_text)}</div></div>
  <div class="card"><div class="name">Cache warm-up</div><div class="value">{html.escape(warmup_text)}</div></div>
</section>
<h2>Runtime comparison</h2>
{runtime_table(results, report_path.parent)}
<h2>Progressive memory usage</h2>
<div class="chart"><div class="legend">{legend}</div>{svg_memory_chart(results)}</div>
<p class="note"><strong>Memory interpretation.</strong> Tree RSS is the sum of resident memory reported for the controller and all workers and can count shared skim pages more than once. Where the operating system permits external access, tree USS is also shown as the sum of private memory and excludes shared pages. macOS commonly restricts child-process USS, in which case the report marks it unavailable instead of substituting an internal profiler. Samples were taken externally every {sample_interval:g} seconds.</p>
<p class="note"><strong>Benchmark controls.</strong> Both measured runs use all households (<code>households_sample_size: 0</code>) and the repository's native multiprocessing configuration. ActivitySim instrumentation, memory profiling, expression profiling, trace output, variability checking, loser logging, and <code>track_skim_usage</code> are disabled.</p>
<p class="note"><strong>Data directory.</strong> <code>{html.escape(str(data_dir))}</code></p>
<p class="note"><strong>Host.</strong> {html.escape(platform.platform())}; {os.cpu_count() or 'unknown'} logical CPUs.</p>
</main></body></html>
"""
    report_path.write_text(report, encoding="utf-8")


def write_json_summary(path: Path, results: list[RunResult], metadata: dict[str, Any]) -> None:
    payload = {
        "metadata": metadata,
        "runs": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_mp_process_count(model_dir: Path) -> int | None:
    try:
        import yaml

        settings = yaml.safe_load(
            (model_dir / "configs_mp" / "settings.yaml").read_text(encoding="utf-8")
        )
        return int(settings.get("num_processes")) if settings.get("num_processes") else None
    except (OSError, TypeError, ValueError):
        return None


def activitysim_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("activitysim")
    except PackageNotFoundError:
        return "unknown"


def run_benchmarks(args: argparse.Namespace, repo_root: Path) -> int:
    if args.sample_interval <= 0:
        raise ValueError("--sample-interval must be greater than zero")

    if args.data_dir is None:
        data_dir = (repo_root / "model" / "data").resolve()
        using_subarea = True
        warnings.warn(
            "--data-dir was not supplied; only the smaller repository subarea data "
            "will be tested.",
            stacklevel=2,
        )
    else:
        data_dir = args.data_dir.expanduser().resolve()
        using_subarea = False
    validate_data_dir(data_dir)

    result_root = make_results_dir(repo_root, args.output_dir)
    model_dir = repo_root / "model"
    definitions = [
        {
            "key": "cache-warmup",
            "label": "Sharrow cache warm-up",
            "sharrow": "require",
            "multiprocess": False,
            "household_sample_size": 5_000,
            "monitor_memory": False,
        },
        {
            "key": "full-sharrow",
            "label": "Full model — Sharrow required",
            "sharrow": "require",
            "multiprocess": True,
            "household_sample_size": 0,
            "monitor_memory": True,
        },
        {
            "key": "full-no-sharrow",
            "label": "Full model — Sharrow disabled",
            "sharrow": False,
            "multiprocess": True,
            "household_sample_size": 0,
            "monitor_memory": True,
        },
    ]

    if not args.quiet:
        print(f"Data directory: {data_dir}", flush=True)
        print(f"Diagnostic outputs: {result_root}", flush=True)

    results: list[RunResult] = []
    for definition in definitions:
        result = launch_run(
            **definition,
            model_dir=model_dir,
            data_dir=data_dir,
            result_root=result_root,
            sample_interval=args.sample_interval,
            quiet=args.quiet,
        )
        results.append(result)
        if not result.succeeded:
            print(
                f"ERROR: {result.label} failed; see {result.log_file}",
                file=sys.stderr,
                flush=True,
            )
            break

    version = activitysim_version()
    process_count = read_mp_process_count(model_dir)
    metadata = {
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "data_dir": str(data_dir),
        "using_subarea": using_subarea,
        "sample_interval_seconds": args.sample_interval,
        "activitysim_version": version,
        "multiprocess_workers": process_count,
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
    }
    write_json_summary(result_root / "metrics.json", results, metadata)
    report_path = result_root / "report.html"
    write_html_report(
        report_path,
        results=results,
        data_dir=data_dir,
        using_subarea=using_subarea,
        sample_interval=args.sample_interval,
        activitysim_version=version,
        process_count=process_count,
    )

    if not args.quiet:
        print(f"Report: {report_path}", flush=True)
        with suppress(Exception):
            if not webbrowser.open(report_path.resolve().as_uri()):
                warnings.warn(f"could not open the browser; report is at {report_path}")

    return 0 if len(results) == len(definitions) and all(r.succeeded for r in results) else 1


def main() -> int:
    args = parse_args()
    repo_root = repository_root()
    sync_locked_environment(repo_root, args.quiet)
    if args._worker_spec is not None:
        return run_model_worker(args._worker_spec.resolve())
    return run_benchmarks(args, repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
