#!/usr/bin/env python3
"""Benchmark full Boston ActivitySim runs with external memory monitoring.

The script first synchronizes the locked uv environment, then runs:

1. A 5,000-household, single-process Sharrow cache warm-up.
2. A full-population (or optionally sampled), multiprocess run with Sharrow required.
3. The same population, multiprocess run without Sharrow.

The two measured runs are monitored by this parent process.  Results are written to
a self-contained HTML report and CSV/JSON data files under ``model/output``.

Sharrow requires zero-based pipeline zone identifiers and realigns OMX matrices
using their named mappings.  The legacy skim reader instead retains original
zone identifiers so its offset mapper also honors those mappings.  This keeps
the two runs semantically equivalent even when OMX rows are not stored in
land-use-table order.
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
import re
import shutil
import statistics
import subprocess
import sys
import time
import warnings
import webbrowser
from collections import Counter
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ENV_READY = "LIGHTHOUSE_DIAGNOSTIC_UV_READY"
BYTES_PER_GIB = 1024**3
DEFAULT_SAMPLE_INTERVAL = 1.0
TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S-%f"
TIMESTAMP_DIR_PATTERN = re.compile(r"^\d{8}-\d{6}-\d{6}$")
COMPONENT_TIMING_PATTERN = re.compile(
    r"^(?P<timestamp>\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}).*?"
    r"time to execute run\.(?P<component>.+?) : "
    r"(?P<seconds>\d+(?:\.\d+)?) seconds",
    re.IGNORECASE,
)
RUN_COMPLETED_PATTERN = re.compile(
    r"Time to execute all models completed\s*:\s*"
    r"(?P<seconds>\d+(?:\.\d+)?) seconds",
    re.IGNORECASE,
)


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
    reused: bool = False
    samples: list[dict[str, float | int]] = field(default_factory=list)
    output_summary: dict[str, Any] = field(default_factory=dict)
    component_timings: dict[str, dict[str, Any]] = field(default_factory=dict)

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
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  %(prog)s --data-dir /path/to/full/data
  %(prog)s --data-dir /path/to/full/data --processes 16
  %(prog)s --sample-households 25000
  %(prog)s --output-dir results/timestamp
  %(prog)s --resume --output-dir results/timestamp
  %(prog)s --quiet

An output path whose final component is literally "timestamp" is replaced by
a timestamped directory name. For example, results/timestamp becomes
results/20260904-153000-123456.

With --resume, a timestamp output selects the newest timestamped directory and
reuses its compatible completed jobs. A designated non-timestamp directory is
resumed directly. Incomplete job directories are preserved with an
"-incomplete-<timestamp>" suffix before they are restarted. If omitted, the
data directory, worker count, and main-run household sample size are recovered
from existing job specifications.
""",
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
        "--output-folder",
        dest="output_dir",
        type=Path,
        help=(
            "new directory for model outputs and the report; defaults to a "
            "timestamped directory under model/output/diagnostics; a final "
            "path component named 'timestamp' is replaced by the current timestamp"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse compatible completed jobs and run only missing or incomplete "
            "jobs; for timestamp output, resume the most recent timestamped directory"
        ),
    )
    parser.add_argument(
        "--sample-households",
        type=positive_int,
        metavar="N",
        help=(
            "sample N households for each main Sharrow and non-Sharrow run; "
            "defaults to the complete population"
        ),
    )
    parser.add_argument(
        "--processes",
        type=positive_int,
        metavar="N",
        help=(
            "number of ActivitySim worker processes for each main multiprocess run; "
            "defaults to num_processes in model/configs_mp/settings.yaml"
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


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


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


def find_table_file(directory: Path, stem: str) -> Path:
    """Find a CSV or Parquet table, preferring Parquet when both are present."""

    for suffix in (".parquet", ".csv"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"could not find {stem}.parquet or {stem}.csv in {directory}"
    )


def table_row_count(path: Path) -> int:
    """Count rows without materializing a table in memory."""

    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)

    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        next(reader, None)
        return sum(1 for row in reader if row)


def table_numeric_totals(
    path: Path, columns: tuple[str, ...]
) -> dict[str, float | None]:
    """Calculate selected column totals in bounded memory."""

    totals: dict[str, float | None] = dict.fromkeys(columns)
    if path.suffix.lower() == ".parquet":
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        names = {name.casefold(): name for name in parquet.schema.names}
        selected = [
            names[name.casefold()] for name in columns if name.casefold() in names
        ]
        running = {name: 0.0 for name in selected}
        for batch in parquet.iter_batches(columns=selected, batch_size=131_072):
            for index, name in enumerate(selected):
                value = pc.sum(batch.column(index)).as_py()
                if value is not None:
                    running[name] += float(value)
        for requested in columns:
            actual = names.get(requested.casefold())
            if actual is not None:
                totals[requested] = running[actual]
        return totals

    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        names = {name.casefold(): name for name in (reader.fieldnames or ())}
        selected = {
            requested: names[requested.casefold()]
            for requested in columns
            if requested.casefold() in names
        }
        running = {name: 0.0 for name in selected}
        for row in reader:
            for requested, actual in selected.items():
                value = row.get(actual)
                if value not in (None, ""):
                    running[requested] += float(value)
        totals.update(running)
    return totals


def summarize_input_data(data_dir: Path) -> dict[str, Any]:
    """Describe the population, zones, and skim inputs used by the benchmark."""

    table_paths = {
        name: find_table_file(data_dir, name)
        for name in ("households", "persons", "land_use")
    }
    table_rows = {name: table_row_count(path) for name, path in table_paths.items()}
    land_use_totals = table_numeric_totals(
        table_paths["land_use"], ("TOTHH", "TOTPOP", "TOTEMP")
    )
    households = table_rows["households"]
    persons = table_rows["persons"]
    skim_paths = sorted(data_dir.glob("*.omx"))
    return {
        "households": households,
        "persons": persons,
        "zones": table_rows["land_use"],
        "persons_per_household": persons / households if households else None,
        "land_use_households": land_use_totals["TOTHH"],
        "land_use_population": land_use_totals["TOTPOP"],
        "land_use_employment": land_use_totals["TOTEMP"],
        "skim_file_count": len(skim_paths),
        "skim_bytes": sum(path.stat().st_size for path in skim_paths),
        "tables": {
            name: {
                "path": str(path),
                "format": path.suffix.removeprefix(".").lower(),
                "rows": table_rows[name],
                "bytes": path.stat().st_size,
            }
            for name, path in table_paths.items()
        },
        "skim_files": [str(path) for path in skim_paths],
    }


def table_column_counts(
    path: Path, columns: tuple[str, ...]
) -> dict[str, dict[str, int]]:
    """Build small categorical distributions while reading in bounded batches."""

    counts = {column: Counter() for column in columns}
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        names = {name.casefold(): name for name in parquet.schema.names}
        selected = [
            names[name.casefold()] for name in columns if name.casefold() in names
        ]
        requested_by_actual = {
            names[name.casefold()]: name for name in columns if name.casefold() in names
        }
        for batch in parquet.iter_batches(columns=selected, batch_size=131_072):
            for index, actual in enumerate(selected):
                requested = requested_by_actual[actual]
                counts[requested].update(
                    "(missing)" if value is None else str(value)
                    for value in batch.column(index).to_pylist()
                )
    else:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            names = {name.casefold(): name for name in (reader.fieldnames or ())}
            selected = {
                requested: names[requested.casefold()]
                for requested in columns
                if requested.casefold() in names
            }
            for row in reader:
                for requested, actual in selected.items():
                    value = row.get(actual)
                    counts[requested][value or "(missing)"] += 1
    return {
        column: dict(sorted(values.items(), key=lambda item: (-item[1], item[0])))
        for column, values in counts.items()
    }


def summarize_model_outputs(output_dir: Path) -> dict[str, Any]:
    """Summarize final tour and trip outputs after a measured run finishes."""

    tours_path = find_table_file(output_dir, "final_tours")
    trips_path = find_table_file(output_dir, "final_trips")
    tours = table_row_count(tours_path)
    trips = table_row_count(trips_path)
    tour_counts = table_column_counts(tours_path, ("tour_category", "tour_type"))
    trip_counts = table_column_counts(trips_path, ("trip_mode", "primary_purpose"))
    return {
        "tours": tours,
        "trips": trips,
        "trips_per_tour": trips / tours if tours else None,
        "tour_categories": tour_counts["tour_category"],
        "tour_types": tour_counts["tour_type"],
        "trip_modes": trip_counts["trip_mode"],
        "trip_primary_purposes": trip_counts["primary_purpose"],
        "files": {
            "final_tours": {
                "path": str(tours_path),
                "bytes": tours_path.stat().st_size,
            },
            "final_trips": {
                "path": str(trips_path),
                "bytes": trips_path.stat().st_size,
            },
        },
    }


def summarize_component_timings(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Aggregate exact ActivitySim component timings from each process log."""

    observations: dict[str, list[float]] = {}
    first_finished: dict[str, dt.datetime] = {}
    for log_path in sorted(output_dir.glob("*activitysim.log")):
        with log_path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                match = COMPONENT_TIMING_PATTERN.search(line)
                if match is None:
                    continue
                component = match.group("component")
                observations.setdefault(component, []).append(
                    float(match.group("seconds"))
                )
                finished = dt.datetime.strptime(
                    match.group("timestamp"), "%d/%m/%Y %H:%M:%S"
                )
                first_finished[component] = min(
                    first_finished.get(component, finished), finished
                )

    ordered_components = sorted(observations, key=first_finished.__getitem__)
    return {
        component: {
            "observations": len(observations[component]),
            "mean_seconds": statistics.fmean(observations[component]),
            "population_standard_deviation_seconds": statistics.pstdev(
                observations[component]
            ),
            "min_seconds": min(observations[component]),
            "max_seconds": max(observations[component]),
            "durations_seconds": sorted(observations[component]),
        }
        for component in ordered_components
    }


def timestamp_now() -> str:
    return dt.datetime.now().strftime(TIMESTAMP_FORMAT)


def latest_timestamped_dir(parent: Path) -> Path | None:
    """Return the newest benchmark timestamp directory under *parent*."""

    if not parent.is_dir():
        return None
    candidates = [
        path
        for path in parent.iterdir()
        if path.is_dir() and TIMESTAMP_DIR_PATTERN.fullmatch(path.name)
    ]
    return max(candidates, key=lambda path: path.name, default=None)


def make_results_dir(
    repo_root: Path, requested: Path | None, *, resume: bool = False
) -> Path:
    """Create a result directory or select the appropriate directory to resume."""

    stamp = timestamp_now()
    timestamped_output = requested is None or requested.name == "timestamp"
    if requested is None:
        timestamp_parent = repo_root / "model" / "output" / "diagnostics"
        result = timestamp_parent / stamp
    elif timestamped_output:
        timestamp_parent = requested.expanduser().parent.resolve()
        result = timestamp_parent / stamp
    else:
        timestamp_parent = None
        result = requested.expanduser().resolve()

    if resume and timestamp_parent is not None:
        result = latest_timestamped_dir(timestamp_parent) or result

    if result.exists():
        if not result.is_dir():
            raise NotADirectoryError(
                f"output path exists but is not a directory: {result}"
            )
        if not resume and any(result.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty output directory: {result}"
            )
    result.mkdir(parents=True, exist_ok=True)
    return result


def existing_worker_specs(result_root: Path) -> list[dict[str, Any]]:
    """Read usable worker specifications from an existing benchmark directory."""

    specs = []
    for key in ("cache-warmup", "full-sharrow", "full-no-sharrow"):
        path = result_root / key / "run-spec.json"
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            specs.append(value)
    return specs


def resumed_data_dir(result_root: Path) -> Path | None:
    for spec in existing_worker_specs(result_root):
        if spec.get("data_dir"):
            return Path(spec["data_dir"]).expanduser().resolve()
    return None


def resumed_process_count(result_root: Path) -> int | None:
    for spec in existing_worker_specs(result_root):
        if spec.get("process_count") is not None:
            return int(spec["process_count"])
    return None


def resumed_household_sample_size(result_root: Path) -> int | None:
    """Recover the main-run household sample size from prior specifications."""

    for spec in existing_worker_specs(result_root):
        if spec.get("multiprocess") and spec.get("household_sample_size") is not None:
            return int(spec["household_sample_size"])
    return None


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
            # Sharrow requires contiguous, zero-based pipeline zone IDs and
            # realigns skim coordinates to land_use.  With those IDs, the
            # legacy Numpy skim reader uses them as raw matrix offsets and
            # cannot honor a shuffled OMX ID mapping.  Retaining original IDs
            # for legacy runs lets OffsetMapper apply that mapping correctly.
            "recode_pipeline_columns": bool(spec["sharrow"]),
        }
    )
    if spec.get("process_count") is not None:
        settings["num_processes"] = int(spec["process_count"])

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
    print(
        "Zone indexing: "
        f"recode_pipeline_columns={bool(spec['sharrow'])} "
        "(OMX mappings honored in both skim implementations)",
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


def worker_spec_data(
    *,
    model_dir: Path,
    data_dir: Path,
    output_dir: Path,
    household_sample_size: int,
    sharrow: str | bool,
    multiprocess: bool,
    process_count: int | None,
) -> dict[str, Any]:
    return {
        "model_dir": str(model_dir),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "household_sample_size": household_sample_size,
        "sharrow": sharrow,
        "multiprocess": multiprocess,
        "process_count": process_count if multiprocess else None,
        "recode_pipeline_columns": bool(sharrow),
    }


def write_worker_spec(
    result_dir: Path,
    *,
    model_dir: Path,
    data_dir: Path,
    output_dir: Path,
    household_sample_size: int,
    sharrow: str | bool,
    multiprocess: bool,
    process_count: int | None,
) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = result_dir / "run-spec.json"
    spec = worker_spec_data(
        model_dir=model_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        household_sample_size=household_sample_size,
        sharrow=sharrow,
        multiprocess=multiprocess,
        process_count=process_count,
    )
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return spec_path


def compatible_worker_spec(path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(actual.get(key) == value for key, value in expected.items())


def completed_model_duration(log_path: Path) -> float | None:
    """Read ActivitySim's final completion duration, if present."""

    if not log_path.is_file():
        return None
    duration = None
    with log_path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = RUN_COMPLETED_PATTERN.search(line)
            if match is not None:
                duration = float(match.group("seconds"))
    return duration


def model_outputs_complete(output_dir: Path) -> bool:
    """Check for the core final tables produced only by a complete model run."""

    try:
        for table in (
            "final_households",
            "final_persons",
            "final_tours",
            "final_trips",
        ):
            if find_table_file(output_dir, table).stat().st_size <= 0:
                return False
    except (FileNotFoundError, OSError):
        return False
    return True


def read_samples_csv(path: Path) -> list[dict[str, float | int]]:
    if not path.is_file():
        return []
    samples: list[dict[str, float | int]] = []
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                try:
                    samples.append(
                        {
                            "elapsed_seconds": float(row["elapsed_seconds"]),
                            "rss_bytes": int(row["rss_bytes"]),
                            "uss_bytes": int(row["uss_bytes"]),
                            "process_count": int(row["process_count"]),
                            "uss_process_count": int(row["uss_process_count"]),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        return []
    return samples


def write_run_result(path: Path, result: RunResult) -> None:
    """Atomically persist one job result so a later invocation can reuse it."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_completed_result(
    *,
    definition: dict[str, Any],
    model_dir: Path,
    data_dir: Path,
    result_root: Path,
    process_count: int | None,
    sample_interval: float,
) -> RunResult | None:
    """Load a compatible completed job, including pre-resume-script outputs."""

    run_dir = result_root / definition["key"]
    output_dir = run_dir / "model-output"
    log_path = run_dir / "console.log"
    expected = worker_spec_data(
        model_dir=model_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        household_sample_size=definition["household_sample_size"],
        sharrow=definition["sharrow"],
        multiprocess=definition["multiprocess"],
        process_count=process_count,
    )
    if not compatible_worker_spec(run_dir / "run-spec.json", expected):
        return None
    if not model_outputs_complete(output_dir):
        return None

    saved_path = run_dir / "run-result.json"
    saved: dict[str, Any] = {}
    if saved_path.is_file():
        try:
            loaded = json.loads(saved_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                saved = loaded
        except (OSError, json.JSONDecodeError):
            saved = {}
    if saved and int(saved.get("return_code", 1)) != 0:
        return None

    samples = saved.get("samples") or read_samples_csv(run_dir / "memory-samples.csv")
    duration = saved.get("duration_seconds")
    if duration is None:
        duration = completed_model_duration(log_path)
        if duration is None:
            return None
        if samples:
            duration = max(
                float(duration),
                max(float(sample["elapsed_seconds"]) for sample in samples)
                + sample_interval,
            )

    started_at = saved.get("started_at")
    if not started_at:
        finished_at = dt.datetime.fromtimestamp(
            log_path.stat().st_mtime, tz=dt.timezone.utc
        )
        started_at = (finished_at - dt.timedelta(seconds=float(duration))).isoformat()

    return RunResult(
        key=definition["key"],
        label=definition["label"],
        sharrow=definition["sharrow"],
        multiprocess=definition["multiprocess"],
        household_sample_size=definition["household_sample_size"],
        output_dir=str(output_dir),
        log_file=str(log_path),
        started_at=str(started_at),
        duration_seconds=float(duration),
        return_code=0,
        reused=True,
        samples=samples,
        output_summary=saved.get("output_summary", {}),
        component_timings=saved.get("component_timings", {}),
    )


def preserve_incomplete_run(run_dir: Path) -> Path | None:
    """Move a stale job aside before restarting it, preserving crash evidence."""

    if not run_dir.exists():
        return None
    suffix = timestamp_now()
    preserved = run_dir.with_name(f"{run_dir.name}-incomplete-{suffix}")
    counter = 1
    while preserved.exists():
        preserved = run_dir.with_name(f"{run_dir.name}-incomplete-{suffix}-{counter}")
        counter += 1
    run_dir.rename(preserved)
    return preserved


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
    process_count: int | None,
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
        process_count=process_count,
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
        scope = (
            "all households"
            if household_sample_size == 0
            else f"{household_sample_size:,} households"
        )
        process_mode = (
            f"multiprocess ({process_count} workers)"
            if multiprocess and process_count is not None
            else "multiprocess"
            if multiprocess
            else "single process"
        )
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
    write_run_result(run_dir / "run-result.json", result)

    if not quiet:
        status = "completed" if result.succeeded else f"failed (exit {return_code})"
        memory = (
            f", peak RSS {format_gib(result.peak_rss_bytes)}" if monitor_memory else ""
        )
        print(
            f"Finished {label}: {status} in {format_duration(duration)}{memory}",
            flush=True,
        )
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


def format_bytes(size: int) -> str:
    if size >= BYTES_PER_GIB:
        return f"{size / BYTES_PER_GIB:.2f} GiB"
    return f"{size / 1024**2:.1f} MiB"


def format_optional_count(value: float | int | None) -> str:
    if value is None:
        return "Unavailable"
    return f"{round(value):,}"


def input_summary_html(summary: dict[str, Any]) -> str:
    persons_per_household = summary.get("persons_per_household")
    ratio = f"{persons_per_household:.3f}" if persons_per_household is not None else "—"
    table_rows = []
    for name, details in summary["tables"].items():
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(name.replace('_', ' ').title())}</td>"
            f"<td>{html.escape(details['format'].upper())}</td>"
            f'<td class="numeric">{details["rows"]:,}</td>'
            f'<td class="numeric">{format_bytes(details["bytes"])}</td>'
            "</tr>"
        )
    return (
        '<section class="cards input-cards">'
        '<div class="card"><div class="name">Households</div>'
        f'<div class="value">{summary["households"]:,}</div></div>'
        '<div class="card"><div class="name">Persons</div>'
        f'<div class="value">{summary["persons"]:,}</div></div>'
        '<div class="card"><div class="name">Zones</div>'
        f'<div class="value">{summary["zones"]:,}</div></div>'
        '<div class="card"><div class="name">Persons / household</div>'
        f'<div class="value">{ratio}</div></div>'
        "</section>"
        "<table><thead><tr><th>Population table</th><th>Format</th>"
        '<th class="numeric">Rows</th><th class="numeric">File size</th>'
        "</tr></thead><tbody>" + "".join(table_rows) + "</tbody></table>"
        '<section class="cards input-cards">'
        '<div class="card"><div class="name">Land-use households</div>'
        f'<div class="value">{format_optional_count(summary.get("land_use_households"))}</div></div>'
        '<div class="card"><div class="name">Land-use population</div>'
        f'<div class="value">{format_optional_count(summary.get("land_use_population"))}</div></div>'
        '<div class="card"><div class="name">Land-use employment</div>'
        f'<div class="value">{format_optional_count(summary.get("land_use_employment"))}</div></div>'
        '<div class="card"><div class="name">Skim OMX files</div>'
        f'<div class="value">{summary["skim_file_count"]:,} · '
        f"{format_bytes(summary['skim_bytes'])}</div></div>"
        "</section>"
    )


def output_overview_table(results: list[RunResult]) -> str:
    rows = []
    for result in results:
        if not result.multiprocess:
            continue
        summary = result.output_summary
        if summary.get("error"):
            values = (
                '<td colspan="3">Summary failed: '
                + html.escape(summary["error"])
                + "</td>"
            )
        elif summary:
            trips_per_tour = summary.get("trips_per_tour")
            ratio = f"{trips_per_tour:.3f}" if trips_per_tour is not None else "—"
            values = (
                f'<td class="numeric">{summary["tours"]:,}</td>'
                f'<td class="numeric">{summary["trips"]:,}</td>'
                f'<td class="numeric">{ratio}</td>'
            )
        else:
            values = '<td colspan="3">Unavailable because the run did not complete</td>'
        rows.append(f"<tr><td>{html.escape(result.label)}</td>{values}</tr>")
    return (
        "<table><thead><tr><th>Run</th>"
        '<th class="numeric">Tours</th><th class="numeric">Trips</th>'
        '<th class="numeric">Trips / tour</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def output_distribution_table(results: list[RunResult], key: str) -> str:
    measured = [
        result
        for result in results
        if result.multiprocess
        and result.output_summary
        and not result.output_summary.get("error")
        and key in result.output_summary
    ]
    categories = sorted(
        {
            category
            for result in measured
            for category in result.output_summary.get(key, {})
        },
        key=lambda category: (
            -sum(
                result.output_summary.get(key, {}).get(category, 0)
                for result in measured
            ),
            category,
        ),
    )
    if not categories:
        return '<p class="empty">No distribution data are available.</p>'

    total_key = "trips" if key.startswith("trip_") else "tours"
    headings = "".join(
        f'<th class="numeric">{html.escape(result.label)}</th>' for result in measured
    )
    rows = []
    for category in categories:
        cells = []
        for result in measured:
            count = int(result.output_summary.get(key, {}).get(category, 0))
            total = int(result.output_summary.get(total_key, 0))
            share = count / total if total else 0.0
            cells.append(
                f'<td class="numeric">{count:,} <small>({share:.1%})</small></td>'
            )
        rows.append(f"<tr><td>{html.escape(category)}</td>{''.join(cells)}</tr>")
    return (
        f"<table><thead><tr><th>Category</th>{headings}</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


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
    y_axis_max = (
        math.ceil(max_y_gib / 5.0) * 5.0 if max_y_gib > 5 else math.ceil(max_y_gib)
    )
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
            f"{seconds / 60:.1f}</text>"
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
        relative = (
            result.duration_seconds / fastest if fastest and result.succeeded else None
        )
        difference = (
            result.duration_seconds - fastest if fastest and result.succeeded else None
        )
        log_link = os.path.relpath(result.log_file, report_dir)
        relative_cell = (
            f"<td>{relative:.3f}×</td>" if relative is not None else "<td>—</td>"
        )
        difference_cell = (
            f"<td>{difference:+.1f} s</td>" if difference is not None else "<td>—</td>"
        )
        uss_cell = (
            f"<td>{format_gib(result.peak_uss_bytes)}</td>"
            if result.uss_available
            else "<td>Unavailable</td>"
        )
        if result.succeeded:
            status = "Reused (completed)" if result.reused else "Completed"
        else:
            status = f"Failed ({result.return_code})"
        row = (
            "<tr>"
            f"<td>{html.escape(result.label)}</td>"
            f"<td>{'require' if result.sharrow else 'False'}</td>"
            f"<td>{status}</td>"
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
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def component_timing_table(results: list[RunResult]) -> str:
    measured = [result for result in results if result.multiprocess]
    components: list[str] = []
    for result in measured:
        for component in result.component_timings:
            if component not in components:
                components.append(component)
    if not components:
        return '<p class="empty">No component timing data are available.</p>'

    run_headings = "".join(
        f'<th colspan="3">{html.escape(result.label)}</th>' for result in measured
    )
    statistic_headings = "".join(
        '<th class="numeric">Mean (s)</th>'
        '<th class="numeric">Population SD (s)</th>'
        '<th class="numeric">N</th>'
        for _ in measured
    )
    rows = []
    for component in components:
        cells = []
        for result in measured:
            timing = result.component_timings.get(component)
            if timing is None:
                cells.append('<td colspan="3">—</td>')
                continue
            observations = int(timing["observations"])
            standard_deviation = (
                f"{timing['population_standard_deviation_seconds']:.3f}"
                if observations > 1
                else "—"
            )
            cells.extend(
                (
                    f'<td class="numeric">{timing["mean_seconds"]:.3f}</td>',
                    f'<td class="numeric">{standard_deviation}</td>',
                    f'<td class="numeric">{observations}</td>',
                )
            )
        rows.append(
            f"<tr><td><code>{html.escape(component)}</code></td>{''.join(cells)}</tr>"
        )
    return (
        '<table><thead><tr><th rowspan="2">Component</th>'
        f"{run_headings}</tr><tr>{statistic_headings}</tr></thead><tbody>"
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
    input_summary: dict[str, Any],
) -> None:
    warmup = next((result for result in results if result.key == "cache-warmup"), None)
    scope = "Repository subarea data" if using_subarea else "Full-scale data"
    scope_class = "warning" if using_subarea else "ok"
    process_text = str(process_count) if process_count else "from configs_mp"
    main_household_sample_size = next(
        (result.household_sample_size for result in results if result.multiprocess),
        0,
    )
    population_text = (
        "all households (<code>households_sample_size: 0</code>)"
        if main_household_sample_size == 0
        else (
            f"a sample of {main_household_sample_size:,} households "
            f"(<code>households_sample_size: {main_household_sample_size}</code>)"
        )
    )
    generated = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    warmup_text = (
        f"{'Reused' if warmup.reused else 'Completed' if warmup.succeeded else 'Failed'} "
        f"in {format_duration(warmup.duration_seconds)}"
        if warmup
        else "Not run"
    )
    resume_note = (
        '<p class="note">Rows marked <strong>Reused</strong> came from a prior '
        "invocation. For benchmark folders created before per-job result files "
        "were available, elapsed runtime is reconstructed from external memory "
        "samples when possible, otherwise from ActivitySim's completion log.</p>"
        if any(result.reused for result in results)
        else ""
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
h3 {{ margin-top:24px; font-size:16px; }}
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
th.numeric, td.numeric {{ text-align:right !important; }}
small {{ color:#64748b; }}
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
<h2>Input data</h2>
<p class="note">Counts describe the files supplied to both measured runs. Land-use totals come from <code>TOTHH</code>, <code>TOTPOP</code>, and <code>TOTEMP</code>; zones are rows in the land-use table.</p>
{input_summary_html(input_summary)}
<h2>Runtime comparison</h2>
{runtime_table(results, report_path.parent)}
{resume_note}
<h2>Per-component runtime</h2>
<p class="note">Each observation is ActivitySim's <code>run.&lt;component&gt;</code> elapsed time for one process and excludes multiprocessing orchestration and checkpoint overhead. Multiprocess components report the mean and population standard deviation across all worker shards; population SD is appropriate because every worker is observed rather than sampled. A dash is shown for serial components with only one observation. Component means are not additive wall-clock times because workers run concurrently.</p>
{component_timing_table(results)}
<h2>Progressive memory usage</h2>
<div class="chart"><div class="legend">{legend}</div>{svg_memory_chart(results)}</div>
<h2>Model output summary</h2>
<p class="note">Output files are scanned only after both timed runs finish, so these summaries are not included in the runtime measurements.</p>
{output_overview_table(results)}
<h3>Trips by mode</h3>
{output_distribution_table(results, "trip_modes")}
<h3>Trips by primary purpose</h3>
{output_distribution_table(results, "trip_primary_purposes")}
<h3>Tours by category</h3>
{output_distribution_table(results, "tour_categories")}
<h3>Tours by type</h3>
{output_distribution_table(results, "tour_types")}
<p class="note"><strong>Memory interpretation.</strong> Tree RSS is the sum of resident memory reported for the controller and all workers and can count shared skim pages more than once. Where the operating system permits external access, tree USS is also shown as the sum of private memory and excludes shared pages. macOS commonly restricts child-process USS, in which case the report marks it unavailable instead of substituting an internal profiler. Samples were taken externally every {sample_interval:g} seconds.</p>
<p class="note"><strong>Benchmark controls.</strong> Both measured runs use {population_text} and {html.escape(process_text)} ActivitySim worker processes. ActivitySim instrumentation, memory profiling, expression profiling, trace output, variability checking, loser logging, and <code>track_skim_usage</code> are disabled. Sharrow runs use zero-based pipeline zone IDs as required; non-Sharrow runs retain original zone IDs so ActivitySim's legacy offset mapper honors OMX mappings even when matrix rows are stored in a different order.</p>
<p class="note"><strong>Data directory.</strong> <code>{html.escape(str(data_dir))}</code></p>
<p class="note"><strong>Host.</strong> {html.escape(platform.platform())}; {os.cpu_count() or "unknown"} logical CPUs.</p>
</main></body></html>
"""
    report_path.write_text(report, encoding="utf-8")


def write_json_summary(
    path: Path, results: list[RunResult], metadata: dict[str, Any]
) -> None:
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
        return (
            int(settings.get("num_processes"))
            if settings.get("num_processes")
            else None
        )
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

    result_root = make_results_dir(repo_root, args.output_dir, resume=args.resume)
    prior_data_dir = resumed_data_dir(result_root) if args.resume else None
    default_data_dir = (repo_root / "model" / "data").resolve()
    if args.data_dir is not None:
        data_dir = args.data_dir.expanduser().resolve()
    elif prior_data_dir is not None:
        data_dir = prior_data_dir
    else:
        data_dir = default_data_dir
    using_subarea = data_dir == default_data_dir
    if args.data_dir is None and using_subarea:
        warnings.warn(
            "--data-dir was not supplied; the smaller repository subarea data "
            "will be tested or resumed.",
            stacklevel=2,
        )
    validate_data_dir(data_dir)

    model_dir = repo_root / "model"
    process_count = (
        args.processes
        or (resumed_process_count(result_root) if args.resume else None)
        or read_mp_process_count(model_dir)
    )
    main_household_sample_size = args.sample_households
    if main_household_sample_size is None and args.resume:
        main_household_sample_size = resumed_household_sample_size(result_root)
    if main_household_sample_size is None:
        main_household_sample_size = 0
    population_label = (
        "Full model"
        if main_household_sample_size == 0
        else f"Model ({main_household_sample_size:,} households)"
    )

    if not args.quiet:
        print("Summarizing input data...", flush=True)
    input_summary = summarize_input_data(data_dir)
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
            "label": f"{population_label} — Sharrow required",
            "sharrow": "require",
            "multiprocess": True,
            "household_sample_size": main_household_sample_size,
            "monitor_memory": True,
        },
        {
            "key": "full-no-sharrow",
            "label": f"{population_label} — Sharrow disabled",
            "sharrow": False,
            "multiprocess": True,
            "household_sample_size": main_household_sample_size,
            "monitor_memory": True,
        },
    ]

    if not args.quiet:
        print(f"Data directory: {data_dir}", flush=True)
        action = "Resuming diagnostic outputs" if args.resume else "Diagnostic outputs"
        print(f"{action}: {result_root}", flush=True)
        print(
            f"Inputs: {input_summary['households']:,} households, "
            f"{input_summary['persons']:,} persons, {input_summary['zones']:,} zones",
            flush=True,
        )
        if process_count is not None:
            print(f"Multiprocess workers: {process_count}", flush=True)
        main_scope = (
            "all households"
            if main_household_sample_size == 0
            else f"{main_household_sample_size:,} sampled households"
        )
        print(f"Main-run population: {main_scope}", flush=True)

    results: list[RunResult] = []
    for definition in definitions:
        if args.resume:
            resumed_result = load_completed_result(
                definition=definition,
                model_dir=model_dir,
                data_dir=data_dir,
                result_root=result_root,
                process_count=process_count,
                sample_interval=args.sample_interval,
            )
            if resumed_result is not None:
                results.append(resumed_result)
                write_run_result(
                    result_root / resumed_result.key / "run-result.json",
                    resumed_result,
                )
                if not args.quiet:
                    print(
                        f"Reusing completed {resumed_result.label}: "
                        f"{format_duration(resumed_result.duration_seconds)}",
                        flush=True,
                    )
                continue

            preserved = preserve_incomplete_run(result_root / definition["key"])
            if preserved is not None and not args.quiet:
                print(
                    f"Preserved incomplete or incompatible job at {preserved}",
                    flush=True,
                )

        result = launch_run(
            **definition,
            model_dir=model_dir,
            data_dir=data_dir,
            result_root=result_root,
            sample_interval=args.sample_interval,
            quiet=args.quiet,
            process_count=process_count,
        )
        results.append(result)
        if not result.succeeded:
            print(
                f"ERROR: {result.label} failed; see {result.log_file}",
                file=sys.stderr,
                flush=True,
            )
            break

    if not args.quiet:
        print("Summarizing component timings and model outputs...", flush=True)
    for result in results:
        if not result.succeeded:
            continue
        try:
            result.component_timings = summarize_component_timings(
                Path(result.output_dir)
            )
        except Exception as error:  # noqa: BLE001 - preserve the benchmark report
            warnings.warn(
                f"could not summarize component timings for {result.label}: {error}",
                stacklevel=2,
            )
        if not result.multiprocess:
            continue
        try:
            result.output_summary = summarize_model_outputs(Path(result.output_dir))
        except Exception as error:  # noqa: BLE001 - preserve the benchmark report
            result.output_summary = {"error": str(error)}
            warnings.warn(
                f"could not summarize outputs for {result.label}: {error}", stacklevel=2
            )
        write_run_result(
            result_root / result.key / "run-result.json",
            result,
        )

    version = activitysim_version()
    metadata = {
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "data_dir": str(data_dir),
        "using_subarea": using_subarea,
        "sample_interval_seconds": args.sample_interval,
        "activitysim_version": version,
        "multiprocess_workers": process_count,
        "household_sample_size": main_household_sample_size,
        "input_summary": input_summary,
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "resume_requested": args.resume,
        "reused_jobs": [result.key for result in results if result.reused],
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
        input_summary=input_summary,
    )

    if not args.quiet:
        print(f"Report: {report_path}", flush=True)
        with suppress(Exception):
            if not webbrowser.open(report_path.resolve().as_uri()):
                warnings.warn(f"could not open the browser; report is at {report_path}")

    return (
        0
        if len(results) == len(definitions) and all(r.succeeded for r in results)
        else 1
    )


def main() -> int:
    args = parse_args()
    repo_root = repository_root()
    sync_locked_environment(repo_root, args.quiet)
    if args._worker_spec is not None:
        return run_model_worker(args._worker_spec.resolve())
    return run_benchmarks(args, repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
