#!/usr/bin/env python3
"""Qualify production Sharrow runs inside memory-limited Linux containers.

The host command builds an image from ``uv.lock``, performs a single-process
cache warm-up, and runs a configurable multiprocess sweep. Raw cgroup-v2 and
host memory metrics are sampled throughout each run.

Examples:
  ./scripts/production-benchmark.py
  ./scripts/production-benchmark.py --sample-households 400000 --processes 2 4
  ./scripts/production-benchmark.py --resume
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


GIB = 1024**3
TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S-%f"
TIMESTAMP_PATTERN = re.compile(r"^\d{8}-\d{6}-\d{6}$")
COMPONENT_COMPLETION_PATTERN = re.compile(
    r"^\[(?P<minutes>\d+):(?P<seconds>\d+(?:\.\d+)?)\].*?INFO:\s+"
    r"(?P<process>mp_[A-Za-z0-9_]+)\s+"
    r"(?P<component>[A-Za-z][A-Za-z0-9_]*)\s*:\s*"
    r"(?P<duration>\d+(?:\.\d+)?)\s+seconds"
)
THREAD_LIMIT_ENV = (
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
ALLOCATOR_ENV = {
    # Bound per-thread glibc arenas and return released NumPy/Pandas buffers
    # promptly. This avoids retained heaps accumulating across trip-purpose
    # segments while leaving calculations and random streams unchanged.
    "MALLOC_ARENA_MAX": "2",
    "MALLOC_TRIM_THRESHOLD_": "65536",
}


@dataclass
class HostSample:
    elapsed_seconds: float
    timestamp_utc: str
    total_bytes: int | None
    available_bytes: int | None
    used_bytes: int | None
    used_percent: float | None
    swap_used_bytes: int | None
    macos_free_percent: float | None


@dataclass
class RunResult:
    key: str
    label: str
    workers: int
    household_sample_size: int
    started_at: str
    duration_seconds: float
    return_code: int
    oom_killed: bool
    safety_abort_reason: str | None
    container_name: str
    container_peak_bytes: int
    container_peak_working_set_bytes: int
    container_peak_swap_bytes: int
    host_min_available_bytes: int | None
    host_peak_used_bytes: int | None
    host_min_macos_free_percent: float | None
    output_dir: str
    console_log: str
    cgroup_samples: str
    host_samples: str
    reused: bool = False
    component_timings: dict[str, dict[str, float | int]] = field(default_factory=dict)
    component_memory: dict[str, dict[str, float | int]] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return (
            self.return_code == 0
            and not self.oom_killed
            and self.safety_abort_reason is None
        )


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or a positive integer")
    return parsed


def bounded_percent(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 100:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 100")
    return parsed


def parse_byte_size(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b?)?\s*", value, re.I)
    if match is None:
        raise argparse.ArgumentTypeError(f"invalid byte size: {value!r}")
    amount = float(match.group(1))
    unit = (match.group(2) or "b").lower().rstrip("b")
    powers = {
        "": 0,
        "k": 1,
        "ki": 1,
        "m": 2,
        "mi": 2,
        "g": 3,
        "gi": 3,
        "t": 4,
        "ti": 4,
    }
    return int(amount * 1024 ** powers[unit])


def docker_size(value: str) -> str:
    parse_byte_size(value)
    return value


def parse_args() -> argparse.Namespace:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description=(
            "Build a locked Linux environment and qualify the full Sharrow model "
            "under cgroup and host-memory monitoring."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  %(prog)s
  %(prog)s --sample-households 400000 --processes 2 4
  %(prog)s --resume
""",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "model" / "data_full",
        help="production data directory (default: model/data_full)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "output directory; defaults to a timestamp below "
            "model/output/production-benchmarks; a final 'timestamp' component "
            "is replaced with the current timestamp"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=root / "model" / "output" / "production-benchmark-cache",
        help="persistent Sharrow/cache directory shared by all runs",
    )
    parser.add_argument(
        "--processes",
        type=positive_int,
        nargs="+",
        default=[2, 4, 6, 8, 10],
        metavar="N",
        help="multiprocess worker counts to test, each at most 10",
    )
    parser.add_argument(
        "--sample-households",
        type=nonnegative_int,
        default=0,
        metavar="N",
        help="households in each measured run; 0 means the full population",
    )
    parser.add_argument(
        "--warmup-households",
        type=positive_int,
        default=5_000,
        metavar="N",
        help="single-process Sharrow precompile sample (default: %(default)s)",
    )
    parser.add_argument(
        "--memory-limit",
        type=docker_size,
        default="52g",
        help="hard memory and memory+swap limit (default: %(default)s)",
    )
    parser.add_argument(
        "--qualification-peak",
        type=docker_size,
        default="50g",
        help="largest peak qualifying for the 64 GB profile (default: %(default)s)",
    )
    parser.add_argument(
        "--shm-size",
        type=docker_size,
        default="16g",
        help="container /dev/shm capacity; actual usage counts toward memory",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="cgroup and host sampling interval (default: %(default)s)",
    )
    parser.add_argument(
        "--host-pressure-interval",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="macOS pressure sampling interval (default: %(default)s)",
    )
    parser.add_argument(
        "--host-min-available-gib",
        type=float,
        default=12.0,
        metavar="GIB",
        help="stop after two samples below this host-available threshold",
    )
    parser.add_argument(
        "--host-min-free-percent",
        type=float,
        default=5.0,
        metavar="PERCENT",
        help="stop after two macOS pressure samples below this percentage",
    )
    parser.add_argument(
        "--container-abort-percent",
        type=bounded_percent,
        default=98.0,
        metavar="PERCENT",
        help="stop before a sustained cgroup hard-limit breach",
    )
    parser.add_argument("--image", help="Docker image name; defaults from uv.lock")
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="use an existing image without rebuilding the locked environment",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse compatible completed jobs from the selected output",
    )
    parser.add_argument(
        "--continue-after-failure",
        action="store_true",
        help="continue to higher worker counts after failure or safety stop",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress and do not open the HTML report",
    )
    parser.add_argument("--_container-worker", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def timestamp_now() -> str:
    return dt.datetime.now().strftime(TIMESTAMP_FORMAT)


def latest_timestamped_dir(parent: Path) -> Path | None:
    if not parent.is_dir():
        return None
    return max(
        (
            path
            for path in parent.iterdir()
            if path.is_dir() and TIMESTAMP_PATTERN.fullmatch(path.name)
        ),
        key=lambda path: path.name,
        default=None,
    )


def result_directory(root: Path, requested: Path | None, resume: bool) -> Path:
    if requested is None:
        parent = root / "model" / "output" / "production-benchmarks"
        result = parent / timestamp_now()
        timestamped = True
    else:
        requested = requested.expanduser().resolve()
        timestamped = requested.name == "timestamp"
        parent = requested.parent if timestamped else requested
        result = parent / timestamp_now() if timestamped else requested
    if resume and timestamped:
        result = latest_timestamped_dir(parent) or result
    if result.exists() and not resume and any(result.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {result}")
    result.mkdir(parents=True, exist_ok=True)
    return result


def run_command(
    command: list[str],
    *,
    check: bool = True,
    capture: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        cwd=cwd,
    )


def require_docker() -> dict[str, Any]:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker CLI is not installed or not on PATH")
    result = run_command(["docker", "info", "--format", "{{json .}}"])
    info = json.loads(result.stdout)
    if not info.get("ServerVersion"):
        raise RuntimeError("Docker engine is not running")
    return info


def locked_image_name(root: Path) -> str:
    digest = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()[:12]
    return f"lighthouse-production-benchmark:{digest}"


def config_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for directory_name in ("configs", "configs_mp", "configs_64gb"):
        directory = root / "model" / directory_name
        if not directory.is_dir():
            continue
        for path in sorted(path for path in directory.rglob("*") if path.is_file()):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def build_image(root: Path, image_name: str, quiet: bool) -> None:
    command = [
        "docker",
        "build",
        "--file",
        str(root / "scripts" / "production-benchmark.Dockerfile"),
        "--tag",
        image_name,
        str(root),
    ]
    if not quiet:
        print(f"Building locked benchmark image {image_name}...", flush=True)
    run_command(command, capture=quiet)


def docker_image_id(image_name: str) -> str:
    result = run_command(
        ["docker", "image", "inspect", image_name, "--format", "{{.Id}}"]
    )
    return result.stdout.strip()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if value == "max":
        return None
    with suppress(ValueError):
        return int(value)
    return None


def _read_key_values(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    for line in lines:
        parts = line.split()
        if len(parts) == 2:
            with suppress(ValueError):
                result[parts[0]] = int(parts[1])
    return result


def _read_pressure(path: Path) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        prefix = parts[0]
        for item in parts[1:]:
            key, _, value = item.partition("=")
            if value:
                with suppress(ValueError):
                    result[f"pressure_{prefix}_{key}"] = (
                        int(value) if key == "total" else float(value)
                    )
    return result


CGROUP_FIELDS = (
    "elapsed_seconds",
    "timestamp_utc",
    "memory_current_bytes",
    "memory_peak_bytes",
    "memory_max_bytes",
    "swap_current_bytes",
    "anon_bytes",
    "file_bytes",
    "shmem_bytes",
    "inactive_file_bytes",
    "pressure_some_avg10",
    "pressure_some_total",
    "pressure_full_avg10",
    "pressure_full_total",
    "event_low",
    "event_high",
    "event_max",
    "event_oom",
    "event_oom_kill",
)


def read_cgroup_v2(root: Path, elapsed: float) -> dict[str, Any]:
    stat = _read_key_values(root / "memory.stat")
    events = _read_key_values(root / "memory.events")
    pressure = _read_pressure(root / "memory.pressure")
    return {
        "elapsed_seconds": round(elapsed, 3),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "memory_current_bytes": _read_int(root / "memory.current") or 0,
        "memory_peak_bytes": _read_int(root / "memory.peak") or 0,
        "memory_max_bytes": _read_int(root / "memory.max") or 0,
        "swap_current_bytes": _read_int(root / "memory.swap.current") or 0,
        "anon_bytes": stat.get("anon", 0),
        "file_bytes": stat.get("file", 0),
        "shmem_bytes": stat.get("shmem", 0),
        "inactive_file_bytes": stat.get("inactive_file", 0),
        "pressure_some_avg10": pressure.get("pressure_some_avg10", 0.0),
        "pressure_some_total": pressure.get("pressure_some_total", 0),
        "pressure_full_avg10": pressure.get("pressure_full_avg10", 0.0),
        "pressure_full_total": pressure.get("pressure_full_total", 0),
        "event_low": events.get("low", 0),
        "event_high": events.get("high", 0),
        "event_max": events.get("max", 0),
        "event_oom": events.get("oom", 0),
        "event_oom_kill": events.get("oom_kill", 0),
    }


def cgroup_sampler(path: Path, interval: float, stop: threading.Event) -> None:
    root = Path("/sys/fs/cgroup")
    start = time.perf_counter()
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CGROUP_FIELDS)
        writer.writeheader()
        while True:
            writer.writerow(read_cgroup_v2(root, time.perf_counter() - start))
            stream.flush()
            if stop.wait(interval):
                writer.writerow(read_cgroup_v2(root, time.perf_counter() - start))
                stream.flush()
                return


def find_table(directory: Path, stem: str) -> Path:
    for suffix in (".parquet", ".csv"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(stem)


def read_table_columns(path: Path, columns: list[str]):
    import pandas as pd

    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns, low_memory=False)


def summarize_input_data(data_dir: Path) -> dict[str, Any]:
    import pandas as pd

    def table_rows(stem: str) -> tuple[int, str, int]:
        path = find_table(data_dir, stem)
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            rows = pq.ParquetFile(path).metadata.num_rows
        else:
            with path.open(encoding="utf-8") as stream:
                rows = sum(1 for _ in stream) - 1
        return rows, path.suffix.lstrip("."), path.stat().st_size

    households, households_format, households_bytes = table_rows("households")
    persons, persons_format, persons_bytes = table_rows("persons")
    land_use = pd.read_csv(find_table(data_dir, "land_use"), low_memory=False)
    skim_files = sorted(data_dir.glob("*.omx"))
    return {
        "households": households,
        "persons": persons,
        "zones": len(land_use),
        "persons_per_household": persons / households if households else None,
        "land_use_households": (
            float(land_use["TOTHH"].sum()) if "TOTHH" in land_use else None
        ),
        "land_use_population": (
            float(land_use["TOTPOP"].sum()) if "TOTPOP" in land_use else None
        ),
        "land_use_employment": (
            float(land_use["TOTEMP"].sum()) if "TOTEMP" in land_use else None
        ),
        "skim_files": len(skim_files),
        "skim_bytes": sum(path.stat().st_size for path in skim_files),
        "tables": {
            "households": {
                "rows": households,
                "format": households_format,
                "bytes": households_bytes,
            },
            "persons": {
                "rows": persons,
                "format": persons_format,
                "bytes": persons_bytes,
            },
        },
    }


def summarize_outputs(output_dir: Path) -> dict[str, Any]:
    tours = read_table_columns(
        find_table(output_dir, "final_tours"), ["tour_category", "tour_type"]
    )
    trips = read_table_columns(
        find_table(output_dir, "final_trips"), ["trip_mode", "primary_purpose"]
    )

    def counts(series) -> dict[str, int]:
        values = series.astype("object").where(series.notna(), "<missing>")
        return {str(key): int(value) for key, value in values.value_counts().items()}

    return {
        "tours": len(tours),
        "trips": len(trips),
        "trips_per_tour": len(trips) / len(tours) if len(tours) else None,
        "tour_categories": counts(tours["tour_category"]),
        "tour_types": counts(tours["tour_type"]),
        "trip_modes": counts(trips["trip_mode"]),
        "trip_primary_purposes": counts(trips["primary_purpose"]),
    }


@contextmanager
def synchronous_sharrow_skim_loading():
    from sharrow.shared_memory import SharedMemDatasetAccessor

    original = SharedMemDatasetAccessor.to_shared_memory

    def synchronous(self, *args, **kwargs):
        kwargs["dask_scheduler"] = "synchronous"
        return original(self, *args, **kwargs)

    SharedMemDatasetAccessor.to_shared_memory = synchronous
    try:
        yield
    finally:
        SharedMemDatasetAccessor.to_shared_memory = original


def container_worker(spec_path: Path) -> int:
    for name in THREAD_LIMIT_ENV:
        os.environ[name] = "1"

    spec = read_json(spec_path)
    result_dir = Path(spec["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    sampler = threading.Thread(
        target=cgroup_sampler,
        args=(
            result_dir / "cgroup-samples.csv",
            float(spec["sample_interval"]),
            stop,
        ),
        daemon=True,
    )
    sampler.start()
    started = time.perf_counter()
    worker_result: dict[str, Any] = {"succeeded": False}
    try:
        import activitysim
        from activitysim.core.workflow import State

        data_dir = Path(spec["data_dir"])
        output_dir = Path(spec["output_dir"])
        cache_dir = Path(spec["cache_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        write_json(result_dir / "input-summary.json", summarize_input_data(data_dir))

        model_dir = Path(spec["model_dir"])
        if spec["multiprocess"]:
            configs: list[Path] = []
            profile_dir = model_dir / "configs_64gb"
            if profile_dir.is_dir():
                configs.append(profile_dir)
            configs.extend((model_dir / "configs_mp", model_dir / "configs"))
            configs_dir = tuple(configs)
        else:
            configs_dir = (model_dir / "configs",)

        settings = {
            "households_sample_size": int(spec["household_sample_size"]),
            "multiprocess": bool(spec["multiprocess"]),
            "num_processes": int(spec["workers"]),
            "sharrow": "require",
            "recode_pipeline_columns": True,
            "instrument": False,
            "memory_profile": False,
            "expression_profile": False,
            "check_for_variability": False,
            "log_alt_losers": False,
            "want_dest_choice_sample_tables": False,
            "keep_chunk_logs": False,
            "keep_mem_logs": False,
        }
        state = State.make_default(
            working_dir=model_dir,
            configs_dir=configs_dir,
            data_dir=data_dir,
            output_dir=output_dir,
            cache_dir=cache_dir,
            settings=settings,
        )
        state.set("imported_extensions", ())
        state.set("run_timestamp", dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
        state.settings.models = [
            model for model in state.settings.models if model != "track_skim_usage"
        ]
        state.settings.sharrow = "require"
        state.filesystem.persist_sharrow_cache()

        print(f"ActivitySim version: {activitysim.__version__}", flush=True)
        print(
            "Production benchmark: sharrow=require, diagnostics disabled, "
            f"workers={spec['workers']}, households={spec['household_sample_size']}",
            flush=True,
        )
        print(
            "Thread limits: " + ", ".join(f"{name}=1" for name in THREAD_LIMIT_ENV),
            flush=True,
        )
        with synchronous_sharrow_skim_loading():
            state.run.all()

        if not spec["multiprocess"]:
            if state.settings.cleanup_pipeline_after_run:
                state.checkpoint.cleanup()
            else:
                state.checkpoint.close_store()

        output_summary = summarize_outputs(output_dir)
        write_json(result_dir / "output-summary.json", output_summary)
        worker_result = {
            "succeeded": True,
            "activitysim_version": activitysim.__version__,
            "duration_seconds": time.perf_counter() - started,
            "output_summary": output_summary,
        }
        return_code = 0
    except BaseException as error:  # noqa: BLE001 - persist model failures
        traceback.print_exc()
        worker_result = {
            "succeeded": False,
            "duration_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        return_code = 1
    finally:
        stop.set()
        sampler.join(timeout=max(5.0, float(spec.get("sample_interval", 1.0)) * 2))
        write_json(result_dir / "container-worker-result.json", worker_result)
    return return_code


def optional_psutil():
    try:
        import psutil
    except ImportError:
        return None
    return psutil


def macos_pressure_free_percent() -> float | None:
    if platform.system() != "Darwin":
        return None
    try:
        result = run_command(["memory_pressure", "-Q"], check=False)
    except OSError:
        return None
    match = re.search(r"free percentage:\s*(\d+(?:\.\d+)?)%", result.stdout)
    return float(match.group(1)) if match else None


def macos_swap_used() -> int | None:
    if platform.system() != "Darwin":
        return None
    try:
        result = run_command(["sysctl", "-n", "vm.swapusage"], check=False)
    except OSError:
        return None
    match = re.search(r"used\s*=\s*([\d.]+)([MG])", result.stdout, re.I)
    if match is None:
        return None
    multiplier = 1024**2 if match.group(2).upper() == "M" else GIB
    return int(float(match.group(1)) * multiplier)


def host_memory_sample(
    elapsed: float, pressure_free_percent: float | None
) -> HostSample:
    psutil = optional_psutil()
    total = available = used = swap_used = None
    used_percent = None
    if psutil is not None:
        with suppress(OSError):
            memory = psutil.virtual_memory()
            total = int(memory.total)
            available = int(memory.available)
            used = int(memory.used)
            used_percent = float(memory.percent)
        with suppress(OSError):
            swap_used = int(psutil.swap_memory().used)
    if swap_used is None:
        swap_used = macos_swap_used()
    return HostSample(
        elapsed_seconds=round(elapsed, 3),
        timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        total_bytes=total,
        available_bytes=available,
        used_bytes=used,
        used_percent=used_percent,
        swap_used_bytes=swap_used,
        macos_free_percent=pressure_free_percent,
    )


def last_cgroup_sample(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as stream:
            header = stream.readline().decode("utf-8").rstrip("\r\n")
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            stream.seek(max(0, end - 4096))
            lines = stream.read().decode("utf-8", errors="ignore").splitlines()
    except OSError:
        return {}
    if not header or not lines:
        return {}
    last = lines[-1]
    if last == header and len(lines) > 1:
        last = lines[-2]
    try:
        keys = next(csv.reader([header]))
        values = next(csv.reader([last]))
    except (csv.Error, StopIteration):
        return {}
    return dict(zip(keys, values, strict=False)) if len(keys) == len(values) else {}


def read_cgroup_samples(path: Path) -> list[dict[str, float | int | str]]:
    numeric = set(CGROUP_FIELDS) - {"timestamp_utc"}
    samples: list[dict[str, float | int | str]] = []
    if not path.is_file():
        return samples
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            parsed: dict[str, float | int | str] = {
                "timestamp_utc": row["timestamp_utc"]
            }
            try:
                for key in numeric:
                    parsed[key] = (
                        float(row[key])
                        if key.endswith("avg10") or key == "elapsed_seconds"
                        else int(row[key])
                    )
            except (KeyError, ValueError):
                continue
            samples.append(parsed)
    return samples


def read_host_samples(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not path.is_file():
        return result
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            parsed: dict[str, Any] = {
                "timestamp_utc": row.get("timestamp_utc", ""),
                "elapsed_seconds": float(row["elapsed_seconds"]),
            }
            for key in (
                "total_bytes",
                "available_bytes",
                "used_bytes",
                "swap_used_bytes",
            ):
                parsed[key] = int(row[key]) if row.get(key) else None
            for key in ("used_percent", "macos_free_percent"):
                parsed[key] = float(row[key]) if row.get(key) else None
            result.append(parsed)
    return result


def component_timings(output_dir: Path) -> dict[str, dict[str, float | int]]:
    path = output_dir / "timing_log.csv"
    grouped: dict[str, list[float]] = {}
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            component = row.get("model_name")
            try:
                seconds = float(row["seconds"])
            except (KeyError, TypeError, ValueError):
                continue
            if component:
                grouped.setdefault(component, []).append(seconds)
    return {
        component: {
            "observations": len(values),
            "mean_seconds": statistics.fmean(values),
            "population_standard_deviation_seconds": (
                statistics.pstdev(values) if len(values) > 1 else 0.0
            ),
            "maximum_seconds": max(values),
        }
        for component, values in grouped.items()
    }


def component_intervals(console_log: Path) -> dict[str, list[tuple[float, float]]]:
    intervals: dict[str, list[tuple[float, float]]] = {}
    if not console_log.is_file():
        return intervals
    with console_log.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = COMPONENT_COMPLETION_PATTERN.search(line)
            if match is None:
                continue
            end = float(match["minutes"]) * 60 + float(match["seconds"])
            duration = float(match["duration"])
            intervals.setdefault(match["component"], []).append(
                (max(0.0, end - duration), end)
            )
    return intervals


def component_memory_summary(
    samples: list[dict[str, Any]], console_log: Path
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for component, intervals in component_intervals(console_log).items():
        matching = [
            sample
            for sample in samples
            if any(
                start <= float(sample["elapsed_seconds"]) <= end
                for start, end in intervals
            )
        ]
        if not matching:
            continue
        result[component] = {
            "samples": len(matching),
            "peak_bytes": max(
                int(sample["memory_current_bytes"]) for sample in matching
            ),
            "peak_working_set_bytes": max(
                int(sample["memory_current_bytes"]) - int(sample["inactive_file_bytes"])
                for sample in matching
            ),
            "peak_swap_bytes": max(
                int(sample["swap_current_bytes"]) for sample in matching
            ),
            "peak_full_pressure_avg10": max(
                float(sample["pressure_full_avg10"]) for sample in matching
            ),
        }
    return result


def compatible_completed_result(
    run_dir: Path, expected_spec: dict[str, Any]
) -> RunResult | None:
    if read_json(run_dir / "run-spec.json") != expected_spec:
        return None
    saved = read_json(run_dir / "run-result.json")
    if not saved:
        return None
    try:
        result = RunResult(**saved)
    except TypeError:
        return None
    if not result.succeeded:
        return None
    result.reused = True
    return result


def preserve_failed_run(run_dir: Path) -> None:
    if run_dir.exists():
        run_dir.rename(
            run_dir.with_name(f"{run_dir.name}-incomplete-{timestamp_now()}")
        )


def docker_mount(source: Path, target: str, readonly: bool = False) -> list[str]:
    value = f"type=bind,source={source},target={target}"
    if readonly:
        value += ",readonly"
    return ["--mount", value]


def inspect_container(name: str) -> dict[str, Any]:
    result = run_command(["docker", "inspect", name])
    values = json.loads(result.stdout)
    return values[0] if values else {}


def stop_container(name: str) -> None:
    run_command(["docker", "stop", "--time", "15", name], check=False)


def launch_container_run(
    *,
    definition: dict[str, Any],
    root: Path,
    data_dir: Path,
    cache_dir: Path,
    result_root: Path,
    image_name: str,
    image_id: str,
    model_config_fingerprint: str,
    memory_limit: str,
    shm_size: str,
    sample_interval: float,
    host_pressure_interval: float,
    host_min_available_gib: float,
    host_min_free_percent: float,
    container_abort_percent: float,
    quiet: bool,
) -> RunResult:
    run_dir = result_root / definition["key"]
    run_dir.mkdir(parents=True)
    output_dir = run_dir / "model-output"
    output_dir.mkdir()
    spec = {
        "schema_version": 2,
        "image_id": image_id,
        "model_config_fingerprint": model_config_fingerprint,
        "model_dir": "/workspace/model",
        "data_dir": "/data",
        "result_dir": "/results",
        "output_dir": "/results/model-output",
        "cache_dir": "/cache/model",
        "multiprocess": definition["multiprocess"],
        "workers": definition["workers"],
        "household_sample_size": definition["household_sample_size"],
        "sample_interval": sample_interval,
        "sharrow": "require",
        "memory_limit": memory_limit,
        "shm_size": shm_size,
        "allocator_environment": ALLOCATOR_ENV,
    }
    write_json(run_dir / "run-spec.json", spec)
    container_name = (
        f"lighthouse-production-{result_root.name}-{definition['key']}".lower().replace(
            "_", "-"
        )
    )[:120]
    command = [
        "docker",
        "run",
        "--detach",
        "--name",
        container_name,
        "--memory",
        memory_limit,
        "--memory-swap",
        memory_limit,
        "--shm-size",
        shm_size,
        "--stop-timeout",
        "15",
        "--env",
        "HOME=/tmp/benchmark-home",
        "--env",
        "XDG_CACHE_HOME=/cache/platform",
        "--env",
        "TZ=UTC",
    ]
    for name in THREAD_LIMIT_ENV:
        command.extend(("--env", f"{name}=1"))
    for name, value in ALLOCATOR_ENV.items():
        command.extend(("--env", f"{name}={value}"))
    if os.name == "posix":
        command.extend(("--user", f"{os.getuid()}:{os.getgid()}"))
    command += docker_mount(root, "/workspace", readonly=True)
    command += docker_mount(data_dir, "/data", readonly=True)
    command += docker_mount(run_dir, "/results")
    command += docker_mount(cache_dir, "/cache")
    command.extend((image_name, "--_container-worker", "/results/run-spec.json"))

    if not quiet:
        scope = (
            "full population"
            if definition["household_sample_size"] == 0
            else f"{definition['household_sample_size']:,} households"
        )
        print(
            f"Starting {definition['label']} ({scope}, limit {memory_limit})...",
            flush=True,
        )
    started_wall = dt.datetime.now(dt.timezone.utc)
    started = time.perf_counter()
    container_id = run_command(command).stdout.strip()
    console_path = run_dir / "console.log"
    console_stream = console_path.open("wb")
    logs = subprocess.Popen(
        ["docker", "logs", "--follow", container_id],
        stdout=console_stream,
        stderr=subprocess.STDOUT,
    )
    waiter = subprocess.Popen(
        ["docker", "wait", container_id],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    host_path = run_dir / "host-samples.csv"
    safety_reason: str | None = None
    unsafe_samples = 0
    next_pressure = 0.0
    pressure_free: float | None = None
    memory_limit_bytes = parse_byte_size(memory_limit)
    last_progress_minute = -1
    with host_path.open("w", newline="", encoding="utf-8") as host_stream:
        host_writer = csv.DictWriter(
            host_stream, fieldnames=list(HostSample.__annotations__)
        )
        host_writer.writeheader()
        while waiter.poll() is None:
            elapsed = time.perf_counter() - started
            if elapsed >= next_pressure:
                pressure_free = macos_pressure_free_percent()
                next_pressure = elapsed + host_pressure_interval
            host_sample = host_memory_sample(elapsed, pressure_free)
            host_writer.writerow(asdict(host_sample))
            host_stream.flush()
            cgroup = last_cgroup_sample(run_dir / "cgroup-samples.csv")
            container_current = int(cgroup.get("memory_current_bytes") or 0)
            reasons = []
            if (
                host_sample.available_bytes is not None
                and host_sample.available_bytes < host_min_available_gib * GIB
            ):
                reasons.append(
                    f"host available memory below {host_min_available_gib:g} GiB"
                )
            if pressure_free is not None and pressure_free < host_min_free_percent:
                reasons.append(
                    f"macOS free pressure percentage below {host_min_free_percent:g}%"
                )
            if container_current >= memory_limit_bytes * container_abort_percent / 100:
                reasons.append(
                    f"container memory reached {container_abort_percent:g}% of limit"
                )
            unsafe_samples = unsafe_samples + 1 if reasons else 0
            if unsafe_samples >= 2:
                safety_reason = "; ".join(reasons)
                print(
                    f"SAFETY STOP for {definition['label']}: {safety_reason}",
                    file=sys.stderr,
                    flush=True,
                )
                stop_container(container_id)
                break
            minute = int(elapsed // 60)
            if not quiet and minute > last_progress_minute and minute > 0:
                last_progress_minute = minute
                available_text = format_gib(host_sample.available_bytes)
                print(
                    f"  {definition['label']}: {format_duration(elapsed)}, "
                    f"container {format_gib(container_current)}, "
                    f"host available {available_text}",
                    flush=True,
                )
            time.sleep(sample_interval)

    try:
        waiter_stdout, _ = waiter.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        stop_container(container_id)
        waiter_stdout, _ = waiter.communicate()
    with suppress(subprocess.TimeoutExpired):
        logs.wait(timeout=30)
    if logs.poll() is None:
        logs.terminate()
        logs.wait(timeout=10)
    console_stream.close()
    duration = time.perf_counter() - started
    state = inspect_container(container_id).get("State", {})
    return_code = int(state.get("ExitCode", waiter_stdout.strip() or 1))
    oom_killed = bool(state.get("OOMKilled", False))
    run_command(["docker", "rm", container_id], check=False)

    cgroup_samples = read_cgroup_samples(run_dir / "cgroup-samples.csv")
    host_samples = read_host_samples(host_path)
    peak = max(
        (int(sample["memory_peak_bytes"]) for sample in cgroup_samples), default=0
    )
    peak_working_set = max(
        (
            int(sample["memory_current_bytes"]) - int(sample["inactive_file_bytes"])
            for sample in cgroup_samples
        ),
        default=0,
    )
    peak_swap = max(
        (int(sample["swap_current_bytes"]) for sample in cgroup_samples), default=0
    )
    host_available = [
        int(sample["available_bytes"])
        for sample in host_samples
        if sample["available_bytes"] is not None
    ]
    host_used = [
        int(sample["used_bytes"])
        for sample in host_samples
        if sample["used_bytes"] is not None
    ]
    host_pressure = [
        float(sample["macos_free_percent"])
        for sample in host_samples
        if sample["macos_free_percent"] is not None
    ]
    result = RunResult(
        key=definition["key"],
        label=definition["label"],
        workers=definition["workers"],
        household_sample_size=definition["household_sample_size"],
        started_at=started_wall.isoformat(),
        duration_seconds=duration,
        return_code=return_code,
        oom_killed=oom_killed,
        safety_abort_reason=safety_reason,
        container_name=container_name,
        container_peak_bytes=peak,
        container_peak_working_set_bytes=peak_working_set,
        container_peak_swap_bytes=peak_swap,
        host_min_available_bytes=min(host_available) if host_available else None,
        host_peak_used_bytes=max(host_used) if host_used else None,
        host_min_macos_free_percent=min(host_pressure) if host_pressure else None,
        output_dir=str(output_dir),
        console_log=str(console_path),
        cgroup_samples=str(run_dir / "cgroup-samples.csv"),
        host_samples=str(host_path),
        component_timings=component_timings(output_dir),
        component_memory=component_memory_summary(cgroup_samples, console_path),
        output_summary=read_json(run_dir / "output-summary.json"),
    )
    write_json(run_dir / "run-result.json", asdict(result))
    if not quiet:
        status = "completed" if result.succeeded else "failed"
        print(
            f"Finished {result.label}: {status}, {format_duration(duration)}, "
            f"cgroup peak {format_gib(peak)}",
            flush=True,
        )
    return result


def format_duration(seconds: float) -> str:
    whole_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return (
        f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"
    )


def format_gib(value: int | None) -> str:
    return "—" if value is None else f"{value / GIB:.2f} GiB"


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{header}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def runtime_table(results: list[RunResult], report_dir: Path, target: int) -> str:
    rows = []
    for result in (result for result in results if result.key != "cache-warmup"):
        if result.succeeded and result.container_peak_bytes <= target:
            status = "QUALIFIES"
        elif result.oom_killed:
            status = "OOM killed"
        elif result.safety_abort_reason:
            status = "Safety stop"
        elif result.succeeded:
            status = "Over target"
        else:
            status = f"Failed ({result.return_code})"
        if result.reused:
            status += " · reused"
        log = os.path.relpath(result.console_log, report_dir)
        rows.append(
            [
                html.escape(result.label),
                str(result.workers),
                status,
                format_duration(result.duration_seconds),
                format_gib(result.container_peak_bytes),
                format_gib(result.container_peak_working_set_bytes),
                format_gib(result.container_peak_swap_bytes),
                format_gib(result.host_min_available_bytes),
                (
                    f"{result.host_min_macos_free_percent:.1f}%"
                    if result.host_min_macos_free_percent is not None
                    else "—"
                ),
                f'<a href="{html.escape(log)}">log</a>',
            ]
        )
    return html_table(
        [
            "Run",
            "Workers",
            "Status",
            "Runtime",
            "Cgroup peak",
            "Peak working set",
            "Peak swap",
            "Min host available",
            "Min macOS free",
            "Console",
        ],
        rows,
    )


def component_table(results: list[RunResult]) -> str:
    rows: list[list[str]] = []
    for result in results:
        if result.key == "cache-warmup":
            continue
        for component, timing in sorted(
            result.component_timings.items(),
            key=lambda item: float(item[1]["mean_seconds"]),
            reverse=True,
        ):
            memory = result.component_memory.get(component, {})
            observations = int(timing["observations"])
            rows.append(
                [
                    str(result.workers),
                    f"<code>{html.escape(component)}</code>",
                    f"{float(timing['mean_seconds']):.3f}",
                    (
                        f"{float(timing['population_standard_deviation_seconds']):.3f}"
                        if observations > 1
                        else "—"
                    ),
                    str(observations),
                    format_gib(int(memory["peak_bytes"])) if memory else "—",
                    (
                        f"{float(memory['peak_full_pressure_avg10']):.2f}%"
                        if memory
                        else "—"
                    ),
                ]
            )
    return html_table(
        [
            "Workers",
            "Component",
            "Mean seconds",
            "Population SD",
            "N",
            "Peak cgroup memory",
            "Peak full PSI avg10",
        ],
        rows,
    )


def memory_chart(results: list[RunResult]) -> str:
    series = []
    colors = ["#2563eb", "#ea580c", "#16a34a", "#9333ea", "#0891b2"]
    measured = [result for result in results if result.key != "cache-warmup"]
    for index, result in enumerate(measured):
        samples = read_cgroup_samples(Path(result.cgroup_samples))
        if samples:
            series.append((result.label, colors[index % len(colors)], samples))
    if not series:
        return '<p class="empty">No cgroup samples available.</p>'
    width, height = 1050, 430
    left, right, top, bottom = 75, 25, 25, 55
    plot_w, plot_h = width - left - right, height - top - bottom
    max_x = max(
        float(sample["elapsed_seconds"])
        for _, _, samples in series
        for sample in samples
    )
    max_y = max(
        int(sample["memory_current_bytes"])
        for _, _, samples in series
        for sample in samples
    )
    max_x = max(max_x, 1.0)
    max_y = max(max_y, GIB)

    def x(value: float) -> float:
        return left + value / max_x * plot_w

    def y(value: int) -> float:
        return top + (1 - value / max_y) * plot_h

    elements = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for tick in range(6):
        value = max_y * tick / 5
        yy = y(int(value))
        elements.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" '
            f'y2="{yy:.1f}" stroke="#e2e8f0"/>'
            f'<text x="{left - 10}" y="{yy + 4:.1f}" text-anchor="end">'
            f"{value / GIB:.0f}</text>"
        )
    for label, color, samples in series:
        points = " ".join(
            f"{x(float(sample['elapsed_seconds'])):.1f},"
            f"{y(int(sample['memory_current_bytes'])):.1f}"
            for sample in samples
        )
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2"/>'
        )
    elements.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" '
        'stroke="#64748b"/>'
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" '
        f'y2="{height - bottom}" stroke="#64748b"/>'
        f'<text x="18" y="{top + plot_h / 2}" '
        f'transform="rotate(-90 18 {top + plot_h / 2})">Cgroup memory (GiB)</text>'
        f'<text x="{left + plot_w / 2}" y="{height - 12}" text-anchor="middle">'
        "Elapsed time (seconds)</text></svg>"
    )
    legend = "".join(
        f'<span><i style="background:{color}"></i>{html.escape(label)}</span>'
        for label, color, _ in series
    )
    return f'<div class="legend">{legend}</div>{"".join(elements)}'


def output_comparison(results: list[RunResult]) -> str:
    rows = []
    for result in results:
        if result.key == "cache-warmup":
            continue
        summary = result.output_summary
        rows.append(
            [
                str(result.workers),
                f"{int(summary['tours']):,}" if "tours" in summary else "—",
                f"{int(summary['trips']):,}" if "trips" in summary else "—",
                (
                    f"{float(summary['trips_per_tour']):.4f}"
                    if summary.get("trips_per_tour") is not None
                    else "—"
                ),
            ]
        )
    return html_table(["Workers", "Tours", "Trips", "Trips / tour"], rows)


def output_stability(results: list[RunResult]) -> str:
    measured = [
        result
        for result in results
        if result.key != "cache-warmup" and result.succeeded and result.output_summary
    ]
    if not measured:
        return '<p class="empty">No successful outputs available.</p>'
    baseline = measured[0]
    dimensions = (
        ("tour_categories", "Tour category"),
        ("tour_types", "Tour type"),
        ("trip_modes", "Trip mode"),
        ("trip_primary_purposes", "Trip primary purpose"),
    )
    rows: list[list[str]] = []
    for result in measured:
        for key, label in dimensions:
            base_counts = baseline.output_summary.get(key, {})
            counts = result.output_summary.get(key, {})
            base_total = sum(int(value) for value in base_counts.values())
            total = sum(int(value) for value in counts.values())
            categories = set(base_counts) | set(counts)
            if not categories or not base_total or not total:
                maximum_delta = total_variation = 0.0
                largest_category = "—"
            else:
                deltas = {
                    category: abs(
                        int(counts.get(category, 0)) / total
                        - int(base_counts.get(category, 0)) / base_total
                    )
                    for category in categories
                }
                largest_category = max(deltas, key=deltas.get)
                maximum_delta = deltas[largest_category] * 100
                total_variation = sum(deltas.values()) * 50
            rows.append(
                [
                    str(result.workers),
                    label,
                    html.escape(str(largest_category)),
                    f"{maximum_delta:.4f} pp",
                    f"{total_variation:.4f} pp",
                    "reference"
                    if result is baseline
                    else f"vs {baseline.workers} workers",
                ]
            )
    return html_table(
        [
            "Workers",
            "Distribution",
            "Largest-shift category",
            "Maximum share shift",
            "Total variation",
            "Comparison",
        ],
        rows,
    )


def input_summary_table(summary: dict[str, Any]) -> str:
    if not summary:
        return '<p class="empty">Input summary unavailable.</p>'
    rows = [
        ["Synthetic households", f"{int(summary.get('households', 0)):,}"],
        ["Synthetic persons", f"{int(summary.get('persons', 0)):,}"],
        [
            "Persons / household",
            f"{float(summary.get('persons_per_household', 0)):.3f}",
        ],
        ["Land-use zones", f"{int(summary.get('zones', 0)):,}"],
        ["Land-use population", f"{float(summary.get('land_use_population', 0)):,.0f}"],
        ["Land-use employment", f"{float(summary.get('land_use_employment', 0)):,.0f}"],
        ["OMX skim files", f"{int(summary.get('skim_files', 0)):,}"],
        ["Compressed skim bytes", format_gib(int(summary.get("skim_bytes", 0)))],
    ]
    return html_table(["Input statistic", "Value"], rows)


def host_memory_chart(results: list[RunResult]) -> str:
    series = []
    colors = ["#2563eb", "#ea580c", "#16a34a", "#9333ea", "#0891b2"]
    measured = [result for result in results if result.key != "cache-warmup"]
    for index, result in enumerate(measured):
        samples = [
            sample
            for sample in read_host_samples(Path(result.host_samples))
            if sample.get("available_bytes") is not None
        ]
        if samples:
            series.append((result.label, colors[index % len(colors)], samples))
    if not series:
        return '<p class="empty">No host samples available.</p>'
    width, height = 1050, 350
    left, right, top, bottom = 75, 25, 25, 55
    plot_w, plot_h = width - left - right, height - top - bottom
    max_x = max(
        float(sample["elapsed_seconds"])
        for _, _, samples in series
        for sample in samples
    )
    max_y = max(
        int(sample["available_bytes"]) for _, _, samples in series for sample in samples
    )
    max_x = max(max_x, 1.0)
    max_y = max(max_y, GIB)

    def x(value: float) -> float:
        return left + value / max_x * plot_w

    def y(value: int) -> float:
        return top + (1 - value / max_y) * plot_h

    elements = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for tick in range(6):
        value = max_y * tick / 5
        yy = y(int(value))
        elements.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" '
            f'y2="{yy:.1f}" stroke="#e2e8f0"/>'
            f'<text x="{left - 10}" y="{yy + 4:.1f}" text-anchor="end">'
            f"{value / GIB:.0f}</text>"
        )
    for _, color, samples in series:
        points = " ".join(
            f"{x(float(sample['elapsed_seconds'])):.1f},"
            f"{y(int(sample['available_bytes'])):.1f}"
            for sample in samples
        )
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2"/>'
        )
    elements.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" '
        'stroke="#64748b"/>'
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" '
        f'y2="{height - bottom}" stroke="#64748b"/>'
        f'<text x="18" y="{top + plot_h / 2}" '
        f'transform="rotate(-90 18 {top + plot_h / 2})">Host available (GiB)</text>'
        f'<text x="{left + plot_w / 2}" y="{height - 12}" text-anchor="middle">'
        "Elapsed time (seconds)</text></svg>"
    )
    legend = "".join(
        f'<span><i style="background:{color}"></i>{html.escape(label)}</span>'
        for label, color, _ in series
    )
    return f'<div class="legend">{legend}</div>{"".join(elements)}'


def write_report(
    path: Path,
    *,
    results: list[RunResult],
    metadata: dict[str, Any],
    qualification_peak: int,
) -> None:
    successful = [
        result
        for result in results
        if result.key != "cache-warmup"
        and result.succeeded
        and result.container_peak_bytes <= qualification_peak
    ]
    fastest = min(successful, key=lambda result: result.duration_seconds, default=None)
    recommendation = (
        f"Fastest qualifying run: <strong>{fastest.workers} workers</strong>, "
        f"{format_duration(fastest.duration_seconds)}, peak "
        f"{format_gib(fastest.container_peak_bytes)}."
        if fastest is not None
        else "No measured run met the qualification criteria."
    )
    requested = metadata["household_sample_size"]
    requested_text = "Full population" if requested == 0 else f"{requested:,}"
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lighthouse production benchmark</title>
<style>
body{{font:14px system-ui,sans-serif;color:#172033;background:#f5f7fa;margin:0;padding:30px}}main{{max-width:1250px;margin:auto}}
h1{{margin-bottom:5px}}h2{{margin-top:30px}}.note{{color:#526071;line-height:1.55}}.card,.chart{{background:#fff;border:1px solid #dbe3ec;border-radius:10px;padding:17px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}.value{{font-size:18px;font-weight:650;margin-top:6px}}
table{{width:100%;border-collapse:collapse;background:#fff;display:block;overflow-x:auto}}th,td{{padding:9px 11px;border:1px solid #e2e8f0;white-space:nowrap;text-align:right}}th{{background:#f8fafc}}th:first-child,td:first-child{{text-align:left}}
.legend{{display:flex;gap:16px;flex-wrap:wrap}}.legend span{{display:flex;align-items:center;gap:6px}}.legend i{{width:20px;height:3px;display:inline-block}}svg{{width:100%;height:auto}}svg text{{font-size:11px;fill:#64748b}}code{{font-family:ui-monospace,monospace}}.empty{{padding:25px;text-align:center;color:#64748b}}
</style></head><body><main>
<h1>Lighthouse production benchmark</h1>
<p class="note">Generated {html.escape(metadata["generated_at"])}. Linux containers use Sharrow <code>require</code>, locked dependencies, synchronous skim loading, and externally enforced memory limits. ActivitySim profiling and diagnostic instrumentation are disabled.</p>
<section class="cards">
<div class="card"><div>Memory limit</div><div class="value">{html.escape(metadata["memory_limit"])}</div></div>
<div class="card"><div>Qualification peak</div><div class="value">{format_gib(qualification_peak)}</div></div>
<div class="card"><div>Households requested</div><div class="value">{requested_text}</div></div>
<div class="card"><div>Recommendation</div><div class="value">{recommendation}</div></div>
</section>
	<h2>Runtime and memory qualification</h2>{runtime_table(results, path.parent, qualification_peak)}
	<p class="note">Cgroup peak counts memory charged to the complete container without multiplying pages shared by workers. Working set subtracts inactive file cache. A run qualifies only when it succeeds below the configured qualification peak.</p>
	<h2>Progressive cgroup memory</h2><div class="chart">{memory_chart(results)}</div>
	<h2>Progressive host memory headroom</h2><div class="chart">{host_memory_chart(results)}</div>
	<p class="note">Host available memory includes Docker Desktop and unrelated macOS processes. It is monitored independently from the container limit and used by the safety stop.</p>
	<h2>Per-component runtime and memory</h2>{component_table(results)}
	<p class="note">Component memory is the highest whole-container sample observed while at least one worker was executing the component. Overlap makes these values diagnostic rather than additive.</p>
	<h2>Input data</h2>{input_summary_table(metadata["input_summary"])}
	<h2>Output totals</h2>{output_comparison(results)}
	<h2>Output stability</h2>{output_stability(results)}
	<p class="note">Share shifts compare every successful worker-count run with the first successful run. Total variation is half the sum of absolute category-share differences.</p>
<p class="note"><strong>Data:</strong> <code>{html.escape(metadata["data_dir"])}</code><br><strong>Image:</strong> <code>{html.escape(metadata["image_id"])}</code><br><strong>Host:</strong> {html.escape(metadata["host_platform"])}</p>
</main></body></html>"""
    path.write_text(report, encoding="utf-8")


def worker_definitions(args: argparse.Namespace) -> list[dict[str, Any]]:
    definitions = [
        {
            "key": "cache-warmup",
            "label": "Sharrow cache warm-up",
            "multiprocess": False,
            "workers": 1,
            "household_sample_size": args.warmup_households,
        }
    ]
    scope = (
        "full" if args.sample_households == 0 else f"sample-{args.sample_households}"
    )
    population = (
        "Full population"
        if args.sample_households == 0
        else f"{args.sample_households:,} households"
    )
    definitions.extend(
        {
            "key": f"{scope}-p{workers:02d}",
            "label": f"{population} — {workers} workers",
            "multiprocess": True,
            "workers": workers,
            "household_sample_size": args.sample_households,
        }
        for workers in sorted(args.processes)
    )
    return definitions


def expected_run_spec(
    definition: dict[str, Any],
    image_id: str,
    model_config_fingerprint: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "image_id": image_id,
        "model_config_fingerprint": model_config_fingerprint,
        "model_dir": "/workspace/model",
        "data_dir": "/data",
        "result_dir": "/results",
        "output_dir": "/results/model-output",
        "cache_dir": "/cache/model",
        "multiprocess": definition["multiprocess"],
        "workers": definition["workers"],
        "household_sample_size": definition["household_sample_size"],
        "sample_interval": args.sample_interval,
        "sharrow": "require",
        "memory_limit": args.memory_limit,
        "shm_size": args.shm_size,
    }


def main_host(args: argparse.Namespace) -> int:
    if args.sample_interval <= 0 or args.host_pressure_interval <= 0:
        raise ValueError("sampling intervals must be greater than zero")
    if max(args.processes) > 10:
        raise ValueError("--processes cannot exceed 10")
    if len(set(args.processes)) != len(args.processes):
        raise ValueError("--processes values must be unique")

    root = repository_root()
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_root = result_directory(root, args.output_dir, args.resume)
    docker_info = require_docker()
    docker_memory = int(docker_info.get("MemTotal") or 0)
    requested_memory = parse_byte_size(args.memory_limit)
    if docker_memory < requested_memory + 2 * GIB:
        raise RuntimeError(
            "Docker Desktop/engine has only "
            f"{format_gib(docker_memory)} available; at least "
            f"{format_gib(requested_memory + 2 * GIB)} is required for a "
            f"{args.memory_limit} container plus VM overhead. Restart Docker Desktop "
            "if you recently raised its allocation."
        )

    image_name = args.image or locked_image_name(root)
    if not args.no_build:
        build_image(root, image_name, args.quiet)
    image_id = docker_image_id(image_name)
    model_config_fingerprint = config_fingerprint(root)
    definitions = worker_definitions(args)

    if not args.quiet:
        print(f"Data: {data_dir}", flush=True)
        print(f"Outputs: {result_root}", flush=True)
        print(
            f"Docker: {docker_info.get('OperatingSystem')} "
            f"{docker_info.get('Architecture')}, {format_gib(docker_memory)} VM memory",
            flush=True,
        )
        print(
            f"Container limit: {args.memory_limit}; /dev/shm: {args.shm_size}; "
            f"workers: {sorted(args.processes)}",
            flush=True,
        )

    results: list[RunResult] = []
    for definition in definitions:
        run_dir = result_root / definition["key"]
        expected_spec = expected_run_spec(
            definition, image_id, model_config_fingerprint, args
        )
        if args.resume:
            reused = compatible_completed_result(run_dir, expected_spec)
            if reused is not None:
                results.append(reused)
                if not args.quiet:
                    print(f"Reusing {reused.label}", flush=True)
                continue
            preserve_failed_run(run_dir)

        result = launch_container_run(
            definition=definition,
            root=root,
            data_dir=data_dir,
            cache_dir=cache_dir,
            result_root=result_root,
            image_name=image_name,
            image_id=image_id,
            model_config_fingerprint=model_config_fingerprint,
            memory_limit=args.memory_limit,
            shm_size=args.shm_size,
            sample_interval=args.sample_interval,
            host_pressure_interval=args.host_pressure_interval,
            host_min_available_gib=args.host_min_available_gib,
            host_min_free_percent=args.host_min_free_percent,
            container_abort_percent=args.container_abort_percent,
            quiet=args.quiet,
        )
        results.append(result)
        if not result.succeeded and not args.continue_after_failure:
            break

    input_summary: dict[str, Any] = {}
    for result in results:
        input_summary = read_json(Path(result.output_dir).parent / "input-summary.json")
        if input_summary:
            break
    metadata = {
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "data_dir": str(data_dir),
        "input_summary": input_summary,
        "household_sample_size": args.sample_households,
        "worker_counts": sorted(args.processes),
        "memory_limit": args.memory_limit,
        "qualification_peak": args.qualification_peak,
        "shm_size": args.shm_size,
        "image_name": image_name,
        "image_id": image_id,
        "model_config_fingerprint": model_config_fingerprint,
        "host_platform": platform.platform(),
        "docker_info": {
            key: docker_info.get(key)
            for key in (
                "ServerVersion",
                "OperatingSystem",
                "Architecture",
                "NCPU",
                "MemTotal",
                "CgroupVersion",
            )
        },
    }
    write_json(
        result_root / "metrics.json",
        {"metadata": metadata, "runs": [asdict(result) for result in results]},
    )
    report_path = result_root / "report.html"
    write_report(
        report_path,
        results=results,
        metadata=metadata,
        qualification_peak=parse_byte_size(args.qualification_peak),
    )
    if not args.quiet:
        print(f"Report: {report_path}", flush=True)
        with suppress(Exception):
            webbrowser.open(report_path.as_uri())
    return (
        0
        if len(results) == len(definitions)
        and all(result.succeeded for result in results)
        else 1
    )


def main() -> int:
    args = parse_args()
    if args._container_worker is not None:
        return container_worker(args._container_worker)
    return main_host(args)


if __name__ == "__main__":
    raise SystemExit(main())
