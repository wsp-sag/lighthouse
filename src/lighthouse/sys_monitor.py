#!/usr/bin/env python3
"""
Simple CPU and memory tracker for Windows (also works cross-platform).

- Default interval: 0.5 seconds
- Prints timestamp, CPU %, Memory Used (GB), Memory Available (GB), Available Memory (%)
- Optional CSV logging with --csv FILE
- Optional program log tracking with --log FILE
- Sequence-aware step inference: only advances when steps complete in order
- Wait-for-all-workers for mp_households (requires all mp_households_N workers to complete a step)
- Phase-aware display (mp_initialize, mp_households, mp_summarize)
- Step status shown (running; for households shows done/total workers)

Usage examples:
    python sys_monitor.py
    python sys_monitor.py --interval 1.0
    python sys_monitor.py --csv sys_usage.csv
    python sys_monitor.py --log path/to/activitysim.log
"""

import sys
import os
import time
import argparse
from datetime import datetime
import re
from typing import Optional, Dict, Set, Tuple

try:
    import psutil
except ImportError:
    print(
        "ERROR: 'psutil' is not installed.\n"
        "Install it with:\n    python -m pip install psutil",
        file=sys.stderr,
    )
    sys.exit(1)


# set working directory to current debug session
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# --- Ordered step list and phase mapping -------------------------------------

STEP_NAMES = [
    ### mp_initialize step
    "initialize_landuse",
    "initialize_households",
    ### mp_accessibility step
    "compute_accessibility",
    ### mp_households step
    "school_location",
    "workplace_location",
    "auto_ownership_simulate",
    "free_parking",
    "cdap_simulate",
    "mandatory_tour_frequency",
    "mandatory_tour_scheduling",
    "non_mandatory_tour_frequency",
    "non_mandatory_tour_destination",
    "non_mandatory_tour_scheduling",
    "tour_mode_choice_simulate",
    "atwork_subtour_frequency",
    "atwork_subtour_destination",
    "atwork_subtour_scheduling",
    "atwork_subtour_mode_choice",
    "stop_frequency",
    "trip_purpose",
    "trip_destination",
    "trip_purpose_and_destination",
    "trip_scheduling",
    "trip_mode_choice",
    ### mp_summarize step
    "write_data_dictionary",
    "write_trip_matrices",
    "write_tables",
]
STEP_TO_PHASE: Dict[str, str] = {}
for name in STEP_NAMES:
    if name in {
        "initialize_landuse",
        "initialize_households",
    }:
        STEP_TO_PHASE[name] = "mp_initialize"
    elif name in {
        "compute_accessibility"
    }:
        STEP_TO_PHASE[name] = "mp_accessibility"
    elif name in {
        "write_data_dictionary",
        "track_skim_usage",
        "write_trip_matrices",
        "write_tables",
    }:
        STEP_TO_PHASE[name] = "mp_summarize"
    else:
        STEP_TO_PHASE[name] = "mp_households"


# --- Regex patterns (strict to avoid false matches) --------------------------

# Completion lines come from activitysim.core.mp_tasks and include a runtime like ": 12.34 seconds"
# Examples:
#  "INFO - activitysim.core.mp_tasks - mp_initialize initialize_landuse : 0.636 seconds"
#  "INFO - activitysim.core.mp_tasks - mp_households_3 school_location : 290.146 seconds"
#  "INFO - activitysim.core.mp_tasks - mp_summarize write_tables : 12.34 seconds"
PAT_COMPLETED_INIT_SUM = re.compile(
    rf"activitysim\.core\.mp_tasks\s*-\s*mp_(?:initialize|accessibility|summarize)\s+"
    rf"({'|'.join(map(re.escape, STEP_NAMES))})\s*:\s*([\d.]+)\s+seconds\b",
    re.IGNORECASE,
)
PAT_COMPLETED_ACCESS = re.compile(
    rf"activitysim\.core\.mp_tasks\s*-\s*mp_accessibility_(\d+)\s+"
    rf"({'|'.join(map(re.escape, STEP_NAMES))})\s*:\s*([\d.]+)\s+seconds\b",
    re.IGNORECASE,
)
PAT_COMPLETED_HH = re.compile(
    rf"activitysim\.core\.mp_tasks\s*-\s*mp_households_(\d+)\s+"
    rf"({'|'.join(map(re.escape, STEP_NAMES))})\s*:\s*([\d.]+)\s+seconds\b",
    re.IGNORECASE,
)

# Worker start lines (to infer the mp_households worker count)
# Example: "INFO - activitysim.core.mp_tasks - start process mp_households_7"
PAT_START_WORKER = re.compile(
    r"activitysim\.core\.mp_tasks\s*-\s*start process mp_households_(\d+)\b",
    re.IGNORECASE,
)
PAT_START_ACCESS_WORKER = re.compile(
    r"activitysim\.core\.mp_tasks\s*-\s*start process mp_accessibility_(\d+)\b",
    re.IGNORECASE,
)

# Phase run lines (optional; for additional robustness/UX)
# Example: "INFO - activitysim.core.mp_tasks - run_sub_simulations step mp_households models ..."
PAT_RUN_PHASE = re.compile(
    r"activitysim\.core\.mp_tasks\s*-\s*run_sub_simulations step (mp_initialize|mp_accessibility|mp_households|mp_summarize)\b",
    re.IGNORECASE,
)


# --- Utilities ---------------------------------------------------------------


def format_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def bytes_to_gb(n_bytes: int) -> float:
    return n_bytes / (1024**3)


def _read_log_tail(log_path: str, max_read_bytes: int) -> str:
    """Read the last max_read_bytes of the log (UTF-8 decode, ignore errors)."""
    try:
        size = os.path.getsize(log_path)
        start = max(0, size - max_read_bytes)
        with open(log_path, "rb") as f:  # read-only, avoids writer contention
            if start:
                f.seek(start)
            chunk = f.read()
        return chunk.decode("utf-8", errors="ignore") if chunk else ""
    except (FileNotFoundError, PermissionError, OSError):
        return ""


# --- State and inference from log -------------------------------------------


class StepTracker:
    """
    Tracks step completion strictly by sequence.
    For mp_households, a step completes only after all known workers complete it.
    """

    def __init__(self) -> None:
        # Index of last completed step in STEP_NAMES (-1 means nothing completed yet)
        self.last_completed_idx: int = -1
        # Set of worker ids (ints) seen via "start process mp_households_N"
        self.active_workers: Set[int] = set()
        # Set of worker ids (ints) seen via "start process mp_accessibility_N"
        self.active_access_workers: Set[int] = set()
        # For accessibility, which workers have completed the step
        self.access_completed_workers: Dict[str, Set[int]] = {}
        # For the current households step, which workers have completed
        self.hh_completed_workers: Dict[str, Set[int]] = {}
        # Keep a cache of seen completion tokens to avoid re-processing duplicates
        # token: tuple('init/sum' or 'hh', phase_id or worker_id, step_name)
        self.seen_completions: Set[Tuple[str, str, str]] = set()
        # Optional: track last observed/announced phase
        self.last_observed_phase: Optional[str] = None

    def _phase_of_index(self, idx: int) -> Optional[str]:
        if 0 <= idx < len(STEP_NAMES):
            return STEP_TO_PHASE[STEP_NAMES[idx]]
        return None

    def update_from_tail(self, text: str) -> None:
        """Parse the tail text and update internal state (workers and completions)."""
        if not text:
            return

        # 1) Record any new workers that started (used to know how many must complete a household step)
        for m in PAT_START_WORKER.finditer(text):
            wid = int(m.group(1))
            self.active_workers.add(wid)

        for m in PAT_START_ACCESS_WORKER.finditer(text):
            wid = int(m.group(1))
            self.active_access_workers.add(wid)

        # 2) Record phase runs (optional, just for UX)
        for m in PAT_RUN_PHASE.finditer(text):
            self.last_observed_phase = m.group(1)

        # 3) Record completions (init/summarize)
        for m in PAT_COMPLETED_INIT_SUM.finditer(text):
            step_name = m.group(1)
            token = ("init/sum", "single", step_name)
            if token in self.seen_completions:
                continue
            self.seen_completions.add(token)
            # We only mark completion during resolution below (respecting order)

        # 3b) Record completions (accessibility per-worker)
        for m in PAT_COMPLETED_ACCESS.finditer(text):
            wid = m.group(1)
            step_name = m.group(2)
            token = ("access", wid, step_name)
            if token in self.seen_completions:
                continue
            self.seen_completions.add(token)
            self.access_completed_workers.setdefault(step_name, set()).add(int(wid))

        # 4) Record completions (households per-worker)
        for m in PAT_COMPLETED_HH.finditer(text):
            wid = m.group(1)
            step_name = m.group(2)
            token = ("hh", wid, step_name)
            if token in self.seen_completions:
                continue
            self.seen_completions.add(token)
            # Cache worker completion for this step
            self.hh_completed_workers.setdefault(step_name, set()).add(int(wid))

        # 5) Resolve sequence strictly, possibly advancing multiple steps if evidence exists
        advanced = True
        while advanced:
            advanced = False
            next_idx = self.last_completed_idx + 1
            if next_idx >= len(STEP_NAMES):
                break

            next_step = STEP_NAMES[next_idx]
            phase = STEP_TO_PHASE[next_step]

            if phase in ("mp_initialize", "mp_summarize"):
                # Completion is true if we have a completion token for this step
                token = ("init/sum", "single", next_step)
                if token in self.seen_completions:
                    self.last_completed_idx = next_idx
                    advanced = True
                    # Clean any stale household worker sets for previous steps
                    self.hh_completed_workers.pop(next_step, None)
                    continue
                # not completed yet -> stop resolving
                break

            if phase == "mp_accessibility":
                completed_workers = self.access_completed_workers.get(next_step, set())
                if self.active_access_workers and completed_workers >= self.active_access_workers:
                    self.last_completed_idx = next_idx
                    advanced = True
                    self.access_completed_workers.pop(next_step, None)
                    continue
                break

            # mp_households: require all active workers to complete the step
            if phase == "mp_households":
                completed_workers = self.hh_completed_workers.get(next_step, set())
                # If we don't yet know active workers, we can't mark complete.
                # We'll wait until starts are seen; still show running status with ?/? below.
                if self.active_workers and completed_workers >= self.active_workers:
                    # step completed by all workers
                    self.last_completed_idx = next_idx
                    advanced = True
                    # prepare for next step: keep hh_completed_workers but this step is done
                    # (we can drop its set to save memory)
                    self.hh_completed_workers.pop(next_step, None)
                    continue
                # not completed yet -> stop resolving
                break

    def current_step_info(
        self,
    ) -> Tuple[Optional[int], Optional[str], str, Optional[int], Optional[int]]:
        """
        Returns:
          - current_step_idx (None if all done),
          - current_step_name (None if all done),
          - phase string ("mp_initialize"|"mp_households"|"mp_summarize"| "DONE"),
          - done_workers (for household phase, else None),
          - total_workers (for household phase, else None)
        """
        idx = self.last_completed_idx + 1
        if idx >= len(STEP_NAMES):
            return None, None, "DONE", None, None

        step_name = STEP_NAMES[idx]
        phase = STEP_TO_PHASE[step_name]

        if phase == "mp_households":
            done = len(self.hh_completed_workers.get(step_name, set()))
            total = len(self.active_workers) if self.active_workers else None
            return idx, step_name, phase, done, total

        return idx, step_name, phase, None, None


# --- Monitor loop ------------------------------------------------------------


def monitor(
    interval: float = 0.5,
    csv_path: Optional[str] = None,
    log_path: Optional[str] = None,
    tail_bytes: int = 256 * 1024,  # bigger default to improve catch-up robustness
    parent_pid: Optional[int] = None,
    delay_seconds: float = 0,
) -> None:
    # Delay start if requested
    if delay_seconds > 0:
        print(f"Delaying sys_monitor start for {delay_seconds} seconds...")
        time.sleep(delay_seconds)
    
    # Warm up psutil's CPU measurement for better first reading
    psutil.cpu_percent(interval=None)

    csv_file = None
    tracker = StepTracker()
    parent_process = None
    
    # If parent PID provided, get parent process object
    if parent_pid is not None:
        try:
            parent_process = psutil.Process(parent_pid)
            print(f"Monitoring parent process PID {parent_pid} ({parent_process.name()})")
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"Warning: Cannot access parent process {parent_pid}: {e}")
            print("Continuing without parent process monitoring.")
            parent_process = None

    try:
        if csv_path:
            # Create file and header if new, else append
            file_exists = False
            try:
                with open(csv_path, "r", encoding="utf-8"):
                    file_exists = True
            except FileNotFoundError:
                file_exists = False

            csv_file = open(csv_path, "a", encoding="utf-8", newline="")
            if not file_exists:
                csv_file.write(
                    "timestamp,cpu_percent,memory_used_gb,memory_available_gb,available_memory_pct,phase,step_index,step_name,status,workers_done,workers_total\n"
                )
                csv_file.flush()

        print("Press Ctrl+C to stop.")
        total_steps = len(STEP_NAMES)

        while True:
            # Check if parent process is still alive
            if parent_process is not None:
                try:
                    if not parent_process.is_running():
                        print(f"\nParent process {parent_pid} has exited. Stopping monitor.")
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    print(f"\nParent process {parent_pid} no longer accessible. Stopping monitor.")
                    break
            ts = format_now()
            cpu = psutil.cpu_percent(interval=None)  # non-blocking snapshot

            vm = psutil.virtual_memory()
            total_gb = bytes_to_gb(vm.total)
            avail_gb = bytes_to_gb(vm.available)
            used_gb = total_gb - avail_gb
            avail_pct = (vm.available / vm.total) * 100 if vm.total else 0.0

            # Update tracker from log (if provided)
            if log_path:
                text = _read_log_tail(log_path, max_read_bytes=tail_bytes)
                tracker.update_from_tail(text)

            curr_idx, curr_step, phase, done_workers, total_workers = (
                tracker.current_step_info()
            )

            # Build status string
            if phase == "DONE":
                status = "finished"
                display_phase = "DONE"
                step_progress = f"{total_steps}/{total_steps}"
                step_display = "DONE"
                workers_display = ""
            else:
                status = "running"
                display_phase = phase
                step_progress = f"{(curr_idx or 0) + 1}/{total_steps}"
                if phase == "mp_households":
                    if total_workers is None:
                        workers_display = " (workers: ?/?)"
                    else:
                        workers_display = f" (workers: {done_workers}/{total_workers})"
                else:
                    workers_display = ""
                step_display = f"{curr_step}{workers_display} ({status})"

            line = (
                f"{ts} | CPU: {cpu:5.1f}% | "
                f"Used: {used_gb:6.2f} GB | "
                f"Available: {avail_gb:6.2f} GB ({avail_pct:5.1f}%) | "
                f"Phase: {display_phase} | Step: {step_progress} | {step_display}"
            )
            print(line)

            if csv_file:
                csv_file.write(
                    f"{ts},{cpu:.1f},{used_gb:.2f},{avail_gb:.2f},{avail_pct:.1f},"
                    f"{display_phase},{'' if curr_idx is None else curr_idx+1},{'' if curr_step is None else curr_step},"
                    f"{status if phase != 'DONE' else 'finished'},"
                    f"{'' if done_workers is None else done_workers},"
                    f"{'' if total_workers is None else total_workers}\n"
                )
                csv_file.flush()

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if csv_file:
            csv_file.close()


# --- CLI ---------------------------------------------------------------------


def positive_float(value: str) -> float:
    try:
        v = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Interval must be a number.")
    if v <= 0:
        raise argparse.ArgumentTypeError("Interval must be > 0.")
    return v


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track CPU/memory usage and current ActivitySim step (phase-aware, waits for all workers)."
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=positive_float,
        default=0.5,
        help="Sampling interval in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="../output/log/sys_usage.csv",
        help="Optional path to CSV file for logging (e.g., sys_usage.csv)",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="../output/log/activitysim.log",
        help="Optional path to an ActivitySim log file to infer the current step.",
    )
    parser.add_argument(
        "--tail-bytes",
        type=int,
        default=256 * 1024,
        help="Number of bytes to tail from the log each tick (default: 262144). Increase if starting mid-run.",
    )
    parser.add_argument(
        "--parent-pid",
        type=int,
        default=None,
        help="Parent process PID to monitor. Monitor will exit if parent process dies.",
    )
    parser.add_argument(
        "--delay",
        type=positive_float,
        default=0,
        help="Delay in seconds before starting monitoring (default: 0)",
    )
    args = parser.parse_args()

    monitor(
        interval=args.interval,
        csv_path=args.csv,
        log_path=args.log,
        tail_bytes=args.tail_bytes,
        parent_pid=args.parent_pid,
        delay_seconds=args.delay,
    )


if __name__ == "__main__":
    main()
