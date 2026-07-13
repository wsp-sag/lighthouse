from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot memory_used_gb and cpu_percent as time series, color-coded by step_name."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to CSV file containing timestamp, memory_used_gb, cpu_percent, and step_name columns.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output image path (default: <csv_name>_memory_timeseries.png).",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(14, 7),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches.",
    )
    return parser.parse_args()


def make_plot(csv_path: Path, output_path: Path, figsize: tuple[float, float]) -> None:
    df = pd.read_csv(csv_path)

    required = {"timestamp", "memory_used_gb", "cpu_percent", "step_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["memory_used_gb"] = pd.to_numeric(df["memory_used_gb"], errors="coerce")
    df["cpu_percent"] = pd.to_numeric(df["cpu_percent"], errors="coerce")
    if "step_index" in df.columns:
        df["step_index"] = pd.to_numeric(df["step_index"], errors="coerce")
    step_name = df["step_name"].astype("string").str.strip()
    if "phase" in df.columns:
        phase = df["phase"].astype("string").str.strip()
        step_name = step_name.fillna(phase.where(phase.eq("DONE"), pd.NA))
    df["step_name"] = step_name.fillna("(missing)").astype(str)
    df = df.dropna(subset=["timestamp", "memory_used_gb", "cpu_percent"]).sort_values(
        "timestamp"
    )

    if df.empty:
        raise ValueError("No valid rows after parsing timestamp and memory_used_gb.")

    fig, ax = plt.subplots(figsize=figsize)

    # Draw a light baseline to preserve continuity of the time series.
    ax.plot(
        df["timestamp"],
        df["memory_used_gb"],
        color="lightgray",
        linewidth=1.2,
        alpha=0.8,
        zorder=1,
        label="overall",
    )

    ax_cpu = ax.twinx()
    ax_cpu.plot(
        df["timestamp"],
        df["cpu_percent"],
        color="#BDBDBD",
        linewidth=1.4,
        alpha=0.85,
        linestyle="--",
        label="CPU %",
        zorder=1,
    )

    step_names = df["step_name"].unique().tolist()
    palette = _distinct_palette(len(step_names))
    step_colors = dict(zip(step_names, palette, strict=False))
    if "step_index" in df.columns:
        first_seen_order = {step: idx for idx, step in enumerate(step_names)}
        step_index_map = (
            df.dropna(subset=["step_index"])
            .groupby("step_name", as_index=True)["step_index"]
            .min()
            .to_dict()
        )
        legend_step_names = sorted(
            step_names,
            key=lambda step: (
                step not in step_index_map,
                step_index_map.get(step, float("inf")),
                first_seen_order[step],
            ),
        )
    else:
        legend_step_names = step_names.copy()
    step_handles: dict[str, plt.Line2D] = {}

    for step in step_names:
        group = df[df["step_name"] == step]
        line = ax.plot(
            group["timestamp"],
            group["memory_used_gb"],
            linewidth=1.1,
            marker="o",
            markersize=2.8,
            color=step_colors[step],
            alpha=0.95,
            label=step,
            zorder=3,
        )[0]
        step_handles[step] = line

    ax.set_title("Memory Used Over Time by Step")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Memory Used (GB)")
    ax_cpu.set_ylabel("CPU Usage (%)")
    ax.grid(True, alpha=0.25)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S"))
    fig.autofmt_xdate(rotation=0, ha="center")

    legend_handles = [step_handles[step] for step in legend_step_names if step in step_handles]
    legend_labels = legend_step_names
    legend_handles.append(ax_cpu.lines[0])
    legend_labels.append("CPU %")
    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            title="step_name",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _distinct_palette(count: int) -> list[tuple[float, float, float]]:
    if count <= 0:
        return []

    vivid_palette = [
        "#E41A1C",
        "#377EB8",
        "#4DAF4A",
        "#984EA3",
        "#FF7F00",
        "#FFFF33",
        "#A65628",
        "#F781BF",
        "#00A6D6",
        "#8DA0CB",
        "#66C2A5",
        "#E78AC3",
        "#A6D854",
        "#FFD92F",
        "#E5C494",
        "#B3B3B3",
    ]

    if count <= len(vivid_palette):
        return vivid_palette[:count]

    palette = vivid_palette.copy()
    cycle_index = 0
    while len(palette) < count:
        palette.append(vivid_palette[cycle_index % len(vivid_palette)])
        cycle_index += 1

    return palette


def main() -> None:
    args = parse_args()

    csv_path = args.csv_path.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else csv_path.with_name(f"{csv_path.stem}_memory_timeseries.png")
    )

    make_plot(csv_path=csv_path, output_path=output_path, figsize=tuple(args.figsize))
    print(f"Saved chart to: {output_path}")


if __name__ == "__main__":
    main()
