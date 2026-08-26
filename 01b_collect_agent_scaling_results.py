from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# PATHS
# ============================================================

THIS_FILE = Path(__file__).resolve()
EXPERIMENT_DIR = THIS_FILE.parent
NEW_ROOT = EXPERIMENT_DIR.parent.parent

SCALABILITY_ROOT = (
    NEW_ROOT
    / "outputs"
    / "scalability"
)

OUTPUT_DIR = (
    SCALABILITY_ROOT
    / "agent_count_consolidated"
)


# ============================================================
# REQUIRED CONDITIONS
# ============================================================

REQUIRED = {
    2: 4,
    3: 9,
    4: 16,
    5: 25,
}


# ============================================================
# HELPERS
# ============================================================

def read_json(path: Path) -> Dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def save_json(
    path: Path,
    data: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def save_csv(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:

    if not rows:

        raise ValueError(
            "No rows to save."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = []

    seen = set()

    for row in rows:

        for key in row.keys():

            if key not in seen:

                seen.add(key)
                fieldnames.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# FIND RESULT FILES
# ============================================================

def find_agent_results() -> List[Path]:

    if not SCALABILITY_ROOT.exists():

        raise FileNotFoundError(
            "Scalability output directory does not exist:\n"
            f"{SCALABILITY_ROOT}"
        )

    candidates = []

    for path in SCALABILITY_ROOT.rglob(
        "results.json"
    ):

        text = str(path).lower()

        if "agent_count_scaling" not in text:
            continue

        candidates.append(path)

    return sorted(
        candidates,
        key=lambda p: p.stat().st_mtime,
    )


# ============================================================
# VALIDATE ONE RESULT
# ============================================================

def normalize_result(
    path: Path,
) -> Dict[str, Any]:

    result = read_json(path)

    if "grid" not in result:
        raise RuntimeError(
            f"Missing grid in:\n{path}"
        )

    if "agent_count" not in result:
        raise RuntimeError(
            f"Missing agent_count in:\n{path}"
        )

    grid = int(
        result["grid"]
    )

    agents = int(
        result["agent_count"]
    )

    if grid not in REQUIRED:

        raise RuntimeError(
            f"Unexpected grid={grid} in:\n{path}"
        )

    if agents != REQUIRED[grid]:

        raise RuntimeError(
            "Grid/agent mismatch.\n"
            f"Grid: {grid}\n"
            f"Expected: {REQUIRED[grid]}\n"
            f"Observed: {agents}\n"
            f"File: {path}"
        )

    reload_difference = float(
        result.get(
            "checkpoint_reload_difference",
            result.get(
                "checkpoint_reload_rmse_difference",
                999.0,
            ),
        )
    )

    if reload_difference > 1e-6:

        raise RuntimeError(
            "Checkpoint restoration was not exact.\n"
            f"Grid: {grid}\n"
            f"Difference: {reload_difference}\n"
            f"File: {path}"
        )

    required_metrics = [
        "parameter_count",
        "best_epoch",
        "best_validation_rmse",
        "test_mae",
        "test_rmse",
        "test_wave_residual",
        "latency_mean_sec",
        "latency_p95_sec",
        "throughput_sequences_per_sec",
        "training_time_sec",
    ]

    for metric in required_metrics:

        if metric not in result:

            raise RuntimeError(
                f"Missing {metric} in:\n{path}"
            )

    normalized = {

        "grid":
            grid,

        "agent_count":
            agents,

        "reference_condition":
            bool(
                grid == 4
            ),

        "parameter_count":
            int(
                result[
                    "parameter_count"
                ]
            ),

        "best_epoch":
            int(
                result[
                    "best_epoch"
                ]
            ),

        "best_validation_rmse":
            float(
                result[
                    "best_validation_rmse"
                ]
            ),

        "test_mae":
            float(
                result[
                    "test_mae"
                ]
            ),

        "test_rmse":
            float(
                result[
                    "test_rmse"
                ]
            ),

        "test_wave_residual":
            float(
                result[
                    "test_wave_residual"
                ]
            ),

        "latency_mean_sec":
            float(
                result[
                    "latency_mean_sec"
                ]
            ),

        "latency_p95_sec":
            float(
                result[
                    "latency_p95_sec"
                ]
            ),

        "throughput_sequences_per_sec":
            float(
                result[
                    "throughput_sequences_per_sec"
                ]
            ),

        "peak_gpu_memory_bytes":
            result.get(
                "peak_gpu_memory_bytes"
            ),

        "training_time_sec":
            float(
                result[
                    "training_time_sec"
                ]
            ),

        "checkpoint_reload_difference":
            reload_difference,

        "source":
            str(path),
    }

    return normalized


# ============================================================
# SELECT LATEST COMPLETED RESULT PER GRID
# ============================================================

def collect_latest_results():

    files = find_agent_results()

    if not files:

        raise RuntimeError(
            "No completed agent-count results found."
        )

    selected: Dict[
        int,
        Dict[str, Any]
    ] = {}

    selected_path: Dict[
        int,
        Path
    ] = {}

    for path in files:

        try:

            row = normalize_result(path)

        except Exception as exc:

            print(
                "[SKIP]",
                path,
            )

            print(
                "       ",
                exc,
            )

            continue

        grid = row[
            "grid"
        ]

        # Sorted by modification time,
        # therefore later valid runs replace earlier ones.
        selected[grid] = row
        selected_path[grid] = path

    missing = [
        grid
        for grid in REQUIRED
        if grid not in selected
    ]

    if missing:

        raise RuntimeError(
            "Missing completed agent-scaling conditions:\n"
            f"{missing}"
        )

    return [
        selected[grid]
        for grid in sorted(
            REQUIRED
        )
    ]


# ============================================================
# ADD SCALING METRICS
# ============================================================

def add_relative_scaling(
    rows: List[Dict[str, Any]],
) -> None:

    reference = next(
        row
        for row in rows
        if row[
            "grid"
        ] == 4
    )

    ref_agents = float(
        reference[
            "agent_count"
        ]
    )

    ref_latency = float(
        reference[
            "latency_mean_sec"
        ]
    )

    ref_memory = reference.get(
        "peak_gpu_memory_bytes"
    )

    ref_parameters = float(
        reference[
            "parameter_count"
        ]
    )

    for row in rows:

        row[
            "agent_ratio_vs_16"
        ] = (
            row[
                "agent_count"
            ]
            / ref_agents
        )

        row[
            "latency_ratio_vs_16"
        ] = (
            row[
                "latency_mean_sec"
            ]
            / ref_latency
        )

        row[
            "parameter_ratio_vs_16"
        ] = (
            row[
                "parameter_count"
            ]
            / ref_parameters
        )

        if (
            ref_memory is not None
            and row.get(
                "peak_gpu_memory_bytes"
            ) is not None
        ):

            row[
                "memory_ratio_vs_16"
            ] = (
                float(
                    row[
                        "peak_gpu_memory_bytes"
                    ]
                )
                / float(
                    ref_memory
                )
            )

        else:

            row[
                "memory_ratio_vs_16"
            ] = None


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 90)

    print(
        "COLLECTING COMPLETED SACU AGENT-SCALING RESULTS"
    )

    print("=" * 90)

    print(
        "Scalability root:"
    )

    print(
        SCALABILITY_ROOT
    )

    rows = collect_latest_results()

    add_relative_scaling(
        rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_csv(
        OUTPUT_DIR
        / "agent_count_scaling_summary.csv",
        rows,
    )

    save_json(
        OUTPUT_DIR
        / "agent_count_scaling_summary.json",
        {

            "conditions":
                rows,

            "required_grids":
                list(
                    REQUIRED.keys()
                ),

            "required_agent_counts":
                list(
                    REQUIRED.values()
                ),

            "reference_grid":
                4,

            "reference_agents":
                16,

            "status":
                "complete",
        },
    )

    print()
    print(
        f"{'Grid':>6}"
        f"{'Agents':>9}"
        f"{'Params':>11}"
        f"{'RMSE':>11}"
        f"{'Residual':>12}"
        f"{'Latency':>12}"
        f"{'Throughput':>13}"
        f"{'GPU MB':>11}"
    )

    print(
        "-" * 86
    )

    for row in rows:

        peak = row.get(
            "peak_gpu_memory_bytes"
        )

        peak_mb = (
            float(peak)
            / 1024**2
            if peak is not None
            else float("nan")
        )

        print(
            f"{row['grid']:>6}"
            f"{row['agent_count']:>9}"
            f"{row['parameter_count']:>11}"
            f"{row['test_rmse']:>11.6f}"
            f"{row['test_wave_residual']:>12.6f}"
            f"{row['latency_mean_sec']:>12.6f}"
            f"{row['throughput_sequences_per_sec']:>13.3f}"
            f"{peak_mb:>11.1f}"
        )

    print()
    print("=" * 90)

    print(
        "PASS: all four SACU agent-count conditions "
        "were found and validated."
    )

    print(
        "Required agents: 4, 9, 16, 25."
    )

    print(
        "Reference: 16 agents."
    )

    print()
    print(
        "Outputs:"
    )

    print(
        OUTPUT_DIR
        / "agent_count_scaling_summary.csv"
    )

    print(
        OUTPUT_DIR
        / "agent_count_scaling_summary.json"
    )


if __name__ == "__main__":

    main()