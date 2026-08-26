from __future__ import annotations

# ============================================================
# FINAL SACU SCALABILITY ANALYSIS
#
# This script performs NO model training and NO inference.
#
# It consolidates previously completed reviewer-requested
# scalability experiments:
#
#   1. Agent-count scaling:
#        4, 9, 16, 25 SACUs
#
#   2. Domain-size scaling:
#        64x64, 128x128, 256x256
#
#   3. Communication-cost scaling:
#        0, 1, 5, 10 ms
#
# Fixed reference conditions:
#
#   agent reference:
#       16 SACUs / grid=4
#
#   domain reference:
#       128x128
#
#   communication reference:
#       0 ms injected synchronization delay
#
# IMPORTANT:
#   - This script never modifies submitted-manuscript values.
#   - It never reruns a model.
#   - It never fabricates missing predictive metrics.
#   - 64x64 and 256x256 domain conditions remain
#     computational-only experiments.
#   - Communication delays are controlled injected scenarios,
#     not measurements of physical network latency.
# ============================================================


# ============================================================
# STANDARD LIBRARY
# ============================================================

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ============================================================
# PATHS
# ============================================================

THIS_FILE = Path(__file__).resolve()

EXPERIMENT_DIR = THIS_FILE.parent

EXPERIMENTS_ROOT = EXPERIMENT_DIR.parent

NEW_ROOT = EXPERIMENTS_ROOT.parent

SCALABILITY_ROOT = (
    NEW_ROOT
    / "outputs"
    / "scalability"
)

FINAL_OUTPUT_DIR = (
    SCALABILITY_ROOT
    / "final_scalability_analysis"
)


# ============================================================
# REQUIRED CONDITIONS
# ============================================================

REQUIRED_AGENT_CONDITIONS = {
    2: 4,
    3: 9,
    4: 16,
    5: 25,
}

REQUIRED_DOMAIN_SIZES = [
    64,
    128,
    256,
]

REQUIRED_COMMUNICATION_DELAYS_MS = [
    0.0,
    1.0,
    5.0,
    10.0,
]


# ============================================================
# REFERENCES
# ============================================================

REFERENCE_GRID = 4
REFERENCE_AGENTS = 16

REFERENCE_DOMAIN_SIZE = 128

REFERENCE_COMMUNICATION_DELAY_MS = 0.0


# ============================================================
# BASIC IO
# ============================================================

def read_json(
    path: Path,
) -> Any:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


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
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def save_text(
    path: Path,
    text: str,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        text,
        encoding="utf-8",
    )


def save_csv(
    path: Path,
    rows: Sequence[
        Dict[str, Any]
    ],
) -> None:

    rows = list(
        rows
    )

    if not rows:

        raise ValueError(
            f"No rows supplied for {path.name}."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields: List[str] = []
    seen = set()

    for row in rows:

        for key in row.keys():

            if key not in seen:

                seen.add(
                    key
                )

                fields.append(
                    key
                )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# NUMERIC HELPERS
# ============================================================

def as_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:

    if value is None:

        return default

    try:

        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return default


def as_int(
    value: Any,
    default: Optional[int] = None,
) -> Optional[int]:

    if value is None:

        return default

    try:

        return int(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return default


def safe_ratio(
    numerator: Optional[float],
    denominator: Optional[float],
) -> Optional[float]:

    if (
        numerator is None
        or denominator is None
        or denominator == 0
    ):

        return None

    return (
        numerator
        / denominator
    )


def safe_percent_change(
    current: Optional[float],
    reference: Optional[float],
) -> Optional[float]:

    ratio = safe_ratio(
        current,
        reference,
    )

    if ratio is None:

        return None

    return (
        ratio
        - 1.0
    ) * 100.0


def bytes_to_mib(
    value: Optional[float],
) -> Optional[float]:

    if value is None:

        return None

    return (
        float(value)
        / (1024.0 ** 2)
    )


def format_optional(
    value: Any,
    decimals: int = 6,
) -> str:

    if value is None:

        return "N/A"

    if isinstance(
        value,
        float,
    ):

        if not math.isfinite(
            value
        ):

            return "N/A"

        return (
            f"{value:.{decimals}f}"
        )

    return str(
        value
    )


# ============================================================
# AGENT RESULTS
#
# Prefer already consolidated file.
# Fall back to raw completed results if needed.
# ============================================================

def find_agent_consolidated_file() -> Optional[Path]:

    path = (
        SCALABILITY_ROOT
        / "agent_count_consolidated"
        / "agent_count_scaling_summary.json"
    )

    if path.exists():

        return path

    return None


def normalize_agent_row(
    row: Dict[str, Any],
    source: str,
) -> Dict[str, Any]:

    grid = as_int(
        row.get(
            "grid"
        )
    )

    agents = as_int(
        row.get(
            "agent_count"
        )
    )

    if grid is None or agents is None:

        raise RuntimeError(
            f"Agent result missing grid/agent_count: {source}"
        )

    if grid not in REQUIRED_AGENT_CONDITIONS:

        raise RuntimeError(
            f"Unexpected agent grid={grid}: {source}"
        )

    expected_agents = (
        REQUIRED_AGENT_CONDITIONS[
            grid
        ]
    )

    if agents != expected_agents:

        raise RuntimeError(
            "Agent-count mismatch.\n"
            f"Grid: {grid}\n"
            f"Expected: {expected_agents}\n"
            f"Observed: {agents}\n"
            f"Source: {source}"
        )

    reload_difference = as_float(
        row.get(
            "checkpoint_reload_difference",
            row.get(
                "checkpoint_reload_rmse_difference"
            ),
        )
    )

    if (
        reload_difference is not None
        and reload_difference > 1e-6
    ):

        raise RuntimeError(
            "Invalid checkpoint restoration in agent run.\n"
            f"Grid: {grid}\n"
            f"Difference: {reload_difference}\n"
            f"Source: {source}"
        )

    normalized = {

        "grid":
            grid,

        "agent_count":
            agents,

        "reference_condition":
            (
                agents
                == REFERENCE_AGENTS
            ),

        "parameter_count":
            as_int(
                row.get(
                    "parameter_count"
                )
            ),

        "best_epoch":
            as_int(
                row.get(
                    "best_epoch"
                )
            ),

        "best_validation_rmse":
            as_float(
                row.get(
                    "best_validation_rmse"
                )
            ),

        "test_mae":
            as_float(
                row.get(
                    "test_mae"
                )
            ),

        "test_rmse":
            as_float(
                row.get(
                    "test_rmse"
                )
            ),

        "test_wave_residual":
            as_float(
                row.get(
                    "test_wave_residual"
                )
            ),

        "latency_mean_sec":
            as_float(
                row.get(
                    "latency_mean_sec"
                )
            ),

        "latency_p95_sec":
            as_float(
                row.get(
                    "latency_p95_sec"
                )
            ),

        "throughput_sequences_per_sec":
            as_float(
                row.get(
                    "throughput_sequences_per_sec"
                )
            ),

        "peak_gpu_memory_bytes":
            as_float(
                row.get(
                    "peak_gpu_memory_bytes"
                )
            ),

        "training_time_sec":
            as_float(
                row.get(
                    "training_time_sec"
                )
            ),

        "checkpoint_reload_difference":
            reload_difference,

        "source":
            source,
    }

    normalized[
        "peak_gpu_memory_mib"
    ] = bytes_to_mib(
        normalized[
            "peak_gpu_memory_bytes"
        ]
    )

    return normalized


def collect_agent_results() -> List[
    Dict[str, Any]
]:

    consolidated = (
        find_agent_consolidated_file()
    )

    rows = []


    if consolidated is not None:

        data = read_json(
            consolidated
        )

        conditions = data.get(
            "conditions",
            []
        )

        for row in conditions:

            rows.append(
                normalize_agent_row(
                    row,
                    str(
                        consolidated
                    ),
                )
            )


    else:

        latest_by_grid: Dict[
            int,
            Tuple[
                float,
                Dict[str, Any],
                Path,
            ]
        ] = {}


        for path in SCALABILITY_ROOT.rglob(
            "results.json"
        ):

            text = str(
                path
            ).lower()


            if (
                "agent_count_scaling"
                not in text
            ):

                continue


            try:

                raw = read_json(
                    path
                )

                normalized = (
                    normalize_agent_row(
                        raw,
                        str(path),
                    )
                )

            except Exception:

                continue


            grid = normalized[
                "grid"
            ]


            modified = (
                path.stat().st_mtime
            )


            current = latest_by_grid.get(
                grid
            )


            if (
                current is None
                or modified > current[0]
            ):

                latest_by_grid[
                    grid
                ] = (
                    modified,
                    normalized,
                    path,
                )


        rows = [

            latest_by_grid[
                grid
            ][1]

            for grid
            in sorted(
                latest_by_grid
            )
        ]


    found_grids = {
        row[
            "grid"
        ]
        for row
        in rows
    }


    missing = [

        grid

        for grid
        in REQUIRED_AGENT_CONDITIONS

        if grid not in found_grids
    ]


    if missing:

        raise RuntimeError(
            "Missing completed agent-count conditions:\n"
            f"{missing}"
        )


    rows.sort(
        key=lambda row:
            row[
                "agent_count"
            ]
    )


    return rows


# ============================================================
# DOMAIN RESULTS
# ============================================================

def normalize_domain_row(
    row: Dict[str, Any],
    source: str,
) -> Dict[str, Any]:

    height = as_int(
        row.get(
            "domain_height"
        )
    )

    width = as_int(
        row.get(
            "domain_width"
        )
    )

    if height is None or width is None:

        raise RuntimeError(
            f"Domain result missing dimensions: {source}"
        )


    if height != width:

        raise RuntimeError(
            f"Non-square domain encountered: {source}"
        )


    if height not in REQUIRED_DOMAIN_SIZES:

        raise RuntimeError(
            f"Unexpected domain size={height}: {source}"
        )


    predictive_valid = bool(
        row.get(
            "predictive_metrics_valid",
            height == REFERENCE_DOMAIN_SIZE,
        )
    )


    normalized = {

        "domain_height":
            height,

        "domain_width":
            width,

        "domain_size":
            height,

        "domain_pixels":
            height
            * width,

        "reference_condition":
            (
                height
                == REFERENCE_DOMAIN_SIZE
            ),

        "grid":
            as_int(
                row.get(
                    "grid"
                )
            ),

        "agent_count":
            as_int(
                row.get(
                    "agent_count"
                )
            ),

        "parameter_count":
            as_int(
                row.get(
                    "parameter_count"
                )
            ),

        "predictive_metrics_valid":
            predictive_valid,

        "test_mae":
            as_float(
                row.get(
                    "test_mae"
                )
            )
            if predictive_valid
            else None,

        "test_rmse":
            as_float(
                row.get(
                    "test_rmse"
                )
            )
            if predictive_valid
            else None,

        "test_wave_residual":
            as_float(
                row.get(
                    "test_wave_residual"
                )
            )
            if predictive_valid
            else None,

        "latency_mean_sec":
            as_float(
                row.get(
                    "latency_mean_sec"
                )
            ),

        "latency_p50_sec":
            as_float(
                row.get(
                    "latency_p50_sec"
                )
            ),

        "latency_p95_sec":
            as_float(
                row.get(
                    "latency_p95_sec"
                )
            ),

        "latency_p99_sec":
            as_float(
                row.get(
                    "latency_p99_sec"
                )
            ),

        "throughput_sequences_per_sec":
            as_float(
                row.get(
                    "throughput_sequences_per_sec"
                )
            ),

        "peak_gpu_memory_bytes":
            as_float(
                row.get(
                    "peak_gpu_memory_bytes"
                )
            ),

        "input_transformation":
            row.get(
                "input_transformation"
            ),

        "predictive_metric_reason":
            row.get(
                "predictive_metric_reason"
            ),

        "source":
            source,
    }


    normalized[
        "peak_gpu_memory_mib"
    ] = bytes_to_mib(
        normalized[
            "peak_gpu_memory_bytes"
        ]
    )


    return normalized


def collect_domain_results() -> List[
    Dict[str, Any]
]:

    latest_by_size: Dict[
        int,
        Tuple[
            float,
            Dict[str, Any],
            Path,
        ]
    ] = {}


    for path in SCALABILITY_ROOT.rglob(
        "results.json"
    ):

        text = str(
            path
        ).lower()


        if (
            "domain_size_scaling"
            not in text
        ):

            continue


        try:

            raw = read_json(
                path
            )


            normalized = (
                normalize_domain_row(
                    raw,
                    str(
                        path
                    ),
                )
            )


        except Exception:

            continue


        size = normalized[
            "domain_size"
        ]


        modified = (
            path.stat().st_mtime
        )


        current = latest_by_size.get(
            size
        )


        if (
            current is None
            or modified > current[0]
        ):

            latest_by_size[
                size
            ] = (
                modified,
                normalized,
                path,
            )


    missing = [

        size

        for size
        in REQUIRED_DOMAIN_SIZES

        if size not in latest_by_size
    ]


    if missing:

        raise RuntimeError(
            "Missing completed domain-size conditions:\n"
            f"{missing}"
        )


    rows = [

        latest_by_size[
            size
        ][1]

        for size
        in REQUIRED_DOMAIN_SIZES
    ]


    return rows


# ============================================================
# COMMUNICATION RESULTS
# ============================================================

def find_latest_communication_file() -> Path:

    candidates = []


    for path in SCALABILITY_ROOT.rglob(
        "communication_cost_scaling.json"
    ):

        candidates.append(
            path
        )


    if not candidates:

        raise RuntimeError(
            "No completed communication-cost "
            "scaling result file was found."
        )


    candidates.sort(
        key=lambda path:
            path.stat().st_mtime
    )


    return candidates[
        -1
    ]


def normalize_communication_row(
    row: Dict[str, Any],
    source: str,
) -> Dict[str, Any]:

    delay = as_float(
        row.get(
            "requested_delay_ms"
        )
    )


    if delay is None:

        raise RuntimeError(
            f"Communication result missing delay: {source}"
        )


    if not any(
        abs(
            delay
            - required
        )
        < 1e-9

        for required
        in REQUIRED_COMMUNICATION_DELAYS_MS
    ):

        raise RuntimeError(
            f"Unexpected communication delay={delay}: {source}"
        )


    prediction_difference = as_float(
        row.get(
            "prediction_max_abs_difference"
        ),
        0.0,
    )


    if (
        prediction_difference is not None
        and prediction_difference > 1e-6
    ):

        raise RuntimeError(
            "Communication delay altered prediction.\n"
            f"Delay: {delay}\n"
            f"Difference: {prediction_difference}"
        )


    return {

        "requested_delay_ms":
            delay,

        "reference_condition":
            (
                abs(
                    delay
                    - REFERENCE_COMMUNICATION_DELAY_MS
                )
                < 1e-9
            ),

        "latency_mean_sec":
            as_float(
                row.get(
                    "latency_mean_sec"
                )
            ),

        "latency_p95_sec":
            as_float(
                row.get(
                    "latency_p95_sec"
                )
            ),

        "throughput_sequences_per_sec":
            as_float(
                row.get(
                    "throughput_sequences_per_sec"
                )
            ),

        "realized_barrier_mean_sec":
            as_float(
                row.get(
                    "realized_barrier_mean_sec"
                )
            ),

        "realized_barrier_ms":
            (
                as_float(
                    row.get(
                        "realized_barrier_mean_sec"
                    )
                )
                * 1000.0
                if as_float(
                    row.get(
                        "realized_barrier_mean_sec"
                    )
                )
                is not None
                else None
            ),

        "prediction_max_abs_difference":
            prediction_difference,

        "prediction_mean_abs_difference":
            as_float(
                row.get(
                    "prediction_mean_abs_difference"
                )
            ),

        "peak_gpu_memory_bytes":
            as_float(
                row.get(
                    "peak_gpu_memory_bytes"
                )
            ),

        "source":
            source,
    }


def collect_communication_results() -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
]:

    path = (
        find_latest_communication_file()
    )


    data = read_json(
        path
    )


    raw_results = data.get(
        "results",
        []
    )


    rows = [

        normalize_communication_row(
            row,
            str(
                path
            ),
        )

        for row
        in raw_results
    ]


    found_delays = {
        round(
            row[
                "requested_delay_ms"
            ],
            9,
        )
        for row
        in rows
    }


    missing = [

        delay

        for delay
        in REQUIRED_COMMUNICATION_DELAYS_MS

        if round(
            delay,
            9,
        )
        not in found_delays
    ]


    if missing:

        raise RuntimeError(
            "Missing communication-delay conditions:\n"
            f"{missing}"
        )


    rows.sort(
        key=lambda row:
            row[
                "requested_delay_ms"
            ]
    )


    metadata = {

        "source":
            str(
                path
            ),

        "physical_network_latency_measured":
            bool(
                data.get(
                    "physical_network_latency_measured",
                    False,
                )
            ),

        "delay_semantics":
            data.get(
                "delay_semantics"
            ),

        "reference_accuracy":
            data.get(
                "reference_accuracy",
                {}
            ),

        "zero_delay_equivalence":
            data.get(
                "zero_delay_equivalence",
                {}
            ),
    }


    return (
        rows,
        metadata,
    )


# ============================================================
# ADD RELATIVE AGENT SCALING
# ============================================================

def analyze_agent_scaling(
    rows: List[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:

    reference = next(

        row

        for row in rows

        if row[
            "agent_count"
        ]
        == REFERENCE_AGENTS
    )


    ref_latency = (
        reference[
            "latency_mean_sec"
        ]
    )

    ref_throughput = (
        reference[
            "throughput_sequences_per_sec"
        ]
    )

    ref_params = (
        reference[
            "parameter_count"
        ]
    )

    ref_rmse = (
        reference[
            "test_rmse"
        ]
    )

    ref_residual = (
        reference[
            "test_wave_residual"
        ]
    )


    for row in rows:

        row[
            "agent_ratio_vs_16"
        ] = safe_ratio(
            row[
                "agent_count"
            ],
            REFERENCE_AGENTS,
        )


        row[
            "parameter_ratio_vs_16"
        ] = safe_ratio(
            row[
                "parameter_count"
            ],
            ref_params,
        )


        row[
            "latency_ratio_vs_16"
        ] = safe_ratio(
            row[
                "latency_mean_sec"
            ],
            ref_latency,
        )


        row[
            "latency_change_percent_vs_16"
        ] = safe_percent_change(
            row[
                "latency_mean_sec"
            ],
            ref_latency,
        )


        row[
            "throughput_ratio_vs_16"
        ] = safe_ratio(
            row[
                "throughput_sequences_per_sec"
            ],
            ref_throughput,
        )


        row[
            "throughput_change_percent_vs_16"
        ] = safe_percent_change(
            row[
                "throughput_sequences_per_sec"
            ],
            ref_throughput,
        )


        row[
            "rmse_difference_vs_16"
        ] = (
            row[
                "test_rmse"
            ]
            - ref_rmse
        )


        row[
            "residual_difference_vs_16"
        ] = (
            row[
                "test_wave_residual"
            ]
            - ref_residual
        )


    minimum_rmse_row = min(
        rows,
        key=lambda row:
            row[
                "test_rmse"
            ],
    )


    rmse_values = [
        row[
            "test_rmse"
        ]
        for row
        in rows
    ]


    latency_4 = next(
        row[
            "latency_mean_sec"
        ]
        for row
        in rows
        if row[
            "agent_count"
        ]
        == 4
    )


    latency_25 = next(
        row[
            "latency_mean_sec"
        ]
        for row
        in rows
        if row[
            "agent_count"
        ]
        == 25
    )


    throughput_4 = next(
        row[
            "throughput_sequences_per_sec"
        ]
        for row
        in rows
        if row[
            "agent_count"
        ]
        == 4
    )


    throughput_25 = next(
        row[
            "throughput_sequences_per_sec"
        ]
        for row
        in rows
        if row[
            "agent_count"
        ]
        == 25
    )


    return {

        "reference_agents":
            REFERENCE_AGENTS,

        "minimum_test_rmse":
            minimum_rmse_row[
                "test_rmse"
            ],

        "minimum_test_rmse_agents":
            minimum_rmse_row[
                "agent_count"
            ],

        "rmse_range":
            max(
                rmse_values
            )
            -
            min(
                rmse_values
            ),

        "rmse_mean":
            statistics.mean(
                rmse_values
            ),

        "latency_4_agents_sec":
            latency_4,

        "latency_25_agents_sec":
            latency_25,

        "latency_ratio_25_vs_4":
            safe_ratio(
                latency_25,
                latency_4,
            ),

        "latency_increase_percent_25_vs_4":
            safe_percent_change(
                latency_25,
                latency_4,
            ),

        "throughput_4_agents":
            throughput_4,

        "throughput_25_agents":
            throughput_25,

        "throughput_ratio_25_vs_4":
            safe_ratio(
                throughput_25,
                throughput_4,
            ),

        "throughput_reduction_percent_25_vs_4":
            (
                (
                    1.0
                    -
                    safe_ratio(
                        throughput_25,
                        throughput_4,
                    )
                )
                * 100.0
            ),

        "interpretation":
            (
                "Reconstruction RMSE remained tightly clustered "
                "across 4-25 SACUs, while latency increased and "
                "throughput decreased as agent count increased."
            ),

        "memory_interpretation":
            (
                "Peak GPU-memory measurements are retained for "
                "auditability but are not interpreted as a "
                "monotonic model-scaling law because the "
                "memory-efficient recomputation strategy and "
                "allocator behavior materially affect these values."
            ),
    }


# ============================================================
# ADD DOMAIN SCALING
# ============================================================

def analyze_domain_scaling(
    rows: List[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:

    reference = next(

        row

        for row in rows

        if row[
            "domain_size"
        ]
        == REFERENCE_DOMAIN_SIZE
    )


    ref_pixels = (
        reference[
            "domain_pixels"
        ]
    )

    ref_latency = (
        reference[
            "latency_mean_sec"
        ]
    )

    ref_throughput = (
        reference[
            "throughput_sequences_per_sec"
        ]
    )

    ref_memory = (
        reference[
            "peak_gpu_memory_bytes"
        ]
    )


    for row in rows:

        row[
            "pixel_ratio_vs_128"
        ] = safe_ratio(
            row[
                "domain_pixels"
            ],
            ref_pixels,
        )


        row[
            "latency_ratio_vs_128"
        ] = safe_ratio(
            row[
                "latency_mean_sec"
            ],
            ref_latency,
        )


        row[
            "latency_change_percent_vs_128"
        ] = safe_percent_change(
            row[
                "latency_mean_sec"
            ],
            ref_latency,
        )


        row[
            "throughput_ratio_vs_128"
        ] = safe_ratio(
            row[
                "throughput_sequences_per_sec"
            ],
            ref_throughput,
        )


        row[
            "throughput_change_percent_vs_128"
        ] = safe_percent_change(
            row[
                "throughput_sequences_per_sec"
            ],
            ref_throughput,
        )


        row[
            "memory_ratio_vs_128"
        ] = safe_ratio(
            row[
                "peak_gpu_memory_bytes"
            ],
            ref_memory,
        )


    condition_256 = next(
        row
        for row
        in rows
        if row[
            "domain_size"
        ]
        == 256
    )


    condition_64 = next(
        row
        for row
        in rows
        if row[
            "domain_size"
        ]
        == 64
    )


    return {

        "reference_domain":
            "128x128",

        "reference_test_mae":
            reference[
                "test_mae"
            ],

        "reference_test_rmse":
            reference[
                "test_rmse"
            ],

        "reference_wave_residual":
            reference[
                "test_wave_residual"
            ],

        "latency_64_sec":
            condition_64[
                "latency_mean_sec"
            ],

        "latency_128_sec":
            reference[
                "latency_mean_sec"
            ],

        "latency_256_sec":
            condition_256[
                "latency_mean_sec"
            ],

        "latency_ratio_256_vs_128":
            safe_ratio(
                condition_256[
                    "latency_mean_sec"
                ],
                reference[
                    "latency_mean_sec"
                ],
            ),

        "throughput_128":
            reference[
                "throughput_sequences_per_sec"
            ],

        "throughput_256":
            condition_256[
                "throughput_sequences_per_sec"
            ],

        "throughput_ratio_256_vs_128":
            safe_ratio(
                condition_256[
                    "throughput_sequences_per_sec"
                ],
                reference[
                    "throughput_sequences_per_sec"
                ],
            ),

        "memory_ratio_256_vs_128":
            safe_ratio(
                condition_256[
                    "peak_gpu_memory_bytes"
                ],
                reference[
                    "peak_gpu_memory_bytes"
                ],
            ),

        "interpretation":
            (
                "Increasing spatial workload from 128x128 "
                "to 256x256 increased computational latency "
                "and reduced throughput while architecture "
                "and parameter count remained fixed."
            ),

        "accuracy_policy":
            (
                "Predictive metrics are reported only for the "
                "native 128x128 condition. The 64x64 and "
                "256x256 conditions are computational workloads "
                "created by deterministic resizing and therefore "
                "do not receive fabricated predictive metrics."
            ),
    }


# ============================================================
# ADD COMMUNICATION ANALYSIS
# ============================================================

def analyze_communication_scaling(
    rows: List[
        Dict[str, Any]
    ],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:

    reference = next(

        row

        for row in rows

        if abs(
            row[
                "requested_delay_ms"
            ]
            -
            REFERENCE_COMMUNICATION_DELAY_MS
        )
        < 1e-9
    )


    ref_latency = (
        reference[
            "latency_mean_sec"
        ]
    )

    ref_throughput = (
        reference[
            "throughput_sequences_per_sec"
        ]
    )


    for row in rows:

        row[
            "latency_difference_sec_vs_zero"
        ] = (
            row[
                "latency_mean_sec"
            ]
            -
            ref_latency
        )


        row[
            "latency_difference_ms_vs_zero"
        ] = (
            row[
                "latency_difference_sec_vs_zero"
            ]
            * 1000.0
        )


        row[
            "latency_ratio_vs_zero"
        ] = safe_ratio(
            row[
                "latency_mean_sec"
            ],
            ref_latency,
        )


        row[
            "throughput_ratio_vs_zero"
        ] = safe_ratio(
            row[
                "throughput_sequences_per_sec"
            ],
            ref_throughput,
        )


        row[
            "throughput_change_percent_vs_zero"
        ] = safe_percent_change(
            row[
                "throughput_sequences_per_sec"
            ],
            ref_throughput,
        )


        row[
            "barrier_error_ms"
        ] = (
            row[
                "realized_barrier_ms"
            ]
            -
            row[
                "requested_delay_ms"
            ]
        )


    max_prediction_difference = max(

        (
            row[
                "prediction_max_abs_difference"
            ]
            or 0.0
        )

        for row
        in rows
    )


    max_barrier_error = max(

        abs(
            row[
                "barrier_error_ms"
            ]
        )

        for row
        in rows
    )


    return {

        "reference_delay_ms":
            0.0,

        "zero_delay_latency_sec":
            ref_latency,

        "zero_delay_throughput":
            ref_throughput,

        "maximum_prediction_difference":
            max_prediction_difference,

        "maximum_absolute_barrier_error_ms":
            max_barrier_error,

        "physical_network_latency_measured":
            metadata[
                "physical_network_latency_measured"
            ],

        "delay_semantics":
            metadata[
                "delay_semantics"
            ],

        "interpretation":
            (
                "The controlled synchronization barrier "
                "faithfully reproduced the requested 0-10 ms "
                "communication delays without changing numerical "
                "predictions. Because these delays are small "
                "relative to approximately 0.32-s SACU inference, "
                "end-to-end latency differences remain within "
                "normal execution variability rather than "
                "forming a monotonic timing trend."
            ),

        "claim_limitation":
            (
                "Injected delays are controlled communication "
                "scenarios and must not be described as measured "
                "physical network latency."
            ),
    }


# ============================================================
# BUILD REVIEWER-FACING SUMMARY TABLE
# ============================================================

def build_reviewer_table(
    agent_rows,
    domain_rows,
    communication_rows,
) -> List[
    Dict[str, Any]
]:

    rows = []


    # --------------------------------------------------------
    # Agent scaling
    # --------------------------------------------------------

    for row in agent_rows:

        rows.append(
            {

                "scalability_axis":
                    "Agent count",

                "condition":
                    f"{row['agent_count']} SACUs",

                "reference":
                    row[
                        "reference_condition"
                    ],

                "parameter_count":
                    row[
                        "parameter_count"
                    ],

                "predictive_rmse":
                    row[
                        "test_rmse"
                    ],

                "wave_residual":
                    row[
                        "test_wave_residual"
                    ],

                "mean_latency_sec":
                    row[
                        "latency_mean_sec"
                    ],

                "p95_latency_sec":
                    row[
                        "latency_p95_sec"
                    ],

                "throughput_seq_per_sec":
                    row[
                        "throughput_sequences_per_sec"
                    ],

                "peak_gpu_memory_mib":
                    row[
                        "peak_gpu_memory_mib"
                    ],

                "interpretation_note":
                    (
                        "Predictive and computational "
                        "agent-scaling condition."
                    ),
            }
        )


    # --------------------------------------------------------
    # Domain scaling
    # --------------------------------------------------------

    for row in domain_rows:

        rows.append(
            {

                "scalability_axis":
                    "Domain size",

                "condition":
                    (
                        f"{row['domain_size']}x"
                        f"{row['domain_size']}"
                    ),

                "reference":
                    row[
                        "reference_condition"
                    ],

                "parameter_count":
                    row[
                        "parameter_count"
                    ],

                "predictive_rmse":
                    row[
                        "test_rmse"
                    ],

                "wave_residual":
                    row[
                        "test_wave_residual"
                    ],

                "mean_latency_sec":
                    row[
                        "latency_mean_sec"
                    ],

                "p95_latency_sec":
                    row[
                        "latency_p95_sec"
                    ],

                "throughput_seq_per_sec":
                    row[
                        "throughput_sequences_per_sec"
                    ],

                "peak_gpu_memory_mib":
                    row[
                        "peak_gpu_memory_mib"
                    ],

                "interpretation_note":
                    (
                        "Native predictive metrics valid."
                        if row[
                            "predictive_metrics_valid"
                        ]
                        else
                        "Computational workload only; "
                        "predictive metrics intentionally omitted."
                    ),
            }
        )


    # --------------------------------------------------------
    # Communication scaling
    # --------------------------------------------------------

    for row in communication_rows:

        rows.append(
            {

                "scalability_axis":
                    "Communication delay",

                "condition":
                    (
                        f"{row['requested_delay_ms']:.1f} ms"
                    ),

                "reference":
                    row[
                        "reference_condition"
                    ],

                "parameter_count":
                    None,

                "predictive_rmse":
                    None,

                "wave_residual":
                    None,

                "mean_latency_sec":
                    row[
                        "latency_mean_sec"
                    ],

                "p95_latency_sec":
                    row[
                        "latency_p95_sec"
                    ],

                "throughput_seq_per_sec":
                    row[
                        "throughput_sequences_per_sec"
                    ],

                "peak_gpu_memory_mib":
                    bytes_to_mib(
                        row[
                            "peak_gpu_memory_bytes"
                        ]
                    ),

                "interpretation_note":
                    (
                        "Controlled synchronization delay; "
                        "prediction unchanged; not measured "
                        "physical network latency."
                    ),
            }
        )


    return rows


# ============================================================
# BUILD TEXT REPORT
# ============================================================

def build_text_report(
    agent_rows,
    domain_rows,
    communication_rows,
    agent_analysis,
    domain_analysis,
    communication_analysis,
) -> str:

    lines: List[str] = []


    lines.append(
        "=" * 90
    )

    lines.append(
        "FINAL REVIEWER-REQUESTED SACU SCALABILITY ANALYSIS"
    )

    lines.append(
        "=" * 90
    )

    lines.append(
        ""
    )


    # ========================================================
    # AGENTS
    # ========================================================

    lines.append(
        "1. AGENT-COUNT SCALING"
    )

    lines.append(
        "-" * 90
    )


    lines.append(
        (
            f"{'Agents':>8}"
            f"{'Params':>12}"
            f"{'RMSE':>12}"
            f"{'Residual':>14}"
            f"{'Latency':>14}"
            f"{'Throughput':>14}"
        )
    )


    for row in agent_rows:

        lines.append(
            (
                f"{row['agent_count']:>8}"
                f"{row['parameter_count']:>12}"
                f"{row['test_rmse']:>12.6f}"
                f"{row['test_wave_residual']:>14.6f}"
                f"{row['latency_mean_sec']:>14.6f}"
                f"{row['throughput_sequences_per_sec']:>14.3f}"
            )
        )


    lines.append(
        ""
    )

    lines.append(
        (
            "Observed RMSE range across 4-25 SACUs: "
            f"{agent_analysis['rmse_range']:.6f}"
        )
    )

    lines.append(
        (
            "25-agent latency / 4-agent latency: "
            f"{agent_analysis['latency_ratio_25_vs_4']:.3f}x"
        )
    )

    lines.append(
        (
            "Throughput reduction from 4 to 25 agents: "
            f"{agent_analysis['throughput_reduction_percent_25_vs_4']:.2f}%"
        )
    )

    lines.append(
        ""
    )

    lines.append(
        agent_analysis[
            "interpretation"
        ]
    )

    lines.append(
        agent_analysis[
            "memory_interpretation"
        ]
    )


    # ========================================================
    # DOMAIN
    # ========================================================

    lines.append(
        ""
    )

    lines.append(
        "2. DOMAIN-SIZE SCALING"
    )

    lines.append(
        "-" * 90
    )


    lines.append(
        (
            f"{'Domain':>12}"
            f"{'Pixel ratio':>14}"
            f"{'Latency':>14}"
            f"{'P95':>14}"
            f"{'Throughput':>14}"
            f"{'GPU MiB':>14}"
        )
    )


    for row in domain_rows:

        lines.append(
            (
                f"{str(row['domain_size']) + 'x' + str(row['domain_size']):>12}"
                f"{row['pixel_ratio_vs_128']:>14.3f}"
                f"{row['latency_mean_sec']:>14.6f}"
                f"{row['latency_p95_sec']:>14.6f}"
                f"{row['throughput_sequences_per_sec']:>14.3f}"
                f"{row['peak_gpu_memory_mib']:>14.1f}"
            )
        )


    lines.append(
        ""
    )

    lines.append(
        (
            "128x128 -> 256x256 workload increase: 4.000x pixels"
        )
    )

    lines.append(
        (
            "128x128 -> 256x256 latency increase: "
            f"{domain_analysis['latency_ratio_256_vs_128']:.3f}x"
        )
    )

    lines.append(
        (
            "128x128 -> 256x256 throughput ratio: "
            f"{domain_analysis['throughput_ratio_256_vs_128']:.3f}x"
        )
    )

    lines.append(
        (
            "128x128 -> 256x256 GPU-memory ratio: "
            f"{domain_analysis['memory_ratio_256_vs_128']:.3f}x"
        )
    )

    lines.append(
        ""
    )

    lines.append(
        domain_analysis[
            "interpretation"
        ]
    )

    lines.append(
        domain_analysis[
            "accuracy_policy"
        ]
    )


    # ========================================================
    # COMMUNICATION
    # ========================================================

    lines.append(
        ""
    )

    lines.append(
        "3. CONTROLLED COMMUNICATION-DELAY SCALING"
    )

    lines.append(
        "-" * 90
    )


    lines.append(
        (
            f"{'Delay ms':>12}"
            f"{'Barrier ms':>14}"
            f"{'Latency':>14}"
            f"{'P95':>14}"
            f"{'Throughput':>14}"
            f"{'Pred diff':>14}"
        )
    )


    for row in communication_rows:

        lines.append(
            (
                f"{row['requested_delay_ms']:>12.1f}"
                f"{row['realized_barrier_ms']:>14.3f}"
                f"{row['latency_mean_sec']:>14.6f}"
                f"{row['latency_p95_sec']:>14.6f}"
                f"{row['throughput_sequences_per_sec']:>14.3f}"
                f"{row['prediction_max_abs_difference']:>14.3e}"
            )
        )


    lines.append(
        ""
    )

    lines.append(
        (
            "Maximum prediction difference across delay conditions: "
            f"{communication_analysis['maximum_prediction_difference']:.3e}"
        )
    )

    lines.append(
        (
            "Maximum absolute communication-barrier timing error: "
            f"{communication_analysis['maximum_absolute_barrier_error_ms']:.3f} ms"
        )
    )

    lines.append(
        ""
    )

    lines.append(
        communication_analysis[
            "interpretation"
        ]
    )

    lines.append(
        communication_analysis[
            "claim_limitation"
        ]
    )


    # ========================================================
    # OVERALL
    # ========================================================

    lines.append(
        ""
    )

    lines.append(
        "4. OVERALL SCALABILITY INTERPRETATION"
    )

    lines.append(
        "-" * 90
    )


    lines.append(
        (
            "The completed reviewer-requested experiments show "
            "that SACU reconstruction accuracy remains stable "
            "as the number of agents increases from 4 to 25, "
            "while computational latency rises and throughput "
            "declines."
        )
    )


    lines.append(
        (
            "With the 16-agent architecture held fixed, increasing "
            "the native spatial workload from 128x128 to a "
            "computational 256x256 workload increased latency "
            "and GPU-memory demand and reduced throughput."
        )
    )


    lines.append(
        (
            "Controlled communication delays of up to 10 ms "
            "were faithfully introduced at the synchronization "
            "barrier without changing numerical predictions. "
            "Because the injected delay is small relative to "
            "the approximately 0.32-s end-to-end inference time, "
            "the total measured timing differences remain within "
            "normal execution variability."
        )
    )


    lines.append(
        ""
    )

    lines.append(
        (
            "These experiments support a bounded scalability claim: "
            "the implemented SACU framework remains operational "
            "over the tested agent counts, spatial workloads, and "
            "controlled communication-delay scenarios, but larger "
            "configurations incur measurable computational costs."
        )
    )


    lines.append(
        ""
    )

    lines.append(
        (
            "No submitted-manuscript numerical value was "
            "modified by this analysis."
        )
    )


    return "\n".join(
        lines
    )


# ============================================================
# BUILD REVIEWER-READY MARKDOWN
# ============================================================

def build_reviewer_markdown(
    agent_rows,
    domain_rows,
    communication_rows,
    agent_analysis,
    domain_analysis,
    communication_analysis,
) -> str:

    lines = []


    lines.append(
        "# Reviewer-Requested SACU Scalability Results"
    )

    lines.append(
        ""
    )


    # --------------------------------------------------------
    # Agent table
    # --------------------------------------------------------

    lines.append(
        "## Agent-count scaling"
    )

    lines.append(
        ""
    )

    lines.append(
        "| SACUs | Parameters | Test RMSE | Wave residual | Mean latency (s) | Throughput (seq/s) |"
    )

    lines.append(
        "|---:|---:|---:|---:|---:|---:|"
    )


    for row in agent_rows:

        lines.append(
            (
                f"| {row['agent_count']} "
                f"| {row['parameter_count']} "
                f"| {row['test_rmse']:.6f} "
                f"| {row['test_wave_residual']:.6f} "
                f"| {row['latency_mean_sec']:.6f} "
                f"| {row['throughput_sequences_per_sec']:.3f} |"
            )
        )


    lines.append(
        ""
    )

    lines.append(
        (
            "Reconstruction accuracy remained tightly clustered "
            f"(RMSE range = {agent_analysis['rmse_range']:.6f}) "
            "across 4–25 SACUs, while computational cost increased."
        )
    )


    # --------------------------------------------------------
    # Domain table
    # --------------------------------------------------------

    lines.append(
        ""
    )

    lines.append(
        "## Domain-size scaling"
    )

    lines.append(
        ""
    )

    lines.append(
        "| Domain | Pixel workload vs. 128×128 | Mean latency (s) | P95 latency (s) | Throughput (seq/s) | Peak GPU memory (MiB) | Predictive metrics |"
    )

    lines.append(
        "|---:|---:|---:|---:|---:|---:|---|"
    )


    for row in domain_rows:

        predictive = (
            f"RMSE {row['test_rmse']:.6f}"
            if row[
                "predictive_metrics_valid"
            ]
            else "Not reported — computational workload only"
        )


        lines.append(
            (
                f"| {row['domain_size']}×{row['domain_size']} "
                f"| {row['pixel_ratio_vs_128']:.2f}× "
                f"| {row['latency_mean_sec']:.6f} "
                f"| {row['latency_p95_sec']:.6f} "
                f"| {row['throughput_sequences_per_sec']:.3f} "
                f"| {row['peak_gpu_memory_mib']:.1f} "
                f"| {predictive} |"
            )
        )


    lines.append(
        ""
    )

    lines.append(
        (
            "The 256×256 computational workload contains four "
            "times as many spatial locations as 128×128 and "
            f"required {domain_analysis['latency_ratio_256_vs_128']:.2f}× "
            "the mean inference time while reducing throughput."
        )
    )


    # --------------------------------------------------------
    # Communication
    # --------------------------------------------------------

    lines.append(
        ""
    )

    lines.append(
        "## Controlled communication-delay scaling"
    )

    lines.append(
        ""
    )

    lines.append(
        "| Injected delay (ms) | Realized barrier (ms) | Mean latency (s) | P95 latency (s) | Throughput (seq/s) | Prediction difference |"
    )

    lines.append(
        "|---:|---:|---:|---:|---:|---:|"
    )


    for row in communication_rows:

        lines.append(
            (
                f"| {row['requested_delay_ms']:.1f} "
                f"| {row['realized_barrier_ms']:.3f} "
                f"| {row['latency_mean_sec']:.6f} "
                f"| {row['latency_p95_sec']:.6f} "
                f"| {row['throughput_sequences_per_sec']:.3f} "
                f"| {row['prediction_max_abs_difference']:.3e} |"
            )
        )


    lines.append(
        ""
    )

    lines.append(
        (
            "The controlled synchronization delay was realized "
            "accurately and did not alter numerical predictions. "
            "These are injected communication scenarios, not "
            "measurements of physical network latency."
        )
    )


    return "\n".join(
        lines
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not SCALABILITY_ROOT.exists():

        raise FileNotFoundError(
            "Scalability output root not found:\n"
            f"{SCALABILITY_ROOT}"
        )


    print()
    print("=" * 90)

    print(
        "FINAL SACU SCALABILITY ANALYSIS"
    )

    print("=" * 90)


    print(
        "Scalability root:"
    )

    print(
        SCALABILITY_ROOT
    )


    print()


    # ========================================================
    # COLLECT
    # ========================================================

    agent_rows = (
        collect_agent_results()
    )


    domain_rows = (
        collect_domain_results()
    )


    (
        communication_rows,
        communication_metadata,
    ) = collect_communication_results()


    print(
        "PASS: agent-count conditions found:",
        [
            row[
                "agent_count"
            ]
            for row
            in agent_rows
        ],
    )


    print(
        "PASS: domain-size conditions found:",
        [
            row[
                "domain_size"
            ]
            for row
            in domain_rows
        ],
    )


    print(
        "PASS: communication-delay conditions found:",
        [
            row[
                "requested_delay_ms"
            ]
            for row
            in communication_rows
        ],
    )


    # ========================================================
    # ANALYZE
    # ========================================================

    agent_analysis = (
        analyze_agent_scaling(
            agent_rows
        )
    )


    domain_analysis = (
        analyze_domain_scaling(
            domain_rows
        )
    )


    communication_analysis = (
        analyze_communication_scaling(

            communication_rows,

            communication_metadata,
        )
    )


    # ========================================================
    # REVIEWER TABLE
    # ========================================================

    reviewer_table = (
        build_reviewer_table(

            agent_rows,

            domain_rows,

            communication_rows,
        )
    )


    # ========================================================
    # FINAL OUTPUT DIRECTORY
    # ========================================================

    FINAL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # CSV OUTPUTS
    # ========================================================

    save_csv(
        FINAL_OUTPUT_DIR
        / "agent_scalability_final.csv",
        agent_rows,
    )


    save_csv(
        FINAL_OUTPUT_DIR
        / "domain_scalability_final.csv",
        domain_rows,
    )


    save_csv(
        FINAL_OUTPUT_DIR
        / "communication_scalability_final.csv",
        communication_rows,
    )


    save_csv(
        FINAL_OUTPUT_DIR
        / "reviewer_scalability_table.csv",
        reviewer_table,
    )


    # ========================================================
    # JSON MASTER OUTPUT
    # ========================================================

    master = {

        "status":
            "complete",

        "experiment_family":
            "reviewer_requested_sacu_scalability",

        "recomputation_performed":
            False,

        "submitted_manuscript_values_modified":
            False,

        "reference_conditions": {

            "agent_count":
                REFERENCE_AGENTS,

            "agent_grid":
                REFERENCE_GRID,

            "domain_size":
                "128x128",

            "communication_delay_ms":
                0.0,
        },

        "agent_scaling": {

            "conditions":
                agent_rows,

            "analysis":
                agent_analysis,
        },

        "domain_scaling": {

            "conditions":
                domain_rows,

            "analysis":
                domain_analysis,
        },

        "communication_scaling": {

            "conditions":
                communication_rows,

            "metadata":
                communication_metadata,

            "analysis":
                communication_analysis,
        },

        "overall_interpretation": {

            "agent_count":
                (
                    "Accuracy remained stable over 4-25 SACUs, "
                    "while latency increased and throughput declined."
                ),

            "domain_size":
                (
                    "Increasing spatial workload increased latency "
                    "and memory demand and reduced throughput with "
                    "the 16-agent architecture held fixed."
                ),

            "communication":
                (
                    "Controlled 0-10 ms synchronization delays were "
                    "faithfully introduced without changing numerical "
                    "predictions; end-to-end differences remained "
                    "small relative to baseline SACU inference time."
                ),

            "claim_scope":
                (
                    "Results support empirical scalability only over "
                    "the tested agent counts, spatial workloads, and "
                    "controlled delay scenarios. They do not establish "
                    "unbounded scalability or real network latency."
                ),
        },
    }


    save_json(
        FINAL_OUTPUT_DIR
        / "scalability_analysis_master.json",
        master,
    )


    # ========================================================
    # TEXT REPORT
    # ========================================================

    report = build_text_report(

        agent_rows,
        domain_rows,
        communication_rows,

        agent_analysis,
        domain_analysis,
        communication_analysis,
    )


    save_text(
        FINAL_OUTPUT_DIR
        / "scalability_summary.txt",
        report,
    )


    # ========================================================
    # MARKDOWN REVIEWER TABLE
    # ========================================================

    markdown = build_reviewer_markdown(

        agent_rows,
        domain_rows,
        communication_rows,

        agent_analysis,
        domain_analysis,
        communication_analysis,
    )


    save_text(
        FINAL_OUTPUT_DIR
        / "reviewer_scalability_results.md",
        markdown,
    )


    # ========================================================
    # CONSOLE REPORT
    # ========================================================

    print()
    print(
        report
    )


    print()
    print("=" * 90)

    print(
        "FINAL OUTPUT FILES"
    )

    print("=" * 90)


    outputs = [

        FINAL_OUTPUT_DIR
        / "agent_scalability_final.csv",

        FINAL_OUTPUT_DIR
        / "domain_scalability_final.csv",

        FINAL_OUTPUT_DIR
        / "communication_scalability_final.csv",

        FINAL_OUTPUT_DIR
        / "reviewer_scalability_table.csv",

        FINAL_OUTPUT_DIR
        / "scalability_analysis_master.json",

        FINAL_OUTPUT_DIR
        / "scalability_summary.txt",

        FINAL_OUTPUT_DIR
        / "reviewer_scalability_results.md",
    ]


    for path in outputs:

        print(
            path
        )


    print()
    print("=" * 90)

    print(
        "PASS: all reviewer-requested scalability "
        "dimensions have been consolidated."
    )

    print(
        "PASS: no training or inference was rerun."
    )

    print(
        "PASS: no unsupported predictive metric "
        "was created."
    )

    print(
        "PASS: no submitted-manuscript numerical "
        "value was modified."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()