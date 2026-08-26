from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# PATHS
# ============================================================

THIS_FILE = Path(__file__).resolve()

EXPERIMENT_DIR = THIS_FILE.parent

NEW_ROOT = (
    EXPERIMENT_DIR
    .parent
    .parent
)

OUTPUT_ROOT = (
    NEW_ROOT
    / "outputs"
)

BASELINE_ROOT = (
    OUTPUT_ROOT
    / "baseline_comparison"
)

MODERN_ROOT = (
    OUTPUT_ROOT
    / "modern_baselines"
)

COMPARISON_ROOT = (
    OUTPUT_ROOT
    / "reviewer_baseline_comparison"
)


# ============================================================
# FIXED MODEL REGISTRY
#
# IMPORTANT:
# This script does NOT search arbitrary experiments and does
# NOT choose models by performance.
#
# It reads only the five reviewer-requested comparator families
# that have already been completed.
# ============================================================

MODEL_REGISTRY = {

    "MLP": {
        "root":
            BASELINE_ROOT
            / "mlp_4x256",

        "display_name":
            "MLP 4x256",

        "family":
            "conventional",

        "physics_in_training":
            False,
    },

    "PINN": {
        "root":
            BASELINE_ROOT
            / "pinn_4x256",

        "display_name":
            "PINN 4x256",

        "family":
            "physics_informed",

        "physics_in_training":
            True,
    },

    "MoE": {
        "root":
            BASELINE_ROOT
            / "moe_4experts",

        "display_name":
            "MoE (4 experts)",

        "family":
            "mixture_of_experts",

        "physics_in_training":
            False,
    },

    "FNO": {
        "root":
            MODERN_ROOT
            / "fno_3d_4x10",

        "display_name":
            "3D FNO",

        "family":
            "neural_operator",

        "physics_in_training":
            False,
    },

    "DD_PINN": {
        "root":
            MODERN_ROOT
            / "dd_pinn_4subdomains",

        "display_name":
            "DD-PINN (4 subdomains)",

        "family":
            "domain_decomposed_physics_informed",

        "physics_in_training":
            True,
    },
}


# ============================================================
# EXPECTED SCIENTIFIC CONTROLS
# ============================================================

REFERENCE_PARAMETER_COUNT = 198401

EXPECTED_MODELS = [
    "MLP",
    "PINN",
    "MoE",
    "FNO",
    "DD_PINN",
]


# ============================================================
# FILE HELPERS
# ============================================================

def read_json(
    path: Path,
) -> Dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


def read_single_row_csv(
    path: Path,
) -> Dict[str, str]:

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(
            f
        )

        rows = list(
            reader
        )


    if len(rows) != 1:

        raise RuntimeError(
            "Expected exactly one row in summary CSV:\n"
            f"{path}\n"
            f"Observed rows: {len(rows)}"
        )


    return rows[0]


def write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:

    if not rows:

        raise ValueError(
            "Cannot write empty comparison table."
        )


    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    fieldnames: List[str] = []

    seen = set()


    for row in rows:

        for key in row.keys():

            if key not in seen:

                seen.add(
                    key
                )

                fieldnames.append(
                    key
                )


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

        for row in rows:

            writer.writerow(
                row
            )


def write_json(
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


# ============================================================
# NUMERIC HELPERS
# ============================================================

def as_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:

    if value is None:
        return default


    if isinstance(
        value,
        (
            float,
            int,
        ),
    ):

        return float(
            value
        )


    text = str(
        value
    ).strip()


    if text == "":
        return default


    try:

        return float(
            text
        )

    except ValueError:

        return default


def as_int(
    value: Any,
    default: Optional[int] = None,
) -> Optional[int]:

    number = as_float(
        value
    )


    if number is None:
        return default


    return int(
        round(
            number
        )
    )


def finite_or_none(
    value: Optional[float],
) -> Optional[float]:

    if value is None:
        return None


    if not math.isfinite(
        value
    ):

        return None


    return value


# ============================================================
# COMPLETED-RUN DISCOVERY
# ============================================================

def is_completed_run(
    run_dir: Path,
) -> bool:

    """
    A run is treated as completed only if it has:

        tables/summary.csv
        results.json

    This deliberately excludes failed/interrupted runs.

    For DD-PINN we additionally require the saved result record
    to report verified checkpoint restoration.
    """

    summary_path = (
        run_dir
        / "tables"
        / "summary.csv"
    )

    results_path = (
        run_dir
        / "results.json"
    )


    if not summary_path.exists():
        return False


    if not results_path.exists():
        return False


    try:

        results = read_json(
            results_path
        )

    except Exception:

        return False


    # --------------------------------------------------------
    # Must be explicitly a reviewer-requested new experiment.
    # --------------------------------------------------------

    if results.get(
        "new_experiment"
    ) is False:

        return False


    if results.get(
        "replaces_existing_manuscript_numbers"
    ) is True:

        return False


    return True


def discover_completed_runs(
    model_root: Path,
) -> List[Path]:

    if not model_root.exists():

        raise FileNotFoundError(
            "Model output directory does not exist:\n"
            f"{model_root}"
        )


    candidates = []


    for path in model_root.iterdir():

        if (
            path.is_dir()
            and is_completed_run(
                path
            )
        ):

            candidates.append(
                path
            )


    return sorted(
        candidates,
        key=lambda p: p.name,
    )


def select_latest_completed_run(
    model_key: str,
) -> Path:

    registry = MODEL_REGISTRY[
        model_key
    ]


    model_root = registry[
        "root"
    ]


    completed_runs = (
        discover_completed_runs(
            model_root
        )
    )


    if not completed_runs:

        raise RuntimeError(
            "No completed run was found for "
            f"{model_key} in:\n{model_root}"
        )


    selected = (
        completed_runs[-1]
    )


    print(
        f"[SELECTED] {model_key}:"
    )

    print(
        f"           {selected}"
    )


    return selected


# ============================================================
# SUMMARY FIELD RESOLUTION
#
# Earlier baseline scripts and modern scripts use slightly
# different column names. These functions normalize them
# without altering any numerical value.
# ============================================================

def first_present(
    row: Dict[str, Any],
    keys: List[str],
) -> Any:

    for key in keys:

        if key in row:

            value = row[
                key
            ]

            if str(
                value
            ).strip() != "":

                return value


    return None


def resolve_parameter_count(
    row: Dict[str, Any],
) -> int:

    value = first_present(
        row,
        [
            "parameter_count",
            "parameters",
        ],
    )


    result = as_int(
        value
    )


    if result is None:

        raise RuntimeError(
            "Parameter count missing from summary."
        )


    return result


def resolve_best_epoch(
    row: Dict[str, Any],
) -> int:

    result = as_int(
        first_present(
            row,
            [
                "best_epoch",
            ],
        )
    )


    if result is None:

        raise RuntimeError(
            "Best epoch missing."
        )


    return result


def resolve_validation_rmse(
    row: Dict[str, Any],
) -> float:

    result = as_float(
        first_present(
            row,
            [
                "best_validation_rmse",
                "validation_RMSE_mean",
                "validation_rmse",
            ],
        )
    )


    if result is None:

        raise RuntimeError(
            "Validation RMSE missing."
        )


    return result


def resolve_test_mae(
    row: Dict[str, Any],
) -> float:

    result = as_float(
        first_present(
            row,
            [
                "test_MAE_mean",
                "test_mae",
                "Test_MAE",
            ],
        )
    )


    if result is None:

        raise RuntimeError(
            "Test MAE missing."
        )


    return result


def resolve_test_rmse(
    row: Dict[str, Any],
) -> float:

    result = as_float(
        first_present(
            row,
            [
                "test_RMSE_mean",
                "test_rmse",
                "Test_RMSE",
            ],
        )
    )


    if result is None:

        raise RuntimeError(
            "Test RMSE missing."
        )


    return result


def resolve_test_residual(
    row: Dict[str, Any],
) -> float:

    result = as_float(
        first_present(
            row,
            [
                "test_residual_mean",
                "test_wave_residual",
                "test_residual",
            ],
        )
    )


    if result is None:

        raise RuntimeError(
            "Test residual missing."
        )


    return result


def resolve_latency_mean(
    row: Dict[str, Any],
) -> float:

    result = as_float(
        first_present(
            row,
            [
                "latency_mean_sec",
                "test_latency_mean_sec",
                "latency_mean",
            ],
        )
    )


    if result is None:

        raise RuntimeError(
            "Mean latency missing."
        )


    return result


def resolve_latency_p95(
    row: Dict[str, Any],
) -> float:

    result = as_float(
        first_present(
            row,
            [
                "latency_p95_sec",
                "test_latency_p95_sec",
                "latency_p95",
            ],
        )
    )


    if result is None:

        raise RuntimeError(
            "P95 latency missing."
        )


    return result


def resolve_training_time(
    row: Dict[str, Any],
) -> Optional[float]:

    return finite_or_none(
        as_float(
            first_present(
                row,
                [
                    "training_time_sec",
                    "total_training_time_sec",
                ],
            )
        )
    )


# ============================================================
# LOAD ONE FINALIZED MODEL RESULT
# ============================================================

def load_model_result(
    model_key: str,
) -> Dict[str, Any]:

    run_dir = (
        select_latest_completed_run(
            model_key
        )
    )


    summary_path = (
        run_dir
        / "tables"
        / "summary.csv"
    )


    results_path = (
        run_dir
        / "results.json"
    )


    summary = read_single_row_csv(
        summary_path
    )


    results = read_json(
        results_path
    )


    parameter_count = (
        resolve_parameter_count(
            summary
        )
    )


    parameter_difference = (
        parameter_count
        - REFERENCE_PARAMETER_COUNT
    )


    parameter_difference_percent = (
        parameter_difference
        / REFERENCE_PARAMETER_COUNT
        * 100.0
    )


    registry = MODEL_REGISTRY[
        model_key
    ]


    result = {

        "model_key":
            model_key,

        "model":
            registry[
                "display_name"
            ],

        "family":
            registry[
                "family"
            ],

        "physics_in_training":
            registry[
                "physics_in_training"
            ],

        "parameter_count":
            parameter_count,

        "parameter_difference":
            parameter_difference,

        "parameter_difference_percent":
            parameter_difference_percent,

        "best_epoch":
            resolve_best_epoch(
                summary
            ),

        "best_validation_rmse":
            resolve_validation_rmse(
                summary
            ),

        "test_mae":
            resolve_test_mae(
                summary
            ),

        "test_rmse":
            resolve_test_rmse(
                summary
            ),

        "test_wave_residual":
            resolve_test_residual(
                summary
            ),

        "latency_mean_sec":
            resolve_latency_mean(
                summary
            ),

        "latency_p95_sec":
            resolve_latency_p95(
                summary
            ),

        "training_time_sec":
            resolve_training_time(
                summary
            ),

        "run_directory":
            str(
                run_dir
            ),

        "summary_source":
            str(
                summary_path
            ),

        "results_source":
            str(
                results_path
            ),

        "new_reviewer_requested_experiment":
            True,

        "replaces_existing_manuscript_numbers":
            False,
    }


    # --------------------------------------------------------
    # DD-PINN-specific checkpoint verification
    # --------------------------------------------------------

    if model_key == "DD_PINN":

        extra = results.get(
            "extra_results",
            {}
        )


        reload_verified = extra.get(
            "checkpoint_reload_verified",
            False,
        )


        reload_difference = as_float(
            extra.get(
                "checkpoint_reload_rmse_difference"
            )
        )


        result[
            "checkpoint_reload_verified"
        ] = bool(
            reload_verified
        )


        result[
            "checkpoint_reload_rmse_difference"
        ] = reload_difference


        if not reload_verified:

            raise RuntimeError(
                "DD-PINN run does not report verified "
                "checkpoint restoration:\n"
                f"{run_dir}"
            )


        if (
            reload_difference is None
            or reload_difference > 1e-6
        ):

            raise RuntimeError(
                "DD-PINN checkpoint restoration mismatch.\n"
                f"Difference: {reload_difference}"
            )


    return result


# ============================================================
# SCIENTIFIC CONSISTENCY CHECKS
# ============================================================

def validate_comparison(
    rows: List[Dict[str, Any]],
) -> None:

    observed_keys = {
        row[
            "model_key"
        ]
        for row
        in rows
    }


    if observed_keys != set(
        EXPECTED_MODELS
    ):

        raise RuntimeError(
            "Comparison does not contain exactly "
            "the five required models.\n"
            f"Observed: {sorted(observed_keys)}"
        )


    for row in rows:

        if row[
            "parameter_count"
        ] <= 0:

            raise RuntimeError(
                f"Invalid parameter count for "
                f"{row['model']}."
            )


        for metric in [

            "best_validation_rmse",
            "test_mae",
            "test_rmse",
            "test_wave_residual",
            "latency_mean_sec",
            "latency_p95_sec",

        ]:

            value = row[
                metric
            ]


            if (
                value is None
                or not math.isfinite(
                    float(
                        value
                    )
                )
            ):

                raise RuntimeError(
                    f"Invalid {metric} "
                    f"for {row['model']}."
                )


        if row[
            "test_rmse"
        ] < 0:

            raise RuntimeError(
                f"Negative test RMSE for "
                f"{row['model']}."
            )


        if row[
            "test_wave_residual"
        ] < 0:

            raise RuntimeError(
                f"Negative residual for "
                f"{row['model']}."
            )


        if row[
            "latency_mean_sec"
        ] <= 0:

            raise RuntimeError(
                f"Invalid latency for "
                f"{row['model']}."
            )


# ============================================================
# RANKING
#
# Rankings are descriptive only.
# No composite score is invented.
# ============================================================

def add_ranks(
    rows: List[Dict[str, Any]],
) -> None:

    rank_specs = {

        "rank_test_mae":
            "test_mae",

        "rank_test_rmse":
            "test_rmse",

        "rank_wave_residual":
            "test_wave_residual",

        "rank_latency":
            "latency_mean_sec",
    }


    for rank_name, metric_name in (
        rank_specs.items()
    ):

        ordered = sorted(
            rows,
            key=lambda row:
                row[
                    metric_name
                ],
        )


        for rank, row in enumerate(
            ordered,
            start=1,
        ):

            row[
                rank_name
            ] = rank


# ============================================================
# REFERENCE-RELATIVE METRICS
#
# These do not replace raw measurements.
# They are provided only for interpretation.
# ============================================================

def add_reference_relative_fields(
    rows: List[Dict[str, Any]],
) -> None:

    mlp_row = next(
        row
        for row
        in rows
        if row[
            "model_key"
        ] == "MLP"
    )


    mlp_rmse = (
        mlp_row[
            "test_rmse"
        ]
    )

    mlp_residual = (
        mlp_row[
            "test_wave_residual"
        ]
    )

    mlp_latency = (
        mlp_row[
            "latency_mean_sec"
        ]
    )


    for row in rows:

        row[
            "rmse_change_vs_mlp_percent"
        ] = (
            (
                row[
                    "test_rmse"
                ]
                - mlp_rmse
            )
            / mlp_rmse
            * 100.0
        )


        if mlp_residual > 0:

            row[
                "residual_ratio_vs_mlp"
            ] = (
                row[
                    "test_wave_residual"
                ]
                / mlp_residual
            )

        else:

            row[
                "residual_ratio_vs_mlp"
            ] = None


        row[
            "latency_ratio_vs_mlp"
        ] = (
            row[
                "latency_mean_sec"
            ]
            / mlp_latency
        )


# ============================================================
# FORMAT REVIEWER TABLE
# ============================================================

def reviewer_table_rows(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    output = []


    for row in rows:

        output.append(
            {

                "Model":
                    row[
                        "model"
                    ],

                "Parameters":
                    row[
                        "parameter_count"
                    ],

                "Parameter Difference (%)":
                    round(
                        row[
                            "parameter_difference_percent"
                        ],
                        3,
                    ),

                "Physics in Training":
                    (
                        "Yes"
                        if row[
                            "physics_in_training"
                        ]
                        else "No"
                    ),

                "Best Epoch":
                    row[
                        "best_epoch"
                    ],

                "Validation RMSE":
                    round(
                        row[
                            "best_validation_rmse"
                        ],
                        8,
                    ),

                "Test MAE":
                    round(
                        row[
                            "test_mae"
                        ],
                        6,
                    ),

                "Test RMSE":
                    round(
                        row[
                            "test_rmse"
                        ],
                        6,
                    ),

                "Wave Residual":
                    round(
                        row[
                            "test_wave_residual"
                        ],
                        6,
                    ),

                "Mean Latency (s)":
                    round(
                        row[
                            "latency_mean_sec"
                        ],
                        6,
                    ),

                "P95 Latency (s)":
                    round(
                        row[
                            "latency_p95_sec"
                        ],
                        6,
                    ),

                "Training Time (s)":
                    (
                        round(
                            row[
                                "training_time_sec"
                            ],
                            3,
                        )
                        if row[
                            "training_time_sec"
                        ]
                        is not None
                        else ""
                    ),

                "RMSE Rank":
                    row[
                        "rank_test_rmse"
                    ],

                "Residual Rank":
                    row[
                        "rank_wave_residual"
                    ],

                "Latency Rank":
                    row[
                        "rank_latency"
                    ],
            }
        )


    return output


# ============================================================
# TEXT SUMMARY
# ============================================================

def create_text_summary(
    rows: List[Dict[str, Any]],
) -> str:

    best_rmse = min(
        rows,
        key=lambda row:
            row[
                "test_rmse"
            ],
    )


    best_residual = min(
        rows,
        key=lambda row:
            row[
                "test_wave_residual"
            ],
    )


    fastest = min(
        rows,
        key=lambda row:
            row[
                "latency_mean_sec"
            ],
    )


    lines = [

        "REVIEWER-REQUESTED BASELINE COMPARISON",
        "=" * 72,
        "",
        "This report consolidates only completed experimental runs.",
        "It does not modify any numerical value in the submitted manuscript.",
        "",
        f"Lowest held-out test RMSE : "
        f"{best_rmse['model']} "
        f"({best_rmse['test_rmse']:.6f})",

        f"Lowest wave residual      : "
        f"{best_residual['model']} "
        f"({best_residual['test_wave_residual']:.6f})",

        f"Lowest mean latency        : "
        f"{fastest['model']} "
        f"({fastest['latency_mean_sec']:.6f} s)",

        "",
        "Per-model results:",
        "",
    ]


    for row in rows:

        lines.extend(
            [
                (
                    f"{row['model']}: "
                    f"params={row['parameter_count']}, "
                    f"MAE={row['test_mae']:.6f}, "
                    f"RMSE={row['test_rmse']:.6f}, "
                    f"residual="
                    f"{row['test_wave_residual']:.6f}, "
                    f"latency="
                    f"{row['latency_mean_sec']:.6f}s"
                )
            ]
        )


    lines.extend(
        [
            "",
            "Interpretation policy:",
            (
                "No composite score is used. Reconstruction error, "
                "physics consistency, and latency remain separate "
                "experimental dimensions."
            ),
            (
                "The held-out test set is not used for checkpoint "
                "selection."
            ),
            (
                "All reported values are new reviewer-requested "
                "experimental results."
            ),
        ]
    )


    return "\n".join(
        lines
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)

    print(
        "CONSOLIDATED REVIEWER BASELINE COMPARISON"
    )

    print("=" * 80)

    print(
        "Project root:"
    )

    print(
        NEW_ROOT
    )

    print()


    # ========================================================
    # LOAD ONLY FINALIZED RUNS
    # ========================================================

    rows: List[
        Dict[str, Any]
    ] = []


    for model_key in EXPECTED_MODELS:

        rows.append(
            load_model_result(
                model_key
            )
        )


    # ========================================================
    # VALIDATION
    # ========================================================

    validate_comparison(
        rows
    )


    add_ranks(
        rows
    )


    add_reference_relative_fields(
        rows
    )


    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    COMPARISON_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # RAW AUDIT TABLE
    # ========================================================

    write_csv(
        COMPARISON_ROOT
        / "baseline_comparison_full.csv",
        rows,
    )


    write_json(
        COMPARISON_ROOT
        / "baseline_comparison_full.json",
        {
            "reference_parameter_count":
                REFERENCE_PARAMETER_COUNT,

            "number_of_models":
                len(
                    rows
                ),

            "models":
                rows,

            "policy":
                (
                    "Only completed reviewer-requested runs "
                    "are consolidated. No manuscript value "
                    "is replaced by this script."
                ),
        },
    )


    # ========================================================
    # CLEAN REVIEWER TABLE
    # ========================================================

    clean_rows = reviewer_table_rows(
        rows
    )


    write_csv(
        COMPARISON_ROOT
        / "reviewer_baseline_table.csv",
        clean_rows,
    )


    # ========================================================
    # TEXT SUMMARY
    # ========================================================

    text_summary = (
        create_text_summary(
            rows
        )
    )


    summary_path = (
        COMPARISON_ROOT
        / "comparison_summary.txt"
    )


    summary_path.write_text(
        text_summary,
        encoding="utf-8",
    )


    # ========================================================
    # CONSOLE TABLE
    # ========================================================

    print()
    print(
        f"{'Model':<25}"
        f"{'Params':>10}"
        f"{'MAE':>11}"
        f"{'RMSE':>11}"
        f"{'Residual':>13}"
        f"{'Latency':>12}"
    )

    print(
        "-" * 82
    )


    for row in rows:

        print(
            f"{row['model']:<25}"
            f"{row['parameter_count']:>10d}"
            f"{row['test_mae']:>11.6f}"
            f"{row['test_rmse']:>11.6f}"
            f"{row['test_wave_residual']:>13.6f}"
            f"{row['latency_mean_sec']:>12.6f}"
        )


    # ========================================================
    # BEST BY INDIVIDUAL DIMENSION
    # ========================================================

    best_mae = min(
        rows,
        key=lambda r:
            r[
                "test_mae"
            ],
    )


    best_rmse = min(
        rows,
        key=lambda r:
            r[
                "test_rmse"
            ],
    )


    best_residual = min(
        rows,
        key=lambda r:
            r[
                "test_wave_residual"
            ],
    )


    fastest = min(
        rows,
        key=lambda r:
            r[
                "latency_mean_sec"
            ],
    )


    print()
    print("=" * 80)

    print(
        "OBSERVED BEST VALUES"
    )

    print("=" * 80)


    print(
        "Lowest Test MAE:",
        best_mae[
            "model"
        ],
        f"({best_mae['test_mae']:.6f})",
    )


    print(
        "Lowest Test RMSE:",
        best_rmse[
            "model"
        ],
        f"({best_rmse['test_rmse']:.6f})",
    )


    print(
        "Lowest Wave Residual:",
        best_residual[
            "model"
        ],
        f"({best_residual['test_wave_residual']:.6f})",
    )


    print(
        "Lowest Mean Latency:",
        fastest[
            "model"
        ],
        f"({fastest['latency_mean_sec']:.6f} sec)",
    )


    print()
    print("=" * 80)

    print(
        "OUTPUT FILES"
    )

    print("=" * 80)


    print(
        COMPARISON_ROOT
        / "baseline_comparison_full.csv"
    )

    print(
        COMPARISON_ROOT
        / "baseline_comparison_full.json"
    )

    print(
        COMPARISON_ROOT
        / "reviewer_baseline_table.csv"
    )

    print(
        COMPARISON_ROOT
        / "comparison_summary.txt"
    )


    print()
    print(
        "PASS: finalized reviewer baseline results "
        "have been consolidated without recomputation."
    )

    print(
        "No submitted-manuscript numerical value "
        "has been modified."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()