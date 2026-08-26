from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\47\472\New-Papers\GIS\Codes\New_Branch")

ORG_ROOT = (
    ROOT
    / "outputs"
    / "organizational_evolution"
)

OUT_ROOT = (
    ROOT
    / "outputs"
    / "organizational_evolution_analysis"
)


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> list[dict]:
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, obj) -> None:
    ensure_dir(path.parent)

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def find_latest_run() -> Path:
    runs = sorted(
        p
        for p in ORG_ROOT.glob(
            "org_evolution_*"
        )
        if p.is_dir()
    )

    if not runs:
        raise FileNotFoundError(
            f"No runs found in {ORG_ROOT}"
        )

    return runs[-1]


def to_int(value):
    return int(float(value))


def to_float(value):
    return float(value)


# ============================================================
# AGENT-EPOCH INFLUENCE SUMMARY
# ============================================================

def summarize_agent_influence(rows):

    grouped = defaultdict(list)

    for row in rows:
        key = (
            to_int(row["epoch"]),
            row["split"],
            to_int(row["agent_id"]),
        )
        grouped[key].append(row)

    result = []

    for (
        epoch,
        split,
        agent_id,
    ), group in sorted(grouped.items()):

        influences = np.array(
            [
                to_float(
                    r["influence_weight"]
                )
                for r in group
            ],
            dtype=float,
        )

        sensor = np.array(
            [
                to_float(
                    r[
                        "sensor_consistency_score"
                    ]
                )
                for r in group
            ],
            dtype=float,
        )

        physics = np.array(
            [
                to_float(
                    r[
                        "local_physics_score"
                    ]
                )
                for r in group
            ],
            dtype=float,
        )

        composite = np.array(
            [
                to_float(
                    r[
                        "composite_influence_score"
                    ]
                )
                for r in group
            ],
            dtype=float,
        )

        result.append(
            {
                "epoch":
                    epoch,

                "split":
                    split,

                "agent_id":
                    agent_id,

                "n_observations":
                    len(group),

                "mean_influence":
                    float(
                        np.mean(
                            influences
                        )
                    ),

                "std_influence":
                    float(
                        np.std(
                            influences,
                            ddof=1,
                        )
                    )
                    if len(influences) > 1
                    else 0.0,

                "min_influence":
                    float(
                        np.min(
                            influences
                        )
                    ),

                "max_influence":
                    float(
                        np.max(
                            influences
                        )
                    ),

                "mean_sensor_score":
                    float(
                        np.mean(
                            sensor
                        )
                    ),

                "mean_physics_score":
                    float(
                        np.mean(
                            physics
                        )
                    ),

                "mean_composite_score":
                    float(
                        np.mean(
                            composite
                        )
                    ),
            }
        )

    return result


# ============================================================
# SYSTEM-LEVEL INFLUENCE DISTRIBUTION
# ============================================================

def summarize_system_distribution(
    rows,
):

    grouped = defaultdict(list)

    for row in rows:
        key = (
            to_int(row["epoch"]),
            row["split"],
            to_int(row["batch_index"]),
        )

        grouped[key].append(row)

    result = []

    for (
        epoch,
        split,
        batch_index,
    ), group in sorted(grouped.items()):

        # One record per agent for this sample.
        weights = np.array(
            [
                to_float(
                    r["influence_weight"]
                )
                for r in group
            ],
            dtype=float,
        )

        # Numerical normalization for safety.
        total = np.sum(weights)

        if total > 0:
            weights = (
                weights
                / total
            )

        positive = weights[
            weights > 0
        ]

        entropy = float(
            -np.sum(
                positive
                * np.log(
                    positive
                )
            )
        )

        max_entropy = (
            np.log(
                len(weights)
            )
            if len(weights) > 1
            else 1.0
        )

        normalized_entropy = (
            entropy
            / max_entropy
        )

        effective_agents = float(
            np.exp(
                entropy
            )
        )

        concentration_hhi = float(
            np.sum(
                weights ** 2
            )
        )

        max_weight = float(
            np.max(
                weights
            )
        )

        min_weight = float(
            np.min(
                weights
            )
        )

        range_weight = (
            max_weight
            - min_weight
        )

        coefficient_variation = (
            float(
                np.std(
                    weights
                )
                / np.mean(
                    weights
                )
            )
            if np.mean(weights) > 0
            else 0.0
        )

        dominant_agent = int(
            group[
                int(
                    np.argmax(
                        weights
                    )
                )
            ][
                "agent_id"
            ]
        )

        result.append(
            {
                "epoch":
                    epoch,

                "split":
                    split,

                "batch_index":
                    batch_index,

                "n_agents":
                    len(weights),

                "influence_entropy":
                    entropy,

                "normalized_influence_entropy":
                    normalized_entropy,

                "effective_number_of_agents":
                    effective_agents,

                "hhi_concentration":
                    concentration_hhi,

                "max_influence":
                    max_weight,

                "min_influence":
                    min_weight,

                "influence_range":
                    range_weight,

                "coefficient_of_variation":
                    coefficient_variation,

                "dominant_agent":
                    dominant_agent,
            }
        )

    return result


# ============================================================
# EPOCH-LEVEL SUMMARY
# ============================================================

def summarize_by_epoch(
    system_rows,
):

    grouped = defaultdict(list)

    for row in system_rows:
        key = (
            row["epoch"],
            row["split"],
        )

        grouped[key].append(row)

    result = []

    for (
        epoch,
        split,
    ), group in sorted(grouped.items()):

        def arr(key):
            return np.array(
                [
                    float(
                        r[key]
                    )
                    for r in group
                ],
                dtype=float,
            )

        entropy = arr(
            "normalized_influence_entropy"
        )

        effective = arr(
            "effective_number_of_agents"
        )

        hhi = arr(
            "hhi_concentration"
        )

        max_inf = arr(
            "max_influence"
        )

        ranges = arr(
            "influence_range"
        )

        cv = arr(
            "coefficient_of_variation"
        )

        result.append(
            {
                "epoch":
                    epoch,

                "split":
                    split,

                "samples":
                    len(group),

                "mean_normalized_influence_entropy":
                    float(
                        np.mean(
                            entropy
                        )
                    ),

                "std_normalized_influence_entropy":
                    float(
                        np.std(
                            entropy,
                            ddof=1,
                        )
                    )
                    if len(group) > 1
                    else 0.0,

                "mean_effective_number_of_agents":
                    float(
                        np.mean(
                            effective
                        )
                    ),

                "mean_hhi_concentration":
                    float(
                        np.mean(
                            hhi
                        )
                    ),

                "mean_max_influence":
                    float(
                        np.mean(
                            max_inf
                        )
                    ),

                "mean_influence_range":
                    float(
                        np.mean(
                            ranges
                        )
                    ),

                "mean_coefficient_of_variation":
                    float(
                        np.mean(
                            cv
                        )
                    ),
            }
        )

    return result


# ============================================================
# DOMINANT-AGENT TRANSITIONS
# ============================================================

def analyze_dominant_agent_changes(
    system_rows,
    split_name="validation",
):

    rows = [
        r
        for r in system_rows
        if r["split"]
        == split_name
    ]

    by_batch = defaultdict(dict)

    for row in rows:
        by_batch[
            row["batch_index"]
        ][
            row["epoch"]
        ] = row

    transitions = []

    for batch_index, epochs in sorted(
        by_batch.items()
    ):

        ordered = sorted(
            epochs
        )

        for a, b in zip(
            ordered[:-1],
            ordered[1:],
        ):

            ra = epochs[a]
            rb = epochs[b]

            transitions.append(
                {
                    "batch_index":
                        batch_index,

                    "epoch_from":
                        a,

                    "epoch_to":
                        b,

                    "dominant_agent_from":
                        ra[
                            "dominant_agent"
                        ],

                    "dominant_agent_to":
                        rb[
                            "dominant_agent"
                        ],

                    "dominant_agent_changed":
                        int(
                            ra[
                                "dominant_agent"
                            ]
                            != rb[
                                "dominant_agent"
                            ]
                        ),

                    "delta_normalized_entropy":
                        rb[
                            "normalized_influence_entropy"
                        ]
                        - ra[
                            "normalized_influence_entropy"
                        ],

                    "delta_max_influence":
                        rb[
                            "max_influence"
                        ]
                        - ra[
                            "max_influence"
                        ],

                    "delta_hhi":
                        rb[
                            "hhi_concentration"
                        ]
                        - ra[
                            "hhi_concentration"
                        ],
                }
            )

    return transitions


# ============================================================
# INITIAL VS FINAL AGENT INFLUENCE
# ============================================================

def compare_initial_final_agents(
    agent_summary,
    split_name="validation",
):

    rows = [
        r
        for r in agent_summary
        if r["split"]
        == split_name
    ]

    epochs = sorted(
        {
            r["epoch"]
            for r in rows
        }
    )

    if len(epochs) < 2:
        return []

    first_epoch = epochs[0]
    final_epoch = epochs[-1]

    first = {
        r["agent_id"]: r
        for r in rows
        if r["epoch"]
        == first_epoch
    }

    final = {
        r["agent_id"]: r
        for r in rows
        if r["epoch"]
        == final_epoch
    }

    result = []

    for agent_id in sorted(
        set(first)
        & set(final)
    ):

        a = first[
            agent_id
        ]

        b = final[
            agent_id
        ]

        result.append(
            {
                "agent_id":
                    agent_id,

                "initial_epoch":
                    first_epoch,

                "final_epoch":
                    final_epoch,

                "initial_mean_influence":
                    a[
                        "mean_influence"
                    ],

                "final_mean_influence":
                    b[
                        "mean_influence"
                    ],

                "delta_mean_influence":
                    b[
                        "mean_influence"
                    ]
                    - a[
                        "mean_influence"
                    ],

                "initial_sensor_score":
                    a[
                        "mean_sensor_score"
                    ],

                "final_sensor_score":
                    b[
                        "mean_sensor_score"
                    ],

                "initial_physics_score":
                    a[
                        "mean_physics_score"
                    ],

                "final_physics_score":
                    b[
                        "mean_physics_score"
                    ],

                "initial_composite_score":
                    a[
                        "mean_composite_score"
                    ],

                "final_composite_score":
                    b[
                        "mean_composite_score"
                    ],
            }
        )

    return result


# ============================================================
# RELATIONSHIP BETWEEN RELIABILITY AND INFLUENCE
# ============================================================

def analyze_score_influence_relationship(
    rows,
    split_name="validation",
):

    filtered = [
        r
        for r in rows
        if r["split"]
        == split_name
    ]

    if not filtered:
        return {}

    influence = np.array(
        [
            to_float(
                r[
                    "influence_weight"
                ]
            )
            for r in filtered
        ],
        dtype=float,
    )

    sensor = np.array(
        [
            to_float(
                r[
                    "sensor_consistency_score"
                ]
            )
            for r in filtered
        ],
        dtype=float,
    )

    physics = np.array(
        [
            to_float(
                r[
                    "local_physics_score"
                ]
            )
            for r in filtered
        ],
        dtype=float,
    )

    composite = np.array(
        [
            to_float(
                r[
                    "composite_influence_score"
                ]
            )
            for r in filtered
        ],
        dtype=float,
    )

    def corr(a, b):
        if (
            np.std(a) < 1e-12
            or np.std(b) < 1e-12
        ):
            return None

        return float(
            np.corrcoef(
                a,
                b,
            )[0, 1]
        )

    return {
        "n_records":
            len(filtered),

        "correlation_influence_sensor_score":
            corr(
                influence,
                sensor,
            ),

        "correlation_influence_physics_score":
            corr(
                influence,
                physics,
            ),

        "correlation_influence_composite_score":
            corr(
                influence,
                composite,
            ),

        "interpretation_note":
            (
                "Scores are lower-is-better, while influence is "
                "higher-is-better. Therefore a negative correlation "
                "with composite score is expected by construction."
            ),
    }


# ============================================================
# FINDINGS
# ============================================================

def build_findings(
    epoch_summary,
    transitions,
    initial_final,
):

    findings = []

    val_rows = sorted(
        [
            r
            for r in epoch_summary
            if r["split"]
            == "validation"
        ],
        key=lambda x:
            x["epoch"],
    )

    if val_rows:

        initial = val_rows[0]
        final = val_rows[-1]

        findings.append(
            {
                "issue":
                    "Normalized influence entropy",

                "status":
                    "OBSERVED",

                "evidence":
                    (
                        f"{initial['mean_normalized_influence_entropy']:.8f}"
                        f" -> "
                        f"{final['mean_normalized_influence_entropy']:.8f}; "
                        f"delta="
                        f"{final['mean_normalized_influence_entropy'] - initial['mean_normalized_influence_entropy']:+.8f}"
                    ),
            }
        )

        findings.append(
            {
                "issue":
                    "Maximum agent influence",

                "status":
                    "OBSERVED",

                "evidence":
                    (
                        f"{initial['mean_max_influence']:.8f}"
                        f" -> "
                        f"{final['mean_max_influence']:.8f}; "
                        f"delta="
                        f"{final['mean_max_influence'] - initial['mean_max_influence']:+.8f}"
                    ),
            }
        )

        findings.append(
            {
                "issue":
                    "Effective number of influential agents",

                "status":
                    "OBSERVED",

                "evidence":
                    (
                        f"{initial['mean_effective_number_of_agents']:.8f}"
                        f" -> "
                        f"{final['mean_effective_number_of_agents']:.8f}"
                    ),
            }
        )

    if transitions:

        changed = sum(
            r[
                "dominant_agent_changed"
            ]
            for r in transitions
        )

        findings.append(
            {
                "issue":
                    "Dominant-agent transitions",

                "status":
                    "OBSERVED",

                "evidence":
                    (
                        f"{changed} of "
                        f"{len(transitions)} "
                        f"epoch-to-epoch validation transitions "
                        f"changed the dominant agent."
                    ),
            }
        )

    if initial_final:

        deltas = np.array(
            [
                r[
                    "delta_mean_influence"
                ]
                for r in initial_final
            ],
            dtype=float,
        )

        findings.append(
            {
                "issue":
                    "Agent-level redistribution magnitude",

                "status":
                    "OBSERVED",

                "evidence":
                    (
                        f"mean absolute change="
                        f"{np.mean(np.abs(deltas)):.8f}; "
                        f"maximum absolute change="
                        f"{np.max(np.abs(deltas)):.8f}"
                    ),
            }
        )

    return findings


# ============================================================
# MAIN
# ============================================================

def main():

    latest_run = (
        find_latest_run()
    )

    source = (
        latest_run
        / "tables"
        / "agent_evolution.csv"
    )

    if not source.exists():
        raise FileNotFoundError(
            source
        )

    out_dir = ensure_dir(
        OUT_ROOT
        / latest_run.name
        / "influence_redistribution"
    )

    print()
    print("=" * 80)
    print(
        "INFLUENCE REDISTRIBUTION ANALYSIS"
    )
    print("=" * 80)

    print(
        "Source:"
    )
    print(
        source
    )

    rows = read_csv(
        source
    )

    print()
    print(
        "Agent-level records:",
        len(rows),
    )

    # --------------------------------------------------------
    # Analysis
    # --------------------------------------------------------

    agent_summary = (
        summarize_agent_influence(
            rows
        )
    )

    system_distribution = (
        summarize_system_distribution(
            rows
        )
    )

    epoch_summary = (
        summarize_by_epoch(
            system_distribution
        )
    )

    transitions = (
        analyze_dominant_agent_changes(
            system_distribution,
            split_name="validation",
        )
    )

    initial_final = (
        compare_initial_final_agents(
            agent_summary,
            split_name="validation",
        )
    )

    relationships = (
        analyze_score_influence_relationship(
            rows,
            split_name="validation",
        )
    )

    findings = (
        build_findings(
            epoch_summary,
            transitions,
            initial_final,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_csv(
        out_dir
        / "agent_influence_summary.csv",
        agent_summary,
    )

    save_csv(
        out_dir
        / "system_influence_distribution.csv",
        system_distribution,
    )

    save_csv(
        out_dir
        / "epoch_influence_summary.csv",
        epoch_summary,
    )

    save_csv(
        out_dir
        / "dominant_agent_transitions.csv",
        transitions,
    )

    save_csv(
        out_dir
        / "initial_vs_final_agent_influence.csv",
        initial_final,
    )

    save_csv(
        out_dir
        / "findings.csv",
        findings,
    )

    save_json(
        out_dir
        / "score_influence_relationship.json",
        relationships,
    )

    save_json(
        out_dir
        / "analysis_summary.json",
        {
            "source_run":
                str(
                    latest_run
                ),

            "analysis":
                "reviewer_requested_influence_redistribution",

            "new_experimental_analysis":
                True,

            "replaces_existing_manuscript_numbers":
                False,

            "findings":
                findings,

            "score_influence_relationship":
                relationships,
        },
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINDINGS")
    print("=" * 80)

    for i, row in enumerate(
        findings,
        start=1,
    ):

        print()
        print(
            f"{i}. "
            f"{row['issue']}"
        )

        print(
            "   Status:",
            row["status"],
        )

        print(
            "   Evidence:",
            row["evidence"],
        )

    print()
    print(
        "Score/influence relationships:"
    )

    for key, value in (
        relationships.items()
    ):

        print(
            f"  {key}: {value}"
        )

    print()
    print("=" * 80)
    print(
        "INFLUENCE ANALYSIS COMPLETED"
    )
    print("=" * 80)

    print(
        "Output directory:"
    )
    print(
        out_dir
    )


if __name__ == "__main__":
    main()