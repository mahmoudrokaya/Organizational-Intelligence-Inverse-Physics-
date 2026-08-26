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
        [
            p
            for p in ORG_ROOT.glob("org_evolution_*")
            if p.is_dir()
        ]
    )

    if not runs:
        raise FileNotFoundError(
            f"No organizational-evolution runs found in:\n{ORG_ROOT}"
        )

    return runs[-1]


def read_csv(path: Path) -> list[dict]:
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


# ============================================================
# CONVERSION
# ============================================================

def to_int(value):
    return int(float(value))


def to_float(value):
    return float(value)


# ============================================================
# AGENT-EPOCH AGGREGATION
# ============================================================

def summarize_agent_specialization(rows):

    grouped = defaultdict(list)

    for row in rows:
        key = (
            to_int(row["epoch"]),
            row["split"],
            to_int(row["agent_id"]),
        )

        grouped[key].append(row)

    summary = []

    for (
        epoch,
        split,
        agent_id,
    ), group in sorted(grouped.items()):

        gate_entropies = np.array(
            [
                to_float(r["gate_entropy"])
                for r in group
            ],
            dtype=float,
        )

        normalized_entropies = np.array(
            [
                to_float(
                    r["gate_entropy_normalized"]
                )
                for r in group
            ],
            dtype=float,
        )

        max_probs = np.array(
            [
                to_float(
                    r["dominant_expert_probability"]
                )
                for r in group
            ],
            dtype=float,
        )

        effective_experts = np.array(
            [
                to_float(
                    r["effective_number_of_experts"]
                )
                for r in group
            ],
            dtype=float,
        )

        influences = np.array(
            [
                to_float(
                    r["influence_weight"]
                )
                for r in group
            ],
            dtype=float,
        )

        dominant = np.array(
            [
                to_int(
                    r["dominant_expert"]
                )
                for r in group
            ],
            dtype=int,
        )

        values, counts = np.unique(
            dominant,
            return_counts=True,
        )

        modal_idx = np.argmax(counts)

        modal_expert = int(
            values[modal_idx]
        )

        modal_fraction = float(
            counts[modal_idx]
            / len(dominant)
        )

        summary.append(
            {
                "epoch": epoch,
                "split": split,
                "agent_id": agent_id,

                "n_observations":
                    len(group),

                "mean_gate_entropy":
                    float(
                        np.mean(
                            gate_entropies
                        )
                    ),

                "std_gate_entropy":
                    float(
                        np.std(
                            gate_entropies,
                            ddof=1,
                        )
                    )
                    if len(group) > 1
                    else 0.0,

                "mean_normalized_gate_entropy":
                    float(
                        np.mean(
                            normalized_entropies
                        )
                    ),

                "mean_dominant_probability":
                    float(
                        np.mean(
                            max_probs
                        )
                    ),

                "mean_effective_experts":
                    float(
                        np.mean(
                            effective_experts
                        )
                    ),

                "mean_influence_weight":
                    float(
                        np.mean(
                            influences
                        )
                    ),

                "modal_expert":
                    modal_expert,

                "modal_expert_fraction":
                    modal_fraction,

                "unique_dominant_experts":
                    int(
                        len(
                            np.unique(
                                dominant
                            )
                        )
                    ),
            }
        )

    return summary


# ============================================================
# SYSTEM-EPOCH SUMMARY
# ============================================================

def summarize_system_specialization(agent_summary):

    grouped = defaultdict(list)

    for row in agent_summary:
        key = (
            row["epoch"],
            row["split"],
        )

        grouped[key].append(row)

    results = []

    for (
        epoch,
        split,
    ), group in sorted(grouped.items()):

        H = np.array(
            [
                r[
                    "mean_normalized_gate_entropy"
                ]
                for r in group
            ],
            dtype=float,
        )

        P = np.array(
            [
                r[
                    "mean_dominant_probability"
                ]
                for r in group
            ],
            dtype=float,
        )

        E = np.array(
            [
                r[
                    "mean_effective_experts"
                ]
                for r in group
            ],
            dtype=float,
        )

        modal_fraction = np.array(
            [
                r[
                    "modal_expert_fraction"
                ]
                for r in group
            ],
            dtype=float,
        )

        results.append(
            {
                "epoch": epoch,
                "split": split,

                "agents":
                    len(group),

                "mean_normalized_gate_entropy":
                    float(
                        np.mean(H)
                    ),

                "std_normalized_gate_entropy":
                    float(
                        np.std(
                            H,
                            ddof=1,
                        )
                    )
                    if len(H) > 1
                    else 0.0,

                "mean_dominant_probability":
                    float(
                        np.mean(P)
                    ),

                "mean_effective_experts":
                    float(
                        np.mean(E)
                    ),

                "mean_modal_expert_fraction":
                    float(
                        np.mean(
                            modal_fraction
                        )
                    ),

                "min_normalized_gate_entropy":
                    float(
                        np.min(H)
                    ),

                "max_normalized_gate_entropy":
                    float(
                        np.max(H)
                    ),
            }
        )

    return results


# ============================================================
# EXPERT UTILIZATION
# ============================================================

def summarize_expert_utilization(rows):

    grouped = defaultdict(list)

    for row in rows:
        key = (
            to_int(row["epoch"]),
            row["split"],
        )

        grouped[key].append(row)

    results = []

    for (
        epoch,
        split,
    ), group in sorted(grouped.items()):

        dominant = np.array(
            [
                to_int(
                    r["dominant_expert"]
                )
                for r in group
            ],
            dtype=int,
        )

        K = max(dominant) + 1

        counts = np.bincount(
            dominant,
            minlength=K,
        )

        total = int(
            np.sum(counts)
        )

        row = {
            "epoch": epoch,
            "split": split,
            "total_agent_observations": total,
        }

        fractions = []

        for k in range(K):

            fraction = (
                float(counts[k] / total)
                if total > 0
                else 0.0
            )

            row[
                f"expert_{k}_count"
            ] = int(
                counts[k]
            )

            row[
                f"expert_{k}_fraction"
            ] = fraction

            fractions.append(
                fraction
            )

        # Utilization entropy across dominant expert assignments.
        fractions = np.array(
            fractions,
            dtype=float,
        )

        positive = fractions[
            fractions > 0
        ]

        utilization_entropy = (
            -np.sum(
                positive
                * np.log(positive)
            )
        )

        max_entropy = (
            np.log(len(fractions))
            if len(fractions) > 1
            else 1.0
        )

        row[
            "dominant_expert_utilization_entropy"
        ] = float(
            utilization_entropy
        )

        row[
            "dominant_expert_utilization_entropy_normalized"
        ] = float(
            utilization_entropy
            / max_entropy
        )

        results.append(row)

    return results


# ============================================================
# TEMPORAL CHANGE
# ============================================================

def calculate_epoch_changes(
    agent_summary,
    split_name="validation",
):

    rows = [
        r
        for r in agent_summary
        if r["split"] == split_name
    ]

    by_agent = defaultdict(dict)

    for row in rows:
        by_agent[
            row["agent_id"]
        ][
            row["epoch"]
        ] = row

    changes = []

    for agent_id, epochs in sorted(
        by_agent.items()
    ):

        ordered_epochs = sorted(
            epochs
        )

        for previous, current in zip(
            ordered_epochs[:-1],
            ordered_epochs[1:],
        ):

            a = epochs[previous]
            b = epochs[current]

            changes.append(
                {
                    "agent_id":
                        agent_id,

                    "split":
                        split_name,

                    "epoch_from":
                        previous,

                    "epoch_to":
                        current,

                    "delta_normalized_gate_entropy":
                        b[
                            "mean_normalized_gate_entropy"
                        ]
                        - a[
                            "mean_normalized_gate_entropy"
                        ],

                    "delta_dominant_probability":
                        b[
                            "mean_dominant_probability"
                        ]
                        - a[
                            "mean_dominant_probability"
                        ],

                    "delta_effective_experts":
                        b[
                            "mean_effective_experts"
                        ]
                        - a[
                            "mean_effective_experts"
                        ],

                    "dominant_expert_changed":
                        int(
                            a["modal_expert"]
                            != b["modal_expert"]
                        ),

                    "modal_expert_from":
                        a["modal_expert"],

                    "modal_expert_to":
                        b["modal_expert"],
                }
            )

    return changes


# ============================================================
# INITIAL VS FINAL ANALYSIS
# ============================================================

def initial_final_comparison(
    agent_summary,
    split_name="validation",
):

    rows = [
        r
        for r in agent_summary
        if r["split"] == split_name
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

    results = []

    for agent_id in sorted(
        set(first)
        & set(final)
    ):

        a = first[agent_id]
        b = final[agent_id]

        results.append(
            {
                "agent_id":
                    agent_id,

                "initial_epoch":
                    first_epoch,

                "final_epoch":
                    final_epoch,

                "initial_normalized_gate_entropy":
                    a[
                        "mean_normalized_gate_entropy"
                    ],

                "final_normalized_gate_entropy":
                    b[
                        "mean_normalized_gate_entropy"
                    ],

                "delta_normalized_gate_entropy":
                    b[
                        "mean_normalized_gate_entropy"
                    ]
                    - a[
                        "mean_normalized_gate_entropy"
                    ],

                "initial_dominant_probability":
                    a[
                        "mean_dominant_probability"
                    ],

                "final_dominant_probability":
                    b[
                        "mean_dominant_probability"
                    ],

                "delta_dominant_probability":
                    b[
                        "mean_dominant_probability"
                    ]
                    - a[
                        "mean_dominant_probability"
                    ],

                "initial_effective_experts":
                    a[
                        "mean_effective_experts"
                    ],

                "final_effective_experts":
                    b[
                        "mean_effective_experts"
                    ],

                "delta_effective_experts":
                    b[
                        "mean_effective_experts"
                    ]
                    - a[
                        "mean_effective_experts"
                    ],

                "initial_modal_expert":
                    a[
                        "modal_expert"
                    ],

                "final_modal_expert":
                    b[
                        "modal_expert"
                    ],

                "modal_expert_changed":
                    int(
                        a[
                            "modal_expert"
                        ]
                        != b[
                            "modal_expert"
                        ]
                    ),
            }
        )

    return results


# ============================================================
# INTERPRETATION
# ============================================================

def build_findings(
    system_summary,
    initial_final,
):

    findings = []

    validation = [
        r
        for r in system_summary
        if r["split"]
        == "validation"
    ]

    validation = sorted(
        validation,
        key=lambda x:
            x["epoch"],
    )

    if validation:

        initial = validation[0]
        final = validation[-1]

        initial_entropy = (
            initial[
                "mean_normalized_gate_entropy"
            ]
        )

        final_entropy = (
            final[
                "mean_normalized_gate_entropy"
            ]
        )

        delta_entropy = (
            final_entropy
            - initial_entropy
        )

        initial_prob = (
            initial[
                "mean_dominant_probability"
            ]
        )

        final_prob = (
            final[
                "mean_dominant_probability"
            ]
        )

        # Descriptive only. No arbitrary statistical
        # significance claim is made here.
        if final_entropy < initial_entropy:

            entropy_direction = (
                "DECREASED"
            )

        elif final_entropy > initial_entropy:

            entropy_direction = (
                "INCREASED"
            )

        else:

            entropy_direction = (
                "UNCHANGED"
            )

        findings.append(
            {
                "issue":
                    "Validation gate entropy from initial to final logged epoch",

                "status":
                    entropy_direction,

                "evidence":
                    (
                        f"normalized entropy: "
                        f"{initial_entropy:.8f} -> "
                        f"{final_entropy:.8f}; "
                        f"delta={delta_entropy:+.8f}"
                    ),
            }
        )

        findings.append(
            {
                "issue":
                    "Validation dominant-expert probability",

                "status":
                    "OBSERVED",

                "evidence":
                    (
                        f"{initial_prob:.8f} -> "
                        f"{final_prob:.8f}; "
                        f"delta="
                        f"{final_prob-initial_prob:+.8f}"
                    ),
            }
        )

    if initial_final:

        changed = sum(
            r["modal_expert_changed"]
            for r in initial_final
        )

        findings.append(
            {
                "issue":
                    "Agents whose modal expert changed between initial and final epoch",

                "status":
                    "OBSERVED",

                "evidence":
                    (
                        f"{changed} of "
                        f"{len(initial_final)} agents"
                    ),
            }
        )

        delta_h = np.array(
            [
                r[
                    "delta_normalized_gate_entropy"
                ]
                for r in initial_final
            ],
            dtype=float,
        )

        findings.append(
            {
                "issue":
                    "Mean agent-level normalized entropy change",

                "status":
                    "OBSERVED",

                "evidence":
                    (
                        f"{np.mean(delta_h):+.8f}"
                    ),
            }
        )

    return findings


# ============================================================
# MAIN
# ============================================================

def main():

    latest_run = find_latest_run()

    source_file = (
        latest_run
        / "tables"
        / "agent_evolution.csv"
    )

    if not source_file.exists():
        raise FileNotFoundError(
            source_file
        )

    out_dir = ensure_dir(
        OUT_ROOT
        / latest_run.name
    )

    print()
    print("=" * 80)
    print("MICRO-EXPERT SPECIALIZATION ANALYSIS")
    print("=" * 80)

    print(
        "Source run:"
    )

    print(
        latest_run
    )

    rows = read_csv(
        source_file
    )

    print()
    print(
        f"Agent-level records loaded: "
        f"{len(rows):,}"
    )

    # --------------------------------------------------------
    # Analyses
    # --------------------------------------------------------

    agent_summary = (
        summarize_agent_specialization(
            rows
        )
    )

    system_summary = (
        summarize_system_specialization(
            agent_summary
        )
    )

    utilization = (
        summarize_expert_utilization(
            rows
        )
    )

    epoch_changes = (
        calculate_epoch_changes(
            agent_summary,
            split_name="validation",
        )
    )

    initial_final = (
        initial_final_comparison(
            agent_summary,
            split_name="validation",
        )
    )

    findings = build_findings(
        system_summary,
        initial_final,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_csv(
        out_dir
        / "agent_specialization_summary.csv",
        agent_summary,
    )

    save_csv(
        out_dir
        / "system_specialization_summary.csv",
        system_summary,
    )

    save_csv(
        out_dir
        / "expert_utilization.csv",
        utilization,
    )

    save_csv(
        out_dir
        / "epoch_to_epoch_specialization_changes.csv",
        epoch_changes,
    )

    save_csv(
        out_dir
        / "initial_vs_final_specialization.csv",
        initial_final,
    )

    save_csv(
        out_dir
        / "findings.csv",
        findings,
    )

    summary = {
        "source_run":
            str(latest_run),

        "source_agent_records":
            len(rows),

        "analysis_type":
            "reviewer_requested_micro_expert_specialization",

        "new_experimental_analysis":
            True,

        "replaces_current_manuscript_numbers":
            False,

        "findings":
            findings,
    }

    save_json(
        out_dir
        / "analysis_summary.json",
        summary,
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
    print("=" * 80)
    print("ANALYSIS COMPLETED")
    print("=" * 80)

    print(
        "Output directory:"
    )

    print(
        out_dir
    )

    print()
    print(
        "Important files:"
    )

    print(
        out_dir
        / "system_specialization_summary.csv"
    )

    print(
        out_dir
        / "expert_utilization.csv"
    )

    print(
        out_dir
        / "initial_vs_final_specialization.csv"
    )

    print(
        out_dir
        / "findings.csv"
    )


if __name__ == "__main__":
    main()