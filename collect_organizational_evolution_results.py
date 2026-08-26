from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


# ============================================================
# PATHS
# ============================================================

THIS_FILE = Path(__file__).resolve()
EXPERIMENTS_DIR = THIS_FILE.parent
NEW_ROOT = EXPERIMENTS_DIR.parent
OUTPUTS_ROOT = NEW_ROOT / "outputs"

ORG_ROOT = OUTPUTS_ROOT / "organizational_evolution"
ORG_ANALYSIS_ROOT = OUTPUTS_ROOT / "organizational_evolution_analysis"

OUTPUT_TXT = EXPERIMENTS_DIR / "organizational_evolution_R.txt"
OUTPUT_JSON = EXPERIMENTS_DIR / "organizational_evolution_R.json"
OUTPUT_CSV = EXPERIMENTS_DIR / "organizational_evolution_agent_summary.csv"


# ============================================================
# BASIC HELPERS
# ============================================================

def is_number(value: Any) -> bool:
    try:
        x = float(value)
        return math.isfinite(x)
    except (TypeError, ValueError):
        return False


def safe_float(value: Any) -> Optional[float]:
    if not is_number(value):
        return None
    return float(value)


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"

    if isinstance(value, (int,)):
        return str(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            return "N/A"
        return f"{value:.{digits}f}"

    return str(value)


def pct_change(initial: Optional[float], final: Optional[float]) -> Optional[float]:
    if initial is None or final is None:
        return None

    if abs(initial) < 1e-15:
        return None

    return 100.0 * (final - initial) / abs(initial)


def absolute_change(initial: Optional[float], final: Optional[float]) -> Optional[float]:
    if initial is None or final is None:
        return None
    return final - initial


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def latest_directory(root: Path, prefix: str = "") -> Optional[Path]:
    if not root.exists():
        return None

    dirs = [
        p for p in root.iterdir()
        if p.is_dir() and (not prefix or p.name.startswith(prefix))
    ]

    if not dirs:
        return None

    return max(dirs, key=lambda p: p.stat().st_mtime)


def find_matching_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    contains: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> Optional[str]:

    normalized = {
        col: normalize_name(col)
        for col in df.columns
    }

    # Exact normalized candidate match.
    for candidate in candidates:
        candidate_n = normalize_name(candidate)

        for col, col_n in normalized.items():
            if col_n == candidate_n:
                return col

    # All required substrings.
    if contains:
        for col, col_n in normalized.items():

            if all(
                normalize_name(term) in col_n
                for term in contains
            ):
                if not any(
                    normalize_name(term) in col_n
                    for term in exclude
                ):
                    return col

    return None


def numeric_columns(df: pd.DataFrame) -> List[str]:
    result = []

    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")

        if converted.notna().sum() > 0:
            result.append(col)

    return result


def first_last_numeric(
    df: pd.DataFrame,
    column: str,
) -> Tuple[Optional[float], Optional[float]]:

    series = pd.to_numeric(df[column], errors="coerce").dropna()

    if len(series) == 0:
        return None, None

    return float(series.iloc[0]), float(series.iloc[-1])


def min_max_numeric(
    df: pd.DataFrame,
    column: str,
) -> Tuple[Optional[float], Optional[float]]:

    series = pd.to_numeric(df[column], errors="coerce").dropna()

    if len(series) == 0:
        return None, None

    return float(series.min()), float(series.max())


def add_schema(lines: List[str], label: str, path: Path, df: pd.DataFrame) -> None:
    lines.append("")
    lines.append(f"{label}")
    lines.append("-" * len(label))
    lines.append(f"Source: {path}")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Columns: {list(df.columns)}")


# ============================================================
# FIND THE COMPLETED RUN
# ============================================================

def find_run() -> Tuple[Path, Path]:

    org_run = latest_directory(
        ORG_ROOT,
        "org_evolution_",
    )

    if org_run is None:
        raise FileNotFoundError(
            f"No organizational-evolution run found under:\n{ORG_ROOT}"
        )

    # Prefer analysis directory with exactly the same run name.
    matching_analysis = (
        ORG_ANALYSIS_ROOT
        / org_run.name
    )

    if matching_analysis.exists():
        analysis_run = matching_analysis
    else:
        analysis_run = latest_directory(
            ORG_ANALYSIS_ROOT,
            "org_evolution_",
        )

    if analysis_run is None:
        raise FileNotFoundError(
            f"No organizational-evolution analysis found under:\n"
            f"{ORG_ANALYSIS_ROOT}"
        )

    return org_run, analysis_run


# ============================================================
# REQUIRED FILES
# ============================================================

def build_paths(
    org_run: Path,
    analysis_run: Path,
) -> Dict[str, Path]:

    return {
        "agent_evolution":
            org_run / "tables" / "agent_evolution.csv",

        "system_evolution":
            org_run / "tables" / "system_evolution.csv",

        "inter_agent_disagreement":
            org_run / "tables" / "inter_agent_disagreement.csv",

        "training_history":
            org_run / "tables" / "training_history.csv",

        "org_results":
            org_run / "results.json",

        "specialization_summary":
            analysis_run / "agent_specialization_summary.csv",

        "initial_vs_final_specialization":
            analysis_run / "initial_vs_final_specialization.csv",

        "system_specialization_summary":
            analysis_run / "system_specialization_summary.csv",

        "specialization_changes":
            analysis_run / "epoch_to_epoch_specialization_changes.csv",

        "expert_utilization":
            analysis_run / "expert_utilization.csv",

        "analysis_summary":
            analysis_run / "analysis_summary.json",

        "analysis_findings":
            analysis_run / "findings.csv",

        "agent_influence_summary":
            analysis_run
            / "influence_redistribution"
            / "agent_influence_summary.csv",

        "initial_vs_final_influence":
            analysis_run
            / "influence_redistribution"
            / "initial_vs_final_agent_influence.csv",

        "dominant_agent_transitions":
            analysis_run
            / "influence_redistribution"
            / "dominant_agent_transitions.csv",

        "epoch_influence_summary":
            analysis_run
            / "influence_redistribution"
            / "epoch_influence_summary.csv",

        "system_influence_distribution":
            analysis_run
            / "influence_redistribution"
            / "system_influence_distribution.csv",

        "score_influence_relationship":
            analysis_run
            / "influence_redistribution"
            / "score_influence_relationship.json",

        "influence_analysis_summary":
            analysis_run
            / "influence_redistribution"
            / "analysis_summary.json",

        "influence_findings":
            analysis_run
            / "influence_redistribution"
            / "findings.csv",
    }


# ============================================================
# SPECIALIZATION EXTRACTION
# ============================================================

def extract_specialization(
    paths: Dict[str, Path],
    lines: List[str],
) -> Dict[str, Any]:

    result: Dict[str, Any] = {}

    path = paths["initial_vs_final_specialization"]

    if not path.exists():
        lines.append("")
        lines.append(
            "WARNING: initial_vs_final_specialization.csv is missing."
        )
        return result

    df = read_csv(path)

    add_schema(
        lines,
        "SPECIALIZATION: INITIAL VS FINAL",
        path,
        df,
    )

    agent_col = find_matching_column(
        df,
        candidates=[
            "agent",
            "agent_id",
            "sacu",
            "sacu_id",
            "unit",
            "unit_id",
        ],
    )

    initial_col = find_matching_column(
        df,
        candidates=[
            "initial_specialization",
            "specialization_initial",
            "initial_entropy",
            "initial_concentration",
            "initial_value",
        ],
        contains=["initial"],
    )

    final_col = find_matching_column(
        df,
        candidates=[
            "final_specialization",
            "specialization_final",
            "final_entropy",
            "final_concentration",
            "final_value",
        ],
        contains=["final"],
    )

    change_col = find_matching_column(
        df,
        candidates=[
            "specialization_change",
            "delta_specialization",
            "change",
            "delta",
        ],
        contains=["change"],
    )

    lines.append("")
    lines.append(f"Detected agent column: {agent_col}")
    lines.append(f"Detected initial column: {initial_col}")
    lines.append(f"Detected final column: {final_col}")
    lines.append(f"Detected change column: {change_col}")

    if initial_col is None or final_col is None:
        lines.append(
            "Could not safely identify initial/final specialization "
            "columns. Raw rows are reported below."
        )

        lines.append(
            df.to_string(index=False)
        )

        result["raw"] = df.to_dict(orient="records")
        return result

    rows = []

    for idx, row in df.iterrows():

        agent = (
            row[agent_col]
            if agent_col is not None
            else idx
        )

        initial = safe_float(row[initial_col])
        final = safe_float(row[final_col])

        if change_col is not None:
            change = safe_float(row[change_col])
        else:
            change = absolute_change(initial, final)

        pct = pct_change(initial, final)

        rows.append({
            "agent": agent,
            "initial_specialization": initial,
            "final_specialization": final,
            "delta_specialization": change,
            "pct_specialization_change": pct,
        })

    valid_changes = [
        row for row in rows
        if row["delta_specialization"] is not None
    ]

    largest_increase = None
    largest_decrease = None
    largest_absolute = None

    if valid_changes:
        largest_increase = max(
            valid_changes,
            key=lambda r: r["delta_specialization"],
        )

        largest_decrease = min(
            valid_changes,
            key=lambda r: r["delta_specialization"],
        )

        largest_absolute = max(
            valid_changes,
            key=lambda r: abs(r["delta_specialization"]),
        )

    lines.append("")
    lines.append("Per-agent specialization change:")

    for row in rows:
        lines.append(
            f"  Agent {row['agent']}: "
            f"initial={fmt(row['initial_specialization'])}, "
            f"final={fmt(row['final_specialization'])}, "
            f"delta={fmt(row['delta_specialization'])}, "
            f"pct={fmt(row['pct_specialization_change'], 3)}%"
        )

    if largest_increase:
        lines.append("")
        lines.append(
            "Largest specialization increase: "
            f"Agent {largest_increase['agent']} "
            f"(delta={fmt(largest_increase['delta_specialization'])})"
        )

        lines.append(
            "Largest specialization decrease: "
            f"Agent {largest_decrease['agent']} "
            f"(delta={fmt(largest_decrease['delta_specialization'])})"
        )

        lines.append(
            "Largest absolute specialization change: "
            f"Agent {largest_absolute['agent']} "
            f"(delta={fmt(largest_absolute['delta_specialization'])})"
        )

    result["agents"] = rows
    result["largest_increase"] = largest_increase
    result["largest_decrease"] = largest_decrease
    result["largest_absolute_change"] = largest_absolute

    # --------------------------------------------------------
    # SYSTEM SPECIALIZATION SUMMARY
    # --------------------------------------------------------

    sys_path = paths["system_specialization_summary"]

    if sys_path.exists():

        sys_df = read_csv(sys_path)

        add_schema(
            lines,
            "SPECIALIZATION: SYSTEM SUMMARY",
            sys_path,
            sys_df,
        )

        lines.append("")
        lines.append(sys_df.to_string(index=False))

        result["system_summary"] = (
            sys_df.to_dict(orient="records")
        )

    # --------------------------------------------------------
    # TEMPORAL SPECIALIZATION
    # --------------------------------------------------------

    temporal_path = paths["specialization_changes"]

    if temporal_path.exists():

        temporal_df = read_csv(temporal_path)

        add_schema(
            lines,
            "SPECIALIZATION: EPOCH-TO-EPOCH CHANGE",
            temporal_path,
            temporal_df,
        )

        epoch_col = find_matching_column(
            temporal_df,
            candidates=[
                "epoch",
                "step",
                "iteration",
                "time",
            ],
        )

        numeric_cols = numeric_columns(temporal_df)

        lines.append("")
        lines.append(f"Detected epoch column: {epoch_col}")
        lines.append(f"Numeric columns: {numeric_cols}")

        temporal_summary = {}

        for col in numeric_cols:

            if col == epoch_col:
                continue

            initial, final = first_last_numeric(
                temporal_df,
                col,
            )

            min_value, max_value = min_max_numeric(
                temporal_df,
                col,
            )

            temporal_summary[col] = {
                "initial": initial,
                "final": final,
                "delta": absolute_change(initial, final),
                "min": min_value,
                "max": max_value,
            }

            lines.append(
                f"  {col}: "
                f"initial={fmt(initial)}, "
                f"final={fmt(final)}, "
                f"delta={fmt(absolute_change(initial, final))}, "
                f"min={fmt(min_value)}, "
                f"max={fmt(max_value)}"
            )

        result["temporal_summary"] = temporal_summary

    return result


# ============================================================
# INFLUENCE REDISTRIBUTION
# ============================================================

def extract_influence(
    paths: Dict[str, Path],
    lines: List[str],
) -> Dict[str, Any]:

    result: Dict[str, Any] = {}

    path = paths["initial_vs_final_influence"]

    if not path.exists():
        lines.append("")
        lines.append(
            "WARNING: initial_vs_final_agent_influence.csv is missing."
        )
        return result

    df = read_csv(path)

    add_schema(
        lines,
        "INFLUENCE: INITIAL VS FINAL",
        path,
        df,
    )

    agent_col = find_matching_column(
        df,
        candidates=[
            "agent",
            "agent_id",
            "sacu",
            "sacu_id",
            "unit",
            "unit_id",
        ],
    )

    initial_col = find_matching_column(
        df,
        candidates=[
            "initial_influence",
            "influence_initial",
            "initial_weight",
            "initial_value",
        ],
        contains=["initial"],
    )

    final_col = find_matching_column(
        df,
        candidates=[
            "final_influence",
            "influence_final",
            "final_weight",
            "final_value",
        ],
        contains=["final"],
    )

    change_col = find_matching_column(
        df,
        candidates=[
            "influence_change",
            "delta_influence",
            "change",
            "delta",
        ],
        contains=["change"],
    )

    lines.append("")
    lines.append(f"Detected agent column: {agent_col}")
    lines.append(f"Detected initial influence column: {initial_col}")
    lines.append(f"Detected final influence column: {final_col}")

    rows = []

    if initial_col is not None and final_col is not None:

        for idx, row in df.iterrows():

            agent = (
                row[agent_col]
                if agent_col is not None
                else idx
            )

            initial = safe_float(row[initial_col])
            final = safe_float(row[final_col])

            if change_col is not None:
                change = safe_float(row[change_col])
            else:
                change = absolute_change(initial, final)

            pct = pct_change(initial, final)

            rows.append({
                "agent": agent,
                "initial_influence": initial,
                "final_influence": final,
                "delta_influence": change,
                "pct_influence_change": pct,
            })

        for row in rows:
            lines.append(
                f"  Agent {row['agent']}: "
                f"initial={fmt(row['initial_influence'])}, "
                f"final={fmt(row['final_influence'])}, "
                f"delta={fmt(row['delta_influence'])}, "
                f"pct={fmt(row['pct_influence_change'], 3)}%"
            )

        valid = [
            row for row in rows
            if row["delta_influence"] is not None
        ]

        if valid:

            largest_gain = max(
                valid,
                key=lambda r: r["delta_influence"],
            )

            largest_loss = min(
                valid,
                key=lambda r: r["delta_influence"],
            )

            lines.append("")
            lines.append(
                "Largest influence gain: "
                f"Agent {largest_gain['agent']} "
                f"(delta={fmt(largest_gain['delta_influence'])})"
            )

            lines.append(
                "Largest influence loss: "
                f"Agent {largest_loss['agent']} "
                f"(delta={fmt(largest_loss['delta_influence'])})"
            )

            result["largest_gain"] = largest_gain
            result["largest_loss"] = largest_loss

    else:
        lines.append("")
        lines.append(
            "Initial/final influence columns could not be identified safely."
        )
        lines.append(df.to_string(index=False))

    result["agents"] = rows

    # --------------------------------------------------------
    # DOMINANT AGENT TRANSITIONS
    # --------------------------------------------------------

    transition_path = paths["dominant_agent_transitions"]

    if transition_path.exists():

        transitions = read_csv(transition_path)

        add_schema(
            lines,
            "INFLUENCE: DOMINANT AGENT TRANSITIONS",
            transition_path,
            transitions,
        )

        lines.append("")
        lines.append(transitions.to_string(index=False))

        result["dominant_transition_count"] = len(transitions)
        result["dominant_transitions"] = (
            transitions.to_dict(orient="records")
        )

        lines.append("")
        lines.append(
            f"Number of recorded dominant-agent transitions: "
            f"{len(transitions)}"
        )

    # --------------------------------------------------------
    # SYSTEM INFLUENCE DISTRIBUTION
    # --------------------------------------------------------

    system_path = paths["system_influence_distribution"]

    if system_path.exists():

        system_df = read_csv(system_path)

        add_schema(
            lines,
            "INFLUENCE: SYSTEM DISTRIBUTION",
            system_path,
            system_df,
        )

        lines.append("")
        lines.append(system_df.to_string(index=False))

        summary = {}

        for col in numeric_columns(system_df):

            initial, final = first_last_numeric(
                system_df,
                col,
            )

            min_v, max_v = min_max_numeric(
                system_df,
                col,
            )

            summary[col] = {
                "initial": initial,
                "final": final,
                "delta": absolute_change(initial, final),
                "min": min_v,
                "max": max_v,
            }

        result["system_distribution"] = summary

    # --------------------------------------------------------
    # EPOCH INFLUENCE SUMMARY
    # --------------------------------------------------------

    epoch_path = paths["epoch_influence_summary"]

    if epoch_path.exists():

        epoch_df = read_csv(epoch_path)

        add_schema(
            lines,
            "INFLUENCE: EPOCH SUMMARY",
            epoch_path,
            epoch_df,
        )

        lines.append("")
        lines.append(epoch_df.to_string(index=False))

        result["epoch_summary"] = (
            epoch_df.to_dict(orient="records")
        )

    # --------------------------------------------------------
    # SCORE-INFLUENCE RELATIONSHIP
    # --------------------------------------------------------

    relationship_path = paths["score_influence_relationship"]

    if relationship_path.exists():

        relationship = read_json(
            relationship_path
        )

        lines.append("")
        lines.append("SCORE-INFLUENCE RELATIONSHIP")
        lines.append("-" * 28)
        lines.append(f"Source: {relationship_path}")
        lines.append(
            json.dumps(
                relationship,
                indent=2,
                ensure_ascii=False,
            )
        )

        result["score_influence_relationship"] = relationship

    return result


# ============================================================
# COMMUNICATION / COOPERATION
# ============================================================

def extract_communication(
    paths: Dict[str, Path],
    lines: List[str],
) -> Dict[str, Any]:

    result: Dict[str, Any] = {}

    disagreement_path = paths["inter_agent_disagreement"]

    if disagreement_path.exists():

        df = read_csv(disagreement_path)

        add_schema(
            lines,
            "COMMUNICATION / COOPERATION: INTER-AGENT DISAGREEMENT",
            disagreement_path,
            df,
        )

        numeric_cols = numeric_columns(df)

        lines.append("")
        lines.append(f"Numeric columns: {numeric_cols}")

        column_summary = {}

        for col in numeric_cols:

            # Skip obvious indexing columns.
            col_n = normalize_name(col)

            if col_n in {
                "epoch",
                "step",
                "iteration",
                "index",
            }:
                continue

            initial, final = first_last_numeric(
                df,
                col,
            )

            minimum, maximum = min_max_numeric(
                df,
                col,
            )

            column_summary[col] = {
                "initial": initial,
                "final": final,
                "delta": absolute_change(initial, final),
                "pct_change": pct_change(initial, final),
                "min": minimum,
                "max": maximum,
            }

            lines.append(
                f"  {col}: "
                f"initial={fmt(initial)}, "
                f"final={fmt(final)}, "
                f"delta={fmt(absolute_change(initial, final))}, "
                f"pct={fmt(pct_change(initial, final), 3)}%, "
                f"min={fmt(minimum)}, "
                f"max={fmt(maximum)}"
            )

        result["inter_agent_disagreement"] = column_summary

    else:
        lines.append("")
        lines.append(
            "WARNING: inter_agent_disagreement.csv is missing."
        )

    # --------------------------------------------------------
    # SYSTEM EVOLUTION
    # --------------------------------------------------------

    system_path = paths["system_evolution"]

    if system_path.exists():

        df = read_csv(system_path)

        add_schema(
            lines,
            "COMMUNICATION / COOPERATION: SYSTEM EVOLUTION",
            system_path,
            df,
        )

        summary = {}

        for col in numeric_columns(df):

            col_n = normalize_name(col)

            if col_n in {
                "epoch",
                "step",
                "iteration",
                "index",
            }:
                continue

            initial, final = first_last_numeric(
                df,
                col,
            )

            minimum, maximum = min_max_numeric(
                df,
                col,
            )

            summary[col] = {
                "initial": initial,
                "final": final,
                "delta": absolute_change(initial, final),
                "pct_change": pct_change(initial, final),
                "min": minimum,
                "max": maximum,
            }

            lines.append(
                f"  {col}: "
                f"initial={fmt(initial)}, "
                f"final={fmt(final)}, "
                f"delta={fmt(absolute_change(initial, final))}, "
                f"pct={fmt(pct_change(initial, final), 3)}%, "
                f"min={fmt(minimum)}, "
                f"max={fmt(maximum)}"
            )

        result["system_evolution"] = summary

    # --------------------------------------------------------
    # IMPORTANT TOPOLOGY-EVIDENCE CHECK
    # --------------------------------------------------------

    topology_keywords = [
        "adjacency",
        "edge",
        "link",
        "neighbor",
        "neighbour",
        "communication",
        "message",
        "topology",
        "active_connection",
        "active_link",
    ]

    topology_evidence = []

    for key in [
        "agent_evolution",
        "system_evolution",
        "inter_agent_disagreement",
    ]:

        path = paths[key]

        if not path.exists():
            continue

        df = read_csv(path)

        for col in df.columns:

            col_n = normalize_name(col)

            if any(
                keyword in col_n
                for keyword in topology_keywords
            ):
                topology_evidence.append({
                    "file": str(path),
                    "column": col,
                })

    result["topology_evidence_columns"] = topology_evidence

    lines.append("")
    lines.append("COMMUNICATION-TOPOLOGY EVIDENCE CHECK")
    lines.append("-" * 37)

    if topology_evidence:

        lines.append(
            "Potential explicit topology/communication variables were found:"
        )

        for item in topology_evidence:
            lines.append(
                f"  {item['column']}  <--  {item['file']}"
            )

        lines.append(
            "These variables must be inspected before claiming "
            "dynamic topology modification."
        )

    else:

        lines.append(
            "No explicit adjacency/link/topology variable was detected "
            "in the inspected tables."
        )

        lines.append(
            "Reviewer-facing wording should therefore describe "
            "'communication/cooperative behavior' rather than claim "
            "experimentally demonstrated topology rewiring."
        )

    return result


# ============================================================
# AGENT EVOLUTION RAW SUMMARY
# ============================================================

def extract_agent_evolution(
    paths: Dict[str, Path],
    lines: List[str],
) -> Dict[str, Any]:

    path = paths["agent_evolution"]

    if not path.exists():
        return {}

    df = read_csv(path)

    add_schema(
        lines,
        "AGENT EVOLUTION: RAW AVAILABLE VARIABLES",
        path,
        df,
    )

    lines.append("")
    lines.append(
        "First 10 rows:"
    )
    lines.append(
        df.head(10).to_string(index=False)
    )

    return {
        "columns": list(df.columns),
        "rows": len(df),
    }


# ============================================================
# OPTIONAL ABLATION SEARCH
# ============================================================

def find_ablation_files() -> List[Path]:

    candidates = []

    for pattern in [
        "*ablation*.csv",
        "*ablation*.json",
        "*component*.csv",
        "*component*.json",
    ]:

        candidates.extend(
            OUTPUTS_ROOT.rglob(pattern)
        )

    return sorted(
        set(candidates),
        key=lambda p: str(p),
    )


# ============================================================
# REVIEWER-READY SUMMARY
# ============================================================

def add_reviewer_ready_summary(
    lines: List[str],
    specialization: Dict[str, Any],
    influence: Dict[str, Any],
    communication: Dict[str, Any],
) -> None:

    lines.append("")
    lines.append("=" * 100)
    lines.append("REVIEWER-READY EXTRACTION SUMMARY")
    lines.append("=" * 100)

    # --------------------------------------------------------
    # Specialization
    # --------------------------------------------------------

    lines.append("")
    lines.append("A. SPECIALIZATION EVOLUTION")
    lines.append("-" * 27)

    largest = specialization.get(
        "largest_absolute_change"
    )

    if largest:
        lines.append(
            "Largest verified agent-level specialization change: "
            f"Agent {largest['agent']}, "
            f"initial={fmt(largest['initial_specialization'])}, "
            f"final={fmt(largest['final_specialization'])}, "
            f"delta={fmt(largest['delta_specialization'])}."
        )

    else:
        lines.append(
            "No reviewer-ready specialization statistic could "
            "be automatically identified."
        )

    # --------------------------------------------------------
    # Influence
    # --------------------------------------------------------

    lines.append("")
    lines.append("B. INFLUENCE REDISTRIBUTION")
    lines.append("-" * 27)

    gain = influence.get("largest_gain")
    loss = influence.get("largest_loss")

    if gain:

        lines.append(
            "Largest influence gain: "
            f"Agent {gain['agent']}, "
            f"initial={fmt(gain['initial_influence'])}, "
            f"final={fmt(gain['final_influence'])}, "
            f"delta={fmt(gain['delta_influence'])}."
        )

        lines.append(
            "Largest influence loss: "
            f"Agent {loss['agent']}, "
            f"initial={fmt(loss['initial_influence'])}, "
            f"final={fmt(loss['final_influence'])}, "
            f"delta={fmt(loss['delta_influence'])}."
        )

    transition_count = influence.get(
        "dominant_transition_count"
    )

    if transition_count is not None:
        lines.append(
            f"Recorded dominant-agent transitions: "
            f"{transition_count}."
        )

    # --------------------------------------------------------
    # Communication
    # --------------------------------------------------------

    lines.append("")
    lines.append("C. COMMUNICATION / COOPERATIVE EVOLUTION")
    lines.append("-" * 38)

    disagreement = communication.get(
        "inter_agent_disagreement",
        {},
    )

    if disagreement:

        for col, values in disagreement.items():
            lines.append(
                f"{col}: "
                f"initial={fmt(values['initial'])}, "
                f"final={fmt(values['final'])}, "
                f"delta={fmt(values['delta'])}, "
                f"min={fmt(values['min'])}, "
                f"max={fmt(values['max'])}."
            )

    else:
        lines.append(
            "No inter-agent disagreement metric was automatically "
            "identified."
        )

    # --------------------------------------------------------
    # Topology claim
    # --------------------------------------------------------

    lines.append("")
    lines.append("D. COMMUNICATION-TOPOLOGY CLAIM SCOPE")
    lines.append("-" * 37)

    topology = communication.get(
        "topology_evidence_columns",
        [],
    )

    if topology:
        lines.append(
            "Explicit communication/topology candidate variables exist. "
            "Inspect them before stating that topology changed dynamically."
        )
    else:
        lines.append(
            "No explicit link/adjacency/topology variable was detected. "
            "Use 'communication/cooperative evolution' and do NOT claim "
            "experimentally verified topology rewiring."
        )


# ============================================================
# AGENT-LEVEL CONSOLIDATED TABLE
# ============================================================

def save_agent_summary(
    specialization: Dict[str, Any],
    influence: Dict[str, Any],
) -> None:

    spec_rows = specialization.get(
        "agents",
        [],
    )

    inf_rows = influence.get(
        "agents",
        [],
    )

    if not spec_rows and not inf_rows:
        return

    spec_df = pd.DataFrame(spec_rows)
    inf_df = pd.DataFrame(inf_rows)

    if not spec_df.empty and not inf_df.empty:

        merged = pd.merge(
            spec_df,
            inf_df,
            on="agent",
            how="outer",
        )

    elif not spec_df.empty:
        merged = spec_df

    else:
        merged = inf_df

    merged.to_csv(
        OUTPUT_CSV,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    org_run, analysis_run = find_run()

    paths = build_paths(
        org_run,
        analysis_run,
    )

    lines: List[str] = []

    lines.append("=" * 100)
    lines.append("ORGANIZATIONAL-EVOLUTION RESULTS FOR REVIEWER RESPONSE")
    lines.append("=" * 100)
    lines.append(f"Project root: {NEW_ROOT}")
    lines.append(f"Organizational run: {org_run}")
    lines.append(f"Analysis run: {analysis_run}")
    lines.append("")
    lines.append(
        "STATUS: READ-ONLY extraction from already completed experiments."
    )
    lines.append(
        "No training, inference, or experiment execution is performed."
    )

    # --------------------------------------------------------
    # Inventory validation
    # --------------------------------------------------------

    lines.append("")
    lines.append("=" * 100)
    lines.append("FILE AVAILABILITY")
    lines.append("=" * 100)

    missing = []

    for key, path in paths.items():

        status = "FOUND" if path.exists() else "MISSING"

        lines.append(
            f"{status:<8} {key:<36} {path}"
        )

        if not path.exists():
            missing.append(key)

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    lines.append("")
    lines.append("=" * 100)
    lines.append("1. AGENT EVOLUTION")
    lines.append("=" * 100)

    agent_evolution = extract_agent_evolution(
        paths,
        lines,
    )

    lines.append("")
    lines.append("=" * 100)
    lines.append("2. SPECIALIZATION EVOLUTION")
    lines.append("=" * 100)

    specialization = extract_specialization(
        paths,
        lines,
    )

    lines.append("")
    lines.append("=" * 100)
    lines.append("3. INFLUENCE REDISTRIBUTION")
    lines.append("=" * 100)

    influence = extract_influence(
        paths,
        lines,
    )

    lines.append("")
    lines.append("=" * 100)
    lines.append("4. COMMUNICATION / COOPERATIVE EVOLUTION")
    lines.append("=" * 100)

    communication = extract_communication(
        paths,
        lines,
    )

    # --------------------------------------------------------
    # Optional ablation inventory
    # --------------------------------------------------------

    lines.append("")
    lines.append("=" * 100)
    lines.append("5. POSSIBLE ABLATION RESULT FILES")
    lines.append("=" * 100)

    ablation_files = find_ablation_files()

    if ablation_files:

        for path in ablation_files:
            lines.append(str(path))

    else:

        lines.append(
            "No ablation-named CSV/JSON file was detected automatically."
        )

        lines.append(
            "Use the already verified manuscript ablation values "
            "for the relationship-to-performance paragraph."
        )

    # --------------------------------------------------------
    # Reviewer-ready summary
    # --------------------------------------------------------

    add_reviewer_ready_summary(
        lines,
        specialization,
        influence,
        communication,
    )

    # --------------------------------------------------------
    # Important interpretation safeguards
    # --------------------------------------------------------

    lines.append("")
    lines.append("=" * 100)
    lines.append("INTERPRETATION SAFEGUARDS")
    lines.append("=" * 100)

    lines.append(
        "1. Do not state that specialization increased unless the "
        "extracted initial/final and temporal values support that direction."
    )

    lines.append(
        "2. Do not state that influence redistribution was large unless "
        "the extracted changes support that characterization."
    )

    lines.append(
        "3. Do not claim communication-topology rewiring unless explicit "
        "adjacency/link activation variables are present in the outputs."
    )

    lines.append(
        "4. Inter-agent disagreement is evidence about cooperative behavior, "
        "not automatically evidence of topology modification."
    )

    lines.append(
        "5. Organizational trajectories show temporal structural evolution; "
        "the ablation analysis provides the separate component-contribution evidence."
    )

    lines.append(
        "6. No causal claim should be inferred solely from temporal association."
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    OUTPUT_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    payload = {
        "project_root": str(NEW_ROOT),
        "organizational_run": str(org_run),
        "analysis_run": str(analysis_run),
        "missing_files": missing,
        "agent_evolution": agent_evolution,
        "specialization": specialization,
        "influence": influence,
        "communication": communication,
        "ablation_files": [
            str(p)
            for p in ablation_files
        ],
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    save_agent_summary(
        specialization,
        influence,
    )

    print()
    print("=" * 100)
    print("COMPLETE")
    print("=" * 100)
    print(f"Reviewer report : {OUTPUT_TXT}")
    print(f"Structured JSON : {OUTPUT_JSON}")

    if OUTPUT_CSV.exists():
        print(f"Agent summary   : {OUTPUT_CSV}")

    print()
    print(
        "No experiment was rerun. "
        "All values were extracted from saved outputs."
    )


if __name__ == "__main__":
    main()