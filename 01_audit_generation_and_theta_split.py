from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\47\472\New-Papers\GIS\Codes")
NEW_BRANCH = PROJECT_ROOT / "New_Branch"

SEQ_DIR = (
    PROJECT_ROOT
    / "data"
    / "sim"
    / "sequences"
)

OUTPUT_DIR = (
    NEW_BRANCH
    / "outputs"
    / "audit"
    / "generation_and_theta_split"
)


# ============================================================
# SETTINGS
# ============================================================

TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".csv",
    ".ps1",
    ".bat",
    ".sh",
}

SKIP_DIR_NAMES = {
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    "node_modules",
}

SEARCH_PATTERNS = {
    "num_sequences": re.compile(
        r"\bnum_sequences\b",
        re.IGNORECASE,
    ),
    "n_sequences": re.compile(
        r"\bn_sequences\b",
        re.IGNORECASE,
    ),
    "10000": re.compile(
        r"\b10000\b",
        re.IGNORECASE,
    ),
    "10_000": re.compile(
        r"\b10_000\b",
        re.IGNORECASE,
    ),
    "10,000": re.compile(
        r"\b10,000\b",
        re.IGNORECASE,
    ),
    "range_10000": re.compile(
        r"range\s*\(\s*10000\s*\)",
        re.IGNORECASE,
    ),
    "theta": re.compile(
        r"\btheta\b",
        re.IGNORECASE,
    ),
    "seq_": re.compile(
        r"\bseq_",
        re.IGNORECASE,
    ),
    "generate": re.compile(
        r"\bgenerat(?:e|ed|ing|ion|or)\w*\b",
        re.IGNORECASE,
    ),
    "simulation": re.compile(
        r"\bsimulat(?:e|ed|ing|ion|ions)\w*\b",
        re.IGNORECASE,
    ),
    "np_savez": re.compile(
        r"\bnp\.savez",
        re.IGNORECASE,
    ),
    "savez_compressed": re.compile(
        r"\bsavez_compressed\b",
        re.IGNORECASE,
    ),
    "sequences_dir": re.compile(
        r"\bsequence[s]?\b",
        re.IGNORECASE,
    ),
}

GENERATOR_HINTS = {
    "generate",
    "generator",
    "simulation",
    "simulate",
    "dataset",
    "sequence",
    "pde",
    "build_data",
    "create_data",
}


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )
    return path


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


def save_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
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


def safe_relative(path: Path) -> str:
    try:
        return str(
            path.relative_to(
                PROJECT_ROOT
            )
        )
    except Exception:
        return str(path)


def should_skip(path: Path) -> bool:
    lower_skip = {
        x.lower()
        for x in SKIP_DIR_NAMES
    }

    return any(
        part.lower()
        in lower_skip
        for part in path.parts
    )


# ============================================================
# PART A
# GENERATION PROVENANCE AUDIT
# ============================================================

def collect_text_files() -> list[Path]:

    files = []

    for root, dirs, names in os.walk(
        PROJECT_ROOT
    ):
        root_path = Path(root)

        dirs[:] = [
            d
            for d in dirs
            if d.lower()
            not in {
                x.lower()
                for x in SKIP_DIR_NAMES
            }
        ]

        for name in names:
            path = root_path / name

            if (
                path.suffix.lower()
                in TEXT_EXTENSIONS
            ):
                files.append(path)

    return files


def search_generation_references(
    text_files: list[Path],
):

    print()
    print("=" * 80)
    print("A. GENERATION PROVENANCE AUDIT")
    print("=" * 80)

    rows = []

    for path in text_files:

        if should_skip(path):
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        lines = text.splitlines()

        for line_no, line in enumerate(
            lines,
            start=1,
        ):

            matches = []

            for label, pattern in (
                SEARCH_PATTERNS.items()
            ):
                if pattern.search(line):
                    matches.append(label)

            if matches:
                rows.append(
                    {
                        "path":
                            str(path),

                        "relative_path":
                            safe_relative(path),

                        "line":
                            line_no,

                        "matched_terms":
                            "|".join(
                                matches
                            ),

                        "text":
                            line.strip()[:1500],
                    }
                )

    print(
        f"Relevant references found: "
        f"{len(rows):,}"
    )

    return rows


def identify_candidate_generator_files(
    references: list[dict],
):

    grouped = defaultdict(
        lambda: {
            "matched_terms": set(),
            "lines": [],
        }
    )

    for row in references:

        path = row["path"]

        grouped[path][
            "matched_terms"
        ].update(
            row["matched_terms"].split("|")
        )

        grouped[path][
            "lines"
        ].append(
            row["line"]
        )

    candidates = []

    for path_str, info in grouped.items():

        path = Path(path_str)
        name_lower = (
            path.name.lower()
        )

        terms = info[
            "matched_terms"
        ]

        score = 0
        reasons = []

        # Filename hints
        for hint in GENERATOR_HINTS:
            if hint in name_lower:
                score += 2
                reasons.append(
                    f"filename:{hint}"
                )

        # Strong code evidence
        if (
            "np_savez" in terms
            or "savez_compressed"
            in terms
        ):
            score += 5
            reasons.append(
                "writes_npz"
            )

        if "num_sequences" in terms:
            score += 4
            reasons.append(
                "num_sequences"
            )

        if "n_sequences" in terms:
            score += 4
            reasons.append(
                "n_sequences"
            )

        if (
            "10000" in terms
            or "10_000" in terms
            or "10,000" in terms
            or "range_10000" in terms
        ):
            score += 3
            reasons.append(
                "mentions_10000"
            )

        if "theta" in terms:
            score += 2
            reasons.append(
                "theta"
            )

        if "generate" in terms:
            score += 2
            reasons.append(
                "generate"
            )

        if "simulation" in terms:
            score += 2
            reasons.append(
                "simulation"
            )

        if score > 0:
            candidates.append(
                {
                    "path":
                        path_str,

                    "relative_path":
                        safe_relative(
                            path
                        ),

                    "score":
                        score,

                    "reasons":
                        "|".join(
                            reasons
                        ),

                    "matched_terms":
                        "|".join(
                            sorted(
                                terms
                            )
                        ),

                    "matching_lines":
                        "|".join(
                            str(x)
                            for x
                            in sorted(
                                set(
                                    info[
                                        "lines"
                                    ]
                                )
                            )
                        ),
                }
            )

    candidates.sort(
        key=lambda x:
            x["score"],
        reverse=True,
    )

    print()
    print(
        "Top candidate generator files:"
    )

    for row in candidates[:20]:

        print(
            f"{row['score']:>3} | "
            f"{row['relative_path']} | "
            f"{row['reasons']}"
        )

    return candidates


def extract_sequence_count_candidates(
    references: list[dict],
):

    rows = []

    count_regexes = [
        re.compile(
            r"(?:num_sequences|n_sequences)"
            r"\s*=\s*(\d[\d_]*)",
            re.IGNORECASE,
        ),

        re.compile(
            r"range\s*\(\s*(\d[\d_]*)\s*\)",
            re.IGNORECASE,
        ),

        re.compile(
            r"\b(\d{3,6})\s+"
            r"(?:simulation\s+)?sequences?\b",
            re.IGNORECASE,
        ),
    ]

    for row in references:

        text = row["text"]

        found = []

        for rx in count_regexes:

            for match in rx.finditer(
                text
            ):
                raw = match.group(1)

                try:
                    value = int(
                        raw.replace(
                            "_",
                            "",
                        )
                    )

                    found.append(
                        value
                    )

                except Exception:
                    pass

        if found:
            for value in found:

                rows.append(
                    {
                        "path":
                            row["path"],

                        "relative_path":
                            row[
                                "relative_path"
                            ],

                        "line":
                            row["line"],

                        "candidate_count":
                            value,

                        "text":
                            text,
                    }
                )

    return rows


# ============================================================
# PART B
# THETA / PARAMETER SPLIT AUDIT
# ============================================================

def load_theta_inventory():

    print()
    print("=" * 80)
    print("B. THETA / PARAMETER-DISJOINT SPLIT AUDIT")
    print("=" * 80)

    files = sorted(
        SEQ_DIR.glob("*.npz")
    )

    if not files:
        raise FileNotFoundError(
            f"No NPZ files found in "
            f"{SEQ_DIR}"
        )

    rows = []

    theta_shapes = defaultdict(int)

    for index, path in enumerate(
        files
    ):

        row = {
            "index":
                index,

            "path":
                str(path),

            "filename":
                path.name,
        }

        try:
            with np.load(
                path,
                allow_pickle=False,
            ) as z:

                row[
                    "keys"
                ] = "|".join(
                    z.files
                )

                if "theta" not in z:
                    row[
                        "theta_status"
                    ] = "MISSING"

                    rows.append(row)
                    continue

                theta = np.asarray(
                    z["theta"],
                    dtype=np.float64,
                ).reshape(-1)

                row[
                    "theta_status"
                ] = "OK"

                row[
                    "theta_shape"
                ] = str(
                    theta.shape
                )

                theta_shapes[
                    str(theta.shape)
                ] += 1

                for i, value in enumerate(
                    theta
                ):
                    row[
                        f"theta_{i}"
                    ] = float(value)

        except Exception as exc:

            row[
                "theta_status"
            ] = (
                f"ERROR: {exc}"
            )

        rows.append(row)

    print(
        f"Total NPZ files: "
        f"{len(rows)}"
    )

    print(
        "Theta shape counts:",
        dict(theta_shapes),
    )

    return rows


def assign_current_split(
    theta_rows: list[dict],
):

    n = len(
        theta_rows
    )

    train_end = int(
        0.70 * n
    )

    val_end = int(
        0.85 * n
    )

    for i, row in enumerate(
        theta_rows
    ):

        if i < train_end:
            split = "train"

        elif i < val_end:
            split = "validation"

        else:
            split = "test"

        row[
            "split"
        ] = split

    return theta_rows


def theta_vector_from_row(row):

    theta_keys = sorted(
        [
            key
            for key in row
            if key.startswith(
                "theta_"
            )
            and key[
                len("theta_"):
            ].isdigit()
        ],
        key=lambda x:
            int(
                x.split("_")[1]
            ),
    )

    if not theta_keys:
        return None

    return np.array(
        [
            row[key]
            for key in theta_keys
        ],
        dtype=np.float64,
    )


def exact_theta_key(
    theta: np.ndarray,
    decimals=12,
):

    rounded = np.round(
        theta,
        decimals=decimals,
    )

    return tuple(
        float(x)
        for x in rounded
    )


def summarize_theta_by_split(
    theta_rows,
):

    split_summary = []

    for split in [
        "train",
        "validation",
        "test",
    ]:

        rows = [
            r
            for r in theta_rows
            if r.get(
                "split"
            ) == split
            and r.get(
                "theta_status"
            ) == "OK"
        ]

        vectors = [
            theta_vector_from_row(
                r
            )
            for r in rows
        ]

        vectors = [
            v
            for v in vectors
            if v is not None
        ]

        if not vectors:
            continue

        matrix = np.vstack(
            vectors
        )

        result = {
            "split":
                split,

            "count":
                len(vectors),

            "theta_dimension":
                matrix.shape[1],
        }

        for j in range(
            matrix.shape[1]
        ):

            result[
                f"theta_{j}_min"
            ] = float(
                np.min(
                    matrix[:, j]
                )
            )

            result[
                f"theta_{j}_max"
            ] = float(
                np.max(
                    matrix[:, j]
                )
            )

            result[
                f"theta_{j}_mean"
            ] = float(
                np.mean(
                    matrix[:, j]
                )
            )

            result[
                f"theta_{j}_std"
            ] = float(
                np.std(
                    matrix[:, j]
                )
            )

        split_summary.append(
            result
        )

    return split_summary


def find_exact_theta_overlap(
    theta_rows,
):

    by_split = {
        "train": defaultdict(list),
        "validation": defaultdict(list),
        "test": defaultdict(list),
    }

    for row in theta_rows:

        if (
            row.get(
                "theta_status"
            )
            != "OK"
        ):
            continue

        theta = theta_vector_from_row(
            row
        )

        if theta is None:
            continue

        key = exact_theta_key(
            theta
        )

        by_split[
            row["split"]
        ][key].append(
            row["filename"]
        )

    comparisons = [
        (
            "train",
            "validation",
        ),
        (
            "train",
            "test",
        ),
        (
            "validation",
            "test",
        ),
    ]

    rows = []

    for a, b in comparisons:

        overlap = (
            set(
                by_split[a].keys()
            )
            &
            set(
                by_split[b].keys()
            )
        )

        for theta_key in sorted(
            overlap
        ):

            rows.append(
                {
                    "split_a":
                        a,

                    "split_b":
                        b,

                    "theta":
                        str(
                            theta_key
                        ),

                    "files_a":
                        "|".join(
                            by_split[
                                a
                            ][
                                theta_key
                            ]
                        ),

                    "files_b":
                        "|".join(
                            by_split[
                                b
                            ][
                                theta_key
                            ]
                        ),
                }
            )

    return rows


def nearest_cross_split_distances(
    theta_rows,
):

    split_vectors = {}

    for split in [
        "train",
        "validation",
        "test",
    ]:

        rows = [
            r
            for r in theta_rows
            if (
                r.get(
                    "split"
                )
                == split
                and r.get(
                    "theta_status"
                )
                == "OK"
            )
        ]

        vectors = []
        names = []

        for r in rows:

            theta = theta_vector_from_row(
                r
            )

            if theta is not None:

                vectors.append(
                    theta
                )

                names.append(
                    r["filename"]
                )

        split_vectors[
            split
        ] = (
            np.vstack(vectors)
            if vectors
            else np.empty(
                (0, 0)
            ),
            names,
        )

    comparisons = [
        (
            "train",
            "validation",
        ),
        (
            "train",
            "test",
        ),
        (
            "validation",
            "test",
        ),
    ]

    summary_rows = []
    pair_rows = []

    for a, b in comparisons:

        A, names_a = (
            split_vectors[a]
        )

        B, names_b = (
            split_vectors[b]
        )

        if (
            len(A) == 0
            or len(B) == 0
        ):
            continue

        # Standardize dimensions using
        # pooled mean/std to make Euclidean
        # distance interpretable across theta dimensions.
        pooled = np.vstack(
            [A, B]
        )

        mu = np.mean(
            pooled,
            axis=0,
        )

        sd = np.std(
            pooled,
            axis=0,
        )

        sd[
            sd < 1e-12
        ] = 1.0

        Az = (
            A - mu
        ) / sd

        Bz = (
            B - mu
        ) / sd

        # Pairwise Euclidean distances.
        diff = (
            Az[:, None, :]
            - Bz[None, :, :]
        )

        dist = np.sqrt(
            np.sum(
                diff ** 2,
                axis=2,
            )
        )

        min_index = np.unravel_index(
            np.argmin(
                dist
            ),
            dist.shape,
        )

        minimum = float(
            dist[
                min_index
            ]
        )

        summary_rows.append(
            {
                "split_a":
                    a,

                "split_b":
                    b,

                "minimum_standardized_theta_distance":
                    minimum,

                "median_standardized_theta_distance":
                    float(
                        np.median(
                            dist
                        )
                    ),

                "mean_standardized_theta_distance":
                    float(
                        np.mean(
                            dist
                        )
                    ),

                "closest_file_a":
                    names_a[
                        min_index[0]
                    ],

                "closest_file_b":
                    names_b[
                        min_index[1]
                    ],
            }
        )

        # Store closest 10 pairs.
        flat = np.argsort(
            dist,
            axis=None,
        )[:10]

        coords = np.unravel_index(
            flat,
            dist.shape,
        )

        for rank, (
            ia,
            ib,
        ) in enumerate(
            zip(
                coords[0],
                coords[1],
            ),
            start=1,
        ):

            pair_rows.append(
                {
                    "split_a":
                        a,

                    "split_b":
                        b,

                    "rank":
                        rank,

                    "file_a":
                        names_a[
                            ia
                        ],

                    "file_b":
                        names_b[
                            ib
                        ],

                    "standardized_theta_distance":
                        float(
                            dist[
                                ia,
                                ib,
                            ]
                        ),

                    "theta_a":
                        str(
                            tuple(
                                float(x)
                                for x
                                in A[
                                    ia
                                ]
                            )
                        ),

                    "theta_b":
                        str(
                            tuple(
                                float(x)
                                for x
                                in B[
                                    ib
                                ]
                            )
                        ),
                }
            )

    return (
        summary_rows,
        pair_rows,
    )


# ============================================================
# INTERPRETATION
# ============================================================

def build_findings(
    count_candidates,
    generator_candidates,
    theta_rows,
    exact_overlap,
    distance_summary,
):

    findings = []

    values = [
        row[
            "candidate_count"
        ]
        for row in count_candidates
    ]

    unique_values = sorted(
        set(values)
    )

    findings.append(
        {
            "issue":
                "Sequence-count values found in code/docs",

            "status":
                "OBSERVED",

            "evidence":
                (
                    ", ".join(
                        str(v)
                        for v in unique_values
                    )
                    if unique_values
                    else
                    "No explicit assignment/range-derived sequence counts found."
                ),
        }
    )

    top_generators = (
        generator_candidates[:10]
    )

    findings.append(
        {
            "issue":
                "Top candidate generator files",

            "status":
                "OBSERVED",

            "evidence":
                " | ".join(
                    row[
                        "relative_path"
                    ]
                    for row
                    in top_generators
                ),
        }
    )

    theta_ok = sum(
        1
        for row in theta_rows
        if row.get(
            "theta_status"
        )
        == "OK"
    )

    findings.append(
        {
            "issue":
                "NPZ files containing theta",

            "status":
                "OBSERVED",

            "evidence":
                f"{theta_ok} of "
                f"{len(theta_rows)} "
                f"NPZ files contain theta.",
        }
    )

    if exact_overlap:

        findings.append(
            {
                "issue":
                    "Exact theta overlap across current 70/15/15 splits",

                "status":
                    "FAIL",

                "evidence":
                    (
                        f"{len(exact_overlap)} "
                        f"cross-split exact theta overlaps found."
                    ),
            }
        )

    else:

        findings.append(
            {
                "issue":
                    "Exact theta overlap across current 70/15/15 splits",

                "status":
                    "PASS",

                "evidence":
                    (
                        "No exact theta vectors are shared "
                        "between train, validation, and test."
                    ),
            }
        )

    for row in distance_summary:

        findings.append(
            {
                "issue":
                    (
                        f"Closest theta pair: "
                        f"{row['split_a']} vs "
                        f"{row['split_b']}"
                    ),

                "status":
                    "OBSERVED",

                "evidence":
                    (
                        f"Minimum standardized distance = "
                        f"{row['minimum_standardized_theta_distance']:.8f}; "
                        f"{row['closest_file_a']} vs "
                        f"{row['closest_file_b']}"
                    ),
            }
        )

    return findings


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_dir(
        OUTPUT_DIR
    )

    start = datetime.now()

    print()
    print("=" * 80)
    print("GENERATION PROVENANCE + THETA SPLIT AUDIT")
    print("=" * 80)

    print(
        "Project root:"
    )
    print(
        PROJECT_ROOT
    )

    print()
    print(
        "Sequence directory:"
    )
    print(
        SEQ_DIR
    )

    # --------------------------------------------------------
    # A. Generation provenance
    # --------------------------------------------------------

    text_files = (
        collect_text_files()
    )

    references = (
        search_generation_references(
            text_files
        )
    )

    generator_candidates = (
        identify_candidate_generator_files(
            references
        )
    )

    count_candidates = (
        extract_sequence_count_candidates(
            references
        )
    )

    # --------------------------------------------------------
    # B. Theta split audit
    # --------------------------------------------------------

    theta_rows = (
        load_theta_inventory()
    )

    theta_rows = (
        assign_current_split(
            theta_rows
        )
    )

    split_summary = (
        summarize_theta_by_split(
            theta_rows
        )
    )

    exact_overlap = (
        find_exact_theta_overlap(
            theta_rows
        )
    )

    (
        distance_summary,
        closest_pairs,
    ) = (
        nearest_cross_split_distances(
            theta_rows
        )
    )

    findings = (
        build_findings(
            count_candidates,
            generator_candidates,
            theta_rows,
            exact_overlap,
            distance_summary,
        )
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    save_csv(
        OUTPUT_DIR
        / "generation_references.csv",
        references,
    )

    save_csv(
        OUTPUT_DIR
        / "candidate_generator_files.csv",
        generator_candidates,
    )

    save_csv(
        OUTPUT_DIR
        / "sequence_count_candidates.csv",
        count_candidates,
    )

    save_csv(
        OUTPUT_DIR
        / "theta_inventory.csv",
        theta_rows,
    )

    save_csv(
        OUTPUT_DIR
        / "theta_split_summary.csv",
        split_summary,
    )

    save_csv(
        OUTPUT_DIR
        / "exact_theta_cross_split_overlap.csv",
        exact_overlap,
    )

    save_csv(
        OUTPUT_DIR
        / "theta_cross_split_distance_summary.csv",
        distance_summary,
    )

    save_csv(
        OUTPUT_DIR
        / "theta_closest_cross_split_pairs.csv",
        closest_pairs,
    )

    save_csv(
        OUTPUT_DIR
        / "findings.csv",
        findings,
    )

    summary = {
        "timestamp":
            datetime.now().isoformat(),

        "project_root":
            str(
                PROJECT_ROOT
            ),

        "sequence_directory":
            str(
                SEQ_DIR
            ),

        "npz_count":
            len(
                theta_rows
            ),

        "text_files_scanned":
            len(
                text_files
            ),

        "generation_reference_count":
            len(
                references
            ),

        "candidate_generator_file_count":
            len(
                generator_candidates
            ),

        "sequence_count_candidates":
            sorted(
                set(
                    row[
                        "candidate_count"
                    ]
                    for row
                    in count_candidates
                )
            ),

        "theta_files_ok":
            sum(
                1
                for row in theta_rows
                if row.get(
                    "theta_status"
                )
                == "OK"
            ),

        "exact_theta_cross_split_overlap_count":
            len(
                exact_overlap
            ),

        "findings":
            findings,
    }

    save_json(
        OUTPUT_DIR
        / "audit_summary.json",
        summary,
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL FINDINGS")
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
            f"   Status: "
            f"{row['status']}"
        )

        print(
            f"   Evidence: "
            f"{row['evidence']}"
        )

    elapsed = (
        datetime.now()
        - start
    ).total_seconds()

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)

    print(
        f"Elapsed: "
        f"{elapsed:.2f} sec"
    )

    print()
    print(
        "Outputs:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Most important files:"
    )

    print(
        OUTPUT_DIR
        / "candidate_generator_files.csv"
    )

    print(
        OUTPUT_DIR
        / "sequence_count_candidates.csv"
    )

    print(
        OUTPUT_DIR
        / "theta_inventory.csv"
    )

    print(
        OUTPUT_DIR
        / "exact_theta_cross_split_overlap.csv"
    )

    print(
        OUTPUT_DIR
        / "theta_cross_split_distance_summary.csv"
    )

    print(
        OUTPUT_DIR
        / "audit_summary.json"
    )


if __name__ == "__main__":
    main()