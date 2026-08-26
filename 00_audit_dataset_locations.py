from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\47\472\New-Papers\GIS\Codes")

EXPECTED_SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "sim"
    / "sequences"
)

NEW_BRANCH = PROJECT_ROOT / "New_Branch"

OUTPUT_DIR = (
    NEW_BRANCH
    / "outputs"
    / "audit"
    / "dataset_location_audit"
)


# ============================================================
# SETTINGS
# ============================================================

# File types whose text will be searched for dataset references.
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

# Archives worth reporting because full datasets might be stored there.
ARCHIVE_EXTENSIONS = {
    ".zip",
    ".7z",
    ".rar",
    ".tar",
    ".gz",
    ".tgz",
}

# Search terms that may reveal where the full dataset was generated,
# stored, copied, downloaded, or described.
SEARCH_PATTERNS = {
    "10000": re.compile(r"\b10000\b", re.IGNORECASE),
    "10,000": re.compile(r"\b10,000\b", re.IGNORECASE),
    "sequence": re.compile(r"\bsequences?\b", re.IGNORECASE),
    "pdebench": re.compile(r"\bpdebench\b", re.IGNORECASE),
    "sim_sequences_path": re.compile(
        r"data[\\/]+sim[\\/]+sequences",
        re.IGNORECASE,
    ),
    "npz": re.compile(r"\.npz\b", re.IGNORECASE),
}

# Avoid wasting time traversing the virtual environment and caches.
SKIP_DIR_NAMES = {
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    "node_modules",
}


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
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
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
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


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)

    for unit in [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{value:.2f} PB"


def safe_relative(path: Path) -> str:
    try:
        return str(
            path.relative_to(
                PROJECT_ROOT
            )
        )
    except Exception:
        return str(path)


def should_skip_path(path: Path) -> bool:
    return any(
        part.lower()
        in {x.lower() for x in SKIP_DIR_NAMES}
        for part in path.parts
    )


# ============================================================
# 1. COMPLETE FILE INVENTORY
# ============================================================

def scan_project_files():
    print()
    print("=" * 80)
    print("SCANNING ENTIRE PROJECT TREE")
    print("=" * 80)

    all_files = []

    for root, dirs, files in os.walk(
        PROJECT_ROOT
    ):
        root_path = Path(root)

        # Prevent traversal into irrelevant large directories.
        dirs[:] = [
            d
            for d in dirs
            if d.lower()
            not in {
                x.lower()
                for x in SKIP_DIR_NAMES
            }
        ]

        for filename in files:
            path = root_path / filename

            try:
                size = path.stat().st_size
            except Exception:
                size = -1

            all_files.append(
                {
                    "path": str(path),
                    "relative_path":
                        safe_relative(path),
                    "name": path.name,
                    "extension":
                        path.suffix.lower(),
                    "size_bytes": size,
                    "size_human":
                        human_size(size)
                        if size >= 0
                        else "UNKNOWN",
                }
            )

    print(
        f"Total files found: "
        f"{len(all_files):,}"
    )

    return all_files


# ============================================================
# 2. FIND ALL NPZ FILES
# ============================================================

def analyze_npz_locations(all_files):
    print()
    print("=" * 80)
    print("NPZ DATASET LOCATION ANALYSIS")
    print("=" * 80)

    npz_files = [
        row
        for row in all_files
        if row["extension"] == ".npz"
    ]

    print(
        f"Total .npz files under project: "
        f"{len(npz_files):,}"
    )

    by_parent = defaultdict(list)

    for row in npz_files:
        parent = str(
            Path(row["path"]).parent
        )

        by_parent[parent].append(row)

    location_rows = []

    for parent, rows in sorted(
        by_parent.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    ):
        total_bytes = sum(
            max(0, r["size_bytes"])
            for r in rows
        )

        location_rows.append(
            {
                "directory": parent,
                "relative_directory":
                    safe_relative(
                        Path(parent)
                    ),
                "npz_count":
                    len(rows),
                "total_size_bytes":
                    total_bytes,
                "total_size_human":
                    human_size(
                        total_bytes
                    ),
            }
        )

    print()
    print("NPZ collections:")

    for row in location_rows:
        print(
            f"{row['npz_count']:>7,} files | "
            f"{row['total_size_human']:>12} | "
            f"{row['directory']}"
        )

    expected_count = 0

    if EXPECTED_SEQUENCE_DIR.exists():
        expected_count = len(
            list(
                EXPECTED_SEQUENCE_DIR.glob(
                    "*.npz"
                )
            )
        )

    print()
    print(
        "Expected current directory:"
    )

    print(
        EXPECTED_SEQUENCE_DIR
    )

    print(
        "Direct .npz count:",
        f"{expected_count:,}",
    )

    return (
        npz_files,
        location_rows,
        expected_count,
    )


# ============================================================
# 3. INSPECT NPZ STRUCTURE
# ============================================================

def inspect_npz_samples(
    location_rows,
    max_samples_per_location=3,
):
    print()
    print("=" * 80)
    print("NPZ STRUCTURE INSPECTION")
    print("=" * 80)

    rows = []

    for location in location_rows:
        folder = Path(
            location["directory"]
        )

        samples = sorted(
            folder.glob("*.npz")
        )[
            :max_samples_per_location
        ]

        for path in samples:
            info = {
                "path": str(path),
                "directory": str(folder),
            }

            try:
                with np.load(
                    path,
                    allow_pickle=False,
                ) as z:

                    keys = list(z.files)

                    info["keys"] = "|".join(
                        keys
                    )

                    for key in keys:
                        try:
                            arr = z[key]

                            info[
                                f"{key}_shape"
                            ] = str(
                                arr.shape
                            )

                            info[
                                f"{key}_dtype"
                            ] = str(
                                arr.dtype
                            )

                        except Exception as exc:
                            info[
                                f"{key}_error"
                            ] = str(exc)

            except Exception as exc:
                info[
                    "load_error"
                ] = str(exc)

            rows.append(info)

    return rows


# ============================================================
# 4. FIND ARCHIVES
# ============================================================

def find_archives(all_files):
    print()
    print("=" * 80)
    print("ARCHIVE SEARCH")
    print("=" * 80)

    archives = []

    for row in all_files:
        if row["extension"] in ARCHIVE_EXTENSIONS:
            archives.append(row)

    print(
        f"Potential archive files found: "
        f"{len(archives):,}"
    )

    for row in archives:
        print(
            f"{row['size_human']:>12} | "
            f"{row['path']}"
        )

    return archives


# ============================================================
# 5. INSPECT ZIP ARCHIVES FOR NPZ CONTENT
# ============================================================

def inspect_zip_archives(archives):
    rows = []

    for item in archives:
        path = Path(
            item["path"]
        )

        if path.suffix.lower() != ".zip":
            continue

        result = {
            "path": str(path),
            "size_human":
                item["size_human"],
            "npz_inside": None,
            "total_entries": None,
            "status": None,
        }

        try:
            with zipfile.ZipFile(
                path,
                "r",
            ) as z:

                names = z.namelist()

                npz_names = [
                    n
                    for n in names
                    if n.lower().endswith(
                        ".npz"
                    )
                ]

                result[
                    "total_entries"
                ] = len(names)

                result[
                    "npz_inside"
                ] = len(npz_names)

                result[
                    "status"
                ] = "OK"

        except Exception as exc:
            result[
                "status"
            ] = f"ERROR: {exc}"

        rows.append(result)

    return rows


# ============================================================
# 6. SEARCH SOURCE FILES FOR DATASET CLAIMS/PATHS
# ============================================================

def scan_text_references(all_files):
    print()
    print("=" * 80)
    print("SEARCHING CODE / CONFIG / README REFERENCES")
    print("=" * 80)

    matches = []

    for row in all_files:
        path = Path(
            row["path"]
        )

        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue

        if should_skip_path(path):
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        for line_number, line in enumerate(
            text.splitlines(),
            start=1,
        ):
            matched_terms = []

            for name, pattern in SEARCH_PATTERNS.items():
                if pattern.search(line):
                    matched_terms.append(
                        name
                    )

            if matched_terms:
                matches.append(
                    {
                        "path": str(path),
                        "relative_path":
                            safe_relative(
                                path
                            ),
                        "line":
                            line_number,
                        "matched_terms":
                            "|".join(
                                matched_terms
                            ),
                        "text":
                            line.strip()[
                                :1000
                            ],
                    }
                )

    print(
        f"Relevant text references found: "
        f"{len(matches):,}"
    )

    return matches


# ============================================================
# 7. SEARCH FOR DIRECTORIES WITH DATA-LIKE NAMES
# ============================================================

def scan_candidate_data_directories():
    print()
    print("=" * 80)
    print("CANDIDATE DATA DIRECTORY SEARCH")
    print("=" * 80)

    keywords = {
        "data",
        "dataset",
        "datasets",
        "sequence",
        "sequences",
        "sim",
        "simulation",
        "simulations",
        "pdebench",
        "samples",
        "generated",
        "output",
        "outputs",
    }

    rows = []

    for root, dirs, _ in os.walk(
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

        for d in dirs:
            path = root_path / d

            if any(
                k in d.lower()
                for k in keywords
            ):
                try:
                    direct_files = [
                        p
                        for p in path.iterdir()
                        if p.is_file()
                    ]

                    direct_npz = [
                        p
                        for p in direct_files
                        if p.suffix.lower()
                        == ".npz"
                    ]

                except Exception:
                    direct_files = []
                    direct_npz = []

                rows.append(
                    {
                        "directory":
                            str(path),
                        "relative_directory":
                            safe_relative(
                                path
                            ),
                        "direct_file_count":
                            len(
                                direct_files
                            ),
                        "direct_npz_count":
                            len(
                                direct_npz
                            ),
                    }
                )

    rows.sort(
        key=lambda x:
            x["direct_npz_count"],
        reverse=True,
    )

    return rows


# ============================================================
# 8. CHECK FOR POSSIBLE DUPLICATE NPZ FILES
# ============================================================

def quick_file_signature(path: Path):
    """
    Fast signature based on size + first/last 64 KB.
    This avoids hashing every large file completely.
    """

    try:
        size = path.stat().st_size

        h = hashlib.sha256()

        with open(path, "rb") as f:
            first = f.read(65536)
            h.update(first)

            if size > 65536:
                f.seek(
                    max(
                        0,
                        size - 65536,
                    )
                )

                h.update(
                    f.read(65536)
                )

        return (
            size,
            h.hexdigest(),
        )

    except Exception:
        return (
            None,
            None,
        )


def analyze_possible_duplicates(npz_files):
    print()
    print("=" * 80)
    print("POSSIBLE DUPLICATE NPZ ANALYSIS")
    print("=" * 80)

    groups = defaultdict(list)

    for i, row in enumerate(
        npz_files,
        start=1,
    ):
        path = Path(
            row["path"]
        )

        signature = (
            quick_file_signature(
                path
            )
        )

        groups[
            signature
        ].append(
            str(path)
        )

        if i % 1000 == 0:
            print(
                f"Checked {i:,} NPZ files..."
            )

    duplicate_rows = []

    for signature, paths in groups.items():
        if (
            signature[0] is not None
            and len(paths) > 1
        ):
            duplicate_rows.append(
                {
                    "size_bytes":
                        signature[0],
                    "signature":
                        signature[1],
                    "duplicate_count":
                        len(paths),
                    "paths":
                        " | ".join(
                            paths
                        ),
                }
            )

    print(
        "Possible duplicate groups:",
        len(
            duplicate_rows
        ),
    )

    return duplicate_rows


# ============================================================
# 9. FINAL INTERPRETATION
# ============================================================

def produce_interpretation(
    expected_count,
    location_rows,
    archives,
    zip_rows,
    references,
):
    findings = []

    # --------------------------------------------------------
    # Current expected directory
    # --------------------------------------------------------

    if expected_count == 100:
        findings.append(
            {
                "question":
                    "Does data\\sim\\sequences contain only 100 sequences?",
                "status":
                    "YES",
                "evidence":
                    (
                        f"The directory contains exactly "
                        f"{expected_count} direct NPZ files."
                    ),
            }
        )

    elif expected_count == 10000:
        findings.append(
            {
                "question":
                    "Does data\\sim\\sequences contain the full 10,000 sequences?",
                "status":
                    "YES",
                "evidence":
                    (
                        f"The directory contains exactly "
                        f"{expected_count} NPZ files."
                    ),
            }
        )

    else:
        findings.append(
            {
                "question":
                    "How many sequences are in data\\sim\\sequences?",
                "status":
                    "OBSERVED",
                "evidence":
                    (
                        f"The directory contains "
                        f"{expected_count} direct NPZ files."
                    ),
            }
        )

    # --------------------------------------------------------
    # Search other locations
    # --------------------------------------------------------

    other_large_locations = [
        row
        for row in location_rows
        if (
            Path(
                row["directory"]
            ).resolve()
            != EXPECTED_SEQUENCE_DIR.resolve()
            and row["npz_count"] >= 1000
        )
    ]

    exact_10000_locations = [
        row
        for row in location_rows
        if row["npz_count"] == 10000
    ]

    if exact_10000_locations:
        findings.append(
            {
                "question":
                    "Do 10,000 NPZ sequences exist elsewhere?",
                "status":
                    "YES",
                "evidence":
                    " | ".join(
                        row["directory"]
                        for row
                        in exact_10000_locations
                    ),
            }
        )

    elif other_large_locations:
        findings.append(
            {
                "question":
                    "Are there large NPZ collections elsewhere?",
                "status":
                    "YES",
                "evidence":
                    " | ".join(
                        (
                            f"{row['directory']} "
                            f"({row['npz_count']:,} files)"
                        )
                        for row
                        in other_large_locations
                    ),
            }
        )

    else:
        findings.append(
            {
                "question":
                    "Do 10,000 NPZ sequences exist elsewhere under Codes?",
                "status":
                    "NOT FOUND",
                "evidence":
                    (
                        "No directory containing 10,000 NPZ files "
                        "was found under the scanned project root."
                    ),
            }
        )

    # --------------------------------------------------------
    # Zip evidence
    # --------------------------------------------------------

    zip_with_many_npz = [
        row
        for row in zip_rows
        if isinstance(
            row.get(
                "npz_inside"
            ),
            int,
        )
        and row[
            "npz_inside"
        ] >= 1000
    ]

    if zip_with_many_npz:
        findings.append(
            {
                "question":
                    "Could the full dataset be stored in a ZIP archive?",
                "status":
                    "POSSIBLE/YES",
                "evidence":
                    " | ".join(
                        (
                            f"{row['path']} "
                            f"({row['npz_inside']:,} NPZ entries)"
                        )
                        for row
                        in zip_with_many_npz
                    ),
            }
        )
    else:
        findings.append(
            {
                "question":
                    "Could the full dataset be stored in a detected ZIP archive?",
                "status":
                    "NO LARGE NPZ ZIP FOUND",
                "evidence":
                    (
                        f"{len(archives)} archive candidates were detected; "
                        "none of the inspectable ZIP files contained "
                        "1,000 or more NPZ entries."
                    ),
            }
        )

    # --------------------------------------------------------
    # Text claims
    # --------------------------------------------------------

    count_claim_refs = [
        row
        for row in references
        if (
            "10000"
            in row[
                "matched_terms"
            ]
            or "10,000"
            in row[
                "matched_terms"
            ]
        )
    ]

    findings.append(
        {
            "question":
                "How many source/code/document references mention 10,000?",
            "status":
                "OBSERVED",
            "evidence":
                str(
                    len(
                        count_claim_refs
                    )
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

    started = datetime.now()

    print()
    print("=" * 80)
    print("DATASET LOCATION AND CLAIM AUDIT")
    print("=" * 80)

    print(
        "Project root:"
    )

    print(
        PROJECT_ROOT
    )

    print()
    print(
        "Expected training sequence directory:"
    )

    print(
        EXPECTED_SEQUENCE_DIR
    )

    # --------------------------------------------------------
    # Scan
    # --------------------------------------------------------

    all_files = (
        scan_project_files()
    )

    (
        npz_files,
        location_rows,
        expected_count,
    ) = analyze_npz_locations(
        all_files
    )

    npz_structure_rows = (
        inspect_npz_samples(
            location_rows
        )
    )

    archives = (
        find_archives(
            all_files
        )
    )

    zip_rows = (
        inspect_zip_archives(
            archives
        )
    )

    references = (
        scan_text_references(
            all_files
        )
    )

    candidate_dirs = (
        scan_candidate_data_directories()
    )

    duplicate_rows = (
        analyze_possible_duplicates(
            npz_files
        )
    )

    findings = (
        produce_interpretation(
            expected_count,
            location_rows,
            archives,
            zip_rows,
            references,
        )
    )

    # --------------------------------------------------------
    # Save evidence
    # --------------------------------------------------------

    save_csv(
        OUTPUT_DIR
        / "all_project_files.csv",
        all_files,
    )

    save_csv(
        OUTPUT_DIR
        / "npz_files.csv",
        npz_files,
    )

    save_csv(
        OUTPUT_DIR
        / "npz_locations.csv",
        location_rows,
    )

    save_csv(
        OUTPUT_DIR
        / "npz_structure_samples.csv",
        npz_structure_rows,
    )

    save_csv(
        OUTPUT_DIR
        / "archives.csv",
        archives,
    )

    save_csv(
        OUTPUT_DIR
        / "zip_contents.csv",
        zip_rows,
    )

    save_csv(
        OUTPUT_DIR
        / "text_references.csv",
        references,
    )

    save_csv(
        OUTPUT_DIR
        / "candidate_data_directories.csv",
        candidate_dirs,
    )

    save_csv(
        OUTPUT_DIR
        / "possible_duplicate_npz.csv",
        duplicate_rows,
    )

    save_csv(
        OUTPUT_DIR
        / "findings.csv",
        findings,
    )

    summary = {
        "audit_timestamp":
            datetime.now().isoformat(),

        "project_root":
            str(
                PROJECT_ROOT
            ),

        "expected_sequence_dir":
            str(
                EXPECTED_SEQUENCE_DIR
            ),

        "expected_sequence_dir_npz_count":
            expected_count,

        "total_project_files":
            len(
                all_files
            ),

        "total_npz_files":
            len(
                npz_files
            ),

        "npz_location_count":
            len(
                location_rows
            ),

        "archive_count":
            len(
                archives
            ),

        "text_reference_count":
            len(
                references
            ),

        "possible_duplicate_groups":
            len(
                duplicate_rows
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
    # Console summary
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
            f"{i}. {row['question']}"
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
        - started
    ).total_seconds()

    print()
    print("=" * 80)
    print(
        "AUDIT COMPLETED"
    )
    print("=" * 80)

    print(
        f"Elapsed time: "
        f"{elapsed:.2f} sec"
    )

    print()
    print(
        "Results directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Most important files to inspect:"
    )

    print(
        OUTPUT_DIR
        / "findings.csv"
    )

    print(
        OUTPUT_DIR
        / "npz_locations.csv"
    )

    print(
        OUTPUT_DIR
        / "text_references.csv"
    )

    print(
        OUTPUT_DIR
        / "audit_summary.json"
    )


if __name__ == "__main__":
    main()