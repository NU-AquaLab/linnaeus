#!/usr/bin/env python3
"""
Release ASDB and ISIC benchmark labels into the aligned data directory.

Reads the raw CSVs from data/, normalizes ASNs, de-duplicates ISIC,
and writes to data/released/202506/labels/{asdb,isic}.{parquet,csv}.
Also copies taxonomy definitions and prompts into the release directory.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from linnaeus.data.alignment import CURRENT_RELEASE, ColumnNormalizer
from linnaeus.models.llm.schema_generation import (
    BUILTIN_TAXONOMIES,
    get_builtin_taxonomy_path,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RELEASE_DIR = DATA_DIR / "released" / CURRENT_RELEASE
LABELS_DIR = RELEASE_DIR / "labels"


def process_asdb() -> pd.DataFrame:
    """Read and normalize the ASDB benchmark labels."""
    csv_path = DATA_DIR / "complete_labeled_data_asdb.csv"
    print(f"Reading ASDB from {csv_path}")
    df = pd.read_csv(csv_path)
    df = ColumnNormalizer.normalize_asn_column(df)
    # Sort by ASN for stable output
    df = df.sort_values("asn").reset_index(drop=True)
    print(f"  ASDB: {len(df)} rows, {len(df.columns) - 1} category columns")
    assert df["asn"].is_unique, "ASDB has duplicate ASNs"
    return df


def process_isic() -> pd.DataFrame:
    """Read, normalize, and de-duplicate the ISIC benchmark labels."""
    csv_path = DATA_DIR / "complete_labeled_data_isic.csv"
    print(f"Reading ISIC from {csv_path}")
    df = pd.read_csv(csv_path)
    df = ColumnNormalizer.normalize_asn_column(df)

    n_before = len(df)
    dupes = df[df.duplicated(subset=["asn"], keep=False)]
    if len(dupes) > 0:
        print(f"  ISIC: merging {len(dupes)} duplicate rows (OR-merge via groupby.max)")
        # OR-merge: for each ASN, take max of each binary column
        cat_cols = [c for c in df.columns if c != "asn"]
        df = df.groupby("asn", as_index=False)[cat_cols].max()
        print(f"  ISIC: {n_before} -> {len(df)} rows after dedup")

    df = df.sort_values("asn").reset_index(drop=True)
    print(f"  ISIC: {len(df)} rows, {len(df.columns) - 1} category columns")
    assert df["asn"].is_unique, "ISIC has duplicate ASNs after dedup"
    return df


def write_labels(df: pd.DataFrame, name: str) -> None:
    """Write a benchmark label DataFrame as both Parquet and CSV."""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = LABELS_DIR / f"{name}.parquet"
    csv_path = LABELS_DIR / f"{name}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"  Wrote {parquet_path}")
    print(f"  Wrote {csv_path}")


def copy_taxonomy_definitions() -> None:
    """Copy the packaged taxonomy JSON files to the release directory."""
    for taxonomy_name in BUILTIN_TAXONOMIES:
        src = get_builtin_taxonomy_path(taxonomy_name)
        dst = RELEASE_DIR / f"{taxonomy_name}_definitions.json"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {src.name} -> {dst.name}")
        else:
            print(f"  WARNING: {src} not found, skipping")


def copy_prompts() -> None:
    """Copy the packaged prompts.yaml to the release directory."""
    from importlib import resources

    src = Path(str(resources.files("linnaeus.resources") / "prompts.yaml"))
    dst = RELEASE_DIR / "prompts.yaml"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  Copied {src.name} -> {dst.name}")
    else:
        print(f"  WARNING: {src} not found, skipping")


def copy_config() -> None:
    """Copy config.yaml to the release directory as a reference config."""
    src = PROJECT_ROOT / "config.yaml"
    dst = RELEASE_DIR / "config.yaml"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  Copied {src.name} -> {dst.name}")
    else:
        print(f"  WARNING: {src} not found, skipping")


def update_metadata(asdb_df: pd.DataFrame, isic_df: pd.DataFrame) -> None:
    """Update metadata.json with benchmark label entries."""
    meta_path = RELEASE_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}

    file_counts = meta.get("file_counts", {})
    file_counts["labels/asdb"] = {"rows": len(asdb_df), "cols": len(asdb_df.columns)}
    file_counts["labels/isic"] = {"rows": len(isic_df), "cols": len(isic_df.columns)}
    meta["file_counts"] = file_counts

    # Record benchmark release info
    meta["benchmark_taxonomies"] = {
        "asdb": {
            "categories": len(asdb_df.columns) - 1,
            "labeled_asns": len(asdb_df),
            "definitions_file": "asdb_definitions.json",
        },
        "isic": {
            "categories": len(isic_df.columns) - 1,
            "labeled_asns": len(isic_df),
            "definitions_file": "isic_definitions.json",
        },
    }
    meta["benchmark_release_date"] = datetime.now().isoformat()

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Updated {meta_path}")


def main() -> None:
    print("=" * 60)
    print("Release benchmark labels")
    print("=" * 60)

    # Process label CSVs
    print("\n--- Processing ASDB ---")
    asdb_df = process_asdb()

    print("\n--- Processing ISIC ---")
    isic_df = process_isic()

    # Write labels
    print("\n--- Writing labels ---")
    write_labels(asdb_df, "asdb")
    write_labels(isic_df, "isic")

    # Copy taxonomy definitions
    print("\n--- Copying taxonomy definitions ---")
    copy_taxonomy_definitions()

    # Copy prompts
    print("\n--- Copying prompts ---")
    copy_prompts()

    # Copy reference config
    print("\n--- Copying config ---")
    copy_config()

    # Update metadata
    print("\n--- Updating metadata ---")
    update_metadata(asdb_df, isic_df)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"  ASDB: {len(asdb_df)} ASNs x {len(asdb_df.columns) - 1} categories")
    print(f"  ISIC: {len(isic_df)} ASNs x {len(isic_df.columns) - 1} categories")
    print(f"  Release dir: {RELEASE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
