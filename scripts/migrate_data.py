#!/usr/bin/env python3
"""
One-off migration script: normalise ``data/`` into a clean, aligned layout.

Usage::

    python scripts/migrate_data.py            # default: project root
    python scripts/migrate_data.py --root /path/to/project

What it does
------------
1. Renames ``data/`` → ``data-legacy/``
2. Reads every file from ``data-legacy/`` and writes normalised copies into:
   - ``data/released/<RELEASE_TAG>/`` – small, committed files (labels,
     baseline/finetuned predictions, splits, metrics, metadata)
   - ``data/local/`` – large, user-generated files (features, complete predictions)
3. Normalisation rules:
   - ASN columns → plain ``int``, column name ``asn``
   - ``Financial*`` → ``Finance*``
   - ``Enterprise_E_commerce`` → ``Enterprise_Ecommerce``
   - ``Enterprise_Industrial_Manufacturing`` → ``Enterprise_IndustrialManufacturing``
   - ``Government_National`` → ``Government_FederalNational``
   - ``Government_SOE`` column dropped
   - Missing top-level columns (``VPNs``, ``Community``) added as zeros
   - Inference chunks concatenated into single files
   - Split assignments derived from original X_train / X_val ASN lists

All original files are preserved in ``data-legacy/``.
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project root so the script can be run from anywhere
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR.parent

# Add src/ to path so we can import linneaus
sys.path.insert(0, str(DEFAULT_ROOT / "src"))

from linneaus.data.alignment import (  # noqa: E402
    CANONICAL_SUBLEVEL,
    CANONICAL_TOPLEVEL,
    CURRENT_RELEASE,
    ColumnNormalizer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("migrate_data")


# ===================================================================
# Helpers
# ===================================================================


def save_dual(df: pd.DataFrame, base: Path, index: bool = False) -> None:
    """Write *df* as both ``.parquet`` and ``.csv``."""
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(f"{base}.parquet", engine="pyarrow", index=index)
    df.to_csv(f"{base}.csv", index=index)
    log.info(
        "  wrote %s.{parquet,csv}  (%d rows, %d cols)",
        base.name,
        len(df),
        len(df.columns),
    )


def norm(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise ASN + category columns."""
    df = ColumnNormalizer.normalize_asn_column(df)
    df = ColumnNormalizer.normalize_category_columns(df)
    return df


def ensure_toplevel_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing canonical top-level columns (filled with 0)."""
    return ColumnNormalizer.ensure_columns(df, CANONICAL_TOPLEVEL, fill=0)


def ensure_sublevel_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing canonical sub-level columns (filled with 0.0)."""
    return ColumnNormalizer.ensure_columns(df, CANONICAL_SUBLEVEL, fill=0.0)


# ===================================================================
# Migration steps
# ===================================================================


def migrate_labels(legacy: Path, out: Path) -> None:
    """Produce ``labels/toplevel``, ``labels/sublevel``, ``labels/hierarchical``."""
    labels_dir = out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load legacy labeled data ----
    csv_path = legacy / "labeled_data" / "labeled_data.csv"
    if not csv_path.exists():
        log.warning("Legacy labeled data not found at %s – skipping labels", csv_path)
        return
    raw = pd.read_csv(csv_path)
    raw.columns = raw.columns.str.strip()

    # ---- sublevel (binary matrix, canonical column names) ----
    sub = raw.copy()
    sub = sub.rename(columns={"ASN": "asn"})
    sub = sub.drop(columns=["ASName / Org."], errors="ignore")
    sub["asn"] = sub["asn"].apply(ColumnNormalizer.normalize_asn)
    tag_cols = [c for c in sub.columns if c != "asn"]
    for c in tag_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce").fillna(0).astype(int)
    sub = ColumnNormalizer.normalize_category_columns(sub)
    sub = ensure_sublevel_cols(sub)
    # Keep only asn + canonical sublevel columns
    sub = sub[["asn"] + [c for c in CANONICAL_SUBLEVEL if c in sub.columns]]
    save_dual(sub, labels_dir / "sublevel")

    # ---- toplevel (collapsed from sublevel) ----
    from linneaus.models.utils.tag_reduction import simplify_tags

    tag_only = sub.drop(columns=["asn"])
    simplified = simplify_tags(tag_only)
    tl = pd.DataFrame({"asn": sub["asn"].values})
    for cat in CANONICAL_TOPLEVEL:
        if cat in simplified.columns:
            tl[cat] = simplified[cat].values
        else:
            tl[cat] = 0
    # Binarise (max across grouped rows – but labels are already one row per ASN)
    for c in CANONICAL_TOPLEVEL:
        tl[c] = tl[c].clip(0, 1).astype(int)
    save_dual(tl, labels_dir / "toplevel")

    # ---- hierarchical (long format) ----
    from linneaus.data.labeled_data import LabeledDataManager

    mgr = LabeledDataManager(legacy)
    legacy_df = mgr.load_legacy_labels(csv_path)
    hier = mgr.convert_to_hierarchical_format(legacy_df)
    # Normalise ASN column name
    if "ASN" in hier.columns:
        hier = hier.rename(columns={"ASN": "asn"})
    hier["asn"] = hier["asn"].apply(ColumnNormalizer.normalize_asn)
    # Lowercase column names for consistency
    hier = hier.rename(
        columns={
            "ASName": "as_name",
            "TopLevelCategory": "top_level_category",
            "SubCategory": "sub_category",
        }
    )
    save_dual(hier, labels_dir / "hierarchical")

    log.info("Labels migration complete")


def migrate_features(legacy: Path, out: Path) -> None:
    """Produce feature files in ``features/``."""
    feat_dir = out / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    proc = legacy / "processed_data"
    if not proc.exists():
        log.warning("No processed_data/ in legacy – skipping features")
        return

    # ipinfo
    ipinfo_path = proc / "ipinfo.csv"
    if ipinfo_path.exists():
        df = pd.read_csv(ipinfo_path)
        df = ColumnNormalizer.normalize_asn_column(df)
        save_dual(df, feat_dir / "ipinfo")

    # peeringdb (netixlan_netfac_features)
    pdb_path = proc / "netixlan_netfac_features.csv"
    if pdb_path.exists():
        df = pd.read_csv(pdb_path)
        df = ColumnNormalizer.normalize_asn_column(df)
        save_dual(df, feat_dir / "peeringdb")

    # llm_input (llm_features)
    llm_path = proc / "llm_features.csv"
    if llm_path.exists():
        df = pd.read_csv(llm_path)
        df = ColumnNormalizer.normalize_asn_column(df)
        save_dual(df, feat_dir / "llm_input")

    log.info("Features migration complete")


def migrate_predictions_toplevel(legacy: Path, released: Path, local: Path) -> None:
    """Normalise TopLevel prediction files.

    Small files (baseline, finetuned, svc) → *released*, large (complete) → *local*.
    """
    released_dir = released / "predictions" / "toplevel"
    released_dir.mkdir(parents=True, exist_ok=True)
    local_dir = local / "predictions" / "toplevel"
    local_dir.mkdir(parents=True, exist_ok=True)

    src = legacy / "predictions" / "TopLevel"
    if not src.exists():
        log.warning("No predictions/TopLevel/ – skipping")
        return

    for name, filename in [
        ("baseline", "y_preds_baseline.csv"),
        ("finetuned", "y_preds_finetuned.csv"),
        ("svc", "y_preds_svc.csv"),
    ]:
        p = src / filename
        if not p.exists():
            log.warning("  %s not found – skipping", p)
            continue
        df = pd.read_csv(p)
        df = norm(df)
        df = ensure_toplevel_cols(df)
        save_dual(df, released_dir / name)

    # complete → local (large, user-generated)
    complete_path = src / "y_preds_complete.csv"
    if complete_path.exists():
        df = pd.read_csv(complete_path)
        df = norm(df)
        df = ensure_toplevel_cols(df)
        save_dual(df, local_dir / "complete")

    log.info("TopLevel predictions migration complete")


def migrate_predictions_sublevel(legacy: Path, released: Path, local: Path) -> None:
    """Normalise SubLevel prediction files.

    Small files (baseline, finetuned) → *released*, large (complete) → *local*.
    """
    released_dir = released / "predictions" / "sublevel"
    released_dir.mkdir(parents=True, exist_ok=True)
    local_dir = local / "predictions" / "sublevel"
    local_dir.mkdir(parents=True, exist_ok=True)

    src = legacy / "predictions" / "SubLevel"
    if not src.exists():
        log.warning("No predictions/SubLevel/ – skipping")
        return

    for name, filename in [
        ("baseline", "y_preds_baseline.csv"),
        ("finetuned", "y_preds_finetuned.csv"),
    ]:
        p = src / filename
        if not p.exists():
            log.warning("  %s not found – skipping", p)
            continue
        df = pd.read_csv(p)
        df = norm(df)
        df = ensure_sublevel_cols(df)
        save_dual(df, released_dir / name)

    # complete → local (large, user-generated)
    complete_path = src / "y_preds_complete.csv"
    if complete_path.exists():
        df = pd.read_csv(complete_path)
        df = norm(df)
        df = ensure_sublevel_cols(df)
        save_dual(df, local_dir / "complete")

    log.info("SubLevel predictions migration complete")


def migrate_splits(legacy: Path, out: Path) -> None:
    """Derive split assignments from X_train.csv / X_val.csv ASN lists."""
    splits_dir = out / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    tl = legacy / "predictions" / "TopLevel"
    train_path = tl / "X_train.csv"
    val_path = tl / "X_val.csv"

    if not train_path.exists() or not val_path.exists():
        log.warning("X_train.csv or X_val.csv not found – skipping splits")
        return

    train_df = pd.read_csv(train_path, usecols=["asn"])
    train_df = ColumnNormalizer.normalize_asn_column(train_df)
    train_asns = set(train_df["asn"].unique())

    val_df = pd.read_csv(val_path, usecols=["asn"])
    val_df = ColumnNormalizer.normalize_asn_column(val_df)
    val_asns = set(val_df["asn"].unique())

    # Remaining labeled ASNs go to test
    labels_csv = legacy / "labeled_data" / "labeled_data.csv"
    all_asns: set = set()
    if labels_csv.exists():
        lab = pd.read_csv(labels_csv, usecols=["ASN"])
        all_asns = set(lab["ASN"].apply(ColumnNormalizer.normalize_asn).unique())

    test_asns = all_asns - train_asns - val_asns

    rows = (
        [{"asn": a, "split": "train"} for a in sorted(train_asns)]
        + [{"asn": a, "split": "val"} for a in sorted(val_asns)]
        + [{"asn": a, "split": "test"} for a in sorted(test_asns)]
    )
    assignments = pd.DataFrame(rows)
    assignments["asn"] = assignments["asn"].astype(int)
    save_dual(assignments, splits_dir / "assignments")

    log.info(
        "Splits: train=%d, val=%d, test=%d",
        len(train_asns),
        len(val_asns),
        len(test_asns),
    )


def migrate_metrics(legacy: Path, out: Path) -> None:
    """Copy metric text files."""
    metrics_dir = out / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for level_name, level_dir in [("toplevel", "TopLevel"), ("sublevel", "SubLevel")]:
        src = legacy / "predictions" / level_dir
        if not src.exists():
            continue
        for txt in src.glob("metrics_*.txt"):
            variant = txt.stem.replace("metrics_", "")
            dst = metrics_dir / f"{level_name}_{variant}.txt"
            shutil.copy2(txt, dst)
            log.info("  copied %s → %s", txt.name, dst.name)


def write_metadata(out: Path) -> None:
    """Write ``metadata.json`` summarising the clean data directory."""
    meta = {
        "schema_version": "aligned_v1",
        "canonical_toplevel": CANONICAL_TOPLEVEL,
        "canonical_sublevel": CANONICAL_SUBLEVEL,
        "toplevel_count": len(CANONICAL_TOPLEVEL),
        "sublevel_count": len(CANONICAL_SUBLEVEL),
        "migration_date": datetime.now().isoformat(),
    }

    # Add row counts for any parquet files we created
    counts: dict = {}
    for pq in sorted(out.rglob("*.parquet")):
        rel = str(pq.relative_to(out)).replace(".parquet", "")
        try:
            df = pd.read_parquet(pq)
            counts[rel] = {"rows": len(df), "cols": len(df.columns)}
        except Exception:
            pass
    meta["file_counts"] = counts

    meta_path = out / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Wrote metadata.json")


# ===================================================================
# Main
# ===================================================================


def main(root: Path) -> None:
    data_dir = root / "data"
    legacy_dir = root / "data-legacy"
    released = data_dir / "released" / CURRENT_RELEASE
    local = data_dir / "local"

    # Idempotency: skip if already migrated
    if (released / "metadata.json").exists():
        log.info("data/released/%s/ already exists – nothing to do.", CURRENT_RELEASE)
        return

    # Step 1: rename data/ → data-legacy/
    if legacy_dir.exists():
        log.info("data-legacy/ already exists – reusing it as source")
    elif data_dir.exists():
        log.info("Renaming data/ → data-legacy/")
        data_dir.rename(legacy_dir)
    else:
        log.error("Neither data/ nor data-legacy/ found in %s", root)
        sys.exit(1)

    # Step 2: create target directories
    released.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    log.info("Created data/released/%s/ and data/local/", CURRENT_RELEASE)

    # Step 3: run migrations
    migrate_labels(legacy_dir, released)
    migrate_features(legacy_dir, local)
    migrate_predictions_toplevel(legacy_dir, released, local)
    migrate_predictions_sublevel(legacy_dir, released, local)
    migrate_splits(legacy_dir, released)
    migrate_metrics(legacy_dir, released)
    write_metadata(released)

    log.info("Migration complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Project root directory (default: auto-detected from script location)",
    )
    args = parser.parse_args()
    main(args.root)
