#!/usr/bin/env python3
"""End-to-end pipeline test: fine-tune GPT-4o-mini + inference + metrics comparison.

Verifies the complete LLM classification workflow after the data restructuring
refactor (data/released/202506/ + data/local/):

1. Load data via AlignedDataset -- labels, splits, features
2. Prepare training JSONL -- convert labeled data to OpenAI fine-tuning format
3. Fine-tune GPT-4o-mini -- upload JSONL, create job, wait for completion
4. Run inference -- baseline (gpt-4o-mini) and fine-tuned model on validation set
5. Compute metrics -- accuracy, precision, recall, F1, Jaccard
6. Compare -- print side-by-side with existing baselines from
   data/released/202506/metrics/

Usage:
    # Full run (fine-tune + inference + compare, ~20-30 min)
    python scripts/test_pipeline_e2e.py

    # Reuse an existing fine-tuned model
    python scripts/test_pipeline_e2e.py --skip-finetune --model-id ft:gpt-4o-mini:...

    # Only run baseline inference (no fine-tuning)
    python scripts/test_pipeline_e2e.py --baseline-only

    # Use larger batches for faster inference
    python scripts/test_pipeline_e2e.py --batch-size 10 \
        --skip-finetune --model-id ft:...
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd
from openai import OpenAI

from linneaus.data.alignment import CANONICAL_TOPLEVEL, AlignedDataset
from linneaus.models.llm.finetuning import (
    DEFAULT_BATCH_SIZE,
    prepare_training_jsonl,
    run_fine_tuning,
    run_inference,
)
from linneaus.models.llm.schema_generation import (
    filter_taxonomy,
    generate_developer_instructions,
    generate_schema,
    load_tags_descriptions,
)
from linneaus.models.utils.metrics import get_global_metrics, get_metrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("e2e_pipeline")

for _lib in ("openai", "httpx", "urllib3"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

FOCUS_CATEGORIES = ["Government", "Access", "Enterprise", "ContentProvider"]
FEATURES_PATH = Path("data/local/features/llm_input.parquet")


# ---------------------------------------------------------------------------
# Connectivity check
# ---------------------------------------------------------------------------


def check_connectivity(client: OpenAI) -> None:
    """Verify API key and network connectivity before starting pipeline."""
    logger.info("Checking OpenAI API connectivity...")
    try:
        client.models.list()
        logger.info("Connectivity check passed")
    except Exception as e:
        err_msg = str(e)
        if "Illegal header" in err_msg or "LocalProtocolError" in err_msg:
            logger.error(
                "API key contains invalid characters. Check OPENAI_API_KEY "
                "for embedded whitespace, newlines, or quotes."
            )
        logger.error("Connectivity check failed: %s", e, exc_info=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_pipeline_data(
    data_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load labels, splits, and features.

    Returns (train_labels, val_labels, train_features, val_features).
    """
    ds = AlignedDataset(data_dir)
    labels = ds.load_labels("toplevel")
    splits = ds.load_splits()

    train_asns = set(splits[splits["split"] == "train"]["asn"])
    val_asns = set(splits[splits["split"] == "val"]["asn"])

    train_labels = labels[labels["asn"].isin(train_asns)].copy()
    val_labels = labels[labels["asn"].isin(val_asns)].copy()

    # Features live in data/local/features/ (not in the released snapshot)
    if not FEATURES_PATH.exists():
        logger.error("Features file not found: %s", FEATURES_PATH)
        logger.error("Run 'linneaus data download' or check data/local/features/.")
        sys.exit(1)

    features = pd.read_parquet(FEATURES_PATH)
    train_features = features[features["asn"].isin(train_asns)].copy()
    val_features = features[features["asn"].isin(val_asns)].copy()

    logger.info(
        "Data loaded: %d train labels, %d val labels, "
        "%d train features, %d val features",
        len(train_labels),
        len(val_labels),
        len(train_features),
        len(val_features),
    )

    return train_labels, val_labels, train_features, val_features


# ---------------------------------------------------------------------------
# Stored metrics parsing
# ---------------------------------------------------------------------------


def compute_reference_metrics(
    ds: AlignedDataset, tags: List[str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Recompute reference metrics for a category subset from the committed
    pre-refactor prediction files.

    The stored ``metrics/*.txt`` files cover the full 20-category run with
    legacy label names, so for a restricted-tags run the fair comparison is
    to re-score the committed baseline/finetuned val-split predictions on
    just ``tags``.

    Returns (baseline_metrics, finetuned_metrics) in ``get_metrics`` format.
    """
    labels = ds.load_labels("toplevel")
    splits = ds.load_splits()
    val_asns = set(splits[splits["split"] == "val"]["asn"])

    refs = {}
    for model_name in ("baseline", "finetuned"):
        preds = ds.load_predictions("toplevel", model_name)
        common = val_asns & set(preds["asn"]) & set(labels["asn"])
        l_df = (
            labels[labels["asn"].isin(common)].sort_values("asn").reset_index(drop=True)
        )
        p_df = (
            preds[preds["asn"].isin(common)].sort_values("asn").reset_index(drop=True)
        )
        refs[model_name] = get_metrics(l_df[tags], p_df[tags], print_results=False)
        logger.info(
            "Reference %s (recomputed on %d val ASNs, %d tags): macro F1 = %.4f",
            model_name,
            len(common),
            len(tags),
            refs[model_name]["macro_f1"],
        )

    return refs["baseline"], refs["finetuned"]


def parse_stored_metrics(metrics_path: Path) -> Dict[str, float]:
    """Parse a stored metrics text file into a dict of scalar metrics."""
    metrics: Dict[str, float] = {}
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if ":" in line and not line.startswith("="):
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if value != "None":
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        pass
    return metrics


# ---------------------------------------------------------------------------
# Metrics computation and comparison
# ---------------------------------------------------------------------------


def compute_and_compare(
    val_labels: pd.DataFrame,
    baseline_preds: pd.DataFrame,
    finetuned_preds: Optional[pd.DataFrame],
    stored_metrics_dir: Path,
    tags: List[str],
    train_count: int,
    finetuned_model_id: Optional[str] = None,
    reference_metrics: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute metrics, print comparison table, return results dict.

    When ``reference_metrics`` is provided (restricted-tags runs), it is used
    as the stored baseline/finetuned comparison instead of the full-run
    ``metrics/*.txt`` files.
    """
    # Align labels and predictions on ASN
    common_asns_bl = set(val_labels["asn"]) & set(baseline_preds["asn"])
    logger.info(
        "Aligning predictions: %d val labels, %d baseline preds, %d common ASNs",
        len(val_labels),
        len(baseline_preds),
        len(common_asns_bl),
    )

    labels_bl = (
        val_labels[val_labels["asn"].isin(common_asns_bl)]
        .sort_values("asn")
        .reset_index(drop=True)
    )
    preds_bl = (
        baseline_preds[baseline_preds["asn"].isin(common_asns_bl)]
        .sort_values("asn")
        .reset_index(drop=True)
    )

    y_true_bl = labels_bl[tags]
    y_pred_bl = preds_bl[tags]

    logger.info("Computing baseline metrics...")
    baseline_metrics = get_metrics(y_true_bl, y_pred_bl, print_results=False)

    finetuned_metrics = None
    if finetuned_preds is not None:
        common_asns_ft = set(val_labels["asn"]) & set(finetuned_preds["asn"])
        labels_ft = (
            val_labels[val_labels["asn"].isin(common_asns_ft)]
            .sort_values("asn")
            .reset_index(drop=True)
        )
        preds_ft = (
            finetuned_preds[finetuned_preds["asn"].isin(common_asns_ft)]
            .sort_values("asn")
            .reset_index(drop=True)
        )
        y_true_ft = labels_ft[tags]
        y_pred_ft = preds_ft[tags]

        logger.info("Computing fine-tuned metrics...")
        finetuned_metrics = get_metrics(y_true_ft, y_pred_ft, print_results=False)

    # Load stored baselines (or subset-aware recomputed references)
    if reference_metrics is not None:
        stored_baseline = get_global_metrics(reference_metrics[0])
        stored_finetuned = get_global_metrics(reference_metrics[1])
    else:
        stored_baseline = parse_stored_metrics(
            stored_metrics_dir / "toplevel_baseline.txt"
        )
        stored_finetuned = parse_stored_metrics(
            stored_metrics_dir / "toplevel_finetuned.txt"
        )

    # ---------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  End-to-End Pipeline Test Results")
    print("=" * 70)

    if finetuned_model_id:
        print(f"\nFine-tuned model: {finetuned_model_id}")
    print(
        f"Training samples: {train_count}    Validation samples: {len(common_asns_bl)}"
    )

    # --- Global metrics ---
    print("\n--- Global Metrics ---")
    has_ft = finetuned_metrics is not None
    header = f"{'Metric':<18} {'Baseline':>12}"
    if has_ft:
        header += f" {'Finetuned':>12}"
    header += f" {'Stored-BL':>12} {'Stored-FT':>12}"
    print(header)
    print("-" * len(header))

    metric_keys = [
        ("accuracy", "Accuracy"),
        ("macro_precision", "Precision"),
        ("macro_recall", "Recall"),
        ("macro_f1", "F1"),
        ("macro_jaccard", "Jaccard"),
    ]
    for key, label in metric_keys:
        bl_val = baseline_metrics.get(key, 0)
        line = f"{label:<18} {bl_val:>11.4f}%"  # not actually percent, fix below

        # Actually format as decimals, not percent
        line = f"{label:<18} {bl_val:>12.4f}"
        if has_ft:
            ft_val = finetuned_metrics.get(key, 0)
            line += f" {ft_val:>12.4f}"
        line += f" {stored_baseline.get(key, 0):>12.4f}"
        line += f" {stored_finetuned.get(key, 0):>12.4f}"
        print(line)

    # --- Focus category breakdown ---
    print("\n--- Focus Category Breakdown (F1 Score) ---")
    bl_label_df = baseline_metrics["label_metrics"]
    header = f"{'Category':<28} {'Baseline':>10}"
    if has_ft:
        ft_label_df = finetuned_metrics["label_metrics"]
        header += f" {'Finetuned':>10} {'Delta':>10}"
    print(header)
    print("-" * len(header))

    breakdown = tags if set(tags) != set(CANONICAL_TOPLEVEL) else FOCUS_CATEGORIES
    for cat in breakdown:
        bl_row = bl_label_df[bl_label_df["Label"] == cat]
        bl_f1 = bl_row["F1 Score"].values[0] if len(bl_row) > 0 else 0.0
        line = f"{cat:<28} {bl_f1:>10.4f}"
        if has_ft:
            ft_row = ft_label_df[ft_label_df["Label"] == cat]
            ft_f1 = ft_row["F1 Score"].values[0] if len(ft_row) > 0 else 0.0
            delta = ft_f1 - bl_f1
            sign = "+" if delta >= 0 else ""
            line += f" {ft_f1:>10.4f} {sign}{delta:>9.4f}"
        print(line)

    # --- Verdict ---
    print("\n--- Verdict ---")
    results: Dict[str, Any] = {"baseline": baseline_metrics}

    if has_ft:
        ft_better = finetuned_metrics["macro_f1"] > baseline_metrics["macro_f1"]
        icon = "PASS" if ft_better else "WARN"
        verb = "improves" if ft_better else "does NOT improve"
        print(f"[{icon}] Fine-tuned model {verb} over baseline (expected behavior)")
        results["finetuned"] = finetuned_metrics

    bl_delta = abs(baseline_metrics["macro_f1"] - stored_baseline.get("macro_f1", 0))
    in_range = bl_delta < 0.05
    icon = "PASS" if in_range else "WARN"
    pct = "<5%" if in_range else ">=5%"
    print(f"[{icon}] Baseline F1 delta from stored: {bl_delta:.4f} ({pct})")

    if has_ft:
        ft_delta = abs(
            finetuned_metrics["macro_f1"] - stored_finetuned.get("macro_f1", 0)
        )
        in_range = ft_delta < 0.05
        icon = "PASS" if in_range else "WARN"
        pct = "<5%" if in_range else ">=5%"
        print(f"[{icon}] Fine-tuned F1 delta from stored: {ft_delta:.4f} ({pct})")

    print()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end pipeline test: fine-tune + inference + metrics comparison"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skip-finetune",
        action="store_true",
        help="Skip fine-tuning step; requires --model-id",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Fine-tuned model ID to reuse (with --skip-finetune)",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only run baseline inference, skip fine-tuning entirely",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"ASNs per inference request (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="e2e-test",
        help="Suffix for the fine-tuned model name (default: e2e-test)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override released data directory (default: data/released/202506)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/local/e2e_results",
        help="Directory to save results (default: data/local/e2e_results)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit validation samples for quick testing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run everything except actual API calls; generates zero predictions",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of top-level categories to restrict "
            "training, prompt, schema, and evaluation to "
            "(e.g. 'Government,Access'). Default: all 20 categories."
        ),
    )
    parser.add_argument(
        "--focus",
        action="store_true",
        help=(
            "Shortcut for --tags with the focus categories: "
            + ",".join(FOCUS_CATEGORIES)
        ),
    )
    args = parser.parse_args()

    # --- Validate args ---
    if args.skip_finetune and not args.model_id:
        parser.error("--skip-finetune requires --model-id")

    if not args.dry_run:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            print("ERROR: Set OPENAI_API_KEY environment variable", file=sys.stderr)
            sys.exit(1)

        # Remove any internal whitespace (bad copy-paste from dashboards)
        cleaned = re.sub(r"\s+", "", api_key)
        if cleaned != api_key:
            logger.warning(
                "OPENAI_API_KEY contains embedded whitespace (newlines/spaces) — "
                "this usually means the key was copy-pasted incorrectly. "
                "Stripping whitespace and continuing."
            )
            api_key = cleaned

        client = OpenAI(
            api_key=api_key,
            timeout=httpx.Timeout(180.0, connect=10.0),
        )
        check_connectivity(client)
    else:
        client = None  # type: ignore[assignment]
        logger.info("Dry-run mode: skipping all API calls")

    if args.tags and args.focus:
        parser.error("Use either --tags or --focus, not both")
    if args.focus:
        tags = list(FOCUS_CATEGORIES)
    elif args.tags:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        unknown = [t for t in tags if t not in CANONICAL_TOPLEVEL]
        if unknown:
            parser.error(f"Unknown categories: {unknown}. Valid: {CANONICAL_TOPLEVEL}")
    else:
        tags = CANONICAL_TOPLEVEL
    restricted = set(tags) != set(CANONICAL_TOPLEVEL)
    if restricted:
        logger.info("Restricted run: %d categories (%s)", len(tags), ", ".join(tags))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_start = time.time()

    # =================================================================
    # Step 1: Load data
    # =================================================================
    logger.info("Step 1/6: Loading data...")
    train_labels, val_labels, train_features, val_features = load_pipeline_data(
        args.data_dir
    )

    # Optionally limit validation samples for quick testing
    if args.max_samples and args.max_samples < len(val_features):
        val_features = val_features.head(args.max_samples)
        val_asns = set(val_features["asn"])
        val_labels = val_labels[val_labels["asn"].isin(val_asns)]
        logger.info("Limited to %d validation samples", args.max_samples)

    # =================================================================
    # Step 2: Build schema and system prompt
    # =================================================================
    logger.info("Step 2/6: Building schema and system prompt...")
    tags_descriptions = load_tags_descriptions()
    if restricted:
        # Describe only the categories the schema can emit — keeps the prompt
        # consistent with the restricted enum and cuts token cost ~proportionally
        tags_descriptions = filter_taxonomy(tags_descriptions, tags)
    system_prompt = generate_developer_instructions(tags_descriptions)
    response_format = generate_schema(tags_descriptions, tags=tags)

    tag_enum_values = sorted(m.value for m in response_format.Tags)
    logger.info("Schema has %d tags: %s", len(tag_enum_values), tag_enum_values[:5])

    # =================================================================
    # Step 3: Prepare training data
    # =================================================================
    train_count = 0
    training_file = output_dir / "training.jsonl"

    if not args.baseline_only:
        logger.info("Step 3/6: Preparing training JSONL...")
        training_file, train_count = prepare_training_jsonl(
            train_labels, train_features, tags, system_prompt, training_file
        )
    else:
        logger.info("Step 3/6: Skipped (--baseline-only)")
        train_count = len(train_labels)

    # =================================================================
    # Step 4: Fine-tune
    # =================================================================
    finetuned_model_id = None

    if args.dry_run:
        logger.info("Step 4/6: Skipped (--dry-run)")
    elif args.baseline_only:
        logger.info("Step 4/6: Skipped (--baseline-only)")
    elif args.skip_finetune:
        finetuned_model_id = args.model_id
        logger.info(
            "Step 4/6: Skipping fine-tuning, using model: %s", finetuned_model_id
        )
    else:
        logger.info("Step 4/6: Fine-tuning GPT-4o-mini (this takes ~10-20 min)...")
        finetuned_model_id = run_fine_tuning(client, training_file, suffix=args.suffix)
        logger.info("Fine-tuned model ID: %s", finetuned_model_id)
        # Save model ID for reuse
        with open(output_dir / "finetuned_model_id.txt", "w") as f:
            f.write(finetuned_model_id + "\n")

    # =================================================================
    # Step 5: Run inference
    # =================================================================
    if args.dry_run:
        logger.info("Step 5/6: Generating zero predictions (--dry-run)")
        zero_data = {"asn": val_features["asn"].tolist()}
        for tag in tags:
            zero_data[tag] = [0] * len(val_features)
        baseline_preds = pd.DataFrame(zero_data)
        baseline_preds.to_parquet(output_dir / "baseline_preds.parquet", index=False)
        finetuned_preds = None
    else:
        logger.info(
            "Step 5/6: Running baseline inference on %d samples...", len(val_features)
        )
        baseline_preds = run_inference(
            client,
            "gpt-4o-mini",
            val_features,
            system_prompt,
            response_format,
            tags,
            batch_size=args.batch_size,
        )
        baseline_preds.to_parquet(output_dir / "baseline_preds.parquet", index=False)
        logger.info("Baseline predictions saved")

        finetuned_preds = None
        if finetuned_model_id:
            logger.info(
                "Step 5b/6: Running fine-tuned inference on %d samples...",
                len(val_features),
            )
            finetuned_preds = run_inference(
                client,
                finetuned_model_id,
                val_features,
                system_prompt,
                response_format,
                tags,
                batch_size=args.batch_size,
            )
            finetuned_preds.to_parquet(
                output_dir / "finetuned_preds.parquet", index=False
            )
            logger.info("Fine-tuned predictions saved")

    # =================================================================
    # Step 6: Compute and compare metrics
    # =================================================================
    logger.info("Step 6/6: Computing and comparing metrics...")
    ds = AlignedDataset(args.data_dir)
    stored_metrics_dir = ds.data_dir / "metrics"

    # For restricted runs the stored full-run metrics are not comparable;
    # recompute references for the chosen tags from committed predictions.
    reference_metrics = compute_reference_metrics(ds, tags) if restricted else None

    results = compute_and_compare(
        val_labels,
        baseline_preds,
        finetuned_preds,
        stored_metrics_dir,
        tags,
        train_count,
        finetuned_model_id,
        reference_metrics=reference_metrics,
    )

    # Save summary
    results_summary: Dict[str, Any] = {}
    for name, m in results.items():
        results_summary[name] = get_global_metrics(m)
    if finetuned_model_id:
        results_summary["finetuned_model_id"] = finetuned_model_id

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_summary, f, indent=2, default=str)

    elapsed = time.time() - pipeline_start
    logger.info("Pipeline complete in %.1f minutes", elapsed / 60)
    logger.info("Results saved to %s/", output_dir)


if __name__ == "__main__":
    main()
