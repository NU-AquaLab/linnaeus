#!/usr/bin/env python3
"""Classify Autonomous Systems using an LLM (OpenAI-compatible API).

This standalone script is the recommended way for students to produce
classification predictions.  It uses the LLM directly — no SVM or
fine-tuned models required.

Usage
-----
    python scripts/classify.py --input my_asns.csv --output my_results.csv

The input CSV must contain an ``asn`` column.  Optional columns
``name`` (or ``organization_name``), ``website``, and ``country``
improve accuracy.

The output CSV matches the ground-truth format::

    asn,Access,Transit,Mobile,...,Community
    15169,0,0,0,...,0
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ---------------------------------------------------------------------------
# Resolve package imports – works whether the package is installed or not.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from linneaus.models.llm.client import create_client  # noqa: E402
from linneaus.models.llm.schema_generation import (  # noqa: E402
    generate_developer_instructions,
    generate_schema,
    get_toplevel_tag_names,
    load_tags_descriptions,
)


def build_tag_column_map(columns: list[str]) -> dict[str, str]:
    """Map schema enum values back to output column names.

    Output columns are the sanitized top-level taxonomy names. The schema
    enum uses the same names as members, but renders display values with
    underscores replaced by spaces — map both forms back to the column.
    """
    mapping: dict[str, str] = {}
    for col in columns:
        mapping[col] = col
        mapping[col.replace("_", " ")] = col
    return mapping


DEFAULT_BATCH_SIZE = 5
DEFAULT_MODEL = "gpt-4o-mini"

logger = logging.getLogger("classify")


# ---------------------------------------------------------------------------
# Core classification logic
# ---------------------------------------------------------------------------


def classify_batch(
    client: OpenAI,
    model: str,
    schema,
    system_prompt: str,
    orgs: list[dict],
    tag_to_column: dict[str, str],
) -> dict[int, list[str]]:
    """Send one batch of organisations to the LLM and return predicted tags.

    Returns a mapping ``{asn: [column_name, ...]}``.
    """
    # Build the user message listing all organisations in this batch
    lines: list[str] = []
    for org in orgs:
        asn = org["asn"]
        name = org.get("name", f"AS{asn}")
        extras = []
        if org.get("website"):
            extras.append(org["website"])
        if org.get("country"):
            extras.append(org["country"])
        detail = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"- ASN {asn}: {name}{detail}")
    user_content = "Classify the following Autonomous Systems:\n" + "\n".join(lines)

    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format=schema,
            temperature=0,
        )
        parsed = completion.choices[0].message.parsed
    except Exception as exc:
        logger.error("API call failed: %s", exc)
        return {org["asn"]: [] for org in orgs}

    # Map parsed responses back to ASNs
    results: dict[int, list[str]] = {}
    responses = getattr(parsed, "responses", []) or []
    for i, org in enumerate(orgs):
        asn = org["asn"]
        if i < len(responses):
            resp = responses[i]
            tag_values = [t.value if hasattr(t, "value") else str(t) for t in resp.tags]
            columns = []
            for tv in tag_values:
                col = tag_to_column.get(tv) or tag_to_column.get(tv.replace(" ", "_"))
                if col:
                    columns.append(col)
                else:
                    logger.warning("Unmapped tag %r for ASN %d", tv, asn)
            results[asn] = columns
        else:
            logger.warning("No response for ASN %d (index %d)", asn, i)
            results[asn] = []

    return results


def run_classification(
    input_path: Path,
    output_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    taxonomy_path: str | Path | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> None:
    """Read input CSV, classify each ASN, write output CSV."""
    # --- Load input --------------------------------------------------------
    df = pd.read_csv(input_path)
    if "asn" not in df.columns:
        logger.error("Input CSV must contain an 'asn' column.")
        sys.exit(1)

    orgs: list[dict] = []
    name_col = "name" if "name" in df.columns else "organization_name"
    for _, row in df.iterrows():
        org: dict = {"asn": int(row["asn"])}
        if name_col in df.columns and pd.notna(row.get(name_col)):
            org["name"] = str(row[name_col])
        if "website" in df.columns and pd.notna(row.get("website")):
            org["website"] = str(row["website"])
        if "country" in df.columns and pd.notna(row.get("country")):
            org["country"] = str(row["country"])
        orgs.append(org)

    logger.info("Loaded %d ASNs from %s", len(orgs), input_path)

    # --- Build schema and prompt -------------------------------------------
    # Output columns are the taxonomy's sanitized top-level names; the schema
    # enum is restricted to exactly those names so every predicted tag maps
    # to a column (for any taxonomy: linneaus, asdb, isic, or custom).
    tags_desc = load_tags_descriptions(taxonomy_path)
    output_columns = get_toplevel_tag_names(tags_desc)
    schema = generate_schema(tags_desc, tags=output_columns)
    system_prompt = generate_developer_instructions(tags_desc)
    tag_to_column = build_tag_column_map(output_columns)

    # --- Create client -----------------------------------------------------
    client = create_client(api_key=api_key, base_url=base_url)

    # --- Classify in batches -----------------------------------------------
    all_results: dict[int, list[str]] = {}
    total_batches = (len(orgs) + batch_size - 1) // batch_size

    for i in range(0, len(orgs), batch_size):
        batch = orgs[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(
            "Processing batch %d/%d (%d ASNs)...", batch_num, total_batches, len(batch)
        )

        batch_results = classify_batch(
            client, model, schema, system_prompt, batch, tag_to_column
        )
        all_results.update(batch_results)

        # Small delay between batches to avoid rate-limits
        if batch_num < total_batches:
            time.sleep(0.5)

    # --- Build output DataFrame --------------------------------------------
    out_rows: list[dict] = []
    for org in orgs:
        asn = org["asn"]
        predicted_cols = set(all_results.get(asn, []))
        row: dict = {"asn": asn}
        for col in output_columns:
            row[col] = 1 if col in predicted_cols else 0
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(output_path, index=False)
    logger.info("Results written to %s (%d rows)", output_path, len(out_df))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify Autonomous Systems using an LLM."
    )
    parser.add_argument(
        "--input", "-i", required=True, type=Path, help="Input CSV with 'asn' column"
    )
    parser.add_argument(
        "--output", "-o", required=True, type=Path, help="Output CSV path"
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--taxonomy",
        "-t",
        type=str,
        default=None,
        help=(
            "Builtin taxonomy name (linneaus, asdb, isic) or path to a "
            "custom taxonomy definitions JSON (default: built-in linneaus)"
        ),
    )
    parser.add_argument(
        "--api-key", default=None, help="API key (or set OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--base-url", default=None, help="Base URL for OpenAI-compatible API"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    run_classification(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        batch_size=args.batch_size,
        taxonomy_path=args.taxonomy,
        api_key=args.api_key,
        base_url=args.base_url,
    )


if __name__ == "__main__":
    main()
