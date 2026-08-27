"""
LLM fine-tuning workflow: training data preparation, job execution, inference.

This module contains the supported end-to-end fine-tuning path, promoted from
the validated ``scripts/test_pipeline_e2e.py`` workflow. It is used by the
``linneaus model fine-tune`` / ``linneaus model prepare-data`` CLI commands
and by the e2e test script.

All functions accept an explicit ``tags`` list, so fine-tuning and inference
can be restricted to a subset of categories (e.g. for cheaper validation runs)
or driven by an alternative taxonomy.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd
from openai import OpenAI

from .fine_tuning import OpenAIFineTuner

logger = logging.getLogger(__name__)

DEFAULT_FINE_TUNE_BASE_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_BATCH_SIZE = 1  # 1 = safest; increase for speed
MAX_RETRIES = 3
BASE_DELAY = 2.0  # seconds


def build_user_message(row: pd.Series) -> str:
    """Format a feature row into a user message for classification."""
    parts = ["Classify this Autonomous System:"]
    parts.append(f"ASN: {row['asn']}")
    if pd.notna(row.get("name")):
        parts.append(f"Organization: {row['name']}")
    if pd.notna(row.get("country")):
        parts.append(f"Country: {row['country']}")
    if pd.notna(row.get("website")):
        parts.append(f"Website: {row['website']}")
    return "\n".join(parts)


def prepare_training_jsonl(
    train_labels: pd.DataFrame,
    train_features: pd.DataFrame,
    tags: List[str],
    system_prompt: str,
    output_path: Path,
) -> Tuple[Path, int]:
    """Create OpenAI fine-tuning JSONL from labels and features.

    Rows whose active tags do not intersect ``tags`` are skipped, so passing
    a subset of categories automatically restricts the training set.

    Returns (output_path, example_count).
    """
    merged = train_labels.merge(train_features, on="asn", how="inner")
    logger.info("Merged %d training samples (labels intersect features)", len(merged))

    examples = []
    for _, row in merged.iterrows():
        active_tags = [tag for tag in tags if row.get(tag, 0) == 1]
        if not active_tags:
            continue

        user_msg = build_user_message(row)
        assistant_response = {"responses": [{"tags": active_tags}]}

        examples.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": json.dumps(assistant_response)},
                ]
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    logger.info("Wrote %d training examples to %s", len(examples), output_path)
    return output_path, len(examples)


def run_fine_tuning(
    client: OpenAI,
    training_file: Path,
    suffix: str = "linneaus",
    base_model: str = DEFAULT_FINE_TUNE_BASE_MODEL,
    poll_interval: int = 30,
) -> str:
    """Upload training data, create fine-tuning job, wait for completion.

    Returns the fine-tuned model ID.
    """
    tuner = OpenAIFineTuner(client)

    file_id = tuner.upload_training_file(training_file)
    logger.info("Uploaded training file: %s", file_id)

    job = tuner.create_fine_tuning_job(
        training_file_id=file_id,
        model=base_model,
        suffix=suffix,
    )
    logger.info("Fine-tuning job created: %s (status: %s)", job.id, job.status)

    result = tuner.wait_for_job_completion(job.id, poll_interval=poll_interval)
    model_id = result.fine_tuned_model
    logger.info("Fine-tuning complete: %s", model_id)

    return model_id


def run_inference(
    client: OpenAI,
    model: str,
    val_features: pd.DataFrame,
    system_prompt: str,
    response_format: Any,
    tags: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """Run inference on a feature set.

    Returns a DataFrame with asn + binary tag columns.
    """
    n = len(val_features)
    asns = val_features["asn"].tolist()

    # Pre-allocate results
    results = {tag: [0] * n for tag in tags}
    results["asn"] = asns

    total_batches = (n + batch_size - 1) // batch_size
    start_time = time.time()

    for batch_idx in range(0, n, batch_size):
        batch = val_features.iloc[batch_idx : batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        # Build user messages for this batch
        user_messages = []
        for _, row in batch.iterrows():
            user_messages.append({"role": "user", "content": build_user_message(row)})

        messages = [{"role": "system", "content": system_prompt}] + user_messages

        for attempt in range(MAX_RETRIES + 1):
            try:
                completion = client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    temperature=0,
                )

                parsed = completion.choices[0].message.parsed
                if parsed and hasattr(parsed, "responses"):
                    for j, resp in enumerate(parsed.responses):
                        if j >= len(batch):
                            break
                        row_idx = batch_idx + j
                        if hasattr(resp, "tags"):
                            for tag in resp.tags:
                                # Enum display values render underscores as
                                # spaces; map back to the sanitized tag name.
                                tag_name = (
                                    tag.value if hasattr(tag, "value") else str(tag)
                                )
                                if tag_name not in results:
                                    tag_name = tag_name.replace(" ", "_")
                                if tag_name in results:
                                    results[tag_name][row_idx] = 1
                break  # success

            except Exception as e:
                if attempt < MAX_RETRIES:
                    delay = BASE_DELAY * (2**attempt)
                    logger.warning(
                        "Batch %d/%d attempt %d/%d failed: %s (retrying in %.0fs)",
                        batch_num,
                        total_batches,
                        attempt + 1,
                        MAX_RETRIES + 1,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Batch %d/%d failed after %d attempts: %s",
                        batch_num,
                        total_batches,
                        MAX_RETRIES + 1,
                        e,
                        exc_info=True,
                    )

        if batch_num % 20 == 0 or batch_num == total_batches:
            elapsed = time.time() - start_time
            rate = (batch_idx + len(batch)) / elapsed if elapsed > 0 else 0
            logger.info(
                "Inference: batch %d/%d (%.1f samples/s)",
                batch_num,
                total_batches,
                rate,
            )

    preds = pd.DataFrame(results)
    return preds
