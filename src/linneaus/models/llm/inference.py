"""
Hierarchical LLM inference module for AS classification.

This module handles batch inference using fine-tuned OpenAI models with the new
hierarchical data schema, supporting both hierarchical and flat classification approaches.
"""

import logging
from math import exp
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel

from linnaeus.models.hierarchical_schemas import CategoryMapping, TopLevelCategory

logger = logging.getLogger(__name__)


def setup_logging(
    name: str = "Linnaeus", level: int = logging.INFO, quiet_libs: bool = True
) -> logging.Logger:
    """
    Set up a logger for use in Jupyter notebooks and silence noisy third-party libraries.

    Parameters
    ----------
    name : str
        The name of your logger.
    level : int
        Logging level (e.g., logging.DEBUG, logging.INFO).
    quiet_libs : bool
        Whether to suppress logs from OpenAI, httpx, urllib3, etc.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if quiet_libs:
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    return logger


def extract_tag_probs(logprobs_content: List, tags: List[str]) -> List[tuple]:
    """
    Extract tag probabilities from OpenAI logprobs content.

    Scans the token sequence produced by the model to find where each tag
    string begins, and records the token-level probability at that position.

    Parameters
    ----------
    logprobs_content : List
        Logprobs content from OpenAI API response
        (``completion.choices[0].logprobs.content``).
    tags : List[str]
        List of tag names to search for in the output sequence.

    Returns
    -------
    List[tuple]
        List of ``(token, probability)`` tuples for each found tag.
    """
    sequence = ""
    top_logprobs: List[tuple] = []
    for item in logprobs_content:
        token: str = item.token
        sequence += token
        top_logprobs += [(token, exp(item.top_logprobs[0].logprob))] * len(token)

    tags_probs: List[tuple] = []
    for i in range(len(sequence)):
        for tag in tags:
            if sequence[i:].startswith(tag):
                tags_probs.append(top_logprobs[i])

    return tags_probs


class HierarchicalBatchInferenceProcessor:
    """Handles batch inference processing for AS classification with hierarchical schema."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        response_format: BaseModel,
        system_prompt: str,
        approach: str = "hierarchical",
        batch_size: int = 10,
    ):
        """
        Initialize the hierarchical batch inference processor.

        Parameters
        ----------
        client : OpenAI
            OpenAI client instance.
        model : str
            Model to use for inference.
        response_format : BaseModel
            Pydantic model defining the response format.
        system_prompt : str
            System prompt for the model.
        approach : str
            Classification approach: "hierarchical" or "flat".
        batch_size : int
            Number of organizations to process per batch.
        """
        self.client = client
        self.model = model
        self.response_format = response_format
        self.system_prompt = system_prompt
        self.approach = approach
        self.batch_size = batch_size

        # Initialize category mappings
        self.category_mappings = CategoryMapping.get_all_mappings()
        self.top_level_categories = [cat.value for cat in TopLevelCategory]

        self.logger = setup_logging()

    def process_batches_hierarchical(
        self,
        features_df: pd.DataFrame,
        message_template: str,
        extract_probs: bool = False,
        progressive_save_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Process batches for hierarchical classification.

        Parameters
        ----------
        features_df : pd.DataFrame
            DataFrame with organization features.
        message_template : str
            Template for formatting input messages.
        extract_probs : bool
            Whether to extract probabilities from logprobs.
        progressive_save_path : Optional[str]
            Path to save intermediate results.

        Returns
        -------
        Tuple[pd.DataFrame, Optional[pd.DataFrame]]
            Hierarchical DataFrame and optionally probabilities DataFrame.
        """
        self.logger.info(
            f"Starting hierarchical batch processing for {len(features_df)} organizations"
        )

        hierarchical_records = []
        probability_records = [] if extract_probs else None

        # Process in batches
        for i in range(0, len(features_df), self.batch_size):
            batch_df = features_df.iloc[i : i + self.batch_size]
            self.logger.info(
                f"Processing batch {i // self.batch_size + 1}/{len(features_df) // self.batch_size + 1}"
            )

            batch_results = self._process_single_batch_hierarchical(
                batch_df, message_template, extract_probs
            )

            if batch_results:
                hierarchical_records.extend(batch_results["hierarchical"])
                if extract_probs and batch_results.get("probabilities"):
                    probability_records.extend(batch_results["probabilities"])

            # Progressive save if requested
            if progressive_save_path and hierarchical_records:
                temp_df = pd.DataFrame(hierarchical_records)
                temp_df.to_parquet(progressive_save_path, index=False)

        # Convert to DataFrames
        hierarchical_df = (
            pd.DataFrame(hierarchical_records)
            if hierarchical_records
            else pd.DataFrame()
        )
        probabilities_df = (
            pd.DataFrame(probability_records) if probability_records else None
        )

        self.logger.info(f"Processed {len(hierarchical_df)} hierarchical records")

        return hierarchical_df, probabilities_df

    def process_batches_flat(
        self,
        features_df: pd.DataFrame,
        message_template: str,
        extract_probs: bool = False,
        progressive_save_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Process batches for flat classification.

        Parameters
        ----------
        features_df : pd.DataFrame
            DataFrame with organization features.
        message_template : str
            Template for formatting input messages.
        extract_probs : bool
            Whether to extract probabilities from logprobs.
        progressive_save_path : Optional[str]
            Path to save intermediate results.

        Returns
        -------
        Tuple[pd.DataFrame, Optional[pd.DataFrame]]
            Flat predictions DataFrame and optionally probabilities DataFrame.
        """
        self.logger.info(
            f"Starting flat batch processing for {len(features_df)} organizations"
        )

        # Initialize results DataFrames
        predictions_df = pd.DataFrame(
            0, index=features_df.index, columns=self.top_level_categories
        )

        probabilities_df = None
        if extract_probs:
            probabilities_df = pd.DataFrame(
                0.0, index=features_df.index, columns=self.top_level_categories
            )

        # Process in batches
        for i in range(0, len(features_df), self.batch_size):
            batch_df = features_df.iloc[i : i + self.batch_size]
            batch_df.index

            self.logger.info(
                f"Processing batch {i // self.batch_size + 1}/{len(features_df) // self.batch_size + 1}"
            )

            batch_results = self._process_single_batch_flat(
                batch_df, message_template, extract_probs
            )

            if batch_results:
                # Update predictions
                for asn, tags in batch_results["predictions"].items():
                    if asn in predictions_df.index:
                        for tag in tags:
                            if tag in self.top_level_categories:
                                predictions_df.at[asn, tag] = 1

                # Update probabilities if available
                if extract_probs and batch_results.get("probabilities"):
                    for asn, probs in batch_results["probabilities"].items():
                        if asn in probabilities_df.index:
                            for tag, prob in probs.items():
                                if tag in self.top_level_categories:
                                    probabilities_df.at[asn, tag] = prob

            # Progressive save if requested
            if progressive_save_path:
                predictions_df.to_csv(progressive_save_path, index=True)

        self.logger.info(f"Processed {len(predictions_df)} flat predictions")

        return predictions_df, probabilities_df

    def _process_single_batch_hierarchical(
        self, batch_df: pd.DataFrame, message_template: str, extract_probs: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Process a single batch for hierarchical classification."""
        try:
            # Build messages for the batch
            messages = self._build_batch_messages(batch_df, message_template)

            # Send API request
            response, logprobs_content = self._send_batch_request(messages)

            if not response:
                return None

            # Extract hierarchical classifications
            hierarchical_records = []
            probability_records = [] if extract_probs else None

            if hasattr(response, "responses") and response.responses:
                for i, (asn, row) in enumerate(batch_df.iterrows()):
                    if i < len(response.responses):
                        asn_name = row.get("name", f"AS{asn}")
                        response_obj = response.responses[i]

                        if hasattr(response_obj, "classifications"):
                            for classification in response_obj.classifications:
                                hierarchical_records.append(
                                    {
                                        "ASN": asn,
                                        "ASName": asn_name,
                                        "TopLevelCategory": classification.get(
                                            "category"
                                        ),
                                        "SubCategory": classification.get(
                                            "subcategory"
                                        ),
                                    }
                                )

                                # Extract probabilities if requested
                                if extract_probs and probability_records is not None:
                                    # This would need to be implemented based on logprobs
                                    probability_records.append(
                                        {
                                            "ASN": asn,
                                            "TopLevelCategory": classification.get(
                                                "category"
                                            ),
                                            "SubCategory": classification.get(
                                                "subcategory"
                                            ),
                                            "Probability": 1.0,  # Placeholder
                                        }
                                    )

            return {
                "hierarchical": hierarchical_records,
                "probabilities": probability_records,
            }

        except Exception as e:
            self.logger.error(f"Error processing hierarchical batch: {e}")
            return None

    def _process_single_batch_flat(
        self, batch_df: pd.DataFrame, message_template: str, extract_probs: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Process a single batch for flat classification."""
        try:
            # Build messages for the batch
            messages = self._build_batch_messages(batch_df, message_template)

            # Send API request
            response, logprobs_content = self._send_batch_request(messages)

            if not response:
                return None

            # Extract flat classifications
            predictions = {}
            probabilities = {} if extract_probs else None

            if hasattr(response, "responses") and response.responses:
                for i, (asn, row) in enumerate(batch_df.iterrows()):
                    if i < len(response.responses):
                        response_obj = response.responses[i]

                        if hasattr(response_obj, "tags"):
                            predictions[asn] = response_obj.tags

                            # Extract probabilities if requested
                            if extract_probs and probabilities is not None:
                                # This would need to be implemented based on logprobs
                                probabilities[asn] = {
                                    tag: 1.0 for tag in response_obj.tags
                                }

            return {"predictions": predictions, "probabilities": probabilities}

        except Exception as e:
            self.logger.error(f"Error processing flat batch: {e}")
            return None

    def _build_batch_messages(
        self, batch_df: pd.DataFrame, message_template: str
    ) -> List[Dict[str, str]]:
        """Build messages for a batch of organizations."""
        messages = []

        for asn, row in batch_df.iterrows():
            # Create organization data
            org_data = {
                "asn": asn,
                "name": row.get("name", f"AS{asn}"),
                "description": row.get("description", f"Autonomous System {asn}"),
            }

            # Format the message
            try:
                content = message_template.format(**org_data)
            except KeyError as e:
                self.logger.warning(f"Template formatting error for ASN {asn}: {e}")
                content = f"Classify this Autonomous System:\nASN: {asn}\nOrganization: {org_data['name']}"

            messages.append({"role": "user", "content": content})

        return messages

    def _send_batch_request(self, messages: List[Dict[str, str]]) -> Tuple[Any, List]:
        """Send a batch request to the OpenAI API."""
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[{"role": "system", "content": self.system_prompt}] + messages,
                response_format=self.response_format,
                temperature=0,
                logprobs=True,
                top_logprobs=1,
                top_p=1,
            )

            return (
                completion.choices[0].message.parsed,
                completion.choices[0].logprobs.content,
            )

        except Exception as e:
            self.logger.error(f"API request failed: {e}")
            return None, []

    def convert_hierarchical_to_flat(
        self, hierarchical_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert hierarchical results to flat format.

        Parameters
        ----------
        hierarchical_df : pd.DataFrame
            DataFrame with hierarchical classifications.

        Returns
        -------
        pd.DataFrame
            Flat predictions DataFrame.
        """
        if hierarchical_df.empty:
            return pd.DataFrame()

        # Get unique ASNs
        unique_asns = hierarchical_df["ASN"].unique()

        # Initialize flat DataFrame
        flat_df = pd.DataFrame(
            0,
            index=pd.Index(unique_asns, name="ASN"),
            columns=self.top_level_categories,
        )

        # Populate flat format
        for _, row in hierarchical_df.iterrows():
            asn = row["ASN"]
            category = row["TopLevelCategory"]

            if category in self.top_level_categories:
                flat_df.at[asn, category] = 1

        return flat_df

    def validate_hierarchical_predictions(
        self, hierarchical_df: pd.DataFrame
    ) -> List[str]:
        """
        Validate hierarchical predictions.

        Parameters
        ----------
        hierarchical_df : pd.DataFrame
            DataFrame with hierarchical predictions.

        Returns
        -------
        List[str]
            List of validation errors.
        """
        errors = []

        required_columns = ["ASN", "ASName", "TopLevelCategory", "SubCategory"]
        missing_columns = [
            col for col in required_columns if col not in hierarchical_df.columns
        ]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            return errors

        # Validate categories and subcategories
        valid_categories = set(self.top_level_categories)
        invalid_categories = hierarchical_df[
            ~hierarchical_df["TopLevelCategory"].isin(valid_categories)
        ]
        if not invalid_categories.empty:
            errors.append(f"Found {len(invalid_categories)} invalid categories")

        # Validate category-subcategory combinations
        category_subcategory_map = {}
        for mapping in self.category_mappings:
            category_subcategory_map[mapping.category.value] = mapping.subcategories

        for _, row in hierarchical_df.iterrows():
            category = row["TopLevelCategory"]
            subcategory = row["SubCategory"]

            if category in category_subcategory_map:
                valid_subcategories = category_subcategory_map[category]

                if not valid_subcategories and pd.notna(subcategory):
                    errors.append(
                        f"Category {category} should not have subcategory (ASN {row['ASN']})"
                    )
                elif valid_subcategories and pd.isna(subcategory):
                    errors.append(
                        f"Category {category} requires subcategory (ASN {row['ASN']})"
                    )
                elif valid_subcategories and subcategory not in valid_subcategories:
                    errors.append(
                        f"Invalid subcategory {subcategory} for {category} (ASN {row['ASN']})"
                    )

        return errors

    def get_prediction_statistics(
        self, hierarchical_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Get statistics about hierarchical predictions.

        Parameters
        ----------
        hierarchical_df : pd.DataFrame
            DataFrame with hierarchical predictions.

        Returns
        -------
        Dict[str, Any]
            Statistics about the predictions.
        """
        if hierarchical_df.empty:
            return {"total_predictions": 0}

        stats = {
            "total_predictions": len(hierarchical_df),
            "unique_asns": hierarchical_df["ASN"].nunique(),
            "category_counts": hierarchical_df["TopLevelCategory"]
            .value_counts()
            .to_dict(),
            "subcategory_counts": hierarchical_df.dropna(subset=["SubCategory"])[
                "SubCategory"
            ]
            .value_counts()
            .to_dict(),
            "predictions_per_asn": hierarchical_df.groupby("ASN")
            .size()
            .describe()
            .to_dict(),
        }

        return stats


# Compatibility wrapper for existing code
class BatchInferenceProcessor(HierarchicalBatchInferenceProcessor):
    """Backward compatibility wrapper for the hierarchical batch inference processor."""

    def __init__(self, *args, **kwargs):
        # Set default approach to flat for backward compatibility
        if "approach" not in kwargs:
            kwargs["approach"] = "flat"
        super().__init__(*args, **kwargs)

    def process_batches(
        self,
        features_df: pd.DataFrame,
        message_template: str,
        output_columns: List[str],
        extract_probs: bool = False,
        progressive_save_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Legacy method for backward compatibility."""
        if self.approach == "hierarchical":
            hierarchical_df, prob_df = self.process_batches_hierarchical(
                features_df, message_template, extract_probs, progressive_save_path
            )
            # Convert to flat format for compatibility
            flat_df = self.convert_hierarchical_to_flat(hierarchical_df)
            return flat_df, prob_df
        else:
            return self.process_batches_flat(
                features_df, message_template, extract_probs, progressive_save_path
            )


if __name__ == "__main__":
    # Example usage
    pass

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # This would be used with actual OpenAI client and model
    print("Hierarchical inference processor ready for use")
    print("Features:")
    print("- Hierarchical and flat classification support")
    print("- Batch processing with configurable batch size")
    print("- Probability extraction (when supported by model)")
    print("- Validation and statistics")
    print("- Backward compatibility with existing code")
