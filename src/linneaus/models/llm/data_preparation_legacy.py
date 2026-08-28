"""
Data preparation module for LLM fine-tuning and inference.

This module handles the conversion of labeled AS data into formats suitable
for OpenAI fine-tuning and batch inference processing.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from linnaeus.data.labeled_data import LabeledDataManager
from linnaeus.models.unified_schemas import HierarchicalTags, TopLevelTags

logger = logging.getLogger(__name__)


class LLMDataPreparation:
    """Handles data preparation for LLM fine-tuning and inference."""

    def __init__(self, data_manager: LabeledDataManager):
        """
        Initialize the LLM data preparation class.

        Parameters
        ----------
        data_manager : LabeledDataManager
            Instance of labeled data manager.
        """
        self.data_manager = data_manager

        # Tag mappings for different classification approaches
        self.hierarchical_tags = [tag.value for tag in HierarchicalTags]
        self.flat_tags = [tag.value for tag in TopLevelTags]

    def prepare_training_data(
        self,
        labeled_df: pd.DataFrame,
        approach: str = "hierarchical",
        system_prompt: str = "",
        message_template: str = "",
        include_examples: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Prepare labeled data for OpenAI fine-tuning format.

        Parameters
        ----------
        labeled_df : pd.DataFrame
            DataFrame with labeled AS data.
        approach : str
            Classification approach: "hierarchical" or "flat".
        system_prompt : str
            System prompt for the model.
        message_template : str
            Template for formatting input messages.
        include_examples : bool
            Whether to include few-shot examples.

        Returns
        -------
        List[Dict[str, Any]]
            List of training examples in OpenAI format.
        """
        logger.info(f"Preparing training data for {approach} approach")
        logger.info(f"Dataset size: {len(labeled_df)}")

        training_examples = []

        for asn, row in labeled_df.iterrows():
            # Get organization data (would need to be enhanced with actual org data)
            org_data = {
                "asn": asn,
                "name": f"AS{asn}",  # Placeholder - would need actual names
                "description": "",  # Placeholder - would need actual descriptions
            }

            # Format the user message
            user_message = self._format_user_message(org_data, message_template)

            # Get the expected tags
            expected_tags = self._get_expected_tags(row, approach)

            # Create the assistant response
            assistant_response = {"responses": [{"tags": expected_tags}]}

            # Create training example
            training_example = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": json.dumps(assistant_response)},
                ]
            }

            training_examples.append(training_example)

        logger.info(f"Created {len(training_examples)} training examples")
        return training_examples

    def _format_user_message(self, org_data: Dict[str, Any], template: str) -> str:
        """
        Format user message using organization data and template.

        Parameters
        ----------
        org_data : Dict[str, Any]
            Organization data dictionary.
        template : str
            Message template string.

        Returns
        -------
        str
            Formatted user message.
        """
        if not template:
            # Default template
            template = "Classify this organization: ASN {asn}, Name: {name}"

        return template.format(**org_data)

    def _get_expected_tags(self, label_row: pd.Series, approach: str) -> List[str]:
        """
        Get expected tags from label row based on approach.

        Parameters
        ----------
        label_row : pd.Series
            Row with label data.
        approach : str
            Classification approach.

        Returns
        -------
        List[str]
            List of expected tag strings.
        """
        expected_tags = []

        if approach == "hierarchical":
            # For hierarchical approach, we'd need to map from flat to hierarchical
            # This is a simplified version - full implementation would need
            # the actual hierarchical labels
            for tag_name in label_row.index:
                if label_row[tag_name] == 1:
                    expected_tags.append(tag_name)
        else:
            # Flat approach
            for tag_name in label_row.index:
                if label_row[tag_name] == 1:
                    expected_tags.append(tag_name)

        return expected_tags

    def save_training_files(
        self,
        training_examples: List[Dict[str, Any]],
        output_dir: Path,
        train_split: float = 0.8,
        filename_prefix: str = "training",
        random_state: int = 42,
    ) -> Tuple[Path, Path]:
        """
        Save training examples to JSONL files for OpenAI fine-tuning.

        Parameters
        ----------
        training_examples : List[Dict[str, Any]]
            Training examples in OpenAI format.
        output_dir : Path
            Output directory for files.
        train_split : float
            Proportion of data for training (rest goes to validation).
        filename_prefix : str
            Prefix for output filenames.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        Tuple[Path, Path]
            Paths to training and validation files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Split into training and validation
        train_examples, val_examples = train_test_split(
            training_examples, train_size=train_split, random_state=random_state
        )

        # Save training file
        train_path = output_dir / f"{filename_prefix}_train.jsonl"
        self._save_jsonl(train_examples, train_path)

        # Save validation file
        val_path = output_dir / f"{filename_prefix}_validation.jsonl"
        self._save_jsonl(val_examples, val_path)

        logger.info(f"Saved {len(train_examples)} training examples to {train_path}")
        logger.info(f"Saved {len(val_examples)} validation examples to {val_path}")

        return train_path, val_path

    def _save_jsonl(self, examples: List[Dict[str, Any]], path: Path) -> None:
        """
        Save examples to JSONL format.

        Parameters
        ----------
        examples : List[Dict[str, Any]]
            List of examples to save.
        path : Path
            Output file path.
        """
        with open(path, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")

    def prepare_inference_batch(
        self,
        organizations: List[Dict[str, Any]],
        message_template: str,
        batch_size: int = 10,
    ) -> List[List[Dict[str, str]]]:
        """
        Prepare organization data for batch inference.

        Parameters
        ----------
        organizations : List[Dict[str, Any]]
            List of organization data dictionaries.
        message_template : str
            Template for formatting messages.
        batch_size : int
            Number of organizations per batch.

        Returns
        -------
        List[List[Dict[str, str]]]
            List of message batches for inference.
        """
        logger.info(f"Preparing {len(organizations)} organizations for inference")
        logger.info(f"Batch size: {batch_size}")

        batches = []

        for i in range(0, len(organizations), batch_size):
            batch_orgs = organizations[i : i + batch_size]
            batch_messages = []

            for org in batch_orgs:
                message = self._format_user_message(org, message_template)
                batch_messages.append({"role": "user", "content": message})

            batches.append(batch_messages)

        logger.info(f"Created {len(batches)} batches for inference")
        return batches

    def validate_training_data(
        self, training_examples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate training data format for OpenAI fine-tuning.

        Parameters
        ----------
        training_examples : List[Dict[str, Any]]
            Training examples to validate.

        Returns
        -------
        Dict[str, Any]
            Validation results and statistics.
        """
        logger.info("Validating training data format")

        validation_results = {
            "total_examples": len(training_examples),
            "valid_examples": 0,
            "invalid_examples": 0,
            "errors": [],
        }

        for i, example in enumerate(training_examples):
            try:
                # Check required structure
                if "messages" not in example:
                    raise ValueError("Missing 'messages' key")

                messages = example["messages"]
                if len(messages) != 3:
                    raise ValueError(
                        "Expected exactly 3 messages (system, user, assistant)"
                    )

                # Check message roles
                expected_roles = ["system", "user", "assistant"]
                for j, msg in enumerate(messages):
                    if msg.get("role") != expected_roles[j]:
                        raise ValueError(f"Message {j} has incorrect role")

                    if "content" not in msg or not msg["content"]:
                        raise ValueError(f"Message {j} missing content")

                # Validate assistant response is valid JSON
                try:
                    json.loads(messages[2]["content"])
                except json.JSONDecodeError as e:
                    raise ValueError(f"Assistant response is not valid JSON: {e}")

                validation_results["valid_examples"] += 1

            except Exception as e:
                validation_results["invalid_examples"] += 1
                validation_results["errors"].append(f"Example {i}: {str(e)}")

        logger.info(
            f"Validation complete: {validation_results['valid_examples']} valid, "
            f"{validation_results['invalid_examples']} invalid examples"
        )

        return validation_results

    def create_prompt_templates(
        self, approach: str = "hierarchical", include_descriptions: bool = True
    ) -> Dict[str, str]:
        """
        Create standard prompt templates for the given approach.

        Parameters
        ----------
        approach : str
            Classification approach: "hierarchical" or "flat".
        include_descriptions : bool
            Whether to include tag descriptions in prompts.

        Returns
        -------
        Dict[str, str]
            Dictionary containing system prompt and message template.
        """
        if approach == "hierarchical":
            system_prompt = self._create_hierarchical_system_prompt(
                include_descriptions
            )
        else:
            system_prompt = self._create_flat_system_prompt(include_descriptions)

        message_template = """Classify this Autonomous System:
ASN: {asn}
Organization Name: {name}
Description: {description}"""

        return {"system_prompt": system_prompt, "message_template": message_template}

    def _create_hierarchical_system_prompt(self, include_descriptions: bool) -> str:
        """Create system prompt for hierarchical classification."""
        prompt = """You are an expert in classifying Internet infrastructure organizations based on their Autonomous System (AS) information.

Your task is to classify organizations using a detailed hierarchical taxonomy. You must analyze the organization's role in the Internet ecosystem and assign appropriate tags.

Classification Guidelines:
- Multiple tags can be assigned if an organization serves multiple functions
- Be specific and use the most detailed applicable subcategory
- Consider the organization's primary business model and services
- Focus on their role in Internet infrastructure

"""

        if include_descriptions:
            prompt += "\nAvailable tags:\n"
            for tag in HierarchicalTags:
                prompt += f"- {tag.value}\n"

        prompt += "\nRespond with a JSON object containing a list of responses, each with a tags array."

        return prompt

    def _create_flat_system_prompt(self, include_descriptions: bool) -> str:
        """Create system prompt for flat classification."""
        prompt = """You are an expert in classifying Internet infrastructure organizations based on their Autonomous System (AS) information.

Your task is to classify organizations using a simplified flat taxonomy. You must analyze the organization's primary role in the Internet ecosystem.

Classification Guidelines:
- Multiple tags can be assigned but prefer the primary function
- Consider the organization's main business model and services
- Focus on their primary role in Internet infrastructure

"""

        if include_descriptions:
            prompt += "\nAvailable categories:\n"
            for tag in TopLevelTags:
                prompt += f"- {tag.value}\n"

        prompt += "\nRespond with a JSON object containing a list of responses, each with a tags array."

        return prompt


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Prepare LLM training data")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="Root data directory"
    )
    parser.add_argument(
        "--approach",
        choices=["hierarchical", "flat"],
        default="hierarchical",
        help="Classification approach",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/llm_training"),
        help="Output directory for training files",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize data manager and preparation
    data_manager = LabeledDataManager(args.data_dir)
    prep = LLMDataPreparation(data_manager)

    # Load labeled data
    labeled_df = data_manager.load_consolidated_labels()

    # Create prompt templates
    templates = prep.create_prompt_templates(args.approach)

    # Prepare training data
    training_examples = prep.prepare_training_data(
        labeled_df,
        approach=args.approach,
        system_prompt=templates["system_prompt"],
        message_template=templates["message_template"],
    )

    # Validate training data
    validation_results = prep.validate_training_data(training_examples)
    print(f"Validation results: {validation_results}")

    # Save training files
    train_path, val_path = prep.save_training_files(training_examples, args.output_dir)

    print(f"Training data saved to:")
    print(f"  Training: {train_path}")
    print(f"  Validation: {val_path}")
