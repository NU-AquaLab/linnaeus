"""
LLM inference module for AS classification.

This module handles batch inference using fine-tuned OpenAI models,
including probability extraction and result processing.
"""

import logging
from math import exp
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

logger = logging.getLogger(__name__)


def setup_logging(
    name: str = "Linneaus", level: int = logging.INFO, quiet_libs: bool = True
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


class BatchInferenceProcessor:
    """Handles batch inference processing for AS classification."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        response_format: BaseModel,
        system_prompt: str,
        batch_size: int = 10,
    ):
        """
        Initialize the batch inference processor.

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
        batch_size : int
            Number of organizations to process per batch.
        """
        self.client = client
        self.model = model
        self.response_format = response_format
        self.system_prompt = system_prompt
        self.batch_size = batch_size

        self.logger = setup_logging()

    def process_batches(
        self,
        features_df: pd.DataFrame,
        message_template: str,
        output_columns: List[str],
        extract_probs: bool = False,
        progressive_save_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Process batches of organizations for classification.

        Parameters
        ----------
        features_df : pd.DataFrame
            DataFrame with organization features.
        message_template : str
            Template for formatting input messages.
        output_columns : List[str]
            List of expected output column names.
        extract_probs : bool
            Whether to extract probabilities from logprobs.
        progressive_save_path : Optional[str]
            Path to save intermediate results.

        Returns
        -------
        Tuple[pd.DataFrame, Optional[pd.DataFrame]]
            Predictions DataFrame and optionally probabilities DataFrame.
        """
        # Initialize results DataFrames
        predictions_df = pd.DataFrame(
            0, index=features_df.index, columns=output_columns
        )
        predictions_probs_df = None

        if extract_probs:
            predictions_probs_df = pd.DataFrame(
                0.0, index=features_df.index, columns=output_columns
            )

        # Prepare batches
        batches = self._prepare_batches(features_df, self.batch_size)

        for asn_batch, batch_rows in tqdm(batches, desc="Processing batches"):
            # Build messages for this batch
            messages = self._build_messages(batch_rows, message_template)

            # Send batch request
            responses, logprobs_content = self._send_batch_request(messages)

            # Extract probabilities if requested
            tags_probs = None
            if extract_probs:
                if hasattr(self.response_format, "Tags"):
                    tags = [tag.value for tag in self.response_format.Tags]
                    tags_probs = self._extract_tag_probs(logprobs_content, tags)

            # Populate predictions
            predictions_df, predictions_probs_df = self._populate_predictions(
                predictions_df,
                predictions_probs_df,
                asn_batch,
                responses,
                output_columns,
                tags_probs,
            )

            # Progressive save if requested
            if progressive_save_path:
                predictions_df.to_csv(progressive_save_path, index=True)

        if extract_probs:
            return predictions_df, predictions_probs_df
        else:
            return predictions_df, None

    def _prepare_batches(
        self, features_df: pd.DataFrame, batch_size: int
    ) -> List[Tuple[pd.Index, pd.DataFrame]]:
        """Prepare data batches for processing."""
        features_df = features_df.sample(frac=1, random_state=23)
        return [
            (
                features_df.index[i : i + batch_size],
                features_df.iloc[i : i + batch_size],
            )
            for i in range(0, len(features_df), batch_size)
        ]

    def _build_messages(
        self, batch_rows: pd.DataFrame, message_template: str
    ) -> List[Dict[str, str]]:
        """Build messages for a batch of organizations."""
        return [
            {"role": "user", "content": message_template.format(**row.to_dict())}
            for _, row in batch_rows.iterrows()
        ]

    def _send_batch_request(self, messages: List[Dict[str, str]]) -> Tuple[Any, List]:
        """Send a batch request to the OpenAI API."""
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

    def _extract_tag_probs(self, logprobs_content, tags):
        """Extract tag probabilities from logprobs content."""
        if not logprobs_content:
            return None

        sequence = ""
        top_logprobs = []
        for item in logprobs_content:
            sequence += item.token
            if item.top_logprobs:
                top_logprobs.extend(item.top_logprobs)

        if not sequence or not top_logprobs:
            return None

        tag_probs = {}
        for tag in tags:
            tag_prob = 0.0
            for logprob_item in top_logprobs:
                if tag.lower() in logprob_item.token.lower():
                    tag_prob = max(tag_prob, exp(logprob_item.logprob))
            tag_probs[tag] = tag_prob

        return tag_probs

    def _populate_predictions(
        self,
        predictions_df: pd.DataFrame,
        predictions_probs_df: Optional[pd.DataFrame],
        asn_batch: pd.Index,
        responses: Any,
        output_columns: List[str],
        tags_probs: Optional[Dict[str, float]],
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Populate prediction DataFrames with batch results."""
        if not responses:
            return predictions_df, predictions_probs_df

        for i, (asn, response) in enumerate(zip(asn_batch, responses.responses)):
            if hasattr(response, "tags") and response.tags:
                for tag in response.tags:
                    tag_value = tag.value if hasattr(tag, "value") else str(tag)
                    if tag_value in output_columns:
                        predictions_df.at[asn, tag_value] = 1

            if predictions_probs_df is not None and tags_probs:
                for tag, prob in tags_probs.items():
                    if tag in output_columns:
                        predictions_probs_df.at[asn, tag] = prob

        return predictions_df, predictions_probs_df


# Main function from helpers for compatibility
def process_batches(
    client,
    features_df,
    message_template,
    response_format,
    model,
    developer_instructions,
    output_columns,
    batch: int = 10,
    probs=False,
    progresive_save_file_path=None,
):
    """
    Process batches for classification (compatibility function).

    This function maintains compatibility with the existing codebase.
    """
    processor = BatchInferenceProcessor(
        client=client,
        model=model,
        response_format=response_format,
        system_prompt=developer_instructions,
        batch_size=batch,
    )

    return processor.process_batches(
        features_df=features_df,
        message_template=message_template,
        output_columns=output_columns,
        extract_probs=probs,
        progressive_save_path=progresive_save_file_path,
    )
