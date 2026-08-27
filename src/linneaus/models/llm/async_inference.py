"""
Async batch processing module for LLM inference.

This module provides asynchronous batch processing capabilities for improved
performance when processing large numbers of AS organizations.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from math import exp
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)


class AsyncBatchInferenceProcessor:
    """Asynchronous batch inference processor for AS classification."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        response_format: BaseModel,
        system_prompt: str,
        batch_size: int = 10,
        max_concurrent_requests: int = 5,
        rate_limit_delay: float = 1.0,
    ):
        """
        Initialize the async batch inference processor.

        Parameters
        ----------
        client : AsyncOpenAI
            Async OpenAI client instance.
        model : str
            Model to use for inference.
        response_format : BaseModel
            Pydantic model defining the response format.
        system_prompt : str
            System prompt for the model.
        batch_size : int
            Number of organizations to process per batch.
        max_concurrent_requests : int
            Maximum number of concurrent API requests.
        rate_limit_delay : float
            Delay between requests to respect rate limits.
        """
        self.client = client
        self.model = model
        self.response_format = response_format
        self.system_prompt = system_prompt
        self.batch_size = batch_size
        self.max_concurrent_requests = max_concurrent_requests
        self.rate_limit_delay = rate_limit_delay

        # Semaphore to control concurrent requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        logger.info(
            f"Initialized async processor with {max_concurrent_requests} concurrent requests"
        )

    async def process_batches_async(
        self,
        features_df: pd.DataFrame,
        message_template: str,
        output_columns: List[str],
        extract_probs: bool = False,
        progressive_save_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Process batches asynchronously for classification.

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
        logger.info(f"Starting async processing of {len(features_df)} organizations")

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
        logger.info(f"Created {len(batches)} batches for processing")

        # Create tasks for all batches
        tasks = []
        for i, (asn_batch, batch_rows) in enumerate(batches):
            task = self._process_single_batch(
                batch_id=i,
                asn_batch=asn_batch,
                batch_rows=batch_rows,
                message_template=message_template,
                output_columns=output_columns,
                extract_probs=extract_probs,
            )
            tasks.append(task)

        # Process all batches concurrently with progress bar
        results = []
        async for result in tqdm.as_completed(tasks, desc="Processing batches"):
            batch_result = await result
            results.append(batch_result)

            # Progressive save if requested
            if progressive_save_path and batch_result:
                self._update_dataframes(
                    predictions_df, predictions_probs_df, batch_result, output_columns
                )
                predictions_df.to_csv(progressive_save_path, index=True)

        # Update final DataFrames with all results
        for batch_result in results:
            if batch_result:
                self._update_dataframes(
                    predictions_df, predictions_probs_df, batch_result, output_columns
                )

        logger.info("Async batch processing completed")

        if extract_probs:
            return predictions_df, predictions_probs_df
        else:
            return predictions_df, None

    async def _process_single_batch(
        self,
        batch_id: int,
        asn_batch: pd.Index,
        batch_rows: pd.DataFrame,
        message_template: str,
        output_columns: List[str],
        extract_probs: bool,
    ) -> Optional[Dict[str, Any]]:
        """Process a single batch asynchronously."""
        async with self.semaphore:
            try:
                # Add delay to respect rate limits
                if batch_id > 0:
                    await asyncio.sleep(self.rate_limit_delay)

                # Build messages
                messages = self._build_messages(batch_rows, message_template)

                # Send API request
                responses, logprobs_content = await self._send_batch_request_async(
                    messages
                )

                # Extract probabilities if requested
                tags_probs = None
                if extract_probs and hasattr(self.response_format, "Tags"):
                    tags = [tag.value for tag in self.response_format.Tags]
                    tags_probs = self._extract_tag_probs(logprobs_content, tags)

                return {
                    "asn_batch": asn_batch,
                    "responses": responses,
                    "tags_probs": tags_probs,
                }

            except Exception as e:
                logger.error(f"Batch {batch_id} failed: {e}")
                return None

    async def _send_batch_request_async(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[Any, List]:
        """Send an async batch request to the OpenAI API."""
        try:
            completion = await self.client.beta.chat.completions.parse(
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
            logger.error(f"API request failed: {e}")
            # Return empty response in case of error
            return None, []

    def _prepare_batches(
        self, features_df: pd.DataFrame, batch_size: int
    ) -> List[Tuple[pd.Index, pd.DataFrame]]:
        """Prepare data batches for processing."""
        # Shuffle with fixed seed for reproducibility
        features_df = features_df.sample(frac=1, random_state=23)

        batches = []
        for i in range(0, len(features_df), batch_size):
            batch_indices = features_df.index[i : i + batch_size]
            batch_data = features_df.iloc[i : i + batch_size]
            batches.append((batch_indices, batch_data))

        return batches

    def _build_messages(
        self, batch_rows: pd.DataFrame, message_template: str
    ) -> List[Dict[str, str]]:
        """Build messages for a batch of organizations."""
        messages = []
        for _, row in batch_rows.iterrows():
            try:
                content = message_template.format(**row.to_dict())
                messages.append({"role": "user", "content": content})
            except KeyError as e:
                logger.warning(f"Template formatting error for ASN {row.name}: {e}")
                messages.append(
                    {
                        "role": "user",
                        "content": f"Classify this organization with ASN {row.name}",
                    }
                )

        return messages

    def _extract_tag_probs(self, logprobs_content, tags):
        """Extract tag probabilities from logprobs content."""
        if not logprobs_content:
            return None

        try:
            # Reconstruct the sequence and extract probabilities
            sequence = ""
            top_logprobs = []

            for item in logprobs_content:
                if hasattr(item, "token"):
                    sequence += item.token
                if hasattr(item, "top_logprobs") and item.top_logprobs:
                    top_logprobs.extend(item.top_logprobs)

            if not sequence or not top_logprobs:
                return None

            # Extract probabilities for each tag
            tag_probs = {}
            for tag in tags:
                tag_prob = 0.0

                # Look for tag mentions in the logprobs
                for logprob_item in top_logprobs:
                    if (
                        hasattr(logprob_item, "token")
                        and tag.lower() in logprob_item.token.lower()
                    ):
                        if hasattr(logprob_item, "logprob"):
                            tag_prob = max(tag_prob, exp(logprob_item.logprob))

                tag_probs[tag] = tag_prob

            return tag_probs

        except Exception as e:
            logger.warning(f"Error extracting probabilities: {e}")
            return None

    def _update_dataframes(
        self,
        predictions_df: pd.DataFrame,
        predictions_probs_df: Optional[pd.DataFrame],
        batch_result: Dict[str, Any],
        output_columns: List[str],
    ) -> None:
        """Update DataFrames with batch results."""
        asn_batch = batch_result["asn_batch"]
        responses = batch_result["responses"]
        tags_probs = batch_result["tags_probs"]

        if not responses or not hasattr(responses, "responses"):
            return

        try:
            # Process each response in the batch
            for i, (asn, response) in enumerate(zip(asn_batch, responses.responses)):
                if hasattr(response, "tags") and response.tags:
                    # Set predictions to 1 for predicted tags
                    for tag in response.tags:
                        tag_value = tag.value if hasattr(tag, "value") else str(tag)
                        if tag_value in output_columns:
                            predictions_df.at[asn, tag_value] = 1

                # Set probabilities if available
                if predictions_probs_df is not None and tags_probs:
                    for tag, prob in tags_probs.items():
                        if tag in output_columns:
                            predictions_probs_df.at[asn, tag] = prob

        except Exception as e:
            logger.error(f"Error updating predictions: {e}")


class ParallelDataPreparation:
    """Parallel data preparation for improved performance."""

    def __init__(self, max_workers: int = 4):
        """
        Initialize parallel data preparation.

        Parameters
        ----------
        max_workers : int
            Maximum number of worker threads.
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def prepare_training_data_parallel(
        self,
        labeled_df: pd.DataFrame,
        approach: str,
        system_prompt: str,
        message_template: str,
        chunk_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Prepare training data in parallel chunks.

        Parameters
        ----------
        labeled_df : pd.DataFrame
            DataFrame with labeled AS data.
        approach : str
            Classification approach.
        system_prompt : str
            System prompt for the model.
        message_template : str
            Template for formatting messages.
        chunk_size : int
            Size of chunks for parallel processing.

        Returns
        -------
        List[Dict[str, Any]]
            List of training examples.
        """
        logger.info(
            f"Preparing training data in parallel with {self.max_workers} workers"
        )

        # Split data into chunks
        chunks = []
        for i in range(0, len(labeled_df), chunk_size):
            chunk = labeled_df.iloc[i : i + chunk_size]
            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks for processing")

        # Process chunks in parallel
        loop = asyncio.get_event_loop()
        tasks = []

        for chunk in chunks:
            task = loop.run_in_executor(
                self.executor,
                self._process_chunk,
                chunk,
                approach,
                system_prompt,
                message_template,
            )
            tasks.append(task)

        # Wait for all chunks to complete
        chunk_results = await asyncio.gather(*tasks)

        # Combine results
        training_examples = []
        for chunk_result in chunk_results:
            training_examples.extend(chunk_result)

        logger.info(f"Prepared {len(training_examples)} training examples")
        return training_examples

    def _process_chunk(
        self,
        chunk_df: pd.DataFrame,
        approach: str,
        system_prompt: str,
        message_template: str,
    ) -> List[Dict[str, Any]]:
        """Process a single chunk of data."""
        training_examples = []

        for asn, row in chunk_df.iterrows():
            # Get organization data
            org_data = {"asn": asn, "name": f"AS{asn}", "description": ""}

            # Format the user message
            user_message = message_template.format(**org_data)

            # Get expected tags
            expected_tags = []
            for tag_name in row.index:
                if row[tag_name] == 1:
                    expected_tags.append(tag_name)

            # Create assistant response
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

        return training_examples

    def __del__(self):
        """Clean up executor."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)


# Compatibility functions for existing code
async def process_batches_async(
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
    max_concurrent_requests: int = 5,
):
    """
    Async version of process_batches for compatibility.

    This function maintains compatibility with the existing codebase while
    providing async processing capabilities.
    """
    # Convert sync client to async if needed
    if not hasattr(client, "beta"):
        from linneaus.models.llm.client import create_async_client

        async_client = create_async_client(
            api_key=client.api_key,
            base_url=str(client.base_url) if client.base_url else None,
        )
    else:
        async_client = client

    processor = AsyncBatchInferenceProcessor(
        client=async_client,
        model=model,
        response_format=response_format,
        system_prompt=developer_instructions,
        batch_size=batch,
        max_concurrent_requests=max_concurrent_requests,
    )

    return await processor.process_batches_async(
        features_df=features_df,
        message_template=message_template,
        output_columns=output_columns,
        extract_probs=probs,
        progressive_save_path=progresive_save_file_path,
    )


if __name__ == "__main__":
    # Example usage
    import argparse
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run async LLM classification")
    parser.add_argument(
        "--model", required=True, help="Model to use for classification"
    )
    parser.add_argument(
        "--input-data", type=Path, required=True, help="Path to input data CSV/Parquet"
    )
    parser.add_argument(
        "--approach",
        choices=["hierarchical", "flat"],
        default="hierarchical",
        help="Classification approach",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Batch size for processing"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=5, help="Maximum concurrent requests"
    )
    parser.add_argument("--output-path", type=Path, help="Path to save results")

    args = parser.parse_args()

    async def main():
        from openai import AsyncOpenAI

        from linneaus.models.llm.schemas import (
            BatchTaggingResponseFlat,
            BatchTaggingResponseHierarchical,
        )

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Initialize async OpenAI client
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Load input data
        if args.input_data.suffix == ".parquet":
            data_df = pd.read_parquet(args.input_data)
        else:
            data_df = pd.read_csv(args.input_data, index_col=0)

        # Set up response format and tags
        if args.approach == "hierarchical":
            response_format = BatchTaggingResponseHierarchical
            from linneaus.models.llm.schemas import HierarchicalTags

            output_columns = [tag.value for tag in HierarchicalTags]
        else:
            response_format = BatchTaggingResponseFlat
            from linneaus.models.llm.schemas import FlatTags

            output_columns = [tag.value for tag in FlatTags]

        # Initialize processor
        processor = AsyncBatchInferenceProcessor(
            client=client,
            model=args.model,
            response_format=response_format,
            system_prompt="You are an expert in classifying Internet infrastructure organizations.",
            batch_size=args.batch_size,
            max_concurrent_requests=args.max_concurrent,
        )

        # Run async processing
        predictions_df, probabilities_df = await processor.process_batches_async(
            features_df=data_df,
            message_template="Classify ASN {asn}: {name}",
            output_columns=output_columns,
            extract_probs=True,
            progressive_save_path=str(args.output_path) if args.output_path else None,
        )

        print(f"Async processing complete!")
        print(f"Predictions shape: {predictions_df.shape}")
        if probabilities_df is not None:
            print(f"Probabilities shape: {probabilities_df.shape}")

        # Close client
        await client.close()

    # Run async main
    asyncio.run(main())
