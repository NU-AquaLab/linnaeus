"""
Fine-tuning module for OpenAI models.

This module handles the complete fine-tuning workflow including:
- Model fine-tuning job creation and management
- Job monitoring and status tracking
- Model evaluation and deployment
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openai.types.fine_tuning import FineTuningJob

logger = logging.getLogger(__name__)


class OpenAIFineTuner:
    """Handles OpenAI model fine-tuning operations."""

    def __init__(
        self, client: OpenAI, max_concurrent_jobs: int = 3, daily_limit: int = 10
    ):
        """
        Initialize the fine-tuner.

        Parameters
        ----------
        client : OpenAI
            OpenAI client instance.
        max_concurrent_jobs : int
            Maximum number of concurrent fine-tuning jobs.
        daily_limit : int
            Daily limit for new fine-tuning jobs.
        """
        self.client = client
        self.max_concurrent_jobs = max_concurrent_jobs
        self.daily_limit = daily_limit

        # Track job submissions today
        self._daily_job_count = 0
        self._last_reset_date = datetime.now().date()

    def _reset_daily_counter_if_needed(self) -> None:
        """Reset daily job counter if it's a new day."""
        today = datetime.now().date()
        if today != self._last_reset_date:
            self._daily_job_count = 0
            self._last_reset_date = today

    def _check_daily_limit(self) -> bool:
        """
        Check if we can submit more jobs today.

        Returns
        -------
        bool
            True if under daily limit, False otherwise.
        """
        self._reset_daily_counter_if_needed()
        return self._daily_job_count < self.daily_limit

    def upload_training_file(self, file_path: Path) -> str:
        """
        Upload training file to OpenAI.

        Parameters
        ----------
        file_path : Path
            Path to the training file (JSONL format).

        Returns
        -------
        str
            File ID from OpenAI.
        """
        logger.info(f"Uploading training file: {file_path}")

        with open(file_path, "rb") as file:
            response = self.client.files.create(file=file, purpose="fine-tune")

        logger.info(f"File uploaded successfully: {response.id}")
        return response.id

    def create_fine_tuning_job(
        self,
        training_file_id: str,
        model: str = "gpt-4o-mini-2024-07-18",
        validation_file_id: Optional[str] = None,
        suffix: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> FineTuningJob:
        """
        Create a fine-tuning job.

        Parameters
        ----------
        training_file_id : str
            OpenAI file ID for training data.
        model : str
            Base model to fine-tune.
        validation_file_id : Optional[str]
            OpenAI file ID for validation data.
        suffix : Optional[str]
            Suffix for the fine-tuned model name.
        hyperparameters : Optional[Dict[str, Any]]
            Hyperparameters for fine-tuning.

        Returns
        -------
        FineTuningJob
            Fine-tuning job object.
        """
        if not self._check_daily_limit():
            raise RuntimeError(f"Daily limit of {self.daily_limit} jobs reached")

        logger.info("Creating fine-tuning job")
        logger.info(f"Base model: {model}")
        logger.info(f"Training file: {training_file_id}")

        job_params = {"training_file": training_file_id, "model": model}

        if validation_file_id:
            job_params["validation_file"] = validation_file_id
            logger.info(f"Validation file: {validation_file_id}")

        if suffix:
            job_params["suffix"] = suffix
            logger.info(f"Model suffix: {suffix}")

        if hyperparameters:
            job_params["hyperparameters"] = hyperparameters
            logger.info(f"Hyperparameters: {hyperparameters}")

        job = self.client.fine_tuning.jobs.create(**job_params)

        self._daily_job_count += 1
        logger.info(f"Fine-tuning job created: {job.id}")
        logger.info(f"Status: {job.status}")

        return job

    def get_job_status(self, job_id: str) -> FineTuningJob:
        """
        Get the status of a fine-tuning job.

        Parameters
        ----------
        job_id : str
            Fine-tuning job ID.

        Returns
        -------
        FineTuningJob
            Fine-tuning job object.
        """
        return self.client.fine_tuning.jobs.retrieve(job_id)

    def list_active_jobs(self) -> List[FineTuningJob]:
        """
        List all active fine-tuning jobs.

        Returns
        -------
        List[FineTuningJob]
            List of active fine-tuning jobs.
        """
        jobs = self.client.fine_tuning.jobs.list()
        active_jobs = [
            job
            for job in jobs.data
            if job.status in ["validating_files", "queued", "running"]
        ]
        return active_jobs

    def can_start_new_job(self) -> bool:
        """
        Check if we can start a new fine-tuning job.

        Returns
        -------
        bool
            True if we can start a new job, False otherwise.
        """
        if not self._check_daily_limit():
            return False

        active_jobs = self.list_active_jobs()
        return len(active_jobs) < self.max_concurrent_jobs

    def wait_for_job_completion(
        self, job_id: str, poll_interval: int = 60, timeout_hours: int = 24
    ) -> FineTuningJob:
        """
        Wait for a fine-tuning job to complete.

        Parameters
        ----------
        job_id : str
            Fine-tuning job ID.
        poll_interval : int
            Polling interval in seconds.
        timeout_hours : int
            Timeout in hours.

        Returns
        -------
        FineTuningJob
            Completed fine-tuning job.
        """
        logger.info(f"Waiting for job completion: {job_id}")

        start_time = datetime.now()
        timeout = timedelta(hours=timeout_hours)

        while datetime.now() - start_time < timeout:
            job = self.get_job_status(job_id)

            logger.info(f"Job {job_id} status: {job.status}")

            if job.status == "succeeded":
                logger.info(f"Job completed successfully: {job.fine_tuned_model}")
                return job
            elif job.status == "failed":
                logger.error(f"Job failed: {job.error}")
                raise RuntimeError(f"Fine-tuning job failed: {job.error}")
            elif job.status == "cancelled":
                logger.warning("Job was cancelled")
                raise RuntimeError("Fine-tuning job was cancelled")

            time.sleep(poll_interval)

        raise TimeoutError(
            f"Job {job_id} did not complete within {timeout_hours} hours"
        )

    def cancel_job(self, job_id: str) -> FineTuningJob:
        """
        Cancel a fine-tuning job.

        Parameters
        ----------
        job_id : str
            Fine-tuning job ID.

        Returns
        -------
        FineTuningJob
            Cancelled fine-tuning job.
        """
        logger.info(f"Cancelling job: {job_id}")
        job = self.client.fine_tuning.jobs.cancel(job_id)
        logger.info(f"Job cancelled: {job.status}")
        return job

    def get_job_events(self, job_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get events for a fine-tuning job.

        Parameters
        ----------
        job_id : str
            Fine-tuning job ID.
        limit : int
            Maximum number of events to retrieve.

        Returns
        -------
        List[Dict[str, Any]]
            List of job events.
        """
        events = self.client.fine_tuning.jobs.list_events(job_id, limit=limit)
        return [event.model_dump() for event in events.data]

    def delete_file(self, file_id: str) -> bool:
        """
        Delete an uploaded file.

        Parameters
        ----------
        file_id : str
            File ID to delete.

        Returns
        -------
        bool
            True if deletion was successful.
        """
        logger.info(f"Deleting file: {file_id}")
        response = self.client.files.delete(file_id)
        success = response.deleted
        logger.info(f"File deletion {'successful' if success else 'failed'}")
        return success

    def estimate_cost(
        self,
        training_examples: int,
        validation_examples: int = 0,
        model: str = "gpt-4o-mini-2024-07-18",
    ) -> Dict[str, float]:
        """
        Estimate the cost of fine-tuning.

        Parameters
        ----------
        training_examples : int
            Number of training examples.
        validation_examples : int
            Number of validation examples.
        model : str
            Base model for fine-tuning.

        Returns
        -------
        Dict[str, float]
            Cost estimation breakdown.
        """
        # OpenAI pricing as of 2024 (subject to change)
        if model.startswith("gpt-4o-mini"):
            cost_per_1k_tokens = 0.0100  # Training cost per 1K tokens
        elif model.startswith("gpt-3.5-turbo"):
            cost_per_1k_tokens = 0.0080
        else:
            # Default estimate
            cost_per_1k_tokens = 0.0100

        # Rough estimate: assume ~1000 tokens per example on average
        # This is a very rough estimate and actual costs may vary significantly
        avg_tokens_per_example = 1000

        total_training_tokens = training_examples * avg_tokens_per_example
        total_validation_tokens = validation_examples * avg_tokens_per_example

        training_cost = (total_training_tokens / 1000) * cost_per_1k_tokens
        validation_cost = (total_validation_tokens / 1000) * cost_per_1k_tokens

        return {
            "training_examples": training_examples,
            "validation_examples": validation_examples,
            "estimated_training_tokens": total_training_tokens,
            "estimated_validation_tokens": total_validation_tokens,
            "estimated_training_cost_usd": training_cost,
            "estimated_validation_cost_usd": validation_cost,
            "estimated_total_cost_usd": training_cost + validation_cost,
            "note": "This is a rough estimate. Actual costs may vary significantly.",
        }


class FineTuningManager:
    """High-level manager for fine-tuning workflows."""

    def __init__(
        self,
        client: OpenAI,
        job_storage_path: Optional[Path] = None,
        max_concurrent_jobs: int = 3,
        daily_limit: int = 10,
    ):
        """
        Initialize the fine-tuning manager.

        Parameters
        ----------
        client : OpenAI
            OpenAI client instance.
        job_storage_path : Optional[Path]
            Path to store job information.
        max_concurrent_jobs : int
            Maximum number of concurrent fine-tuning jobs.
        daily_limit : int
            Daily limit for new fine-tuning jobs.
        """
        self.fine_tuner = OpenAIFineTuner(client, max_concurrent_jobs, daily_limit)
        self.job_storage_path = job_storage_path or Path("fine_tuning_jobs.json")
        self.jobs_data = self._load_jobs_data()

    def _load_jobs_data(self) -> Dict[str, Dict[str, Any]]:
        """Load jobs data from storage."""
        if self.job_storage_path.exists():
            with open(self.job_storage_path, "r") as f:
                return json.load(f)
        return {}

    def _save_jobs_data(self) -> None:
        """Save jobs data to storage."""
        with open(self.job_storage_path, "w") as f:
            json.dump(self.jobs_data, f, indent=2, default=str)

    def start_fine_tuning_workflow(
        self,
        training_file_path: Path,
        validation_file_path: Optional[Path] = None,
        model: str = "gpt-4o-mini-2024-07-18",
        suffix: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        cleanup_files: bool = True,
    ) -> str:
        """
        Start a complete fine-tuning workflow.

        Parameters
        ----------
        training_file_path : Path
            Path to training data file.
        validation_file_path : Optional[Path]
            Path to validation data file.
        model : str
            Base model to fine-tune.
        suffix : Optional[str]
            Suffix for the fine-tuned model name.
        hyperparameters : Optional[Dict[str, Any]]
            Hyperparameters for fine-tuning.
        cleanup_files : bool
            Whether to delete uploaded files after job creation.

        Returns
        -------
        str
            Fine-tuning job ID.
        """
        logger.info("Starting fine-tuning workflow")

        if not self.fine_tuner.can_start_new_job():
            raise RuntimeError("Cannot start new job: limits exceeded")

        # Upload files
        training_file_id = self.fine_tuner.upload_training_file(training_file_path)

        validation_file_id = None
        if validation_file_path:
            validation_file_id = self.fine_tuner.upload_training_file(
                validation_file_path
            )

        try:
            # Create fine-tuning job
            job = self.fine_tuner.create_fine_tuning_job(
                training_file_id=training_file_id,
                model=model,
                validation_file_id=validation_file_id,
                suffix=suffix,
                hyperparameters=hyperparameters,
            )

            # Store job information
            self.jobs_data[job.id] = {
                "id": job.id,
                "status": job.status,
                "model": model,
                "training_file_id": training_file_id,
                "validation_file_id": validation_file_id,
                "created_at": datetime.now().isoformat(),
                "training_file_path": str(training_file_path),
                "validation_file_path": (
                    str(validation_file_path) if validation_file_path else None
                ),
                "suffix": suffix,
                "hyperparameters": hyperparameters,
                "cleanup_files": cleanup_files,
            }
            self._save_jobs_data()

            return job.id

        except Exception as e:
            # Clean up files on error
            logger.error(f"Error creating fine-tuning job: {e}")
            self.fine_tuner.delete_file(training_file_id)
            if validation_file_id:
                self.fine_tuner.delete_file(validation_file_id)
            raise

    def monitor_job(self, job_id: str, poll_interval: int = 60) -> Dict[str, Any]:
        """
        Monitor a fine-tuning job until completion.

        Parameters
        ----------
        job_id : str
            Fine-tuning job ID.
        poll_interval : int
            Polling interval in seconds.

        Returns
        -------
        Dict[str, Any]
            Job completion information.
        """
        logger.info(f"Monitoring job: {job_id}")

        job = self.fine_tuner.wait_for_job_completion(job_id, poll_interval)

        # Update stored job data
        if job_id in self.jobs_data:
            self.jobs_data[job_id].update(
                {
                    "status": job.status,
                    "fine_tuned_model": job.fine_tuned_model,
                    "finished_at": datetime.now().isoformat(),
                    "result_files": [file.id for file in (job.result_files or [])],
                    "trained_tokens": job.trained_tokens,
                }
            )
            self._save_jobs_data()

        # Clean up files if requested
        job_data = self.jobs_data.get(job_id, {})
        if job_data.get("cleanup_files", True):
            if job_data.get("training_file_id"):
                self.fine_tuner.delete_file(job_data["training_file_id"])
            if job_data.get("validation_file_id"):
                self.fine_tuner.delete_file(job_data["validation_file_id"])

        return {
            "job_id": job.id,
            "status": job.status,
            "fine_tuned_model": job.fine_tuned_model,
            "trained_tokens": job.trained_tokens,
        }

    def list_jobs(self, active_only: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        List stored fine-tuning jobs.

        Parameters
        ----------
        active_only : bool
            Only return active jobs.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary of job data.
        """
        if not active_only:
            return self.jobs_data

        active_jobs = {}
        for job_id, job_data in self.jobs_data.items():
            if job_data.get("status") in ["validating_files", "queued", "running"]:
                active_jobs[job_id] = job_data

        return active_jobs

    def get_job_summary(self, job_id: str) -> Dict[str, Any]:
        """
        Get a summary of a fine-tuning job.

        Parameters
        ----------
        job_id : str
            Fine-tuning job ID.

        Returns
        -------
        Dict[str, Any]
            Job summary information.
        """
        if job_id not in self.jobs_data:
            # Try to fetch from OpenAI
            job = self.fine_tuner.get_job_status(job_id)
            return {
                "id": job.id,
                "status": job.status,
                "model": job.model,
                "fine_tuned_model": job.fine_tuned_model,
                "trained_tokens": job.trained_tokens,
            }

        return self.jobs_data[job_id]


if __name__ == "__main__":
    # Example usage
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Fine-tune OpenAI models")
    parser.add_argument(
        "--training-file", type=Path, required=True, help="Path to training JSONL file"
    )
    parser.add_argument(
        "--validation-file", type=Path, help="Path to validation JSONL file"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini-2024-07-18", help="Base model to fine-tune"
    )
    parser.add_argument("--suffix", help="Suffix for fine-tuned model name")
    parser.add_argument("--wait", action="store_true", help="Wait for job completion")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Initialize fine-tuning manager
    manager = FineTuningManager(client)

    # Start fine-tuning
    job_id = manager.start_fine_tuning_workflow(
        training_file_path=args.training_file,
        validation_file_path=args.validation_file,
        model=args.model,
        suffix=args.suffix,
    )

    print(f"Fine-tuning job started: {job_id}")

    if args.wait:
        result = manager.monitor_job(job_id)
        print(f"Job completed: {result}")
