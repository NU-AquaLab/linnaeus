"""
Hierarchical sublevel classifier for Stage 2 of two-stage pipeline.

This module implements the LLM-only classifier that predicts subcategories
within each top-level category predicted by Stage 1.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set

from openai import OpenAI
from pydantic import BaseModel

from ...data.access import DataAccessLayer
from ..hierarchical_schemas import CategoryMapping, TopLevelCategory
from ..llm.inference import HierarchicalBatchInferenceProcessor
from ..two_stage_schemas import (
    ConfidenceScore,
    ModelType,
    PipelineConfig,
    SublevelPrediction,
    TopLevelPrediction,
)

logger = logging.getLogger(__name__)


class CategorySpecificResponse(BaseModel):
    """Response format for category-specific sublevel predictions."""

    responses: List[Dict[str, Any]]


class HierarchicalSublevelClassifier:
    """
    LLM-only classifier for Stage 2 sublevel category prediction.

    This classifier takes the top-level predictions from Stage 1 and predicts
    specific subcategories within each predicted category using specialized
    fine-tuned models or prompts for each category.
    """

    def __init__(
        self,
        openai_client: OpenAI,
        category_models: Dict[str, str],
        category_prompts: Optional[Dict[str, str]] = None,
        data_access: Optional[DataAccessLayer] = None,
        config: Optional[PipelineConfig] = None,
        batch_size: int = 10,
        temperature: float = 0.0001,
    ):
        """
        Initialize hierarchical sublevel classifier.

        Parameters
        ----------
        openai_client : OpenAI
            OpenAI client for API calls.
        category_models : Dict[str, str]
            Mapping from category name to fine-tuned model ID.
        category_prompts : Optional[Dict[str, str]]
            Category-specific system prompts.
        data_access : Optional[DataAccessLayer]
            Data access layer for organization information.
        config : Optional[PipelineConfig]
            Pipeline configuration.
        batch_size : int
            Batch size for processing.
        temperature : float
            Temperature for LLM generation.
        """
        self.client = openai_client
        self.category_models = category_models
        self.category_prompts = category_prompts or {}
        self.data_access = data_access or DataAccessLayer()
        self.config = config
        self.batch_size = batch_size
        self.temperature = temperature

        # Initialize category mappings
        self.category_mappings = {
            mapping.category.value: mapping
            for mapping in CategoryMapping.get_all_mappings()
        }

        # Create category-specific processors
        self.processors: Dict[str, HierarchicalBatchInferenceProcessor] = {}
        self._initialize_processors()

        logger.info(
            f"Initialized HierarchicalSublevelClassifier with {len(self.category_models)} category models"
        )

    def _initialize_processors(self) -> None:
        """Initialize category-specific LLM processors."""
        for category, model_id in self.category_models.items():
            try:
                # Get category-specific prompt
                system_prompt = self.category_prompts.get(
                    category, self._get_default_category_prompt(category)
                )

                # Create processor for this category
                processor = HierarchicalBatchInferenceProcessor(
                    client=self.client,
                    model=model_id,
                    response_format=CategorySpecificResponse,
                    system_prompt=system_prompt,
                    approach="hierarchical",
                    batch_size=self.batch_size,
                )

                self.processors[category] = processor
                logger.info(f"Initialized processor for category: {category}")

            except Exception as e:
                logger.error(
                    f"Failed to initialize processor for category {category}: {e}"
                )

    def _get_default_category_prompt(self, category: str) -> str:
        """
        Generate default system prompt for a category.

        Parameters
        ----------
        category : str
            Category name.

        Returns
        -------
        str
            Default system prompt for the category.
        """
        mapping = self.category_mappings.get(category)
        if not mapping:
            return f"Classify the subcategory for {category} organizations."

        subcategories_text = "\n".join([f"- {sub}" for sub in mapping.subcategories])

        return f"""You are an expert in classifying Internet infrastructure organizations
specifically within the {category} category.

Your task is to predict the most specific subcategory for organizations that have
already been classified as {category} organizations.

Available subcategories for {category}:
{subcategories_text}

Respond with a JSON object containing a list of responses, each with a classifications
array containing objects with "category" (always "{category}") and "subcategory" fields."""

    def predict(
        self,
        organizations: List[Dict[str, Any]],
        stage1_predictions: List[TopLevelPrediction],
    ) -> List[SublevelPrediction]:
        """
        Predict sublevel categories based on Stage 1 predictions.

        Parameters
        ----------
        organizations : List[Dict[str, Any]]
            Organization data with ASN, name, description.
        stage1_predictions : List[TopLevelPrediction]
            Top-level predictions from Stage 1.

        Returns
        -------
        List[SublevelPrediction]
            Sublevel predictions for each organization and category.
        """
        logger.info(
            f"Processing {len(organizations)} organizations for sublevel classification"
        )
        start_time = time.time()

        results = []

        try:
            # Group predictions by category
            category_org_map = self._group_by_category(
                organizations, stage1_predictions
            )

            # Process each category separately
            for category, org_list in category_org_map.items():
                if category not in self.processors:
                    logger.warning(f"No processor available for category: {category}")
                    continue

                if not org_list:
                    continue

                logger.info(
                    f"Processing {len(org_list)} organizations for category: {category}"
                )

                try:
                    category_results = self._predict_category(category, org_list)
                    results.extend(category_results)

                except Exception as e:
                    logger.error(f"Error processing category {category}: {e}")
                    # Create fallback predictions
                    fallback_results = self._create_fallback_predictions(
                        category, org_list
                    )
                    results.extend(fallback_results)

            prediction_time = time.time() - start_time
            logger.info(
                f"Generated {len(results)} sublevel predictions in {prediction_time:.2f}s"
            )

            return results

        except Exception as e:
            logger.error(f"Error in sublevel prediction: {e}")
            raise

    def _group_by_category(
        self,
        organizations: List[Dict[str, Any]],
        stage1_predictions: List[TopLevelPrediction],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group organizations by their predicted top-level categories.

        Parameters
        ----------
        organizations : List[Dict[str, Any]]
            Organization data.
        stage1_predictions : List[TopLevelPrediction]
            Stage 1 predictions.

        Returns
        -------
        Dict[str, List[Dict[str, Any]]]
            Organizations grouped by category.
        """
        category_org_map = {}

        # Create ASN to organization mapping
        org_by_asn = {org["asn"]: org for org in organizations}

        # Group by category
        for prediction in stage1_predictions:
            category = prediction.category.value

            # Find corresponding organization (assuming ASN is available in prediction context)
            # This is a simplification - in practice, we'd need better association
            for org in organizations:
                if category not in category_org_map:
                    category_org_map[category] = []

                # Add organization with prediction context
                org_with_context = org.copy()
                org_with_context["stage1_confidence"] = prediction.confidence.value
                org_with_context["predicted_category"] = category

                category_org_map[category].append(org_with_context)
                break  # For now, just take first organization per prediction

        return category_org_map

    def _predict_category(
        self, category: str, organizations: List[Dict[str, Any]]
    ) -> List[SublevelPrediction]:
        """
        Predict subcategories for organizations within a specific category.

        Parameters
        ----------
        category : str
            Top-level category.
        organizations : List[Dict[str, Any]]
            Organizations to classify.

        Returns
        -------
        List[SublevelPrediction]
            Sublevel predictions.
        """
        processor = self.processors[category]
        results = []

        try:
            # Process in batches
            for i in range(0, len(organizations), self.batch_size):
                batch = organizations[i : i + self.batch_size]

                # Process batch
                batch_results = processor.process_batch(batch)

                # Convert to SublevelPrediction objects
                for j, org in enumerate(batch):
                    if j < len(batch_results):
                        result = batch_results[j]

                        # Extract classifications
                        classifications = result.get("responses", [{}])[0].get(
                            "classifications", []
                        )

                        for classification in classifications:
                            subcategory = classification.get("subcategory")
                            if subcategory:
                                # Validate subcategory
                                if self._is_valid_subcategory(category, subcategory):
                                    confidence = ConfidenceScore(
                                        value=classification.get("confidence", 0.8),
                                        model_type=ModelType.LLM,
                                    )

                                    prediction = SublevelPrediction(
                                        subcategory=subcategory,
                                        confidence=confidence,
                                        parent_category=TopLevelCategory(category),
                                    )

                                    results.append(prediction)
                                else:
                                    logger.warning(
                                        f"Invalid subcategory '{subcategory}' for category '{category}'"
                                    )

        except Exception as e:
            logger.error(f"Error in category {category} prediction: {e}")
            raise

        return results

    def _is_valid_subcategory(self, category: str, subcategory: str) -> bool:
        """
        Validate that a subcategory is valid for the given category.

        Parameters
        ----------
        category : str
            Top-level category.
        subcategory : str
            Proposed subcategory.

        Returns
        -------
        bool
            True if valid, False otherwise.
        """
        mapping = self.category_mappings.get(category)
        if not mapping:
            return False

        return subcategory in mapping.subcategories

    def _create_fallback_predictions(
        self, category: str, organizations: List[Dict[str, Any]]
    ) -> List[SublevelPrediction]:
        """
        Create fallback predictions when LLM processing fails.

        Parameters
        ----------
        category : str
            Top-level category.
        organizations : List[Dict[str, Any]]
            Organizations that need fallback predictions.

        Returns
        -------
        List[SublevelPrediction]
            Fallback predictions.
        """
        results = []
        mapping = self.category_mappings.get(category)

        if not mapping or not mapping.subcategories:
            return results

        # Use most common subcategory as fallback
        fallback_subcategory = mapping.subcategories[0]  # First subcategory as fallback

        for org in organizations:
            confidence = ConfidenceScore(
                value=0.1, model_type=ModelType.LLM  # Low confidence for fallback
            )

            prediction = SublevelPrediction(
                subcategory=fallback_subcategory,
                confidence=confidence,
                parent_category=TopLevelCategory(category),
            )

            results.append(prediction)

        logger.info(
            f"Created {len(results)} fallback predictions for category {category}"
        )
        return results

    def get_available_categories(self) -> Set[str]:
        """
        Get categories that have available processors.

        Returns
        -------
        Set[str]
            Available category names.
        """
        return set(self.processors.keys())

    def add_category_processor(
        self, category: str, model_id: str, system_prompt: Optional[str] = None
    ) -> None:
        """
        Add a processor for a new category.

        Parameters
        ----------
        category : str
            Category name.
        model_id : str
            Fine-tuned model ID.
        system_prompt : Optional[str]
            Custom system prompt for the category.
        """
        try:
            prompt = system_prompt or self._get_default_category_prompt(category)

            processor = HierarchicalBatchInferenceProcessor(
                client=self.client,
                model=model_id,
                response_format=CategorySpecificResponse,
                system_prompt=prompt,
                approach="hierarchical",
                batch_size=self.batch_size,
            )

            self.processors[category] = processor
            self.category_models[category] = model_id

            logger.info(f"Added processor for category: {category}")

        except Exception as e:
            logger.error(f"Failed to add processor for category {category}: {e}")
            raise

    def remove_category_processor(self, category: str) -> None:
        """
        Remove a category processor.

        Parameters
        ----------
        category : str
            Category name to remove.
        """
        if category in self.processors:
            del self.processors[category]

        if category in self.category_models:
            del self.category_models[category]

        logger.info(f"Removed processor for category: {category}")


# Export the main class
__all__ = ["HierarchicalSublevelClassifier", "CategorySpecificResponse"]
