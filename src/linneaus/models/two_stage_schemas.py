"""
Enhanced Pydantic schemas for two-stage hierarchical AS classification.

This module provides comprehensive data models for the two-stage hierarchical
classification pipeline, including validation, confidence scores, and model
attribution for both stages.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .hierarchical_schemas import CategoryMapping, TopLevelCategory


class ModelType(str, Enum):
    """Types of models used in prediction."""

    LLM = "llm"
    SVM = "svm"
    ASSEMBLED = "assembled"
    TWO_STAGE = "two_stage"


class ConfidenceScore(BaseModel):
    """
    Confidence score for a prediction with model attribution.

    Attributes
    ----------
    value : float
        Confidence score between 0 and 1.
    model_type : ModelType
        Type of model that generated this confidence.
    source_scores : Optional[Dict[str, float]]
        Breakdown of confidence by component models (for assembled models).
    """

    value: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )
    model_type: ModelType = Field(
        ..., description="Type of model that generated this confidence"
    )
    source_scores: Optional[Dict[str, float]] = Field(
        default=None, description="Component model scores"
    )

    @field_validator("source_scores")
    @classmethod
    def validate_source_scores(cls, v):
        """Validate that all source scores are between 0 and 1."""
        if v is not None:
            for score in v.values():
                if not (0.0 <= score <= 1.0):
                    raise ValueError(f"Source score {score} must be between 0 and 1")
        return v


class TopLevelPrediction(BaseModel):
    """
    Top-level category prediction from Stage 1.

    Attributes
    ----------
    category : TopLevelCategory
        Predicted top-level category.
    confidence : ConfidenceScore
        Confidence score for this prediction.
    feature_importance : Optional[Dict[str, float]]
        Feature importance scores (for SVM components).
    """

    category: TopLevelCategory = Field(..., description="Predicted top-level category")
    confidence: ConfidenceScore = Field(
        ..., description="Confidence score for prediction"
    )
    feature_importance: Optional[Dict[str, float]] = Field(
        default=None, description="Feature importance scores for SVM components"
    )


class SublevelPrediction(BaseModel):
    """
    Sublevel category prediction from Stage 2.

    Attributes
    ----------
    subcategory : str
        Predicted subcategory within the top-level category.
    confidence : ConfidenceScore
        Confidence score for this prediction.
    parent_category : TopLevelCategory
        Top-level category this subcategory belongs to.
    """

    subcategory: str = Field(..., description="Predicted subcategory")
    confidence: ConfidenceScore = Field(
        ..., description="Confidence score for prediction"
    )
    parent_category: TopLevelCategory = Field(
        ..., description="Parent top-level category"
    )

    @model_validator(mode="after")
    def validate_category_consistency(self):
        """Validate that subcategory is valid for the parent category."""
        category_mappings = CategoryMapping.get_all_mappings()

        # Find mapping for parent category
        parent_mapping = None
        for mapping in category_mappings:
            if mapping.category == self.parent_category:
                parent_mapping = mapping
                break

        if parent_mapping is None:
            raise ValueError(
                f"No subcategories found for category {self.parent_category}"
            )

        if self.subcategory not in parent_mapping.subcategories:
            raise ValueError(
                f"Subcategory '{self.subcategory}' is not valid for category '{self.parent_category}'. "
                f"Valid subcategories: {parent_mapping.subcategories}"
            )

        return self


class TwoStageASClassification(BaseModel):
    """
    Complete two-stage hierarchical classification result for an AS.

    Attributes
    ----------
    asn : int
        Autonomous System Number.
    organization_name : str
        Organization name.
    stage1_predictions : List[TopLevelPrediction]
        Top-level category predictions from Stage 1.
    stage2_predictions : List[SublevelPrediction]
        Sublevel predictions from Stage 2.
    overall_confidence : float
        Overall confidence score combining both stages.
    processing_time_seconds : Optional[float]
        Total processing time for both stages.
    timestamp : datetime
        When the prediction was made.
    model_versions : Dict[str, str]
        Version information for models used.
    """

    asn: int = Field(..., gt=0, description="Autonomous System Number")
    organization_name: str = Field(..., description="Organization name")
    stage1_predictions: List[TopLevelPrediction] = Field(
        ..., description="Stage 1 predictions"
    )
    stage2_predictions: List[SublevelPrediction] = Field(
        default=[], description="Stage 2 predictions"
    )
    overall_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Overall confidence score"
    )
    processing_time_seconds: Optional[float] = Field(
        default=None, ge=0.0, description="Processing time"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Prediction timestamp"
    )
    model_versions: Dict[str, str] = Field(
        default={}, description="Model version information"
    )

    @field_validator("stage1_predictions")
    @classmethod
    def validate_stage1_predictions(cls, v):
        """Ensure at least one stage1 prediction exists."""
        if not v:
            raise ValueError("At least one stage1 prediction is required")
        return v

    @model_validator(mode="after")
    def validate_stage_consistency(self):
        """Validate consistency between Stage 1 and Stage 2 predictions."""
        if not self.stage2_predictions:
            return self

        # Get all predicted top-level categories from Stage 1
        stage1_categories = {pred.category for pred in self.stage1_predictions}

        # Check that all Stage 2 predictions have corresponding Stage 1 predictions
        for stage2_pred in self.stage2_predictions:
            if stage2_pred.parent_category not in stage1_categories:
                raise ValueError(
                    f"Stage 2 prediction for category '{stage2_pred.parent_category}' "
                    f"has no corresponding Stage 1 prediction"
                )

        return self

    def get_hierarchical_tags(self) -> List[str]:
        """Get all hierarchical tags in 'Category:Subcategory' format."""
        tags = []

        # Create mapping from category to subcategories
        category_subcat_map = {}
        for stage2_pred in self.stage2_predictions:
            if stage2_pred.parent_category not in category_subcat_map:
                category_subcat_map[stage2_pred.parent_category] = []
            category_subcat_map[stage2_pred.parent_category].append(
                stage2_pred.subcategory
            )

        # Generate hierarchical tags
        for stage1_pred in self.stage1_predictions:
            if stage1_pred.category in category_subcat_map:
                # Has subcategory predictions
                for subcategory in category_subcat_map[stage1_pred.category]:
                    tags.append(f"{stage1_pred.category.value}:{subcategory}")
            else:
                # No subcategory, use category only
                tags.append(stage1_pred.category.value)

        return tags

    def get_top_level_tags(self) -> List[str]:
        """Get only the top-level category tags."""
        return [pred.category.value for pred in self.stage1_predictions]


class BatchTwoStageClassificationResponse(BaseModel):
    """
    Response for batch processing of AS organizations through two-stage pipeline.

    Attributes
    ----------
    results : List[TwoStageASClassification]
        Individual classification results.
    batch_size : int
        Number of organizations in this batch.
    total_processing_time_seconds : Optional[float]
        Total time to process the entire batch.
    success_count : int
        Number of successfully processed organizations.
    error_count : int
        Number of organizations that failed processing.
    model_performance : Optional[Dict[str, Any]]
        Performance metrics for the batch.
    """

    results: List[TwoStageASClassification] = Field(
        ..., description="Individual classification results"
    )
    batch_size: int = Field(..., ge=0, description="Batch size")
    total_processing_time_seconds: Optional[float] = Field(
        default=None, ge=0.0, description="Total processing time"
    )
    success_count: int = Field(
        ..., ge=0, description="Number of successful predictions"
    )
    error_count: int = Field(..., ge=0, description="Number of failed predictions")
    model_performance: Optional[Dict[str, Any]] = Field(
        default=None, description="Batch performance metrics"
    )

    @model_validator(mode="after")
    def validate_counts(self):
        """Validate that success + error count equals batch size."""
        if self.success_count + self.error_count != self.batch_size:
            raise ValueError(
                f"Success count ({self.success_count}) + error count ({self.error_count}) "
                f"must equal batch size ({self.batch_size})"
            )

        if len(self.results) != self.success_count:
            raise ValueError(
                f"Number of results ({len(self.results)}) must equal success count ({self.success_count})"
            )

        return self


class PipelineConfig(BaseModel):
    """
    Configuration for the two-stage hierarchical pipeline.

    Attributes
    ----------
    stage1 : Dict[str, Any]
        Stage 1 configuration (assembled model).
    stage2 : Dict[str, Any]
        Stage 2 configuration (sublevel LLM models).
    general : Dict[str, Any]
        General pipeline configuration.
    prompts : Dict[str, str]
        Prompt templates for different stages.
    """

    stage1: Dict[str, Any] = Field(default={}, description="Stage 1 configuration")
    stage2: Dict[str, Any] = Field(default={}, description="Stage 2 configuration")
    general: Dict[str, Any] = Field(default={}, description="General configuration")
    prompts: Dict[str, str] = Field(default={}, description="Prompt templates")

    @field_validator("stage1")
    @classmethod
    def validate_stage1_config(cls, v):
        """Validate Stage 1 configuration has required fields."""
        required_fields = ["svm_weight", "llm_weight", "meta_learner"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Stage 1 config missing required field: {field}")

        # Validate weights sum to reasonable value
        if "svm_weight" in v and "llm_weight" in v:
            total_weight = v["svm_weight"] + v["llm_weight"]
            if not (0.8 <= total_weight <= 1.2):  # Allow some flexibility
                raise ValueError(
                    f"SVM weight ({v['svm_weight']}) + LLM weight ({v['llm_weight']}) "
                    f"should sum to approximately 1.0"
                )

        return v

    @field_validator("stage2")
    @classmethod
    def validate_stage2_config(cls, v):
        """Validate Stage 2 configuration has category models."""
        if "category_models" not in v:
            raise ValueError("Stage 2 config missing 'category_models' field")
        return v


class ModelPerformanceMetrics(BaseModel):
    """
    Performance metrics for model evaluation.

    Attributes
    ----------
    accuracy : float
        Overall accuracy score.
    precision : float
        Macro-averaged precision.
    recall : float
        Macro-averaged recall.
    f1_score : float
        Macro-averaged F1 score.
    stage1_accuracy : Optional[float]
        Stage 1 specific accuracy.
    stage2_accuracy : Optional[float]
        Stage 2 specific accuracy.
    hierarchical_consistency : Optional[float]
        Percentage of predictions with consistent hierarchy.
    exact_match_rate : Optional[float]
        Percentage of exactly correct full hierarchical predictions.
    partial_match_rate : Optional[float]
        Percentage of partially correct predictions.
    """

    accuracy: float = Field(..., ge=0.0, le=1.0, description="Overall accuracy")
    precision: float = Field(
        ..., ge=0.0, le=1.0, description="Macro-averaged precision"
    )
    recall: float = Field(..., ge=0.0, le=1.0, description="Macro-averaged recall")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="Macro-averaged F1 score")
    stage1_accuracy: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Stage 1 accuracy"
    )
    stage2_accuracy: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Stage 2 accuracy"
    )
    hierarchical_consistency: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Hierarchical consistency rate"
    )
    exact_match_rate: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Exact match rate"
    )
    partial_match_rate: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Partial match rate"
    )


# Export all classes
__all__ = [
    "ModelType",
    "ConfidenceScore",
    "TopLevelPrediction",
    "SublevelPrediction",
    "TwoStageASClassification",
    "BatchTwoStageClassificationResponse",
    "PipelineConfig",
    "ModelPerformanceMetrics",
]
