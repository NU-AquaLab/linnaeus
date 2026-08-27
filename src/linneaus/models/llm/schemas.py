"""
Pydantic schemas for LLM fine-tuning and inference.

This module defines all the data models used for OpenAI API interactions,
including request/response formats for both hierarchical and flat classification.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class HierarchicalTags(str, Enum):
    """Hierarchical tags for detailed AS classification."""

    # Access
    Access_LargeISP = "Access Large ISP"
    Access_SmallISP = "Access Small ISP"

    # Transit
    Transit_Global = "Transit Global"
    Transit_Regional = "Transit Regional"
    Transit_Domestic = "Transit Domestic"

    # Mobile
    Mobile = "Mobile"

    # Satellite
    Satellite = "Satellite"

    # Content Provider
    ContentProvider_Cloud = "ContentProvider Cloud"
    ContentProvider_Hosting = "ContentProvider Hosting"
    ContentProvider_CDN = "ContentProvider CDN"

    # Educational Research
    EducationalResearch_University = "Educational Research University"
    EducationalResearch_AcademicBackbone = "Educational Research Academic Backbone"
    EducationalResearch_Schools = "Educational Research Schools"
    EducationalResearch_ResearchInstitutes = "Educational Research Research Institutes"

    # Government
    Government_FederalNational = "Government Federal National"
    Government_StateProvince = "Government State Province"
    Government_CityCountyMunicipality = "Government City County Municipality"
    Government_Legislative = "Government Legislative"
    Government_Judiciary = "Government Judiciary"
    Government_Regulators = "Government Regulators"
    Government_Agencies = "Government Agencies"
    Government_Agencies_Space = "Government Agencies Space"
    Government_Agencies_CentralBanks = "Government Agencies CentralBanks"
    Government_PoliticalParties = "Government Political Parties"

    # IXP
    IXP = "Internet Exchange Point"

    # DNS
    DNS_Roots = "DNS Roots"
    DNS_ccTLD = "DNS ccTLD"
    DNS_ANS = "DNS ANS"

    # Utility
    Utility = "Utility"

    # Enterprise
    Enterprise_E_commerce = "Enterprise E-commerce"
    Enterprise_Entertainment = "Enterprise Entertainment"
    Enterprise_Industrial_Manufacturing = "Enterprise Industrial Manufacturing"
    Enterprise_Technology = "Enterprise Technology"

    # Energy
    Energy = "Energy"

    # Financial
    Bank = "Bank"
    CreditUnion = "CreditUnion"
    Financial_Entities = "Financial Entities"
    Financial_StockExchanges = "Financial StockExchanges"

    # Law Enforcement
    LawEnforcement = "LawEnforcement"

    # Health
    Health_Insurances = "Health Insurances"
    Health_Hospitals = "Health Hospitals"

    # Cooperatives
    Cooperatives = "Cooperatives"

    # TV/Radio/Cultural
    TvRadioCulturalAmenities = "TV/Radio and Cultural Amenities"

    # Transportation
    Transportation_Trains = "Transportation Trains"
    Transportation_Ships = "Transportation Ships"
    Transportation_Buses = "Transportation Buses"
    Transportation_TransitAuthority = "Transportation Transit Authority"
    Transportation_Airports = "Transportation Airports"

    # VPNs
    VPNs = "VPNs"

    # Personal
    Personal = "Personal"

    # Community
    Community = "Community"


class FlatTags(str, Enum):
    """Flat tags for simplified AS classification."""

    Access = "Access"
    Transit = "Transit"
    Mobile = "Mobile"
    Satellite = "Satellite"
    ContentProvider = "Content Provider"
    EducationalResearch = "Educational Research"
    Government = "Government"
    IXP = "Internet Exchange Point"
    DNS = "DNS"
    EnergyAndUtility = "Energy & Utility"  # Merged tag
    Enterprise = "Enterprise"
    Finance = "Finance"
    LawEnforcement = "Law Enforcement"
    Health = "Health"
    Cooperatives = "Cooperatives"
    TvRadioCulturalAmenities = "TV/Radio and Cultural Amenities"
    Transportation = "Transportation"
    VPNs = "Virtual Private Networks"
    Personal = "Personal"
    Community = "Community"


class TaggingResponseHierarchical(BaseModel):
    """Single organization tagging response for hierarchical classification."""

    tags: List[HierarchicalTags] = Field(
        ..., description="List of hierarchical tags assigned to the organization"
    )


class BatchTaggingResponseHierarchical(BaseModel):
    """Batch response for hierarchical classification."""

    responses: List[TaggingResponseHierarchical] = Field(
        ..., description="List of tagging responses for each organization in the batch"
    )


class TaggingResponseFlat(BaseModel):
    """Single organization tagging response for flat classification."""

    tags: List[FlatTags] = Field(
        ..., description="List of flat tags assigned to the organization"
    )


class BatchTaggingResponseFlat(BaseModel):
    """Batch response for flat classification."""

    responses: List[TaggingResponseFlat] = Field(
        ..., description="List of tagging responses for each organization in the batch"
    )


class OrganizationData(BaseModel):
    """Organization data for classification input."""

    asn: int = Field(..., description="Autonomous System Number")
    name: Optional[str] = Field(None, description="Organization name")
    description: Optional[str] = Field(None, description="Organization description")
    country: Optional[str] = Field(None, description="Country code")
    website: Optional[str] = Field(None, description="Organization website")

    # Additional metadata
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @field_validator("asn")
    @classmethod
    def validate_asn(cls, v):
        """Validate ASN is positive integer."""
        if v <= 0:
            raise ValueError("ASN must be a positive integer")
        return v


class ClassificationRequest(BaseModel):
    """Request for AS classification."""

    organizations: List[OrganizationData] = Field(
        ..., description="Organizations to classify"
    )
    approach: str = Field(
        default="hierarchical",
        description="Classification approach: 'hierarchical' or 'flat'",
    )
    model: Optional[str] = Field(None, description="Model to use for classification")
    batch_size: int = Field(default=10, description="Batch size for processing")
    include_probabilities: bool = Field(
        default=False, description="Whether to include prediction probabilities"
    )

    @field_validator("approach")
    @classmethod
    def validate_approach(cls, v):
        """Validate classification approach."""
        if v not in ["hierarchical", "flat"]:
            raise ValueError("Approach must be 'hierarchical' or 'flat'")
        return v

    @field_validator("organizations")
    @classmethod
    def validate_organizations(cls, v):
        """Validate we have organizations to classify."""
        if not v:
            raise ValueError("At least one organization must be provided")
        return v


class ClassificationResult(BaseModel):
    """Result of AS classification for a single organization."""

    asn: int = Field(..., description="Autonomous System Number")
    predicted_tags: List[Union[HierarchicalTags, FlatTags]] = Field(
        ..., description="Predicted tags"
    )
    tag_probabilities: Optional[Dict[str, float]] = Field(
        None, description="Prediction probabilities for each tag"
    )
    confidence_score: Optional[float] = Field(
        None, description="Overall confidence score"
    )
    model_used: str = Field(..., description="Model used for classification")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Classification timestamp"
    )


class BatchClassificationResponse(BaseModel):
    """Response for batch classification request."""

    results: List[ClassificationResult] = Field(
        ..., description="Classification results for each organization"
    )
    total_processed: int = Field(..., description="Total organizations processed")
    successful: int = Field(..., description="Number of successful classifications")
    failed: int = Field(..., description="Number of failed classifications")
    processing_time_seconds: float = Field(..., description="Total processing time")
    approach_used: str = Field(..., description="Classification approach used")
    model_used: str = Field(..., description="Model used for classification")
    batch_id: Optional[str] = Field(None, description="Batch processing ID")


class TrainingExample(BaseModel):
    """Training example for fine-tuning."""

    messages: List[Dict[str, str]] = Field(
        ..., description="List of messages (system, user, assistant)"
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        """Validate message structure."""
        if len(v) != 3:
            raise ValueError("Training example must have exactly 3 messages")

        expected_roles = ["system", "user", "assistant"]
        for i, msg in enumerate(v):
            if msg.get("role") != expected_roles[i]:
                raise ValueError(f"Message {i} must have role '{expected_roles[i]}'")
            if "content" not in msg or not msg["content"]:
                raise ValueError(f"Message {i} must have non-empty content")

        return v


class FineTuningJobInfo(BaseModel):
    """Information about a fine-tuning job."""

    job_id: str = Field(..., description="OpenAI fine-tuning job ID")
    status: str = Field(..., description="Job status")
    model: str = Field(..., description="Base model being fine-tuned")
    fine_tuned_model: Optional[str] = Field(None, description="Fine-tuned model name")
    training_file_id: str = Field(..., description="Training file ID")
    validation_file_id: Optional[str] = Field(None, description="Validation file ID")
    created_at: datetime = Field(..., description="Job creation timestamp")
    finished_at: Optional[datetime] = Field(
        None, description="Job completion timestamp"
    )
    trained_tokens: Optional[int] = Field(None, description="Number of tokens trained")
    hyperparameters: Optional[Dict[str, Any]] = Field(
        None, description="Training hyperparameters"
    )
    error: Optional[str] = Field(None, description="Error message if job failed")
    result_files: Optional[List[str]] = Field(None, description="Result file IDs")


class EvaluationMetrics(BaseModel):
    """Model evaluation metrics."""

    accuracy: float = Field(..., description="Overall accuracy")
    precision: float = Field(..., description="Macro-averaged precision")
    recall: float = Field(..., description="Macro-averaged recall")
    f1_score: float = Field(..., description="Macro-averaged F1 score")

    # Per-tag metrics
    per_tag_metrics: Optional[Dict[str, Dict[str, float]]] = Field(
        None, description="Per-tag precision, recall, and F1 scores"
    )

    # Additional metrics
    jaccard_score: Optional[float] = Field(None, description="Jaccard similarity score")
    pr_auc: Optional[float] = Field(None, description="Precision-Recall AUC")

    # Metadata
    evaluation_dataset_size: int = Field(..., description="Size of evaluation dataset")
    model_name: str = Field(..., description="Model evaluated")
    evaluation_timestamp: datetime = Field(
        default_factory=datetime.now, description="Evaluation timestamp"
    )


class LabeledDataset(BaseModel):
    """Labeled dataset information."""

    total_samples: int = Field(..., description="Total number of labeled samples")
    num_tags: int = Field(..., description="Number of possible tags")
    tag_distribution: Dict[str, int] = Field(
        ..., description="Distribution of samples per tag"
    )
    data_version: str = Field(..., description="Dataset version identifier")
    creation_timestamp: datetime = Field(
        default_factory=datetime.now, description="Dataset creation timestamp"
    )

    # Split information
    train_size: Optional[int] = Field(None, description="Training set size")
    validation_size: Optional[int] = Field(None, description="Validation set size")
    test_size: Optional[int] = Field(None, description="Test set size")
    split_random_state: Optional[int] = Field(
        None, description="Random seed for splits"
    )


class ModelConfiguration(BaseModel):
    """Model configuration settings."""

    model_name: str = Field(..., description="Model name")
    approach: str = Field(..., description="Classification approach")
    temperature: float = Field(default=0.0, description="Generation temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    batch_size: int = Field(default=10, description="Batch size for processing")

    # Fine-tuning specific
    base_model: Optional[str] = Field(None, description="Base model for fine-tuning")
    learning_rate: Optional[float] = Field(None, description="Learning rate")
    n_epochs: Optional[int] = Field(None, description="Number of training epochs")

    # Prompts
    system_prompt: Optional[str] = Field(None, description="System prompt")
    message_template: Optional[str] = Field(None, description="Message template")

    @field_validator("approach")
    @classmethod
    def validate_approach(cls, v):
        """Validate classification approach."""
        if v not in ["hierarchical", "flat", "hybrid"]:
            raise ValueError("Approach must be 'hierarchical', 'flat', or 'hybrid'")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        """Validate temperature range."""
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v


class PipelineStatus(BaseModel):
    """Status of the classification pipeline."""

    status: str = Field(..., description="Pipeline status")
    current_stage: str = Field(..., description="Current processing stage")
    progress: float = Field(..., description="Progress percentage (0-100)")
    total_organizations: int = Field(..., description="Total organizations to process")
    processed_organizations: int = Field(
        ..., description="Organizations processed so far"
    )

    # Timing information
    started_at: datetime = Field(..., description="Pipeline start time")
    estimated_completion: Optional[datetime] = Field(
        None, description="Estimated completion time"
    )

    # Error information
    errors: List[str] = Field(
        default_factory=list, description="List of errors encountered"
    )
    warnings: List[str] = Field(default_factory=list, description="List of warnings")

    # Resource usage
    api_calls_made: int = Field(default=0, description="Number of API calls made")
    tokens_used: int = Field(default=0, description="Total tokens used")
    estimated_cost: Optional[float] = Field(None, description="Estimated cost in USD")


# Legacy compatibility models (for backward compatibility with existing code)
class BatchTaggingResponseAllTags(BatchTaggingResponseHierarchical):
    """Legacy model for hierarchical classification."""


class BatchTaggingResponseReduced(BatchTaggingResponseFlat):
    """Legacy model for flat classification (21 tags version)."""

    class Tags(str, Enum):
        """Legacy flat tags (21 tags version)."""

        Access = "Access"
        Transit = "Transit"
        Mobile = "Mobile"
        Satellite = "Satellite"
        ContentProvider = "Content Provider"
        EducationalResearch = "Educational Research"
        Government = "Government"
        IXP = "Internet Exchange Point"
        DNS = "DNS"
        Utility = "Utility"
        Enterprise = "Enterprise"
        Energy = "Energy"
        Finance = "Finance"
        LawEnforcement = "Law Enforcement"
        Health = "Health"
        Cooperatives = "Cooperatives"
        TvRadioCulturalAmenities = "TV/Radio and Cultural Amenities"
        Transportation = "Transportation"
        VPNs = "Virtual Private Networks"
        Personal = "Personal"
        Community = "Community"


class BatchTaggingResponseReduced_v2(BatchTaggingResponseFlat):
    """Legacy model for flat classification (20 tags version - current)."""

    # Uses the same tags as FlatTags enum above


if __name__ == "__main__":
    # Example usage and validation

    # Create a sample organization
    org = OrganizationData(
        asn=7922,
        name="Comcast Cable Communications",
        description="Large cable internet and TV provider",
        country="US",
    )

    print(f"Organization: {org.name} (ASN {org.asn})")

    # Create a classification result
    result = ClassificationResult(
        asn=org.asn,
        predicted_tags=[
            HierarchicalTags.Access_LargeISP,
            HierarchicalTags.ContentProvider_Cloud,
        ],
        model_used="gpt-4o-mini-ft-2024",
        confidence_score=0.95,
    )

    print(f"Predicted tags: {[tag.value for tag in result.predicted_tags]}")

    # Create evaluation metrics
    metrics = EvaluationMetrics(
        accuracy=0.754,
        precision=0.723,
        recall=0.681,
        f1_score=0.701,
        evaluation_dataset_size=297,
        model_name="gpt-4o-mini-ft-hierarchical",
    )

    print(
        f"Model performance - F1: {metrics.f1_score:.3f}, Accuracy: {metrics.accuracy:.3f}"
    )
