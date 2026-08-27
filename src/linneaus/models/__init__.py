"""Machine learning models and data schemas for Linneaus."""

# Legacy components (for backward compatibility)
from .evaluation import ClassificationEvaluator, export_evaluation_report
from .hybrid.hierarchical_tagger import ModernHierarchicalTagger

# Hybrid models
from .hybrid.stacking_classifier import HybridASClassifier
from .llm.fine_tuning import OpenAIFineTuner

# LLM models
from .llm.inference import HierarchicalBatchInferenceProcessor

# Original schemas (deprecated - use unified_schemas instead)
from .schemas import (
    ASClassification,
    ASPOPData,
    ASRankData,
    BatchClassificationResponse,
    ClassificationTags,
    OrganizationData,
    PeeringDBData,
)
from .sklearn_interface import ASNClassifier, ASNDataTransformer, HybridASNClassifier

# SVM models
from .svm.feature_engineering import ASNFeatureEngineer
from .svm.svm_models import HierarchicalSVMClassifier, SVMClassifier

# New unified schema system
from .unified_schemas import (
    BatchUnifiedClassificationResponse,
    ClassificationRequest,
    HierarchicalTags,
    TagHierarchy,
    TopLevelTags,
    UnifiedASClassification,
)

__all__ = [
    # Legacy components
    "ASClassification",
    "OrganizationData",
    "ClassificationTags",
    "BatchClassificationResponse",
    "ASRankData",
    "PeeringDBData",
    "ASPOPData",
    "HierarchicalBatchInferenceProcessor",
    "ClassificationEvaluator",
    "export_evaluation_report",
    "ASNClassifier",
    "ASNDataTransformer",
    "HybridASNClassifier",
    # New unified system
    "TopLevelTags",
    "HierarchicalTags",
    "TagHierarchy",
    "UnifiedASClassification",
    "BatchUnifiedClassificationResponse",
    "ClassificationRequest",
    # LLM models
    "OpenAIFineTuner",
    # SVM models
    "ASNFeatureEngineer",
    "SVMClassifier",
    "HierarchicalSVMClassifier",
    # Hybrid models
    "HybridASClassifier",
    "ModernHierarchicalTagger",
]
