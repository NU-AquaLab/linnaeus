"""
Comprehensive error handling and graceful degradation for two-stage pipeline.

This module provides robust error handling, fallback mechanisms, and detailed
error reporting for the two-stage hierarchical classification system.
"""

import logging
import traceback
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel

from ..hierarchical_schemas import TopLevelCategory
from ..two_stage_schemas import (
    ConfidenceScore,
    ModelType,
    SublevelPrediction,
    TopLevelPrediction,
    TwoStageASClassification,
)

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Categories of errors that can occur."""

    DATA_ERROR = "data_error"
    MODEL_ERROR = "model_error"
    API_ERROR = "api_error"
    CONFIGURATION_ERROR = "configuration_error"
    VALIDATION_ERROR = "validation_error"
    SYSTEM_ERROR = "system_error"


class PipelineError(BaseModel):
    """
    Structured error information for pipeline failures.

    Attributes
    ----------
    error_id : str
        Unique identifier for the error.
    category : ErrorCategory
        Category of the error.
    severity : ErrorSeverity
        Severity level of the error.
    message : str
        Human-readable error message.
    details : Dict[str, Any]
        Additional error details.
    component : str
        Pipeline component where error occurred.
    asn : Optional[int]
        ASN being processed when error occurred.
    recovery_action : Optional[str]
        Description of recovery action taken.
    """

    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    component: str
    asn: Optional[int] = None
    recovery_action: Optional[str] = None


class FallbackStrategy(str, Enum):
    """Strategies for handling component failures."""

    SKIP = "skip"  # Skip the failed component
    DEFAULT = "default"  # Use default values
    PREVIOUS = "previous"  # Use previous successful result
    ALTERNATIVE = "alternative"  # Use alternative model/approach


class ErrorHandler:
    """
    Comprehensive error handler with graceful degradation.

    This class provides centralized error handling, logging, and recovery
    mechanisms for the two-stage pipeline.
    """

    def __init__(
        self,
        enable_graceful_degradation: bool = True,
        max_retry_attempts: int = 3,
        fallback_strategy: FallbackStrategy = FallbackStrategy.DEFAULT,
    ):
        """
        Initialize error handler.

        Parameters
        ----------
        enable_graceful_degradation : bool
            Whether to enable graceful degradation on errors.
        max_retry_attempts : int
            Maximum number of retry attempts for recoverable errors.
        fallback_strategy : FallbackStrategy
            Default fallback strategy for component failures.
        """
        self.enable_graceful_degradation = enable_graceful_degradation
        self.max_retry_attempts = max_retry_attempts
        self.fallback_strategy = fallback_strategy

        # Error tracking
        self.errors: List[PipelineError] = []
        self.error_counts: Dict[ErrorCategory, int] = {cat: 0 for cat in ErrorCategory}
        self.recovery_stats: Dict[str, int] = {}

        logger.info(
            f"Initialized ErrorHandler with graceful degradation: {enable_graceful_degradation}"
        )

    def handle_error(
        self,
        error: Exception,
        component: str,
        asn: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> PipelineError:
        """
        Handle and log an error with detailed information.

        Parameters
        ----------
        error : Exception
            The exception that occurred.
        component : str
            Component where the error occurred.
        asn : Optional[int]
            ASN being processed when error occurred.
        context : Optional[Dict[str, Any]]
            Additional context information.

        Returns
        -------
        PipelineError
            Structured error information.
        """
        # Categorize error
        error_category = self._categorize_error(error)
        severity = self._determine_severity(error, error_category)

        # Create error record
        pipeline_error = PipelineError(
            error_id=f"{component}_{len(self.errors) + 1}",
            category=error_category,
            severity=severity,
            message=str(error),
            details={
                "exception_type": type(error).__name__,
                "traceback": traceback.format_exc(),
                "context": context or {},
            },
            component=component,
            asn=asn,
        )

        # Log error
        log_level = (
            logging.ERROR
            if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
            else logging.WARNING
        )
        logger.log(
            log_level,
            f"Error in {component} (ASN: {asn}): {pipeline_error.message} "
            f"[Category: {error_category}, Severity: {severity}]",
        )

        # Track error
        self.errors.append(pipeline_error)
        self.error_counts[error_category] += 1

        return pipeline_error

    def with_error_handling(
        self,
        component: str,
        fallback_strategy: Optional[FallbackStrategy] = None,
        max_retries: Optional[int] = None,
    ):
        """
        Decorator for adding error handling to pipeline methods.

        Parameters
        ----------
        component : str
            Name of the component being wrapped.
        fallback_strategy : Optional[FallbackStrategy]
            Fallback strategy to use for this component.
        max_retries : Optional[int]
            Maximum retry attempts for this component.

        Returns
        -------
        Callable
            Decorated function with error handling.
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                strategy = fallback_strategy or self.fallback_strategy
                retries = max_retries or self.max_retry_attempts

                last_error = None

                for attempt in range(retries + 1):
                    try:
                        result = func(*args, **kwargs)

                        # Log successful retry if this wasn't the first attempt
                        if attempt > 0:
                            logger.info(f"Successful retry {attempt} for {component}")
                            self.recovery_stats[f"{component}_retry_success"] = (
                                self.recovery_stats.get(f"{component}_retry_success", 0)
                                + 1
                            )

                        return result

                    except Exception as e:
                        last_error = e

                        # Handle the error
                        pipeline_error = self.handle_error(
                            e,
                            component,
                            asn=kwargs.get("asn") or (args[0] if args else None),
                        )

                        # If this is the last attempt or graceful degradation is disabled
                        if attempt == retries or not self.enable_graceful_degradation:
                            if self.enable_graceful_degradation:
                                return self._create_fallback_result(
                                    strategy, component, pipeline_error
                                )
                            else:
                                raise e

                        # Wait before retry (simple linear backoff)
                        if attempt < retries:
                            import time

                            wait_time = (attempt + 1) * 0.5
                            logger.info(
                                f"Retrying {component} in {wait_time}s (attempt {attempt + 1}/{retries})"
                            )
                            time.sleep(wait_time)

                # All retries failed
                if self.enable_graceful_degradation and last_error:
                    pipeline_error = self.handle_error(last_error, component)
                    return self._create_fallback_result(
                        strategy, component, pipeline_error
                    )
                else:
                    raise last_error

            return wrapper

        return decorator

    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """
        Categorize an error based on its type and message.

        Parameters
        ----------
        error : Exception
            The error to categorize.

        Returns
        -------
        ErrorCategory
            The error category.
        """
        error_type = type(error).__name__
        error_message = str(error).lower()

        # API-related errors
        if any(
            keyword in error_message
            for keyword in ["api", "openai", "rate limit", "quota", "connection"]
        ):
            return ErrorCategory.API_ERROR

        # Data-related errors
        if any(
            keyword in error_message
            for keyword in ["dataframe", "column", "index", "missing", "empty"]
        ):
            return ErrorCategory.DATA_ERROR

        # Model-related errors
        if any(
            keyword in error_message
            for keyword in ["model", "prediction", "inference", "classifier"]
        ):
            return ErrorCategory.MODEL_ERROR

        # Validation errors
        if "validation" in error_message or error_type in [
            "ValidationError",
            "ValueError",
        ]:
            return ErrorCategory.VALIDATION_ERROR

        # Configuration errors
        if "config" in error_message or error_type == "KeyError":
            return ErrorCategory.CONFIGURATION_ERROR

        # Default to system error
        return ErrorCategory.SYSTEM_ERROR

    def _determine_severity(
        self, error: Exception, category: ErrorCategory
    ) -> ErrorSeverity:
        """
        Determine error severity based on error and category.

        Parameters
        ----------
        error : Exception
            The error.
        category : ErrorCategory
            Error category.

        Returns
        -------
        ErrorSeverity
            Error severity level.
        """
        error_message = str(error).lower()

        # Critical errors
        if any(
            keyword in error_message for keyword in ["critical", "fatal", "corrupt"]
        ):
            return ErrorSeverity.CRITICAL

        # High severity errors
        if category == ErrorCategory.SYSTEM_ERROR:
            return ErrorSeverity.HIGH

        if any(keyword in error_message for keyword in ["timeout", "memory", "disk"]):
            return ErrorSeverity.HIGH

        # Medium severity errors
        if category in [ErrorCategory.MODEL_ERROR, ErrorCategory.API_ERROR]:
            return ErrorSeverity.MEDIUM

        # Low severity errors (recoverable)
        return ErrorSeverity.LOW

    def _create_fallback_result(
        self, strategy: FallbackStrategy, component: str, pipeline_error: PipelineError
    ) -> Any:
        """
        Create a fallback result based on the strategy.

        Parameters
        ----------
        strategy : FallbackStrategy
            Fallback strategy to use.
        component : str
            Component that failed.
        pipeline_error : PipelineError
            Error information.

        Returns
        -------
        Any
            Fallback result.
        """
        recovery_action = f"Applied {strategy.value} fallback strategy"
        pipeline_error.recovery_action = recovery_action

        self.recovery_stats[f"{component}_{strategy.value}"] = (
            self.recovery_stats.get(f"{component}_{strategy.value}", 0) + 1
        )

        logger.info(f"Applied fallback strategy '{strategy.value}' for {component}")

        if component.startswith("stage1"):
            return self._create_stage1_fallback(strategy, pipeline_error)
        elif component.startswith("stage2"):
            return self._create_stage2_fallback(strategy, pipeline_error)
        else:
            return self._create_generic_fallback(strategy, pipeline_error)

    def _create_stage1_fallback(
        self, strategy: FallbackStrategy, pipeline_error: PipelineError
    ) -> List[TopLevelPrediction]:
        """
        Create Stage 1 fallback predictions.

        Parameters
        ----------
        strategy : FallbackStrategy
            Fallback strategy.
        pipeline_error : PipelineError
            Error information.

        Returns
        -------
        List[TopLevelPrediction]
            Fallback Stage 1 predictions.
        """
        if strategy == FallbackStrategy.DEFAULT:
            # Return default Enterprise classification with low confidence
            confidence = ConfidenceScore(value=0.1, model_type=ModelType.ASSEMBLED)
            fallback_prediction = TopLevelPrediction(
                category=TopLevelCategory.ENTERPRISE, confidence=confidence
            )
            return [fallback_prediction]

        elif strategy == FallbackStrategy.SKIP:
            return []

        # Default case
        confidence = ConfidenceScore(value=0.05, model_type=ModelType.ASSEMBLED)
        fallback_prediction = TopLevelPrediction(
            category=TopLevelCategory.ENTERPRISE, confidence=confidence
        )
        return [fallback_prediction]

    def _create_stage2_fallback(
        self, strategy: FallbackStrategy, pipeline_error: PipelineError
    ) -> List[SublevelPrediction]:
        """
        Create Stage 2 fallback predictions.

        Parameters
        ----------
        strategy : FallbackStrategy
            Fallback strategy.
        pipeline_error : PipelineError
            Error information.

        Returns
        -------
        List[SublevelPrediction]
            Fallback Stage 2 predictions.
        """
        if strategy == FallbackStrategy.SKIP:
            return []

        # For Stage 2, we typically don't have enough context to create meaningful fallbacks
        # Return empty list (Stage 2 predictions are optional)
        return []

    def _create_generic_fallback(
        self, strategy: FallbackStrategy, pipeline_error: PipelineError
    ) -> Any:
        """
        Create generic fallback result.

        Parameters
        ----------
        strategy : FallbackStrategy
            Fallback strategy.
        pipeline_error : PipelineError
            Error information.

        Returns
        -------
        Any
            Generic fallback result.
        """
        if strategy == FallbackStrategy.SKIP:
            return None

        # Return empty dict as generic fallback
        return {}

    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get summary of all errors encountered.

        Returns
        -------
        Dict[str, Any]
            Error summary statistics.
        """
        severity_counts = {}
        component_counts = {}

        for error in self.errors:
            # Count by severity
            severity_counts[error.severity] = severity_counts.get(error.severity, 0) + 1

            # Count by component
            component_counts[error.component] = (
                component_counts.get(error.component, 0) + 1
            )

        return {
            "total_errors": len(self.errors),
            "error_counts_by_category": dict(self.error_counts),
            "error_counts_by_severity": severity_counts,
            "error_counts_by_component": component_counts,
            "recovery_stats": dict(self.recovery_stats),
            "recent_errors": [
                {
                    "component": error.component,
                    "category": error.category,
                    "severity": error.severity,
                    "message": error.message,
                    "asn": error.asn,
                    "recovery_action": error.recovery_action,
                }
                for error in self.errors[-10:]  # Last 10 errors
            ],
        }

    def reset_error_tracking(self) -> None:
        """Reset error tracking statistics."""
        self.errors.clear()
        self.error_counts = {cat: 0 for cat in ErrorCategory}
        self.recovery_stats.clear()
        logger.info("Error tracking statistics reset")

    def export_error_log(self, output_path: str) -> None:
        """
        Export detailed error log to file.

        Parameters
        ----------
        output_path : str
            Path to save error log.
        """
        import json
        from datetime import datetime

        error_log = {
            "export_timestamp": datetime.now().isoformat(),
            "total_errors": len(self.errors),
            "summary": self.get_error_summary(),
            "detailed_errors": [
                {
                    "error_id": error.error_id,
                    "category": error.category,
                    "severity": error.severity,
                    "message": error.message,
                    "component": error.component,
                    "asn": error.asn,
                    "recovery_action": error.recovery_action,
                    "details": error.details,
                }
                for error in self.errors
            ],
        }

        with open(output_path, "w") as f:
            json.dump(error_log, f, indent=2, default=str)

        logger.info(f"Error log exported to {output_path}")


class RobustPipelineWrapper:
    """
    Wrapper that adds comprehensive error handling to pipeline components.

    This class provides a unified interface for making pipeline components
    more robust and fault-tolerant.
    """

    def __init__(self, error_handler: ErrorHandler):
        """
        Initialize robust pipeline wrapper.

        Parameters
        ----------
        error_handler : ErrorHandler
            Error handler instance to use.
        """
        self.error_handler = error_handler

    def wrap_stage1_classifier(self, classifier):
        """
        Wrap Stage 1 classifier with error handling.

        Parameters
        ----------
        classifier
            Stage 1 classifier to wrap.

        Returns
        -------
        Wrapped classifier with error handling.
        """
        original_predict = classifier.predict

        @self.error_handler.with_error_handling("stage1_classifier")
        def robust_predict(X: pd.DataFrame) -> List[TopLevelPrediction]:
            return original_predict(X)

        classifier.predict = robust_predict
        return classifier

    def wrap_stage2_classifier(self, classifier):
        """
        Wrap Stage 2 classifier with error handling.

        Parameters
        ----------
        classifier
            Stage 2 classifier to wrap.

        Returns
        -------
        Wrapped classifier with error handling.
        """
        original_predict = classifier.predict

        @self.error_handler.with_error_handling("stage2_classifier")
        def robust_predict(
            organizations, stage1_predictions
        ) -> List[SublevelPrediction]:
            return original_predict(organizations, stage1_predictions)

        classifier.predict = robust_predict
        return classifier

    def wrap_pipeline(self, pipeline):
        """
        Wrap entire pipeline with error handling.

        Parameters
        ----------
        pipeline
            Pipeline to wrap.

        Returns
        -------
        Wrapped pipeline with error handling.
        """
        # Wrap individual methods
        if hasattr(pipeline, "predict_single"):
            original_predict_single = pipeline.predict_single

            @self.error_handler.with_error_handling("pipeline_predict_single")
            def robust_predict_single(*args, **kwargs) -> TwoStageASClassification:
                return original_predict_single(*args, **kwargs)

            pipeline.predict_single = robust_predict_single

        if hasattr(pipeline, "predict_batch"):
            original_predict_batch = pipeline.predict_batch

            @self.error_handler.with_error_handling("pipeline_predict_batch")
            def robust_predict_batch(*args, **kwargs):
                return original_predict_batch(*args, **kwargs)

            pipeline.predict_batch = robust_predict_batch

        return pipeline


# Export main classes
__all__ = [
    "ErrorSeverity",
    "ErrorCategory",
    "PipelineError",
    "FallbackStrategy",
    "ErrorHandler",
    "RobustPipelineWrapper",
]
