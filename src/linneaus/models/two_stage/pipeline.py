"""
Two-stage hierarchical classification pipeline controller.

This module implements the main controller that orchestrates both Stage 1
(top-level classification) and Stage 2 (sublevel classification) with
comprehensive error handling, monitoring, and performance optimization.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd

from ...data.access import DataAccessLayer

if TYPE_CHECKING:
    from ...config.two_stage_config import LinneausConfig
from ..two_stage_schemas import (
    BatchTwoStageClassificationResponse,
    ConfidenceScore,
    ModelType,
    PipelineConfig,
    SublevelPrediction,
    TopLevelPrediction,
    TwoStageASClassification,
)
from .assembled_classifier import AssembledTopLevelClassifier
from .error_handling import ErrorHandler, RobustPipelineWrapper
from .monitoring import PerformanceMonitor, PerformanceOptimizer
from .sublevel_classifier import HierarchicalSublevelClassifier

logger = logging.getLogger(__name__)


class TwoStageHierarchicalPipeline:
    """
    Main controller for two-stage hierarchical AS classification pipeline.

    This pipeline orchestrates both stages of classification:
    1. Stage 1: Top-level category prediction using assembled LLM+SVM model
    2. Stage 2: Sublevel category prediction using category-specific LLM models

    Features:
    - Parallel processing for batch inference
    - Comprehensive error handling and graceful degradation
    - Performance monitoring and logging
    - Confidence score tracking across stages
    - Model version tracking and attribution
    """

    def __init__(
        self,
        stage1_classifier: AssembledTopLevelClassifier,
        stage2_classifier: HierarchicalSublevelClassifier,
        config: PipelineConfig,
        data_access: Optional[DataAccessLayer] = None,
        max_workers: int = 4,
        enable_monitoring: bool = True,
    ):
        """
        Initialize the two-stage pipeline.

        Parameters
        ----------
        stage1_classifier : AssembledTopLevelClassifier
            Stage 1 assembled classifier.
        stage2_classifier : HierarchicalSublevelClassifier
            Stage 2 sublevel classifier.
        config : PipelineConfig
            Pipeline configuration.
        data_access : Optional[DataAccessLayer]
            Data access layer for organization data.
        max_workers : int
            Maximum number of parallel workers.
        enable_monitoring : bool
            Whether to enable performance monitoring.
        """
        self.stage1_classifier = stage1_classifier
        self.stage2_classifier = stage2_classifier
        self.config = config
        self.data_access = data_access or DataAccessLayer(enable_auto_download=True)
        self.max_workers = max_workers
        self.enable_monitoring = enable_monitoring

        # Performance monitoring
        self.monitor = PerformanceMonitor() if enable_monitoring else None
        self.optimizer = (
            PerformanceOptimizer(self.monitor) if enable_monitoring else None
        )

        # Initialize error handling
        self.error_handler = ErrorHandler(
            enable_graceful_degradation=True, max_retry_attempts=3
        )

        # Wrap components with error handling if monitoring enabled
        if enable_monitoring and self.monitor:
            wrapper = RobustPipelineWrapper(self.error_handler)
            self.stage1_classifier = wrapper.wrap_stage1_classifier(
                self.stage1_classifier
            )
            self.stage2_classifier = wrapper.wrap_stage2_classifier(
                self.stage2_classifier
            )

            # Start system monitoring
            self.monitor.start_system_monitoring(interval=2.0)

        # Performance tracking (legacy)
        self.performance_stats = {
            "total_predictions": 0,
            "stage1_time": 0.0,
            "stage2_time": 0.0,
            "total_time": 0.0,
            "errors": 0,
            "success_rate": 0.0,
        }

        # Model versions
        self.model_versions = {
            "pipeline_version": "1.0.0",
            "stage1_model": "assembled_v1",
            "stage2_models": ",".join(stage2_classifier.category_models.keys()),
        }

        logger.info("Initialized TwoStageHierarchicalPipeline")
        logger.info(f"Stage 1: {type(stage1_classifier).__name__}")
        logger.info(
            f"Stage 2: {len(stage2_classifier.category_models)} category models"
        )

    @classmethod
    def from_config(
        cls,
        config: "LinneausConfig",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> "TwoStageHierarchicalPipeline":
        """Construct the full pipeline from a :class:`LinneausConfig`.

        Parameters
        ----------
        config : LinneausConfig
            Full Linneaus configuration object.
        api_key : str, optional
            Override API key (falls back to config / env vars).
        base_url : str, optional
            Override base URL for OpenAI-compatible API.
        """
        from ...models.llm.client import create_client
        from ...models.llm.inference import HierarchicalBatchInferenceProcessor
        from ...models.llm.schemas import BatchTaggingResponseHierarchical

        ts = config.two_stage_pipeline
        llm = config.openai

        # Create OpenAI-compatible client
        client = create_client(
            api_key=api_key or llm.api_key,
            base_url=base_url or llm.base_url,
        )

        # Stage 1: Assembled top-level classifier
        processor = HierarchicalBatchInferenceProcessor(
            client=client,
            model=llm.default_model,
            response_format=BatchTaggingResponseHierarchical,
            system_prompt=ts.prompts.stage1_system,
            batch_size=ts.stage1.parallel_batch_size,
        )
        stage1_classifier = AssembledTopLevelClassifier(
            llm_processor=processor,
            svm_weight=ts.stage1.svm_weight,
            llm_weight=ts.stage1.llm_weight,
            parallel_batch_size=ts.stage1.parallel_batch_size,
        )

        # Stage 2: Sublevel classifier
        stage2_classifier = HierarchicalSublevelClassifier(
            openai_client=client,
            category_models=ts.stage2.category_models,
            category_prompts=ts.prompts.stage2_system_templates,
            batch_size=ts.stage2.batch_size,
            temperature=ts.stage2.temperature,
        )

        # Convert TwoStagePipelineConfig -> PipelineConfig (dict-based)
        pipeline_config = PipelineConfig(
            stage1=ts.stage1.model_dump(),
            stage2=ts.stage2.model_dump(),
            general=ts.general.model_dump(),
            prompts={
                "stage1_system": ts.prompts.stage1_system,
                **ts.prompts.stage2_system_templates,
            },
        )

        return cls(
            stage1_classifier=stage1_classifier,
            stage2_classifier=stage2_classifier,
            config=pipeline_config,
            max_workers=ts.general.max_workers,
            enable_monitoring=ts.general.enable_monitoring,
        )

    def predict_single(
        self,
        asn: int,
        organization_name: Optional[str] = None,
        include_timing: bool = False,
    ) -> TwoStageASClassification:
        """
        Predict classification for a single AS organization.

        Parameters
        ----------
        asn : int
            Autonomous System Number.
        organization_name : Optional[str]
            Organization name (will be fetched if not provided).
        include_timing : bool
            Whether to include timing information.

        Returns
        -------
        TwoStageASClassification
            Complete two-stage classification result.
        """
        # Monitor entire prediction process
        monitor_context = (
            self.monitor.monitor_component("pipeline_predict_single", asn=asn)
            if self.monitor
            else None
        )

        with monitor_context or self._null_context():
            start_time = time.time()

            try:
                # Get organization data
                with (
                    self.monitor.monitor_component("data_access", asn=asn)
                    if self.monitor
                    else self._null_context()
                ):
                    org_data = self._get_organization_data(asn, organization_name)

                # Stage 1: Top-level classification
                with (
                    self.monitor.monitor_component("stage1_prediction", asn=asn)
                    if self.monitor
                    else self._null_context()
                ) as stage1_metrics:
                    stage1_start = time.time()
                    stage1_predictions = self._predict_stage1([org_data])
                    stage1_time = time.time() - stage1_start

                    if stage1_metrics:
                        stage1_metrics.metadata.update(
                            {
                                "predictions_count": len(stage1_predictions),
                                "processing_time": stage1_time,
                            }
                        )

                # Stage 2: Sublevel classification
                with (
                    self.monitor.monitor_component("stage2_prediction", asn=asn)
                    if self.monitor
                    else self._null_context()
                ) as stage2_metrics:
                    stage2_start = time.time()
                    stage2_predictions = self._predict_stage2(
                        [org_data], stage1_predictions
                    )
                    stage2_time = time.time() - stage2_start

                    if stage2_metrics:
                        stage2_metrics.metadata.update(
                            {
                                "predictions_count": len(stage2_predictions),
                                "processing_time": stage2_time,
                            }
                        )

                # Calculate overall confidence
                overall_confidence = self._calculate_overall_confidence(
                    stage1_predictions, stage2_predictions
                )

                # Create result
                total_time = time.time() - start_time

                result = TwoStageASClassification(
                    asn=asn,
                    organization_name=org_data["name"],
                    stage1_predictions=stage1_predictions,
                    stage2_predictions=stage2_predictions,
                    overall_confidence=overall_confidence,
                    processing_time_seconds=total_time if include_timing else None,
                    model_versions=self.model_versions.copy(),
                )

                # Update performance stats
                if self.enable_monitoring:
                    self._update_performance_stats(
                        stage1_time=stage1_time,
                        stage2_time=stage2_time,
                        total_time=total_time,
                        success=True,
                    )

                logger.debug(f"Successfully predicted ASN {asn} in {total_time:.3f}s")
                return result

            except Exception as e:
                logger.error(f"Error predicting ASN {asn}: {e}")

                # Update error stats
                if self.enable_monitoring:
                    self._update_performance_stats(
                        stage1_time=0, stage2_time=0, total_time=0, success=False
                    )

                raise

    def predict_batch(
        self,
        organizations: List[Dict[str, Any]],
        parallel: bool = True,
        include_timing: bool = False,
    ) -> BatchTwoStageClassificationResponse:
        """
        Predict classifications for a batch of organizations.

        Parameters
        ----------
        organizations : List[Dict[str, Any]]
            List of organization data with 'asn', 'name', 'description'.
        parallel : bool
            Whether to use parallel processing.
        include_timing : bool
            Whether to include timing information.

        Returns
        -------
        BatchTwoStageClassificationResponse
            Batch classification response.
        """
        logger.info(f"Processing batch of {len(organizations)} organizations")
        batch_start_time = time.time()

        results = []
        errors = 0

        try:
            if parallel and len(organizations) > 1:
                results, errors = self._predict_batch_parallel(
                    organizations, include_timing
                )
            else:
                results, errors = self._predict_batch_sequential(
                    organizations, include_timing
                )

            total_time = time.time() - batch_start_time

            # Create batch response
            response = BatchTwoStageClassificationResponse(
                results=results,
                batch_size=len(organizations),
                total_processing_time_seconds=total_time if include_timing else None,
                success_count=len(results),
                error_count=errors,
                model_performance=(
                    self._get_batch_performance_metrics(results)
                    if self.enable_monitoring
                    else None
                ),
            )

            logger.info(
                f"Batch completed: {len(results)} successes, {errors} errors "
                f"in {total_time:.2f}s"
            )

            return response

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise

    def _predict_batch_parallel(
        self, organizations: List[Dict[str, Any]], include_timing: bool
    ) -> Tuple[List[TwoStageASClassification], int]:
        """
        Process batch in parallel.

        Parameters
        ----------
        organizations : List[Dict[str, Any]]
            Organizations to process.
        include_timing : bool
            Whether to include timing.

        Returns
        -------
        Tuple[List[TwoStageASClassification], int]
            Results and error count.
        """
        results = []
        errors = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_org = {
                executor.submit(
                    self.predict_single, org["asn"], org.get("name"), include_timing
                ): org
                for org in organizations
            }

            # Collect results
            for future in as_completed(future_to_org):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    org = future_to_org[future]
                    logger.error(f"Error processing ASN {org['asn']}: {e}")
                    errors += 1

        return results, errors

    def _predict_batch_sequential(
        self, organizations: List[Dict[str, Any]], include_timing: bool
    ) -> Tuple[List[TwoStageASClassification], int]:
        """
        Process batch sequentially.

        Parameters
        ----------
        organizations : List[Dict[str, Any]]
            Organizations to process.
        include_timing : bool
            Whether to include timing.

        Returns
        -------
        Tuple[List[TwoStageASClassification], int]
            Results and error count.
        """
        results = []
        errors = 0

        for org in organizations:
            try:
                result = self.predict_single(
                    org["asn"], org.get("name"), include_timing
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing ASN {org['asn']}: {e}")
                errors += 1

        return results, errors

    def _get_organization_data(
        self, asn: int, organization_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get organization data for prediction.

        Parameters
        ----------
        asn : int
            ASN to look up.
        organization_name : Optional[str]
            Pre-provided organization name.

        Returns
        -------
        Dict[str, Any]
            Organization data.
        """
        if organization_name:
            return {
                "asn": asn,
                "name": organization_name,
                "description": (
                    f"Autonomous System {asn}" f" operated by {organization_name}"
                ),
            }

        # Try to get from data access layer
        try:
            org_data = self.data_access.get_organization_data(asn)
            if org_data:
                # Build comprehensive description from available data
                description_parts = []

                # Add organization type if available
                if org_data.organization_type:
                    description_parts.append(
                        f"{org_data.organization_type} organization"
                    )

                # Add geographic info
                if org_data.country:
                    description_parts.append(f"based in {org_data.country}")

                # Add network scope if available
                if org_data.geographic_scope:
                    description_parts.append(f"with {org_data.geographic_scope} scope")

                # Construct description
                if description_parts:
                    org_name = org_data.name or f"AS{asn}"
                    parts = ", ".join(description_parts)
                    description = f"Autonomous System {asn}" f" - {org_name}: {parts}"
                else:
                    org_name = org_data.name or f"AS{asn}"
                    description = f"Autonomous System {asn}" f" operated by {org_name}"

                # Prepare comprehensive data for classification
                result = {
                    "asn": asn,
                    "name": org_data.name or f"AS{asn}",
                    "description": description,
                }

                # Add additional context from various sources
                if org_data.asrank:
                    result["asrank_rank"] = org_data.asrank.rank
                    result["organization_id"] = org_data.asrank.organization_id

                if org_data.peeringdb:
                    result["network_type"] = org_data.peeringdb.info_type
                    result["network_scope"] = org_data.peeringdb.info_scope

                if org_data.ipinfo:
                    result["ipinfo_type"] = org_data.ipinfo.type
                    result["ipinfo_allocation"] = org_data.ipinfo.allocation
                    result["registry"] = org_data.ipinfo.registry

                if org_data.aspop:
                    result["rir"] = org_data.aspop.rir

                return result

        except Exception as e:
            logger.warning(f"Could not fetch organization data for ASN {asn}: {e}")

        # Fallback
        return {
            "asn": asn,
            "name": f"AS{asn}",
            "description": f"Autonomous System {asn}",
        }

    def _predict_stage1(
        self, organizations: List[Dict[str, Any]]
    ) -> List[TopLevelPrediction]:
        """
        Execute Stage 1 prediction.

        Parameters
        ----------
        organizations : List[Dict[str, Any]]
            Organization data.

        Returns
        -------
        List[TopLevelPrediction]
            Stage 1 predictions.
        """
        try:
            # Convert to DataFrame format expected by classifier
            df = pd.DataFrame(organizations)

            # Run Stage 1 classifier
            predictions = self.stage1_classifier.predict(df)

            logger.debug(f"Stage 1 generated {len(predictions)} top-level predictions")
            return predictions

        except Exception as e:
            logger.error(f"Stage 1 prediction failed: {e}")

            # Fallback: create minimal predictions
            fallback_predictions = []
            for org in organizations:
                confidence = ConfidenceScore(value=0.1, model_type=ModelType.ASSEMBLED)
                # Use a default category as fallback
                from ..hierarchical_schemas import TopLevelCategory

                prediction = TopLevelPrediction(
                    category=TopLevelCategory.ENTERPRISE,  # Default fallback
                    confidence=confidence,
                )
                fallback_predictions.append(prediction)

            logger.warning(
                f"Using {len(fallback_predictions)} fallback predictions for Stage 1"
            )
            return fallback_predictions

    def _predict_stage2(
        self,
        organizations: List[Dict[str, Any]],
        stage1_predictions: List[TopLevelPrediction],
    ) -> List[SublevelPrediction]:
        """
        Execute Stage 2 prediction.

        Parameters
        ----------
        organizations : List[Dict[str, Any]]
            Organization data.
        stage1_predictions : List[TopLevelPrediction]
            Stage 1 predictions.

        Returns
        -------
        List[SublevelPrediction]
            Stage 2 predictions.
        """
        try:
            # Run Stage 2 classifier
            predictions = self.stage2_classifier.predict(
                organizations, stage1_predictions
            )

            logger.debug(f"Stage 2 generated {len(predictions)} sublevel predictions")
            return predictions

        except Exception as e:
            logger.error(f"Stage 2 prediction failed: {e}")

            # Return empty list - sublevel predictions are optional
            logger.warning("Stage 2 failed, proceeding without sublevel predictions")
            return []

    def _calculate_overall_confidence(
        self,
        stage1_predictions: List[TopLevelPrediction],
        stage2_predictions: List[SublevelPrediction],
    ) -> float:
        """
        Calculate overall confidence combining both stages.

        Parameters
        ----------
        stage1_predictions : List[TopLevelPrediction]
            Stage 1 predictions.
        stage2_predictions : List[SublevelPrediction]
            Stage 2 predictions.

        Returns
        -------
        float
            Overall confidence score.
        """
        if not stage1_predictions:
            return 0.0

        # Average Stage 1 confidences
        stage1_confidence = sum(
            pred.confidence.value for pred in stage1_predictions
        ) / len(stage1_predictions)

        if not stage2_predictions:
            return stage1_confidence

        # Average Stage 2 confidences
        stage2_confidence = sum(
            pred.confidence.value for pred in stage2_predictions
        ) / len(stage2_predictions)

        # Combine stages (weighted average)
        stage1_weight = self.config.stage1.get("confidence_weight", 0.6)
        stage2_weight = self.config.stage2.get("confidence_weight", 0.4)

        overall_confidence = (
            stage1_weight * stage1_confidence + stage2_weight * stage2_confidence
        )

        return min(1.0, max(0.0, overall_confidence))

    def _update_performance_stats(
        self, stage1_time: float, stage2_time: float, total_time: float, success: bool
    ) -> None:
        """Update performance statistics."""
        self.performance_stats["total_predictions"] += 1
        self.performance_stats["stage1_time"] += stage1_time
        self.performance_stats["stage2_time"] += stage2_time
        self.performance_stats["total_time"] += total_time

        if not success:
            self.performance_stats["errors"] += 1

        # Update success rate
        total = self.performance_stats["total_predictions"]
        errors = self.performance_stats["errors"]
        self.performance_stats["success_rate"] = (
            (total - errors) / total if total > 0 else 0.0
        )

    def _get_batch_performance_metrics(
        self, results: List[TwoStageASClassification]
    ) -> Dict[str, Any]:
        """
        Calculate batch performance metrics.

        Parameters
        ----------
        results : List[TwoStageASClassification]
            Batch results.

        Returns
        -------
        Dict[str, Any]
            Performance metrics.
        """
        if not results:
            return {}

        # Calculate average confidences
        avg_overall_confidence = sum(r.overall_confidence for r in results) / len(
            results
        )

        stage1_confidences = []
        stage2_confidences = []

        for result in results:
            stage1_confidences.extend(
                [p.confidence.value for p in result.stage1_predictions]
            )
            stage2_confidences.extend(
                [p.confidence.value for p in result.stage2_predictions]
            )

        avg_stage1_confidence = (
            sum(stage1_confidences) / len(stage1_confidences)
            if stage1_confidences
            else 0.0
        )
        avg_stage2_confidence = (
            sum(stage2_confidences) / len(stage2_confidences)
            if stage2_confidences
            else 0.0
        )

        return {
            "avg_overall_confidence": avg_overall_confidence,
            "avg_stage1_confidence": avg_stage1_confidence,
            "avg_stage2_confidence": avg_stage2_confidence,
            "stage1_predictions": len(stage1_confidences),
            "stage2_predictions": len(stage2_confidences),
            "pipeline_stats": self.performance_stats.copy(),
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get overall pipeline performance summary.

        Returns
        -------
        Dict[str, Any]
            Performance summary.
        """
        stats = self.performance_stats.copy()

        # Calculate averages
        total = stats["total_predictions"]
        if total > 0:
            stats["avg_stage1_time"] = stats["stage1_time"] / total
            stats["avg_stage2_time"] = stats["stage2_time"] / total
            stats["avg_total_time"] = stats["total_time"] / total
        else:
            stats["avg_stage1_time"] = 0.0
            stats["avg_stage2_time"] = 0.0
            stats["avg_total_time"] = 0.0

        return stats

    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self.performance_stats = {
            "total_predictions": 0,
            "stage1_time": 0.0,
            "stage2_time": 0.0,
            "total_time": 0.0,
            "errors": 0,
            "success_rate": 0.0,
        }
        logger.info("Performance statistics reset")

    def _null_context(self):
        """Null context manager for when monitoring is disabled."""
        from contextlib import nullcontext

        return nullcontext()

    def get_performance_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get performance optimization recommendations.

        Returns
        -------
        List[Dict[str, Any]]
            List of recommendations from the optimizer.
        """
        if not self.optimizer:
            return [
                {
                    "type": "info",
                    "message": "Monitoring disabled - no recommendations available",
                }
            ]

        return self.optimizer.generate_recommendations()

    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """
        Get detailed bottleneck analysis.

        Returns
        -------
        Dict[str, Any]
            Bottleneck analysis results.
        """
        if not self.optimizer:
            return {"error": "Monitoring disabled"}

        return self.optimizer.analyze_bottlenecks()

    def export_performance_metrics(self, output_path: str):
        """
        Export performance metrics to file.

        Parameters
        ----------
        output_path : str
            Path to save metrics.
        """
        if not self.monitor:
            logger.warning("Monitoring disabled - no metrics to export")
            return

        self.monitor.export_metrics(output_path, include_system_metrics=True)
        logger.info(f"Performance metrics exported to {output_path}")

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring summary.

        Returns
        -------
        Dict[str, Any]
            Complete monitoring summary including recommendations.
        """
        if not self.monitor:
            return {"error": "Monitoring disabled"}

        summary = {
            "pipeline_summary": self.monitor.get_pipeline_summary(),
            "system_metrics": self.monitor.get_system_metrics_summary(),
            "recommendations": self.get_performance_recommendations(),
            "bottlenecks": self.get_bottleneck_analysis(),
            "legacy_stats": self.get_performance_summary(),
        }

        return summary

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "monitor") and self.monitor:
            self.monitor.stop_system_monitoring()


# Export the main class
__all__ = ["TwoStageHierarchicalPipeline"]
