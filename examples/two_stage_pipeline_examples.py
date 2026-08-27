#!/usr/bin/env python3
"""
Two-Stage Hierarchical AS Classification Pipeline - Usage Examples

This script demonstrates various usage patterns and capabilities of the
two-stage hierarchical classification pipeline for AS organizations.
"""

import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from linneaus.config import load_config
from linneaus.data.two_stage_data import DataSplitManager

# Import pipeline components
from linneaus.models.two_stage import TwoStageHierarchicalPipeline
from linneaus.models.two_stage.error_handling import ErrorHandler, FallbackStrategy
from linneaus.models.two_stage.evaluation import HierarchicalEvaluator


def example_1_basic_single_prediction():
    """
    Example 1: Basic single AS prediction

    Demonstrates the simplest use case - classifying a single AS organization.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Single AS Prediction")
    print("=" * 60)

    # Initialize pipeline
    pipeline = TwoStageHierarchicalPipeline()

    # Well-known AS examples
    test_cases = [
        {"asn": 15169, "name": "Google LLC", "expected": "Content"},
        {"asn": 7922, "name": "Comcast Cable Communications", "expected": "Access"},
        {"asn": 174, "name": "Cogent Communications", "expected": "Transit"},
        {"asn": 16509, "name": "Amazon.com, Inc.", "expected": "Content"},
    ]

    for case in test_cases:
        print(f"\nClassifying ASN {case['asn']} - {case['name']}")
        print("-" * 50)

        try:
            # Predict
            result = pipeline.predict_single(
                asn=case["asn"], organization_name=case["name"], include_timing=True
            )

            # Display results
            print(f"Organization: {result.organization_name}")
            print(f"Processing time: {result.processing_time:.3f}s")
            print(f"Overall confidence: {result.overall_confidence:.3f}")

            print("\nStage 1 Predictions (Top-Level):")
            for pred in result.stage1_predictions:
                marker = "✓" if pred.category.value == case["expected"] else " "
                print(f"  {marker} {pred.category.value}: {pred.confidence.value:.3f}")

            if result.stage2_predictions:
                print("\nStage 2 Predictions (Sublevel):")
                for pred in result.stage2_predictions:
                    print(
                        f"    {pred.parent_category.value} → {pred.subcategory}: {pred.confidence.value:.3f}"
                    )
            else:
                print("\nNo Stage 2 predictions generated")

        except Exception as e:
            print(f"ERROR: Failed to classify ASN {case['asn']}: {e}")


def example_2_batch_processing():
    """
    Example 2: Batch processing with performance monitoring

    Demonstrates batch processing capabilities and performance monitoring.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Batch Processing with Performance Monitoring")
    print("=" * 60)

    # Initialize pipeline
    pipeline = TwoStageHierarchicalPipeline()

    # Create a diverse batch of organizations
    organizations = [
        {"asn": 15169, "organization_name": "Google LLC"},
        {"asn": 32934, "organization_name": "Facebook, Inc."},
        {"asn": 16509, "organization_name": "Amazon.com, Inc."},
        {"asn": 7922, "organization_name": "Comcast Cable Communications"},
        {"asn": 174, "organization_name": "Cogent Communications"},
        {"asn": 1239, "organization_name": "Sprint"},
        {"asn": 3356, "organization_name": "Level 3 Parent, LLC"},
        {"asn": 8075, "organization_name": "Microsoft Corporation"},
        {"asn": 20940, "organization_name": "Akamai International B.V."},
        {"asn": 13335, "organization_name": "Cloudflare, Inc."},
    ]

    print(f"Processing batch of {len(organizations)} organizations...")

    # Test different batch processing modes
    modes = [
        {"parallel": False, "name": "Sequential"},
        {"parallel": True, "name": "Parallel"},
    ]

    for mode in modes:
        print(f"\n{mode['name']} Processing:")
        print("-" * 30)

        start_time = time.time()

        try:
            batch_result = pipeline.predict_batch(
                organizations=organizations,
                parallel=mode["parallel"],
                include_timing=True,
            )

            processing_time = time.time() - start_time

            # Display performance metrics
            print(f"Total processing time: {processing_time:.2f}s")
            print(
                f"Pipeline reported time: {batch_result.overall_performance.total_processing_time:.2f}s"
            )
            print(
                f"Average confidence: {batch_result.overall_performance.average_confidence:.3f}"
            )
            print(f"Success rate: {len(batch_result.results)} / {len(organizations)}")
            print(
                f"Throughput: {len(batch_result.results) / processing_time:.1f} predictions/sec"
            )

            # Show sample results
            print(f"\nSample Results (first 3):")
            for i, result in enumerate(batch_result.results[:3]):
                categories = [p.category.value for p in result.stage1_predictions]
                print(
                    f"  ASN {result.asn}: {categories} (conf: {result.overall_confidence:.3f})"
                )

        except Exception as e:
            print(f"ERROR: Batch processing failed: {e}")


def example_3_error_handling_and_resilience():
    """
    Example 3: Error handling and system resilience

    Demonstrates robust error handling with problematic inputs.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Error Handling and System Resilience")
    print("=" * 60)

    # Initialize with custom error handling
    error_handler = ErrorHandler(
        enable_graceful_degradation=True,
        max_retry_attempts=2,
        fallback_strategy=FallbackStrategy.DEFAULT,
    )

    pipeline = TwoStageHierarchicalPipeline(error_handler=error_handler)

    # Test cases with potential issues
    problematic_cases = [
        {"asn": 999999, "name": "Non-existent ASN", "issue": "Missing data"},
        {"asn": 888888, "name": "Another fake ASN", "issue": "Missing data"},
        {
            "asn": 15169,
            "name": "",
            "issue": "Empty organization name",
        },  # Valid ASN, empty name
        {"asn": 0, "name": "Invalid ASN Zero", "issue": "Invalid ASN"},
        {"asn": -1, "name": "Negative ASN", "issue": "Invalid ASN"},
    ]

    print("Testing error handling with problematic inputs...")

    successful_predictions = 0
    fallback_predictions = 0

    for case in problematic_cases:
        print(f"\nTesting: ASN {case['asn']} - {case['issue']}")
        print("-" * 40)

        try:
            result = pipeline.predict_single(
                asn=case["asn"], organization_name=case["name"]
            )

            # Check if this was a fallback result (very low confidence)
            if result.overall_confidence < 0.2:
                fallback_predictions += 1
                print(
                    f"  Result: Fallback prediction (confidence: {result.overall_confidence:.3f})"
                )
                print(
                    f"  Categories: {[p.category.value for p in result.stage1_predictions]}"
                )
            else:
                successful_predictions += 1
                print(
                    f"  Result: Normal prediction (confidence: {result.overall_confidence:.3f})"
                )
                print(
                    f"  Categories: {[p.category.value for p in result.stage1_predictions]}"
                )

        except Exception as e:
            print(f"  Result: Exception raised - {e}")

    # Display error statistics
    print(f"\n=== Error Handling Summary ===")
    error_summary = error_handler.get_error_summary()
    print(f"Total errors encountered: {error_summary['total_errors']}")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Fallback predictions: {fallback_predictions}")
    print(f"Recovery statistics: {error_summary['recovery_stats']}")

    if error_summary["recent_errors"]:
        print(f"\nRecent errors:")
        for error in error_summary["recent_errors"][:3]:
            print(f"  • {error['component']}: {error['message'][:50]}...")


def example_4_evaluation_framework():
    """
    Example 4: Using the evaluation framework

    Demonstrates comprehensive evaluation using test data.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Evaluation Framework")
    print("=" * 60)

    try:
        # Load test data
        data_manager = DataSplitManager()
        train_data, val_data, test_data = data_manager.get_splits()

        print(f"Loaded data splits:")
        print(f"  Training: {len(train_data)} samples")
        print(f"  Validation: {len(val_data)} samples")
        print(f"  Test: {len(test_data)} samples")

        # Use a small subset for demonstration
        test_subset = test_data.head(20)  # Use first 20 test samples
        print(f"\nUsing test subset of {len(test_subset)} samples for demonstration")

        # Initialize pipeline and evaluator
        pipeline = TwoStageHierarchicalPipeline()
        evaluator = HierarchicalEvaluator()

        # Generate predictions
        print("\nGenerating predictions...")
        predictions = []
        failed_predictions = 0

        for _, row in test_subset.iterrows():
            asn = row["ASN"]
            try:
                pred = pipeline.predict_single(asn)
                predictions.append(pred)
            except Exception as e:
                print(f"  Failed to predict ASN {asn}: {e}")
                failed_predictions += 1

        print(f"Generated {len(predictions)} predictions ({failed_predictions} failed)")

        if predictions:
            # Evaluate performance
            print("\nEvaluating performance...")
            results = evaluator.evaluate(
                predictions=predictions,
                ground_truth=test_subset,
                include_detailed_analysis=True,
            )

            # Display key metrics
            print(f"\n=== Performance Results ===")
            print(f"Stage 1 Accuracy: {results['stage1']['accuracy']:.3f}")
            print(f"Stage 1 F1-Score (Macro): {results['stage1']['f1_macro']:.3f}")
            print(
                f"Stage 1 Exact Match: {results['stage1']['exact_match_accuracy']:.3f}"
            )

            print(f"\nStage 2 Accuracy: {results['stage2']['overall_accuracy']:.3f}")
            print(f"Stage 2 Coverage: {results['stage2']['coverage']:.3f}")

            print(
                f"\nHierarchical Consistency: {results['consistency']['consistency_rate']:.3f}"
            )
            print(
                f"Consistency Violations: {results['consistency']['total_violations']}"
            )

            print(f"\nSystem Exact Match: {results['system']['exact_match_rate']:.3f}")
            print(
                f"System Partial Match: {results['system']['partial_match_rate']:.3f}"
            )

            # Top performing categories
            if "per_category" in results["stage1"]:
                sorted_cats = sorted(
                    results["stage1"]["per_category"].items(),
                    key=lambda x: x[1].get("f1", 0),
                    reverse=True,
                )

                print(f"\nTop Performing Categories (Stage 1):")
                for cat, metrics in sorted_cats[:5]:
                    print(
                        f"  {cat}: F1={metrics.get('f1', 0):.3f}, Support={metrics.get('support', 0)}"
                    )

            # Generate and save detailed report
            report_path = "evaluation_report_example.txt"
            evaluator.create_performance_report(results, report_path)
            print(f"\nDetailed evaluation report saved to: {report_path}")

        else:
            print("No successful predictions to evaluate")

    except Exception as e:
        print(f"ERROR: Evaluation example failed: {e}")
        print("This might be due to missing training data or configuration issues")


def example_5_performance_benchmarking():
    """
    Example 5: Performance benchmarking

    Demonstrates performance testing with different configurations.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Performance Benchmarking")
    print("=" * 60)

    # Test ASNs for benchmarking
    test_asns = [15169, 32934, 16509, 7922, 174, 1239, 3356, 8075, 20940, 13335]

    # Different batch sizes to test
    batch_sizes = [1, 5, 10]

    print("Benchmarking different batch sizes...")

    benchmark_results = []

    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        print("-" * 30)

        # Create organizations list
        organizations = [{"asn": asn} for asn in test_asns]

        # Initialize fresh pipeline for each test
        pipeline = TwoStageHierarchicalPipeline()

        # Warm up (exclude from timing)
        try:
            pipeline.predict_single(test_asns[0])
            print("  Pipeline warmed up")
        except:
            print("  Warning: Warmup failed")

        # Benchmark
        start_time = time.time()
        successful_predictions = 0
        total_confidence = 0.0

        # Process in batches
        batches = [
            organizations[i : i + batch_size]
            for i in range(0, len(organizations), batch_size)
        ]

        for batch in batches:
            try:
                batch_result = pipeline.predict_batch(batch, parallel=(batch_size > 1))
                successful_predictions += len(batch_result.results)
                total_confidence += sum(
                    r.overall_confidence for r in batch_result.results
                )
            except Exception as e:
                print(f"    Batch failed: {e}")

        elapsed_time = time.time() - start_time

        # Calculate metrics
        throughput = successful_predictions / elapsed_time if elapsed_time > 0 else 0
        avg_confidence = (
            total_confidence / successful_predictions
            if successful_predictions > 0
            else 0
        )

        # Store results
        result = {
            "batch_size": batch_size,
            "total_time": elapsed_time,
            "successful_predictions": successful_predictions,
            "throughput": throughput,
            "avg_confidence": avg_confidence,
        }
        benchmark_results.append(result)

        # Display results
        print(f"  Total time: {elapsed_time:.2f}s")
        print(
            f"  Successful predictions: {successful_predictions}/{len(organizations)}"
        )
        print(f"  Throughput: {throughput:.1f} predictions/sec")
        print(f"  Average confidence: {avg_confidence:.3f}")

    # Summary table
    print(f"\n=== Benchmark Summary ===")
    print(
        f"{'Batch Size':<12} {'Time (s)':<10} {'Success':<8} {'Throughput':<12} {'Avg Conf':<10}"
    )
    print("-" * 55)

    for result in benchmark_results:
        print(
            f"{result['batch_size']:<12} {result['total_time']:<10.2f} "
            f"{result['successful_predictions']:<8} {result['throughput']:<12.1f} "
            f"{result['avg_confidence']:<10.3f}"
        )


def example_6_custom_configuration():
    """
    Example 6: Custom configuration and model settings

    Demonstrates how to customize pipeline behavior through configuration.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Custom Configuration")
    print("=" * 60)

    # Load base configuration
    config = load_config()

    # Customize configuration
    print("Customizing pipeline configuration...")

    # Adjust Stage 1 ensemble weights
    config.two_stage_pipeline.stage1.svm_weight = 0.3
    config.two_stage_pipeline.stage1.llm_weight = 0.7

    # Adjust confidence thresholds
    config.two_stage_pipeline.stage1.confidence_threshold = 0.4
    config.two_stage_pipeline.stage2.confidence_threshold = 0.3

    # Customize error handling
    config.two_stage_pipeline.error_handling.max_retry_attempts = 1
    config.two_stage_pipeline.error_handling.enable_graceful_degradation = True

    print(f"  SVM weight: {config.two_stage_pipeline.stage1.svm_weight}")
    print(f"  LLM weight: {config.two_stage_pipeline.stage1.llm_weight}")
    print(
        f"  Stage 1 confidence threshold: {config.two_stage_pipeline.stage1.confidence_threshold}"
    )
    print(
        f"  Stage 2 confidence threshold: {config.two_stage_pipeline.stage2.confidence_threshold}"
    )

    # Initialize pipeline with custom config
    pipeline = TwoStageHierarchicalPipeline(config=config)

    # Test with custom configuration
    test_asns = [15169, 7922, 174]

    print(f"\nTesting custom configuration with {len(test_asns)} ASNs...")

    for asn in test_asns:
        try:
            result = pipeline.predict_single(asn)
            print(f"\nASN {asn}:")
            print(f"  Overall confidence: {result.overall_confidence:.3f}")
            print(f"  Stage 1 predictions: {len(result.stage1_predictions)}")
            print(f"  Stage 2 predictions: {len(result.stage2_predictions)}")

            for pred in result.stage1_predictions:
                print(f"    {pred.category.value}: {pred.confidence.value:.3f}")

        except Exception as e:
            print(f"ASN {asn} failed: {e}")


def main():
    """
    Main function to run all examples.
    """
    print("Two-Stage Hierarchical AS Classification Pipeline - Usage Examples")
    print("=" * 70)
    print("\nThis script demonstrates various capabilities of the pipeline:")
    print("1. Basic single predictions")
    print("2. Batch processing")
    print("3. Error handling and resilience")
    print("4. Evaluation framework")
    print("5. Performance benchmarking")
    print("6. Custom configuration")

    try:
        # Run all examples
        example_1_basic_single_prediction()
        example_2_batch_processing()
        example_3_error_handling_and_resilience()
        example_4_evaluation_framework()
        example_5_performance_benchmarking()
        example_6_custom_configuration()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"\nExample execution failed: {e}")
        logger.exception("Example execution failed")


if __name__ == "__main__":
    main()
