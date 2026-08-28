"""
Tests for hybrid model CLI functionality.

Tests the actual CLI commands that exist in linnaeus.cli.main:
- model predict: Classify organizations using various approaches
- benchmark: Benchmark multiple classification approaches
- model train: Train a new classification model (stub)
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from click.testing import CliRunner

from linnaeus.cli.main import main
from linnaeus.models.unified_schemas import (
    HierarchicalTags,
    TopLevelTags,
    UnifiedASClassification,
)


class TestHybridCLI:
    """Test CLI functionality for hybrid models."""

    @pytest.fixture
    def cli_runner(self):
        """Create CLI runner for testing."""
        return CliRunner()

    @pytest.fixture
    def sample_input_csv(self):
        """Create sample input CSV content."""
        return """asn
174
15169
32934"""

    @pytest.fixture
    def sample_unified_results(self):
        """Create sample unified classification results."""
        return [
            UnifiedASClassification(
                asn=174,
                organization_name="Cogent Communications",
                top_level_tags=[TopLevelTags.TRANSIT],
                hierarchical_tags=[HierarchicalTags.TRANSIT_GLOBAL],
                classification_approach="hybrid",
                model_used="HybridClassifier",
                top_level_confidence={"Transit": 0.95},
            ),
            UnifiedASClassification(
                asn=15169,
                organization_name="Google LLC",
                top_level_tags=[TopLevelTags.CONTENT_PROVIDER],
                hierarchical_tags=[HierarchicalTags.CONTENT_PROVIDER_CLOUD],
                classification_approach="hybrid",
                model_used="HybridClassifier",
                top_level_confidence={"Content Provider": 0.87},
            ),
        ]

    # -----------------------------------------------------------------------
    # Tests for: model predict
    # -----------------------------------------------------------------------

    def test_predict_command_hybrid_approach(
        self, cli_runner, sample_input_csv, sample_unified_results
    ):
        """Test predict command with hybrid approach."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.csv"
            output_file = Path(tmpdir) / "output.json"

            input_file.write_text(sample_input_csv)

            mock_classifier = Mock()
            mock_classifier.predict_unified.return_value = sample_unified_results

            with patch(
                "linnaeus.models.hybrid.stacking_classifier.HybridASClassifier",
                return_value=mock_classifier,
            ) as mock_hybrid_class:
                result = cli_runner.invoke(
                    main,
                    [
                        "model",
                        "predict",
                        "--approach",
                        "hybrid",
                        "--input",
                        str(input_file),
                        "--output",
                        str(output_file),
                        "--format",
                        "json",
                    ],
                )

                # The command should run and produce output about classification
                assert (
                    result.exit_code == 0
                ), f"Exit code: {result.exit_code}, Output: {result.output}"
                # Check output mentions hybrid approach
                assert "hybrid" in result.output.lower()
                # Check output mentions classified organizations
                assert "classified" in result.output.lower()

    def test_predict_command_svm_only(self, cli_runner, sample_input_csv):
        """Test predict command with SVM-only approach."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.csv"
            output_file = Path(tmpdir) / "output.csv"

            input_file.write_text(sample_input_csv)

            mock_classifier = Mock()
            mock_classifier.predict_unified = Mock(return_value=[])
            # Also mock hasattr behavior
            mock_classifier.__class__ = type("SVMClassifier", (), {})

            with patch(
                "linnaeus.models.svm.svm_models.SVMClassifier",
                return_value=mock_classifier,
            ):
                result = cli_runner.invoke(
                    main,
                    [
                        "model",
                        "predict",
                        "--approach",
                        "svm-only",
                        "--input",
                        str(input_file),
                        "--output",
                        str(output_file),
                        "--format",
                        "csv",
                    ],
                )

                assert (
                    result.exit_code == 0
                ), f"Exit code: {result.exit_code}, Output: {result.output}"
                assert "svm-only" in result.output.lower()

    def test_predict_command_hierarchical(
        self, cli_runner, sample_input_csv, sample_unified_results
    ):
        """Test predict command with hierarchical approach."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.csv"
            output_file = Path(tmpdir) / "output.json"

            input_file.write_text(sample_input_csv)

            mock_classifier = Mock()
            mock_classifier.predict_unified.return_value = sample_unified_results

            with patch(
                "linnaeus.models.hybrid.stacking_classifier.HybridASClassifier",
                return_value=mock_classifier,
            ) as mock_hybrid_class:
                result = cli_runner.invoke(
                    main,
                    [
                        "model",
                        "predict",
                        "--approach",
                        "hierarchical",
                        "--input",
                        str(input_file),
                        "--output",
                        str(output_file),
                    ],
                )

                assert (
                    result.exit_code == 0
                ), f"Exit code: {result.exit_code}, Output: {result.output}"
                assert "hierarchical" in result.output.lower()

    def test_predict_command_llm_only_not_available(self, cli_runner, sample_input_csv):
        """Test predict command with LLM-only approach shows not available message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.csv"
            output_file = Path(tmpdir) / "output.json"

            input_file.write_text(sample_input_csv)

            # The actual CLI exits with code 1 and prints a warning for llm-only
            result = cli_runner.invoke(
                main,
                [
                    "model",
                    "predict",
                    "--approach",
                    "llm-only",
                    "--input",
                    str(input_file),
                    "--output",
                    str(output_file),
                ],
            )

            # llm-only approach exits with sys.exit(1) in the actual CLI
            assert result.exit_code != 0

    def test_predict_command_json_output(
        self, cli_runner, sample_input_csv, sample_unified_results
    ):
        """Test JSON output format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.csv"
            output_file = Path(tmpdir) / "output.json"

            input_file.write_text(sample_input_csv)

            mock_classifier = Mock()
            mock_classifier.predict_unified.return_value = sample_unified_results

            with patch(
                "linnaeus.models.hybrid.stacking_classifier.HybridASClassifier",
                return_value=mock_classifier,
            ):
                result = cli_runner.invoke(
                    main,
                    [
                        "model",
                        "predict",
                        "--approach",
                        "hybrid",
                        "--input",
                        str(input_file),
                        "--output",
                        str(output_file),
                        "--format",
                        "json",
                    ],
                )

                assert (
                    result.exit_code == 0
                ), f"Exit code: {result.exit_code}, Output: {result.output}"

                # Verify JSON output file was created with correct structure
                assert output_file.exists(), "Output JSON file should be created"
                with open(output_file) as f:
                    json_data = json.load(f)

                assert len(json_data) == 2
                for item in json_data:
                    assert "asn" in item
                    assert "organization_name" in item
                    assert "top_level_tags" in item
                    assert "hierarchical_tags" in item
                    assert "model_used" in item

    def test_predict_command_csv_output(
        self, cli_runner, sample_input_csv, sample_unified_results
    ):
        """Test CSV output format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.csv"
            output_file = Path(tmpdir) / "output.csv"

            input_file.write_text(sample_input_csv)

            mock_classifier = Mock()
            mock_classifier.predict_unified.return_value = sample_unified_results

            with patch(
                "linnaeus.models.hybrid.stacking_classifier.HybridASClassifier",
                return_value=mock_classifier,
            ):
                result = cli_runner.invoke(
                    main,
                    [
                        "model",
                        "predict",
                        "--approach",
                        "hybrid",
                        "--input",
                        str(input_file),
                        "--output",
                        str(output_file),
                        "--format",
                        "csv",
                    ],
                )

                assert (
                    result.exit_code == 0
                ), f"Exit code: {result.exit_code}, Output: {result.output}"

                # Verify CSV output file was created
                assert output_file.exists(), "Output CSV file should be created"
                df = pd.read_csv(output_file)
                assert len(df) == 2
                assert "asn" in df.columns

    def test_predict_command_error_handling(self, cli_runner):
        """Test error handling in predict command for non-existent input file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Non-existent input file
            input_file = Path(tmpdir) / "nonexistent.csv"
            output_file = Path(tmpdir) / "output.json"

            result = cli_runner.invoke(
                main,
                [
                    "model",
                    "predict",
                    "--input",
                    str(input_file),
                    "--output",
                    str(output_file),
                ],
            )

            # Should fail due to missing input file (click validates exists=True)
            assert result.exit_code != 0

    def test_predict_command_import_error_handling(self, cli_runner, sample_input_csv):
        """Test handling of import errors for hybrid features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.csv"
            output_file = Path(tmpdir) / "output.json"

            input_file.write_text(sample_input_csv)

            # Simulate import failure for the hybrid classifier module
            with patch.dict(
                "sys.modules",
                {
                    "linnaeus.models.hybrid": None,
                    "linnaeus.models.hybrid.stacking_classifier": None,
                },
            ):
                result = cli_runner.invoke(
                    main,
                    [
                        "model",
                        "predict",
                        "--approach",
                        "hybrid",
                        "--input",
                        str(input_file),
                        "--output",
                        str(output_file),
                    ],
                )

                # Should handle import error gracefully (the CLI catches ImportError)
                assert "failed to import" in result.output.lower()

    # -----------------------------------------------------------------------
    # Tests for: model train (stub command)
    # -----------------------------------------------------------------------

    def test_train_command_stub(self, cli_runner):
        """Test that train command exists and shows not-yet-implemented message."""
        result = cli_runner.invoke(main, ["model", "train"])

        assert result.exit_code == 0
        assert (
            "training" in result.output.lower()
            or "not yet implemented" in result.output.lower()
        )

    # -----------------------------------------------------------------------
    # Tests for: benchmark (top-level command)
    # -----------------------------------------------------------------------

    def test_benchmark_command_basic(self, cli_runner):
        """Test benchmark command basic functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_file = Path(tmpdir) / "dataset.csv"
            output_file = Path(tmpdir) / "benchmark.json"

            # Create sample dataset
            dataset_file.write_text("asn\n174\n15169\n32934")

            mock_classifier = Mock()
            mock_classifier.predict_unified.return_value = [
                UnifiedASClassification(
                    asn=174,
                    organization_name="Test",
                    top_level_tags=[],
                    hierarchical_tags=[],
                    classification_approach="hybrid",
                    model_used="HybridClassifier",
                )
            ] * 3

            with (
                patch(
                    "linnaeus.models.hybrid.stacking_classifier.HybridASClassifier",
                    return_value=mock_classifier,
                ),
                patch(
                    "linnaeus.models.svm.svm_models.SVMClassifier",
                    return_value=mock_classifier,
                ),
            ):
                result = cli_runner.invoke(
                    main,
                    [
                        "benchmark",
                        "--models",
                        "hybrid",
                        "--dataset",
                        str(dataset_file),
                        "--output",
                        str(output_file),
                        "--sample-size",
                        "3",
                    ],
                )

                assert (
                    result.exit_code == 0
                ), f"Exit code: {result.exit_code}, Output: {result.output}"
                # The CLI outputs "Benchmark Summary" in the output
                assert "benchmark" in result.output.lower()

    def test_benchmark_command_model_selection(self, cli_runner):
        """Test benchmark command with specific model selection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_file = Path(tmpdir) / "dataset.csv"
            dataset_file.write_text("asn\n174\n15169")

            mock_classifier = Mock()
            mock_classifier.predict_unified.return_value = []

            with patch(
                "linnaeus.models.hybrid.stacking_classifier.HybridASClassifier",
                return_value=mock_classifier,
            ):
                result = cli_runner.invoke(
                    main,
                    [
                        "benchmark",
                        "--models",
                        "hybrid",
                        "--dataset",
                        str(dataset_file),
                        "--sample-size",
                        "2",
                    ],
                )

                assert (
                    result.exit_code == 0
                ), f"Exit code: {result.exit_code}, Output: {result.output}"
                assert "hybrid" in result.output.lower()

    def test_benchmark_command_invalid_model(self, cli_runner):
        """Test benchmark command with invalid model name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_file = Path(tmpdir) / "dataset.csv"
            dataset_file.write_text("asn\n174")

            result = cli_runner.invoke(
                main,
                [
                    "benchmark",
                    "--models",
                    "invalid-model",
                    "--dataset",
                    str(dataset_file),
                ],
            )

            # Should fail with invalid model (sys.exit(1))
            assert result.exit_code != 0
            assert "invalid model" in result.output.lower()

    def test_benchmark_command_sampling(self, cli_runner):
        """Test benchmark command with sampling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_file = Path(tmpdir) / "dataset.csv"

            # Create dataset with many ASNs
            asns = "\n".join([str(i) for i in range(174, 274)])  # 100 ASNs
            dataset_file.write_text(f"asn\n{asns}")

            mock_classifier = Mock()
            mock_classifier.predict_unified.return_value = []

            with patch(
                "linnaeus.models.hybrid.stacking_classifier.HybridASClassifier",
                return_value=mock_classifier,
            ):
                result = cli_runner.invoke(
                    main,
                    [
                        "benchmark",
                        "--models",
                        "hybrid",
                        "--dataset",
                        str(dataset_file),
                        "--sample-size",
                        "10",
                    ],
                )

                assert (
                    result.exit_code == 0
                ), f"Exit code: {result.exit_code}, Output: {result.output}"
                assert (
                    "sampled 10" in result.output.lower()
                    or "10" in result.output.lower()
                )

    # -----------------------------------------------------------------------
    # Tests for: CLI structure and help
    # -----------------------------------------------------------------------

    def test_main_help(self, cli_runner):
        """Test that main help command works."""
        result = cli_runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert (
            "linnaeus" in result.output.lower() or "autonomous" in result.output.lower()
        )

    def test_model_group_help(self, cli_runner):
        """Test that model group help works."""
        result = cli_runner.invoke(main, ["model", "--help"])
        assert result.exit_code == 0
        assert "predict" in result.output.lower()
        assert "train" in result.output.lower()

    def test_predict_help(self, cli_runner):
        """Test that predict help shows available approaches."""
        result = cli_runner.invoke(main, ["model", "predict", "--help"])
        assert result.exit_code == 0
        assert "hybrid" in result.output.lower()
        assert "hierarchical" in result.output.lower()
        assert "flat" in result.output.lower()

    def test_benchmark_help(self, cli_runner):
        """Test that benchmark help works."""
        result = cli_runner.invoke(main, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "models" in result.output.lower()
        assert "dataset" in result.output.lower()
