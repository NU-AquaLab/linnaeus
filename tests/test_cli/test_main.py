"""
Tests for CLI interface.
"""

from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from linneaus.cli.main import data, main, model


class TestCLI:
    """Test CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_main_help(self, runner):
        """Test main help command."""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "Linneaus" in result.output
        assert "AI-powered classification" in result.output
        assert "data" in result.output
        assert "model" in result.output

    def test_version(self, runner):
        """Test version command."""
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_info_command(self, runner):
        """Test info command."""
        result = runner.invoke(main, ["info"])

        assert result.exit_code == 0
        assert "Linneaus v0.1.0" in result.output
        assert "Author: Esteban" in result.output


class TestDataCommands:
    """Test data management commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_data_help(self, runner):
        """Test data command help."""
        result = runner.invoke(data, ["--help"])

        assert result.exit_code == 0
        assert "Data management commands" in result.output
        assert "download" in result.output
        assert "process" in result.output
        assert "status" in result.output

    @patch("linneaus.data.access.DataAccessLayer")
    def test_data_status(self, mock_data_layer, runner, mock_config):
        """Test data status command."""
        # Mock data layer
        mock_instance = Mock()
        mock_instance._asrank_cache = {"174": {}, "15169": {}}
        mock_instance._peeringdb_cache = {"174": {}, "15169": {}}
        mock_instance._aspop_cache = {"174": {}}
        mock_instance._load_asrank_cache = Mock()
        mock_instance._load_peeringdb_cache = Mock()
        mock_instance._load_aspop_cache = Mock()

        mock_data_layer.return_value = mock_instance

        with patch("linneaus.config.get_config", return_value=mock_config):
            result = runner.invoke(data, ["status"])

        assert result.exit_code == 0
        assert "Data Status Summary" in result.output
        assert "ASRank" in result.output
        assert "PeeringDB" in result.output
        assert "ASPOP" in result.output

    @patch("linneaus.data.downloaders.download_all_sources")
    @patch("asyncio.run")
    def test_data_download(
        self, mock_asyncio_run, mock_download, runner, mock_config, temp_dir
    ):
        """Test data download command."""
        # Mock successful download
        mock_download_result = {
            "peeringdb": temp_dir / "peeringdb.json",
            "asrank": temp_dir / "asrank.jsonl",
        }
        mock_asyncio_run.return_value = mock_download_result

        with patch("linneaus.config.get_config", return_value=mock_config):
            result = runner.invoke(
                data,
                ["download", "--sources", "peeringdb,asrank", "--date", "2024-01-01"],
            )

        assert result.exit_code == 0
        assert "Download completed successfully" in result.output
        mock_asyncio_run.assert_called_once()

    def test_data_download_failure(self, runner, mock_config):
        """Test data download failure handling."""
        with (
            patch("linneaus.config.get_config", return_value=mock_config),
            patch("asyncio.run", side_effect=Exception("Download failed")),
        ):

            result = runner.invoke(data, ["download"])

            assert result.exit_code == 1
            assert "Download failed" in result.output

    def test_data_process(self, runner, mock_config):
        """Test data process command."""
        with patch("linneaus.config.get_config", return_value=mock_config):
            result = runner.invoke(data, ["process"])

        assert result.exit_code == 0
        assert "Processing raw data files" in result.output
        assert "not yet implemented" in result.output


class TestModelCommands:
    """Test model commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_model_help(self, runner):
        """Test model command help."""
        result = runner.invoke(model, ["--help"])

        assert result.exit_code == 0
        assert "Model training and inference commands" in result.output
        assert "train" in result.output
        assert "predict" in result.output
        assert "evaluate" in result.output

    def test_model_train(self, runner, mock_config):
        """Test model train command."""
        with patch("linneaus.config.get_config", return_value=mock_config):
            result = runner.invoke(model, ["train"])

        assert result.exit_code == 0
        assert "Training stacking classifier" in result.output

    def test_model_predict(self, runner, mock_config, temp_dir):
        """Test model predict command."""
        # Create dummy input file
        input_file = temp_dir / "input.csv"
        input_file.write_text("asn,name\n174,Test\n")

        output_file = temp_dir / "output.json"

        with patch("linneaus.config.get_config", return_value=mock_config):
            result = runner.invoke(
                model,
                [
                    "predict",
                    "--input",
                    str(input_file),
                    "--output",
                    str(output_file),
                    "--format",
                    "json",
                ],
            )

        assert result.exit_code == 0
        assert "Classifying organizations" in result.output

    def test_model_evaluate(self, runner, mock_config, temp_dir):
        """Test model evaluate command."""
        # Create dummy CSV files with matching columns
        pred_file = temp_dir / "predictions.csv"
        pred_file.write_text("asn,Access,Transit\n174,1,0\n")

        truth_file = temp_dir / "truth.csv"
        truth_file.write_text("asn,Access,Transit\n174,1,1\n")

        with patch("linneaus.config.get_config", return_value=mock_config):
            result = runner.invoke(
                model,
                [
                    "evaluate",
                    "--predictions",
                    str(pred_file),
                    "--labels",
                    str(truth_file),
                ],
            )

        assert result.exit_code == 0
        assert "Evaluating predictions" in result.output


class TestConfigIntegration:
    """Test configuration integration with CLI."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_custom_config_file(self, runner, temp_dir):
        """Test using custom configuration file."""
        import sys

        # Create custom config
        config_file = temp_dir / "custom_config.yaml"
        config_data = """
        environment: testing
        openai:
          api_key: test_key
          default_model: gpt-4o-mini
        """
        config_file.write_text(config_data)

        # The CLI imports get_config at module level. Since the module attribute
        # 'main' conflicts with the click group name, we need to use the module
        # directly from sys.modules.
        cli_module = sys.modules["linneaus.cli.main"]
        with patch.object(cli_module, "get_config") as mock_get_config:
            result = runner.invoke(main, ["--config", str(config_file), "info"])

        assert result.exit_code == 0
        mock_get_config.assert_called_with(config_file)

    def test_missing_config_file(self, runner):
        """Test behavior with missing config file."""
        result = runner.invoke(main, ["--config", "/nonexistent/config.yaml", "info"])

        # Should fail gracefully
        assert result.exit_code != 0
