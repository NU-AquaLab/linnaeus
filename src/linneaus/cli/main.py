"""
Main CLI entry point for Linneaus.

This module provides the command-line interface for the Linneaus autonomous
systems classification tool.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from ..config import get_config
from ..data.access import DataAccessLayer
from ..data.downloaders import download_all_sources
from ..data.validation import DataValidator, validate_all_data_sources
from ..models.llm.schema_generation import BUILTIN_TAXONOMIES as _BUILTIN_TAXONOMIES
from ..models.llm.schema_generation import get_builtin_taxonomy_path

console = Console()


def llm_options(func):
    """Add ``--base-url`` and ``--api-key`` options for LLM provider selection."""
    func = click.option(
        "--base-url",
        envvar="LLM_BASE_URL",
        default=None,
        help=(
            "Base URL for an OpenAI-compatible API. Examples: "
            "https://api.anthropic.com/v1/ (Claude), "
            "http://localhost:11434/v1 (Ollama)"
        ),
    )(func)
    func = click.option(
        "--api-key",
        envvar=["LLM_API_KEY", "OPENAI_API_KEY"],
        default=None,
        help="API key for the LLM provider",
    )(func)
    return func


def _resolve_taxonomy_path(
    taxonomy: Optional[str], taxonomy_file: Optional[str]
) -> Optional[Path]:
    """Resolve ``--taxonomy`` / ``--taxonomy-file`` to a filesystem path.

    Returns ``None`` when the default linneaus taxonomy should be used
    (i.e. loaded from the package resource).
    """
    if taxonomy_file:
        return Path(taxonomy_file)
    if taxonomy and taxonomy != "linneaus":
        return get_builtin_taxonomy_path(taxonomy)
    return None


@click.group()
@click.version_option()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
def main(config: Optional[Path] = None) -> None:
    """
    Linneaus: AI-powered classification of Internet Autonomous Systems.

    A tool for automatically classifying and analyzing Autonomous Systems (AS)
    that make up the Internet using machine learning and large language models.
    """
    if config:
        # Load custom config file
        get_config(config)


@main.command()
def init() -> None:
    """Create a starter config.yaml in the current directory."""
    import shutil

    dest = Path("config.yaml")
    if dest.exists():
        console.print("[yellow]config.yaml already exists, skipping.[/yellow]")
        return

    template = (
        Path(__file__).resolve().parent.parent / "resources" / "config.yaml.example"
    )
    if not template.exists():
        console.print("[red]Template config not found in package resources.[/red]")
        sys.exit(1)

    shutil.copy2(template, dest)
    console.print(f"[green]Created config.yaml in {dest.resolve()}[/green]")


@main.group()
def data():
    """Data management commands."""


@main.group()
def model():
    """Model training and inference commands."""


@main.group()
def two_stage():
    """Two-stage hierarchical classification pipeline commands."""


@data.command()
@click.option(
    "--sources",
    default="peeringdb,asrank,aspop,ipinfo",
    help="Comma-separated list of data sources to download",
)
@click.option("--date", help="Target date for data snapshots (YYYY-MM-DD)")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Output directory for downloaded data",
)
@click.option(
    "--force-refresh", is_flag=True, help="Force refresh even if cached data exists"
)
def download(
    sources: str, date: Optional[str], output_dir: Optional[Path], force_refresh: bool
) -> None:
    """
    Download data from external sources.

    Downloads fresh data snapshots from various sources including PeeringDB,
    ASRank, APNIC ASPOP, and IPinfo. By default, downloads the latest available data.

    Note: IPinfo requires specific ASN(s) for meaningful data. Use individual
    downloader methods for targeted IPinfo downloads.
    """
    try:
        source_list = [s.strip() for s in sources.split(",")]

        console.print(
            f"[blue]Downloading data sources: {', '.join(source_list)}[/blue]"
        )
        if date:
            console.print(f"[blue]Target date: {date}[/blue]")

        # Run async download
        downloaded_files = asyncio.run(
            download_all_sources(source_list, date, output_dir)
        )

        if downloaded_files:
            console.print("\n[green]✓ Download completed successfully![/green]")

            # Show summary table
            table = Table(title="Downloaded Files")
            table.add_column("Source", style="cyan")
            table.add_column("File", style="green")

            for source, file_path in downloaded_files.items():
                table.add_row(source, str(file_path))

            console.print(table)
        else:
            console.print("[red]✗ No files were downloaded successfully[/red]")
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]✗ Download failed: {e}[/red]")
        sys.exit(1)


@data.command()
@click.option(
    "--asn", type=int, required=True, help="Autonomous System Number to download"
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Output directory for downloaded data",
)
def download_ipinfo(asn: int, output_dir: Optional[Path]) -> None:
    """
    Download IPinfo data for a specific ASN.

    Downloads organization and network information for a single ASN
    from IPinfo's API. Requires IPINFO_TOKEN environment variable
    for premium features.

    Examples:
    \b
    # Download data for Google's ASN
    linneaus data download-ipinfo --asn 15169

    \b
    # Download with custom output directory
    linneaus data download-ipinfo --asn 7922 --output-dir /path/to/data
    """
    try:
        from ..data.downloaders import IPinfoDownloader

        console.print(f"[blue]Downloading IPinfo data for ASN {asn}...[/blue]")

        # Initialize downloader
        downloader = IPinfoDownloader(output_dir)

        # Download data
        data = asyncio.run(downloader.download_asn(asn))

        if "error" in data:
            console.print(f"[red]✗ Failed to download ASN {asn}: {data['error']}[/red]")
            sys.exit(1)

        # Save individual file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = (
            output_dir or downloader.output_dir
        ) / f"ipinfo_AS{asn}_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        console.print("[green]✓ IPinfo data downloaded successfully[/green]")
        console.print(f"[green]✓ Saved to: {output_file}[/green]")

        # Show key information
        if "org" in data:
            console.print(f"Organization: {data['org']}")
        if "country" in data:
            console.print(f"Country: {data['country']}")
        if "region" in data:
            console.print(f"Region: {data['region']}")

    except ImportError as e:
        console.print(f"[red]✗ Failed to import IPinfo downloader: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]✗ IPinfo download failed: {e}[/red]")
        sys.exit(1)


@data.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing data files to validate",
)
@click.option(
    "--output-report",
    type=click.Path(path_type=Path),
    help="Path to save validation report",
)
@click.option(
    "--source",
    type=click.Choice(["asrank", "peeringdb", "aspop", "ipinfo", "all"]),
    default="all",
    help="Specific data source to validate",
)
def validate(
    data_dir: Optional[Path], output_report: Optional[Path], source: str
) -> None:
    """
    Validate data quality and generate quality reports.

    Performs comprehensive validation of downloaded and processed data,
    checking for completeness, consistency, and accuracy. Generates
    detailed quality reports with recommendations for improvement.

    Examples:
    \b
    # Validate all data sources
    linneaus data validate

    \b
    # Validate specific source
    linneaus data validate --source asrank

    \b
    # Generate report to file
    linneaus data validate --output-report quality_report.md
    """
    try:
        config = get_config()

        # Use provided data directory or default from config
        if data_dir is None:
            data_dir = config.data.processed_data_dir

        if not data_dir.exists():
            console.print(f"[red]✗ Data directory not found: {data_dir}[/red]")
            sys.exit(1)

        console.print(f"[blue]Validating data in: {data_dir}[/blue]")

        if source == "all":
            # Validate all available sources
            results = validate_all_data_sources(data_dir, output_report)

            if not results:
                console.print("[yellow]⚠ No data files found to validate[/yellow]")
                sys.exit(0)

            # Summary statistics
            total_sources = len(results)
            passed_sources = sum(1 for r in results if r.is_valid)
            failed_sources = total_sources - passed_sources

            console.print("\n[blue]Validation Summary:[/blue]")
            console.print(f"  Total sources: {total_sources}")
            console.print(f"  [green]Passed: {passed_sources}[/green]")
            console.print(f"  [red]Failed: {failed_sources}[/red]")

            if failed_sources > 0:
                console.print(
                    "\n[yellow]⚠ Some data sources failed"
                    " validation. Check the detailed"
                    " report.[/yellow]"
                )
                sys.exit(1)
            else:
                console.print("\n[green]✓ All data sources passed validation![/green]")

        else:
            # Validate specific source
            validator = DataValidator()

            # Find the most recent file for the specified source
            if source == "asrank":
                files = list(data_dir.glob("as_rank_*.csv"))
                if files:
                    latest_file = max(files, key=lambda x: x.stat().st_mtime)
                    result = validator.validate_asrank_data(latest_file)
                else:
                    console.print(f"[red]✗ No ASRank files found in {data_dir}[/red]")
                    sys.exit(1)

            elif source == "peeringdb":
                files = list(data_dir.glob("peeringdb_*.json"))
                if files:
                    latest_file = max(files, key=lambda x: x.stat().st_mtime)
                    result = validator.validate_peeringdb_data(latest_file)
                else:
                    console.print(
                        f"[red]✗ No PeeringDB files found in {data_dir}[/red]"
                    )
                    sys.exit(1)

            elif source == "aspop":
                files = list(data_dir.glob("aspop_*.csv")) + list(
                    data_dir.glob("aspop_*.json")
                )
                if files:
                    latest_file = max(files, key=lambda x: x.stat().st_mtime)
                    result = validator.validate_aspop_data(latest_file)
                else:
                    console.print(f"[red]✗ No ASPOP files found in {data_dir}[/red]")
                    sys.exit(1)

            elif source == "ipinfo":
                files = list(data_dir.glob("ipinfo_*.json"))
                if files:
                    latest_file = max(files, key=lambda x: x.stat().st_mtime)
                    result = validator.validate_ipinfo_data(latest_file)
                else:
                    console.print(f"[red]✗ No IPinfo files found in {data_dir}[/red]")
                    sys.exit(1)

            # Display results
            metrics = result.quality_metrics
            status = "PASS" if result.is_valid else "FAIL"
            status_style = "green" if result.is_valid else "red"

            console.print(f"\n[blue]{source.upper()} Validation Results:[/blue]")
            console.print(f"  Status: [{status_style}]{status}[/{status_style}]")
            console.print(
                f"  Records: {metrics.valid_records:,}/{metrics.total_records:,}"
            )
            console.print(f"  Quality Score: {metrics.overall_quality_score:.1%}")
            console.print(f"  Completeness: {metrics.completeness_score:.1%}")
            console.print(f"  Consistency: {metrics.consistency_score:.1%}")
            console.print(f"  Accuracy: {metrics.accuracy_score:.1%}")

            if result.errors:
                console.print("\n[red]Errors:[/red]")
                for error in result.errors:
                    console.print(f"  - {error}")

            if result.warnings:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in result.warnings:
                    console.print(f"  - {warning}")

            if result.recommendations:
                console.print("\n[blue]Recommendations:[/blue]")
                for rec in result.recommendations:
                    console.print(f"  - {rec}")

            # Save report if requested
            if output_report:
                report = validator.generate_quality_report([result])
                with open(output_report, "w", encoding="utf-8") as f:
                    f.write(report)
                console.print(f"\n[green]✓ Report saved to {output_report}[/green]")

            if not result.is_valid:
                sys.exit(1)

    except Exception as e:
        console.print(f"[red]✗ Validation failed: {e}[/red]")
        sys.exit(1)


@data.command()
@click.option(
    "--asns",
    required=True,
    help="Comma-separated list of ASNs or path to file with ASNs",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Output directory for downloaded data",
)
@click.option(
    "--max-concurrent", type=int, default=5, help="Maximum concurrent requests"
)
def download_ipinfo_batch(
    asns: str, output_dir: Optional[Path], max_concurrent: int
) -> None:
    """
    Download IPinfo data for multiple ASNs.

    Downloads organization and network information for multiple ASNs
    from IPinfo's API using concurrent requests.

    Examples:
    \b
    # Download data for multiple ASNs
    linneaus data download-ipinfo-batch --asns "15169,7922,8075"

    \b
    # Download from file (one ASN per line)
    linneaus data download-ipinfo-batch --asns asn_list.txt --max-concurrent 10
    """
    try:
        pass

        from ..data.downloaders import IPinfoDownloader

        # Parse ASN input
        asn_list = []
        if asns.endswith(".txt") or asns.endswith(".csv"):
            # Read from file
            asn_file = Path(asns)
            if not asn_file.exists():
                console.print(f"[red]✗ ASN file not found: {asns}[/red]")
                sys.exit(1)

            with open(asn_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and line.isdigit():
                        asn_list.append(int(line))
        else:
            # Parse comma-separated list
            try:
                asn_list = [int(asn.strip()) for asn in asns.split(",")]
            except ValueError:
                console.print(
                    "[red]✗ Invalid ASN format. Use"
                    " comma-separated numbers or file path[/red]"
                )
                sys.exit(1)

        if not asn_list:
            console.print("[red]✗ No valid ASNs found[/red]")
            sys.exit(1)

        console.print(
            f"[blue]Downloading IPinfo data for {len(asn_list)} ASNs...[/blue]"
        )

        # Initialize downloader and download
        downloader = IPinfoDownloader(output_dir)
        output_file = asyncio.run(downloader.download_batch(asn_list, max_concurrent))

        console.print("[green]✓ Batch download completed[/green]")
        console.print(f"[green]✓ Results saved to: {output_file}[/green]")

    except ImportError as e:
        console.print(f"[red]✗ Failed to import IPinfo downloader: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]✗ IPinfo batch download failed: {e}[/red]")
        sys.exit(1)


@data.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Input directory containing raw data files",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Output directory for processed data",
)
def process(input_dir: Optional[Path], output_dir: Optional[Path]) -> None:
    """
    Process raw data files into structured formats.

    Processes downloaded raw data files and converts them into the structured
    formats used by the classification pipeline.
    """
    console.print("[blue]Processing raw data files...[/blue]")

    # TODO: Implement data processing pipeline
    console.print("[yellow]⚠ Data processing not yet implemented[/yellow]")


@data.command()
def status() -> None:
    """
    Show data availability and status.

    Displays information about available data sources, their freshness,
    and overall data status.
    """
    try:
        data_layer = DataAccessLayer()

        console.print("[blue]Data Status Summary[/blue]\n")

        # Create status table
        table = Table(title="Data Source Status")
        table.add_column("Source", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Records", justify="right")
        table.add_column("Last Updated")

        # Check each data source
        sources_info = []

        # ASRank status
        try:
            data_layer._load_asrank_cache()
            if data_layer._asrank_cache:
                record_count = len(data_layer._asrank_cache)
                status = "✓ Available"
                sources_info.append(("ASRank", status, str(record_count), "Unknown"))
            else:
                sources_info.append(("ASRank", "✗ Missing", "0", "N/A"))
        except Exception:
            sources_info.append(("ASRank", "✗ Error", "0", "N/A"))

        # PeeringDB status
        try:
            data_layer._load_peeringdb_cache()
            if data_layer._peeringdb_cache:
                record_count = len(data_layer._peeringdb_cache)
                status = "✓ Available"
                sources_info.append(("PeeringDB", status, str(record_count), "Unknown"))
            else:
                sources_info.append(("PeeringDB", "✗ Missing", "0", "N/A"))
        except Exception:
            sources_info.append(("PeeringDB", "✗ Error", "0", "N/A"))

        # ASPOP status
        try:
            data_layer._load_aspop_cache()
            if data_layer._aspop_cache:
                record_count = len(data_layer._aspop_cache)
                status = "✓ Available"
                sources_info.append(("ASPOP", status, str(record_count), "Unknown"))
            else:
                sources_info.append(("ASPOP", "✗ Missing", "0", "N/A"))
        except Exception:
            sources_info.append(("ASPOP", "✗ Error", "0", "N/A"))

        # Add rows to table
        for source, status, count, updated in sources_info:
            table.add_row(source, status, count, updated)

        console.print(table)

        # Overall status
        available_sources = sum(
            1 for _, status, _, _ in sources_info if "Available" in status
        )
        total_sources = len(sources_info)

        if available_sources == total_sources:
            console.print(
                f"\n[green]✓ All {total_sources} data sources are available[/green]"
            )
        elif available_sources > 0:
            console.print(
                f"\n[yellow]⚠ {available_sources}/"
                f"{total_sources} data sources are"
                " available[/yellow]"
            )
        else:
            console.print("\n[red]✗ No data sources are available[/red]")
            console.print("[blue]Run 'linneaus data download' to fetch data[/blue]")

    except Exception as e:
        console.print(f"[red]✗ Failed to check data status: {e}[/red]")
        sys.exit(1)


@model.command()
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=Path("data"),
    help="Data directory",
)
@click.option(
    "--select-tags",
    default="reduce_tags",
    help="Tag selection: 'all_tags', 'reduce_tags', or a category prefix",
)
@click.option(
    "--llm-preds",
    type=click.Path(exists=True, path_type=Path),
    help="Path to LLM predictions CSV",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("models"),
    help="Output directory for model",
)
@click.option("--svm-c", type=float, default=5.0, help="SVM C parameter")
@click.option("--svm-gamma", type=float, default=0.0001, help="SVM gamma parameter")
@click.option("--svm-kernel", default="rbf", help="SVM kernel type")
@click.option("--random-state", type=int, default=42, help="Random seed")
@click.option(
    "--cv-folds", type=int, default=2, help="Cross-validation folds for stacking"
)
def train(
    data_dir,
    select_tags,
    llm_preds,
    output_dir,
    svm_c,
    svm_gamma,
    svm_kernel,
    random_state,
    cv_folds,
):
    """Train the stacking classifier (SVM + LLM)."""
    import json
    from datetime import datetime

    import pandas as pd
    from sklearn.impute import KNNImputer
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC

    from ..data.labeled_data import LabeledDataManager
    from ..models.hybrid.svc_llm_estimator import SVCLLMEstimator

    console.print("[bold]Training stacking classifier...[/bold]")

    # Load labeled data
    manager = LabeledDataManager(data_dir)
    manager.load_legacy_labels()

    # Load features (this would need actual feature extraction)
    console.print(f"Tag selection: {select_tags}")

    # Load LLM predictions
    if llm_preds:
        llm_preds_df = pd.read_csv(llm_preds)
        if (
            "asn" not in llm_preds_df.columns
            and llm_preds_df.columns[0].lower() == "asn"
        ):
            llm_preds_df.columns = ["asn"] + list(llm_preds_df.columns[1:])
    else:
        console.print(
            "[yellow]No LLM predictions provided. Training SVM-only model.[/yellow]"
        )
        llm_preds_df = None

    # Setup parameters
    svm_params = {
        "C": svm_c,
        "gamma": svm_gamma,
        "kernel": svm_kernel,
        "probability": True,
    }

    meta_learner = SVC(kernel="linear", probability=True, random_state=random_state)
    cv_stacking = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )
    imputer = KNNImputer(n_neighbors=5)

    # Create and train estimator
    SVCLLMEstimator(
        svm_params=svm_params,
        meta_learner=meta_learner,
        cv_stacking=cv_stacking,
        llm_preds_df=llm_preds_df,
        imputer=imputer,
        random_state=random_state,
    )

    console.print("[bold green]Training pipeline ready.[/bold green]")
    console.print(
        "Note: Full training requires feature-extracted"
        " data. Use 'linneaus data prepare' first."
    )

    # Save metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(output_dir) / f"model_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp": timestamp,
        "svm_params": svm_params,
        "select_tags": select_tags,
        "random_state": random_state,
        "cv_folds": cv_folds,
    }

    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    console.print(f"Model metadata saved to: {metadata_path}")


@model.command()
@click.option("--model", help="Model identifier or path to use for prediction")
@click.option(
    "--approach",
    type=click.Choice(
        ["flat", "hierarchical", "hybrid", "svm-only", "llm-only", "two-stage"]
    ),
    default="two-stage",
    help="Classification approach to use",
)
@click.option(
    "--input",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input file with ASNs to classify (CSV with 'asn' column)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output file for classification results",
)
@click.option(
    "--format",
    type=click.Choice(["json", "csv", "excel"]),
    default="json",
    help="Output format",
)
@click.option(
    "--batch-size",
    type=int,
    default=10,
    help="Batch size for processing",
)
@click.option(
    "--taxonomy",
    type=click.Choice(list(_BUILTIN_TAXONOMIES)),
    default="linneaus",
    help="Taxonomy to use for classification",
)
@click.option(
    "--taxonomy-file",
    type=click.Path(exists=True),
    default=None,
    help="Custom taxonomy definitions JSON file",
)
@llm_options
def predict(
    model: Optional[str],
    approach: str,
    input: Path,
    output: Path,
    format: str,
    batch_size: int,
    taxonomy: str,
    taxonomy_file: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
) -> None:
    """
    Classify organizations using a trained model.

    Takes a list of AS organizations and produces classification predictions
    using the specified approach: flat, hierarchical, hybrid, svm-only, or llm-only.

    Examples:
    \b
    # Hybrid classification (default)
    linneaus model predict --input asns.csv --output results.json

    \b
    # SVM-only approach
    linneaus model predict --approach svm-only \\
        --input asns.csv --output results.csv

    \b
    # Hierarchical classification
    linneaus model predict --approach hierarchical \\
        --input asns.csv --output results.json

    \b
    # Two-stage hierarchical classification (recommended)
    linneaus model predict --approach two-stage --input asns.csv --output results.json
    """
    tax_path = _resolve_taxonomy_path(taxonomy, taxonomy_file)
    if tax_path:
        console.print(f"[blue]Using taxonomy: {tax_path}[/blue]")
        if approach != "llm-only":
            console.print(
                "[red]✗ Non-default taxonomies are only supported with "
                "--approach llm-only (the other approaches use models trained "
                "on the linneaus taxonomy).[/red]"
            )
            sys.exit(1)

    console.print(
        f"[blue]Classifying organizations from {input}"
        f" using {approach} approach...[/blue]"
    )

    try:
        # Import hybrid models
        import json
        import os

        import pandas as pd

        from ..models.hybrid.stacking_classifier import HybridASClassifier
        from ..models.svm.svm_models import SVMClassifier
        from ..models.unified_schemas import UnifiedASClassification

        # Propagate CLI overrides so downstream client factories pick them up
        if api_key:
            os.environ["LLM_API_KEY"] = api_key
        if base_url:
            os.environ["LLM_BASE_URL"] = base_url

        # Read input data
        console.print(f"[blue]Reading input data from {input}...[/blue]")
        if input.suffix.lower() == ".csv":
            df = pd.read_csv(input)
        elif input.suffix.lower() == ".json":
            df = pd.read_json(input)
        else:
            console.print(f"[red]✗ Unsupported input format: {input.suffix}[/red]")
            sys.exit(1)

        if "asn" not in df.columns:
            console.print("[red]✗ Input file must contain 'asn' column[/red]")
            sys.exit(1)

        console.print(f"[green]✓ Loaded {len(df)} ASNs for classification[/green]")

        # Initialize classifier based on approach
        classifier = None

        if approach == "hybrid":
            console.print("[blue]Initializing hybrid classifier...[/blue]")
            classifier = HybridASClassifier(
                approach="hybrid", llm_model_id=model, use_hierarchical_svm=True
            )
        elif approach == "hierarchical":
            console.print("[blue]Initializing hierarchical classifier...[/blue]")
            classifier = HybridASClassifier(
                approach="hierarchical", llm_model_id=model, use_hierarchical_svm=True
            )
        elif approach == "flat":
            console.print("[blue]Initializing flat classifier...[/blue]")
            classifier = HybridASClassifier(
                approach="flat", llm_model_id=model, use_hierarchical_svm=False
            )
        elif approach == "svm-only":
            console.print("[blue]Initializing SVM-only classifier...[/blue]")
            classifier = SVMClassifier(approach="flat")
        elif approach == "llm-only":
            # Direct LLM classification with structured output. Works with any
            # taxonomy (builtin or custom) and any OpenAI-compatible model.
            from ..models.llm.client import create_client
            from ..models.llm.finetuning import run_inference
            from ..models.llm.schema_generation import (
                generate_developer_instructions,
                generate_schema,
                get_toplevel_tag_names,
                load_tags_descriptions,
            )

            tags_descriptions = load_tags_descriptions(tax_path)
            output_columns = get_toplevel_tag_names(tags_descriptions)
            response_format = generate_schema(tags_descriptions, tags=output_columns)
            system_prompt = generate_developer_instructions(tags_descriptions)

            features = df.copy()
            if (
                "name" not in features.columns
                and "organization_name" in features.columns
            ):
                features = features.rename(columns={"organization_name": "name"})

            llm_model = model or "gpt-4o-mini"
            console.print(
                f"[blue]Running LLM classification with {llm_model} "
                f"({len(output_columns)} categories)...[/blue]"
            )
            client = create_client(api_key=api_key, base_url=base_url)
            preds = run_inference(
                client,
                llm_model,
                features,
                system_prompt,
                response_format,
                output_columns,
                batch_size=batch_size,
            )

            if format == "csv":
                preds.to_csv(output, index=False)
            elif format == "excel":
                preds.to_excel(output, index=False)
            else:
                with open(output, "w") as f:
                    json.dump(preds.to_dict(orient="records"), f, indent=2)

            console.print(
                f"[green]✓ Classified {len(preds)} organizations -> {output}[/green]"
            )
            return
        elif approach == "two-stage":
            console.print(
                "[blue]Initializing two-stage hierarchical classifier...[/blue]"
            )
            from ..config import load_config
            from ..models.two_stage import TwoStageHierarchicalPipeline

            # Load configuration
            pipeline_config = load_config()
            pipeline_config.two_stage_pipeline.stage1.parallel_batch_size = batch_size

            classifier = TwoStageHierarchicalPipeline.from_config(pipeline_config)

        # Make predictions
        console.print(f"[blue]Running {approach} classification...[/blue]")

        if approach == "two-stage":
            # Use two-stage pipeline
            organizations = []
            for _, row in df.iterrows():
                org = {"asn": int(row["asn"])}
                if "organization_name" in row and pd.notna(row["organization_name"]):
                    org["organization_name"] = str(row["organization_name"])
                organizations.append(org)

            batch_result = classifier.predict_batch(organizations, parallel=True)
            results = batch_result.results

        elif hasattr(classifier, "predict_unified"):
            # Use unified prediction interface
            results = classifier.predict_unified(df[["asn"]])
        else:
            # Fallback to basic prediction
            console.print("[yellow]⚠ Using basic prediction interface[/yellow]")
            results = []
            for _, row in df.iterrows():
                result = UnifiedASClassification(
                    asn=int(row["asn"]),
                    organization_name=f"ASN{row['asn']}",
                    top_level_tags=[],
                    hierarchical_tags=[],
                    model_used=approach,
                    classification_approach=approach,
                )
                results.append(result)

        console.print(f"[green]✓ Classified {len(results)} organizations[/green]")

        # Save results
        console.print(f"[blue]Saving results to {output} in {format} format...[/blue]")

        if format == "json":
            # Convert to JSON serializable format
            json_results = []
            for result in results:
                if approach == "two-stage":
                    # Handle two-stage results
                    json_result = {
                        "asn": result.asn,
                        "organization_name": result.organization_name,
                        "stage1_predictions": [
                            {
                                "category": pred.category.value,
                                "confidence": pred.confidence.value,
                                "model_type": pred.confidence.model_type.value,
                            }
                            for pred in result.stage1_predictions
                        ],
                        "stage2_predictions": [
                            {
                                "parent_category": pred.parent_category.value,
                                "subcategory": pred.subcategory,
                                "confidence": pred.confidence.value,
                            }
                            for pred in result.stage2_predictions
                        ],
                        "overall_confidence": result.overall_confidence,
                        "classification_approach": approach,
                    }
                else:
                    # Handle unified results
                    json_result = {
                        "asn": result.asn,
                        "organization_name": result.organization_name,
                        "top_level_tags": [tag.value for tag in result.top_level_tags],
                        "hierarchical_tags": [
                            tag.value for tag in result.hierarchical_tags
                        ],
                        "model_used": result.model_used,
                        "classification_approach": result.classification_approach,
                        "timestamp": result.timestamp.isoformat(),
                    }
                    if result.top_level_confidence:
                        json_result["confidence_scores"] = result.top_level_confidence
                json_results.append(json_result)

            with open(output, "w") as f:
                json.dump(json_results, f, indent=2)

        elif format == "csv":
            # Convert to DataFrame for CSV export
            rows = []
            for result in results:
                if approach == "two-stage":
                    # Handle two-stage results
                    stage1_categories = [
                        p.category.value for p in result.stage1_predictions
                    ]
                    stage2_data = [
                        f"{p.parent_category.value}:{p.subcategory}"
                        for p in result.stage2_predictions
                    ]

                    row = {
                        "asn": result.asn,
                        "organization_name": result.organization_name,
                        "approach": approach,
                        "stage1_categories": "|".join(stage1_categories),
                        "stage2_subcategories": "|".join(stage2_data),
                        "overall_confidence": result.overall_confidence,
                    }
                else:
                    # Handle unified results
                    row = {
                        "asn": result.asn,
                        "organization_name": result.organization_name,
                        "approach": result.classification_approach,
                        "model_used": result.model_used,
                        "timestamp": result.timestamp.isoformat(),
                    }

                    # Add tag columns
                    all_tags = []
                    all_tags.extend([tag.value for tag in result.top_level_tags])
                    all_tags.extend([tag.value for tag in result.hierarchical_tags])
                    row["tags"] = "|".join(all_tags)

                rows.append(row)

            results_df = pd.DataFrame(rows)
            results_df.to_csv(output, index=False)

        elif format == "excel":
            # Similar to CSV but save as Excel
            rows = []
            for result in results:
                if approach == "two-stage":
                    # Handle two-stage results
                    stage1_categories = [
                        p.category.value for p in result.stage1_predictions
                    ]
                    stage2_data = [
                        f"{p.parent_category.value}:{p.subcategory}"
                        for p in result.stage2_predictions
                    ]

                    row = {
                        "asn": result.asn,
                        "organization_name": result.organization_name,
                        "approach": approach,
                        "stage1_categories": "|".join(stage1_categories),
                        "stage2_subcategories": "|".join(stage2_data),
                        "overall_confidence": result.overall_confidence,
                    }
                else:
                    # Handle unified results
                    row = {
                        "asn": result.asn,
                        "organization_name": result.organization_name,
                        "approach": result.classification_approach,
                        "model_used": result.model_used,
                        "timestamp": result.timestamp.isoformat(),
                    }

                    # Add tag columns
                    all_tags = []
                    all_tags.extend([tag.value for tag in result.top_level_tags])
                    all_tags.extend([tag.value for tag in result.hierarchical_tags])
                    row["tags"] = "|".join(all_tags)

                rows.append(row)

            results_df = pd.DataFrame(rows)
            results_df.to_excel(output, index=False)

        console.print(f"[green]✓ Results saved to {output}[/green]")

    except ImportError as e:
        console.print(f"[red]✗ Failed to import required modules: {e}[/red]")
        console.print(
            "[yellow]⚠ Some hybrid features may not be available yet[/yellow]"
        )
    except Exception as e:
        console.print(f"[red]✗ Prediction failed: {e}[/red]")
        sys.exit(1)


# Two-Stage Pipeline Commands
@two_stage.command()
@click.option(
    "--asn", type=int, required=True, help="Autonomous System Number to classify"
)
@click.option(
    "--organization-name",
    help="Organization name (optional, improves classification accuracy)",
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Custom configuration file",
)
@click.option(
    "--output-format",
    type=click.Choice(["json", "table", "simple"]),
    default="table",
    help="Output format",
)
@click.option(
    "--include-timing", is_flag=True, help="Include processing timing information"
)
@llm_options
def classify_single(
    asn: int,
    organization_name: Optional[str],
    config: Optional[Path],
    output_format: str,
    include_timing: bool,
    base_url: Optional[str],
    api_key: Optional[str],
) -> None:
    """
    Classify a single AS organization using the two-stage pipeline.

    Uses the advanced two-stage hierarchical classification system that combines
    LLM and SVM approaches for optimal accuracy.

    Examples:
    \b
    # Basic classification
    linneaus two-stage classify-single --asn 15169

    \b
    # With organization name for better accuracy
    linneaus two-stage classify-single --asn 15169 --organization-name "Google LLC"

    \b
    # JSON output with timing
    linneaus two-stage classify-single --asn 7922 --output-format json --include-timing
    """
    console.print(f"[blue]Classifying ASN {asn} using two-stage pipeline...[/blue]")

    try:
        import json

        from ..config import load_config
        from ..models.two_stage import TwoStageHierarchicalPipeline

        # Load configuration
        if config:
            pipeline_config = load_config(config)
        else:
            pipeline_config = load_config()

        # Initialize pipeline
        console.print("[blue]Initializing two-stage pipeline...[/blue]")
        pipeline = TwoStageHierarchicalPipeline.from_config(
            pipeline_config, api_key=api_key, base_url=base_url
        )

        # Make prediction
        result = pipeline.predict_single(
            asn=asn, organization_name=organization_name, include_timing=include_timing
        )

        # Display results based on format
        if output_format == "json":
            # Convert to JSON-serializable format
            json_result = {
                "asn": result.asn,
                "organization_name": result.organization_name,
                "stage1_predictions": [
                    {
                        "category": pred.category.value,
                        "confidence": pred.confidence.value,
                        "model_type": pred.confidence.model_type.value,
                    }
                    for pred in result.stage1_predictions
                ],
                "stage2_predictions": [
                    {
                        "parent_category": pred.parent_category.value,
                        "subcategory": pred.subcategory,
                        "confidence": pred.confidence.value,
                    }
                    for pred in result.stage2_predictions
                ],
                "overall_confidence": result.overall_confidence,
            }

            if include_timing and result.processing_time:
                json_result["processing_time"] = result.processing_time

            console.print(json.dumps(json_result, indent=2))

        elif output_format == "table":
            # Rich table format
            from rich.table import Table

            console.print(
                f"\n[bold green]Classification Results for ASN {asn}[/bold green]"
            )
            console.print(f"Organization: {result.organization_name}")
            console.print(f"Overall Confidence: {result.overall_confidence:.3f}")

            if include_timing and result.processing_time:
                console.print(f"Processing Time: {result.processing_time:.3f}s")

            # Stage 1 table
            if result.stage1_predictions:
                table1 = Table(title="Stage 1: Top-Level Categories")
                table1.add_column("Category", style="cyan")
                table1.add_column("Confidence", style="green")
                table1.add_column("Model", style="yellow")

                for pred in result.stage1_predictions:
                    table1.add_row(
                        pred.category.value,
                        f"{pred.confidence.value:.3f}",
                        pred.confidence.model_type.value,
                    )

                console.print(table1)

            # Stage 2 table
            if result.stage2_predictions:
                table2 = Table(title="Stage 2: Sublevel Categories")
                table2.add_column("Top-Level", style="cyan")
                table2.add_column("Subcategory", style="green")
                table2.add_column("Confidence", style="yellow")

                for pred in result.stage2_predictions:
                    table2.add_row(
                        pred.parent_category.value,
                        pred.subcategory,
                        f"{pred.confidence.value:.3f}",
                    )

                console.print(table2)
            else:
                console.print("[yellow]No Stage 2 predictions generated[/yellow]")

        elif output_format == "simple":
            # Simple text format
            console.print(f"ASN {asn}: {result.organization_name}")
            console.print(
                f"Categories: {[p.category.value for p in result.stage1_predictions]}"
            )

            if result.stage2_predictions:
                subcats = [
                    f"{p.parent_category.value}:{p.subcategory}"
                    for p in result.stage2_predictions
                ]
                console.print(f"Subcategories: {subcats}")

            console.print(f"Confidence: {result.overall_confidence:.3f}")

        console.print("[green]✓ Classification completed successfully[/green]")

    except Exception as e:
        console.print(f"[red]✗ Classification failed: {e}[/red]")
        sys.exit(1)


@two_stage.command()
@click.option(
    "--input",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input CSV file with ASN column",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output file for results",
)
@click.option(
    "--format",
    type=click.Choice(["json", "csv", "excel"]),
    default="json",
    help="Output format",
)
@click.option("--batch-size", type=int, default=10, help="Batch size for processing")
@click.option("--parallel/--sequential", default=True, help="Use parallel processing")
@click.option(
    "--include-timing", is_flag=True, help="Include processing timing information"
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Custom configuration file",
)
@llm_options
def classify_batch(
    input: Path,
    output: Path,
    format: str,
    batch_size: int,
    parallel: bool,
    include_timing: bool,
    config: Optional[Path],
    base_url: Optional[str],
    api_key: Optional[str],
) -> None:
    """
    Classify multiple AS organizations using the two-stage pipeline.

    Processes a CSV file containing ASNs and produces comprehensive
    classification results using the two-stage hierarchical approach.

    Input CSV should have columns: asn, organization_name (optional)

    Examples:
    \b
    # Basic batch processing
    linneaus two-stage classify-batch --input asns.csv --output results.json

    \b
    # Large batch with custom settings
    linneaus two-stage classify-batch \\
        --input large_dataset.csv --output results.csv \\
        --format csv --batch-size 20 --parallel

    \b
    # Performance analysis
    linneaus two-stage classify-batch \\
        --input test.csv --output results.json \\
        --include-timing
    """
    console.print(f"[blue]Processing batch classification from {input}...[/blue]")

    try:
        import json

        import pandas as pd

        from ..config import load_config
        from ..models.two_stage import TwoStageHierarchicalPipeline

        # Load configuration
        if config:
            pipeline_config = load_config(config)
        else:
            pipeline_config = load_config()

        # Update batch size in config
        pipeline_config.two_stage_pipeline.stage1.parallel_batch_size = batch_size

        # Read input data
        console.print(f"[blue]Reading input data from {input}...[/blue]")
        df = pd.read_csv(input)

        if "asn" not in df.columns:
            console.print("[red]✗ Input file must contain 'asn' column[/red]")
            sys.exit(1)

        console.print(
            f"[green]✓ Loaded {len(df)} organizations for classification[/green]"
        )

        # Prepare organizations data
        organizations = []
        for _, row in df.iterrows():
            org = {"asn": int(row["asn"])}
            if "organization_name" in row and pd.notna(row["organization_name"]):
                org["organization_name"] = str(row["organization_name"])
            organizations.append(org)

        # Initialize pipeline
        console.print("[blue]Initializing two-stage pipeline...[/blue]")
        pipeline = TwoStageHierarchicalPipeline.from_config(
            pipeline_config, api_key=api_key, base_url=base_url
        )

        # Process batch
        processing_mode = "parallel" if parallel else "sequential"
        console.print(
            f"[blue]Running {processing_mode} batch"
            f" processing (batch size:"
            f" {batch_size})...[/blue]"
        )

        batch_result = pipeline.predict_batch(
            organizations=organizations,
            parallel=parallel,
            include_timing=include_timing,
        )

        console.print(
            f"[green]✓ Processed {len(batch_result.results)}"
            " organizations successfully[/green]"
        )

        # Display performance summary
        perf = batch_result.overall_performance
        console.print(f"Average confidence: {perf.average_confidence:.3f}")

        if include_timing:
            console.print(f"Total processing time: {perf.total_processing_time:.2f}s")
            console.print(
                "Throughput: "
                f"{len(batch_result.results) / perf.total_processing_time:.1f}"
                " predictions/second"
            )

        # Save results
        console.print(f"[blue]Saving results to {output} in {format} format...[/blue]")

        if format == "json":
            # Convert to JSON format
            json_results = []
            for result in batch_result.results:
                json_result = {
                    "asn": result.asn,
                    "organization_name": result.organization_name,
                    "stage1_predictions": [
                        {
                            "category": pred.category.value,
                            "confidence": pred.confidence.value,
                            "model_type": pred.confidence.model_type.value,
                        }
                        for pred in result.stage1_predictions
                    ],
                    "stage2_predictions": [
                        {
                            "parent_category": pred.parent_category.value,
                            "subcategory": pred.subcategory,
                            "confidence": pred.confidence.value,
                        }
                        for pred in result.stage2_predictions
                    ],
                    "overall_confidence": result.overall_confidence,
                }

                if include_timing and result.processing_time:
                    json_result["processing_time"] = result.processing_time

                json_results.append(json_result)

            # Include batch metadata
            output_data = {
                "metadata": {
                    "total_predictions": len(json_results),
                    "average_confidence": perf.average_confidence,
                    "processing_mode": processing_mode,
                    "batch_size": batch_size,
                },
                "results": json_results,
            }

            if include_timing:
                output_data["metadata"][
                    "total_processing_time"
                ] = perf.total_processing_time
                output_data["metadata"]["throughput"] = (
                    len(json_results) / perf.total_processing_time
                )

            with open(output, "w") as f:
                json.dump(output_data, f, indent=2)

        elif format == "csv":
            # Convert to flat CSV format
            rows = []
            for result in batch_result.results:
                # Main prediction row
                stage1_categories = [
                    p.category.value for p in result.stage1_predictions
                ]
                stage1_confidences = [
                    p.confidence.value for p in result.stage1_predictions
                ]

                stage2_data = []
                for pred in result.stage2_predictions:
                    stage2_data.append(
                        f"{pred.parent_category.value}:{pred.subcategory}"
                    )

                row = {
                    "asn": result.asn,
                    "organization_name": result.organization_name,
                    "stage1_categories": "|".join(stage1_categories),
                    "stage1_confidences": "|".join(map(str, stage1_confidences)),
                    "stage2_subcategories": "|".join(stage2_data),
                    "overall_confidence": result.overall_confidence,
                }

                if include_timing and result.processing_time:
                    row["processing_time"] = result.processing_time

                rows.append(row)

            results_df = pd.DataFrame(rows)
            results_df.to_csv(output, index=False)

        elif format == "excel":
            # Excel format with multiple sheets
            rows = []
            for result in batch_result.results:
                stage1_categories = [
                    p.category.value for p in result.stage1_predictions
                ]
                stage1_confidences = [
                    p.confidence.value for p in result.stage1_predictions
                ]

                stage2_data = []
                for pred in result.stage2_predictions:
                    stage2_data.append(
                        f"{pred.parent_category.value}:{pred.subcategory}"
                    )

                row = {
                    "asn": result.asn,
                    "organization_name": result.organization_name,
                    "stage1_categories": "|".join(stage1_categories),
                    "stage1_confidences": "|".join(map(str, stage1_confidences)),
                    "stage2_subcategories": "|".join(stage2_data),
                    "overall_confidence": result.overall_confidence,
                }

                if include_timing and result.processing_time:
                    row["processing_time"] = result.processing_time

                rows.append(row)

            results_df = pd.DataFrame(rows)

            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                results_df.to_excel(writer, sheet_name="Classifications", index=False)

                # Add summary sheet
                summary_data = {
                    "Metric": [
                        "Total Predictions",
                        "Average Confidence",
                        "Processing Mode",
                        "Batch Size",
                    ],
                    "Value": [
                        len(batch_result.results),
                        f"{perf.average_confidence:.3f}",
                        processing_mode,
                        batch_size,
                    ],
                }

                if include_timing:
                    summary_data["Metric"].extend(
                        ["Total Time (s)", "Throughput (pred/s)"]
                    )
                    throughput = len(batch_result.results) / perf.total_processing_time
                    summary_data["Value"].extend(
                        [
                            f"{perf.total_processing_time:.2f}",
                            f"{throughput:.1f}",
                        ]
                    )

                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="Summary", index=False)

        console.print(f"[green]✓ Results saved to {output}[/green]")

    except Exception as e:
        console.print(f"[red]✗ Batch classification failed: {e}[/red]")
        sys.exit(1)


@two_stage.command()
@click.option(
    "--predictions",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="JSON file with two-stage predictions",
)
@click.option(
    "--ground-truth",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="CSV file with ground truth labels",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Output file for evaluation report",
)
@click.option(
    "--detailed-analysis/--summary-only",
    default=True,
    help="Include detailed error analysis",
)
def evaluate(
    predictions: Path,
    ground_truth: Path,
    output: Optional[Path],
    detailed_analysis: bool,
) -> None:
    """
    Evaluate two-stage pipeline predictions against ground truth.

    Computes comprehensive metrics including stage-specific performance,
    hierarchical consistency, and overall system accuracy.

    Examples:
    \b
    # Basic evaluation
    linneaus two-stage evaluate --predictions results.json --ground-truth labels.csv

    \b
    # Detailed evaluation with report
    linneaus two-stage evaluate \\
        --predictions results.json \\
        --ground-truth labels.csv \\
        --output eval_report.txt --detailed-analysis
    """
    console.print("[blue]Evaluating two-stage pipeline predictions...[/blue]")

    try:
        import json

        import pandas as pd

        from ..models.hierarchical_schemas import TopLevelCategory
        from ..models.two_stage.evaluation import HierarchicalEvaluator
        from ..models.two_stage_schemas import (
            ConfidenceScore,
            ModelType,
            SublevelPrediction,
            TopLevelPrediction,
            TwoStageASClassification,
        )

        # Load predictions
        console.print(f"[blue]Loading predictions from {predictions}...[/blue]")
        with open(predictions, "r") as f:
            pred_data = json.load(f)

        # Handle both direct results and metadata format
        if "results" in pred_data:
            pred_results = pred_data["results"]
        else:
            pred_results = pred_data

        # Convert to TwoStageASClassification objects
        console.print("[blue]Converting predictions to evaluation format...[/blue]")
        prediction_objects = []

        for pred in pred_results:
            # Convert stage 1 predictions
            stage1_preds = []
            for s1_pred in pred.get("stage1_predictions", []):
                stage1_preds.append(
                    TopLevelPrediction(
                        category=TopLevelCategory(s1_pred["category"]),
                        confidence=ConfidenceScore(
                            value=s1_pred["confidence"],
                            model_type=ModelType(
                                s1_pred.get("model_type", "assembled")
                            ),
                        ),
                    )
                )

            # Convert stage 2 predictions
            stage2_preds = []
            for s2_pred in pred.get("stage2_predictions", []):
                stage2_preds.append(
                    SublevelPrediction(
                        parent_category=TopLevelCategory(s2_pred["parent_category"]),
                        subcategory=s2_pred["subcategory"],
                        confidence=ConfidenceScore(
                            value=s2_pred["confidence"], model_type=ModelType.LLM
                        ),
                    )
                )

            prediction_objects.append(
                TwoStageASClassification(
                    asn=pred["asn"],
                    organization_name=pred.get("organization_name", ""),
                    stage1_predictions=stage1_preds,
                    stage2_predictions=stage2_preds,
                    overall_confidence=pred["overall_confidence"],
                )
            )

        console.print(f"[green]✓ Loaded {len(prediction_objects)} predictions[/green]")

        # Load ground truth
        console.print(f"[blue]Loading ground truth from {ground_truth}...[/blue]")
        gt_df = pd.read_csv(ground_truth)

        required_columns = ["ASN", "TopLevelCategory"]
        missing_cols = [col for col in required_columns if col not in gt_df.columns]
        if missing_cols:
            console.print(
                "[red]✗ Ground truth file missing"
                f" required columns: {missing_cols}[/red]"
            )
            sys.exit(1)

        console.print(f"[green]✓ Loaded {len(gt_df)} ground truth samples[/green]")

        # Initialize evaluator and run evaluation
        console.print("[blue]Running hierarchical evaluation...[/blue]")
        evaluator = HierarchicalEvaluator()

        results = evaluator.evaluate(
            predictions=prediction_objects,
            ground_truth=gt_df,
            include_detailed_analysis=detailed_analysis,
        )

        # Display results
        console.print("\n[bold green]Evaluation Results[/bold green]")

        # Stage 1 metrics
        stage1 = results["stage1"]
        console.print("\n[bold blue]Stage 1 (Top-Level) Performance:[/bold blue]")
        console.print(f"  Accuracy: {stage1['accuracy']:.3f}")
        console.print(f"  Precision (Macro): {stage1['precision_macro']:.3f}")
        console.print(f"  Recall (Macro): {stage1['recall_macro']:.3f}")
        console.print(f"  F1-Score (Macro): {stage1['f1_macro']:.3f}")
        console.print(f"  Exact Match: {stage1['exact_match_accuracy']:.3f}")

        # Stage 2 metrics
        stage2 = results["stage2"]
        console.print("\n[bold blue]Stage 2 (Sublevel) Performance:[/bold blue]")
        console.print(f"  Accuracy: {stage2['overall_accuracy']:.3f}")
        console.print(f"  Coverage: {stage2['coverage']:.3f}")
        console.print(f"  Total Samples: {stage2['total_samples']}")

        # Consistency metrics
        consistency = results["consistency"]
        console.print("\n[bold blue]Hierarchical Consistency:[/bold blue]")
        console.print(f"  Consistency Rate: {consistency['consistency_rate']:.3f}")
        console.print(f"  Total Violations: {consistency['total_violations']}")

        # Overall system metrics
        system = results["system"]
        console.print("\n[bold blue]Overall System Performance:[/bold blue]")
        console.print(f"  Exact Match Rate: {system['exact_match_rate']:.3f}")
        console.print(f"  Partial Match Rate: {system['partial_match_rate']:.3f}")

        # Top performing categories
        if "per_category" in stage1 and stage1["per_category"]:
            console.print(
                "\n[bold blue]Top Performing Categories (Stage 1):[/bold blue]"
            )
            sorted_cats = sorted(
                stage1["per_category"].items(),
                key=lambda x: x[1].get("f1", 0),
                reverse=True,
            )

            for cat, metrics in sorted_cats[:5]:
                console.print(
                    f"  {cat}: F1={metrics.get('f1', 0):.3f}"
                    f" (Support: {metrics.get('support', 0)})"
                )

        # Generate detailed report if requested
        if output:
            console.print("\n[blue]Generating detailed evaluation report...[/blue]")
            evaluator.create_performance_report(results, output)
            console.print(f"[green]✓ Detailed report saved to {output}[/green]")

        console.print("\n[green]✓ Evaluation completed successfully[/green]")

    except Exception as e:
        console.print(f"[red]✗ Evaluation failed: {e}[/red]")
        sys.exit(1)


@two_stage.command()
@click.option(
    "--test-dataset",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="CSV file with ASNs for benchmarking",
)
@click.option(
    "--sample-size",
    type=int,
    default=50,
    help="Number of ASNs to sample for benchmarking",
)
@click.option(
    "--batch-sizes",
    default="1,5,10,20",
    help="Comma-separated list of batch sizes to test",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Output file for benchmark results",
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Custom configuration file",
)
def benchmark(
    test_dataset: Path,
    sample_size: int,
    batch_sizes: str,
    output: Optional[Path],
    config: Optional[Path],
) -> None:
    """
    Benchmark two-stage pipeline performance.

    Tests the pipeline with different batch sizes and configurations
    to measure throughput and processing characteristics.

    Examples:
    \b
    # Basic benchmark
    linneaus two-stage benchmark --test-dataset test_asns.csv

    \b
    # Custom benchmark with results
    linneaus two-stage benchmark \\
        --test-dataset large_test.csv \\
        --sample-size 100 \\
        --batch-sizes "5,10,25,50" \\
        --output benchmark.json
    """
    console.print("[blue]Running two-stage pipeline benchmark...[/blue]")

    try:
        import json
        import time
        from datetime import datetime

        import pandas as pd

        from ..config import load_config
        from ..models.two_stage import TwoStageHierarchicalPipeline

        # Load configuration
        if config:
            pipeline_config = load_config(config)
        else:
            pipeline_config = load_config()

        # Parse batch sizes
        batch_size_list = [int(size.strip()) for size in batch_sizes.split(",")]
        console.print(f"[blue]Testing batch sizes: {batch_size_list}[/blue]")

        # Load and sample test data
        console.print(f"[blue]Loading test dataset from {test_dataset}...[/blue]")
        df = pd.read_csv(test_dataset)

        if "asn" not in df.columns:
            console.print("[red]✗ Test dataset must contain 'asn' column[/red]")
            sys.exit(1)

        # Sample data if needed
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            console.print(f"[blue]Sampled {sample_size} ASNs for benchmarking[/blue]")

        test_asns = df["asn"].tolist()
        console.print(f"[green]✓ Using {len(test_asns)} ASNs for benchmark[/green]")

        # Benchmark results
        benchmark_results = {
            "timestamp": datetime.now().isoformat(),
            "test_dataset": str(test_dataset),
            "sample_size": len(test_asns),
            "batch_sizes_tested": batch_size_list,
            "results": {},
        }

        # Initialize pipeline once
        console.print("[blue]Initializing two-stage pipeline...[/blue]")
        pipeline = TwoStageHierarchicalPipeline.from_config(pipeline_config)

        # Warmup
        console.print("[blue]Warming up pipeline...[/blue]")
        try:
            pipeline.predict_single(test_asns[0])
            console.print("[green]✓ Pipeline warmed up[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠ Warmup failed: {e}[/yellow]")

        # Test each batch size
        from rich.table import Table

        results_table = Table(title="Benchmark Results")
        results_table.add_column("Batch Size", style="cyan")
        results_table.add_column("Mode", style="yellow")
        results_table.add_column("Time (s)", style="green")
        results_table.add_column("Success", style="magenta")
        results_table.add_column("Throughput", style="blue")
        results_table.add_column("Avg Confidence", style="white")

        for batch_size in batch_size_list:
            console.print(f"\n[bold blue]Testing batch size: {batch_size}[/bold blue]")

            # Test both parallel and sequential modes
            for parallel in [True, False]:
                mode = "Parallel" if parallel else "Sequential"
                console.print(f"  Testing {mode.lower()} processing...")

                # Prepare organizations
                organizations = [{"asn": asn} for asn in test_asns]

                # Update config
                pipeline_config.two_stage_pipeline.stage1.parallel_batch_size = (
                    batch_size
                )

                start_time = time.time()
                successful_predictions = 0
                total_confidence = 0.0

                try:
                    batch_result = pipeline.predict_batch(
                        organizations=organizations,
                        parallel=parallel,
                        include_timing=True,
                    )

                    successful_predictions = len(batch_result.results)
                    total_confidence = sum(
                        r.overall_confidence for r in batch_result.results
                    )

                except Exception as e:
                    console.print(f"    [red]✗ Failed: {e}[/red]")

                end_time = time.time()
                processing_time = end_time - start_time

                # Calculate metrics
                throughput = (
                    successful_predictions / processing_time
                    if processing_time > 0
                    else 0
                )
                avg_confidence = (
                    total_confidence / successful_predictions
                    if successful_predictions > 0
                    else 0
                )

                # Store results
                key = f"{batch_size}_{mode.lower()}"
                benchmark_results["results"][key] = {
                    "batch_size": batch_size,
                    "parallel": parallel,
                    "processing_time": processing_time,
                    "successful_predictions": successful_predictions,
                    "total_predictions": len(test_asns),
                    "throughput": throughput,
                    "average_confidence": avg_confidence,
                    "success_rate": successful_predictions / len(test_asns),
                }

                # Add to table
                results_table.add_row(
                    str(batch_size),
                    mode,
                    f"{processing_time:.2f}",
                    f"{successful_predictions}/{len(test_asns)}",
                    f"{throughput:.1f} pred/s",
                    f"{avg_confidence:.3f}",
                )

                console.print(
                    f"    Time: {processing_time:.2f}s,"
                    f" Throughput: {throughput:.1f} pred/s"
                )

        # Display results table
        console.print("\n")
        console.print(results_table)

        # Find optimal configuration
        best_config = max(
            benchmark_results["results"].items(), key=lambda x: x[1]["throughput"]
        )

        console.print("\n[bold green]Optimal Configuration:[/bold green]")
        console.print(f"  Configuration: {best_config[0]}")
        console.print(
            f"  Throughput: {best_config[1]['throughput']:.1f} predictions/second"
        )
        console.print(f"  Success Rate: {best_config[1]['success_rate']:.1%}")

        # Save results if output specified
        if output:
            with open(output, "w") as f:
                json.dump(benchmark_results, f, indent=2)
            console.print(f"\n[green]✓ Benchmark results saved to {output}[/green]")

        console.print("\n[green]✓ Benchmark completed successfully[/green]")

    except Exception as e:
        console.print(f"[red]✗ Benchmark failed: {e}[/red]")
        sys.exit(1)


@two_stage.command()
def help_commands() -> None:
    """
    Show help for two-stage pipeline commands.

    Provides an overview of all available two-stage pipeline commands
    and their primary use cases.
    """
    console.print(
        "[bold blue]Two-Stage Hierarchical Classification"
        " Pipeline Commands[/bold blue]\n"
    )

    console.print("[bold green]Classification Commands:[/bold green]")
    console.print(
        "  • [cyan]classify-single[/cyan] - Classify a single AS organization"
    )
    console.print(
        "  • [cyan]classify-batch[/cyan]  -"
        " Classify multiple AS organizations from CSV"
    )
    console.print()

    console.print("[bold green]Evaluation Commands:[/bold green]")
    console.print(
        "  • [cyan]evaluate[/cyan]        -"
        " Evaluate predictions against ground truth"
    )
    console.print(
        "  • [cyan]benchmark[/cyan]       -"
        " Performance benchmark with different"
        " configurations"
    )
    console.print(
        "  • [cyan]monitor-performance[/cyan] -"
        " Real-time performance monitoring"
        " and optimization"
    )
    console.print()

    console.print("[bold green]Quick Start Examples:[/bold green]")
    console.print("  # Classify a single organization")
    console.print(
        "  [yellow]linneaus two-stage classify-single"
        ' --asn 15169 --organization-name "Google'
        ' LLC"[/yellow]'
    )
    console.print()
    console.print("  # Batch processing")
    console.print(
        "  [yellow]linneaus two-stage classify-batch"
        " --input asns.csv"
        " --output results.json[/yellow]"
    )
    console.print()
    console.print("  # Evaluate performance")
    console.print(
        "  [yellow]linneaus two-stage evaluate"
        " --predictions results.json"
        " --ground-truth labels.csv[/yellow]"
    )
    console.print()
    console.print("  # Benchmark performance")
    console.print(
        "  [yellow]linneaus two-stage benchmark"
        " --test-dataset test.csv"
        " --output benchmark.json[/yellow]"
    )
    console.print()
    console.print("  # Monitor performance in real-time")
    console.print(
        "  [yellow]linneaus two-stage"
        " monitor-performance"
        " --test-dataset test.csv"
        " --output metrics.json[/yellow]"
    )
    console.print()

    console.print("[bold green]Features:[/bold green]")
    console.print("  • [green]✓[/green] Hybrid LLM + SVM approach for optimal accuracy")
    console.print("  • [green]✓[/green] Two-stage hierarchical classification")
    console.print("  • [green]✓[/green] Parallel batch processing")
    console.print("  • [green]✓[/green] Comprehensive error handling")
    console.print("  • [green]✓[/green] Multiple output formats (JSON, CSV, Excel)")
    console.print("  • [green]✓[/green] Performance monitoring and benchmarking")
    console.print()

    console.print("For detailed help on any command, use:")
    console.print("[yellow]linneaus two-stage COMMAND --help[/yellow]")


@two_stage.command()
@click.option(
    "--test-dataset",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="CSV file with ASNs for performance testing",
)
@click.option(
    "--sample-size",
    type=int,
    default=20,
    help="Number of ASNs to process for monitoring",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Output file for performance metrics",
)
@click.option(
    "--monitoring-interval",
    type=float,
    default=1.0,
    help="System monitoring interval in seconds",
)
def monitor_performance(
    test_dataset: Path,
    sample_size: int,
    output: Optional[Path],
    monitoring_interval: float,
) -> None:
    """
    Monitor two-stage pipeline performance in real-time.

    Runs the pipeline on a sample dataset while collecting detailed
    performance metrics, system resource usage, and optimization recommendations.

    Examples:
    \b
    # Basic performance monitoring
    linneaus two-stage monitor-performance --test-dataset test.csv

    \b
    # Detailed monitoring with output
    linneaus two-stage monitor-performance \\
        --test-dataset test.csv --sample-size 50 \\
        --output metrics.json
    """
    console.print("[blue]Starting performance monitoring session...[/blue]")

    try:
        import json
        import time

        import pandas as pd

        from ..config import load_config
        from ..models.two_stage import TwoStageHierarchicalPipeline

        # Load test data
        console.print(f"[blue]Loading test dataset from {test_dataset}...[/blue]")
        df = pd.read_csv(test_dataset)

        if "asn" not in df.columns:
            console.print("[red]✗ Test dataset must contain 'asn' column[/red]")
            sys.exit(1)

        # Sample data
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            console.print(f"[blue]Sampled {sample_size} ASNs for monitoring[/blue]")

        test_asns = df["asn"].tolist()
        console.print(
            f"[green]✓ Using {len(test_asns)} ASNs for performance monitoring[/green]"
        )

        # Initialize pipeline with monitoring enabled
        config = load_config()
        console.print("[blue]Initializing two-stage pipeline with monitoring...[/blue]")
        pipeline = TwoStageHierarchicalPipeline.from_config(config)

        # Start intensive monitoring
        if hasattr(pipeline, "monitor") and pipeline.monitor:
            pipeline.monitor.start_system_monitoring(interval=monitoring_interval)
            console.print(
                f"[green]✓ System monitoring started"
                f" (interval: {monitoring_interval}s)"
                "[/green]"
            )

        # Process organizations with monitoring
        console.print(
            "[blue]Processing organizations with performance monitoring...[/blue]"
        )

        organizations = [{"asn": asn} for asn in test_asns]

        # Run predictions
        start_time = time.time()
        batch_result = pipeline.predict_batch(
            organizations, parallel=True, include_timing=True
        )
        total_time = time.time() - start_time

        console.print(
            f"[green]✓ Processed"
            f" {len(batch_result.results)} organizations"
            f" in {total_time:.2f}s[/green]"
        )

        # Get comprehensive monitoring summary
        if hasattr(pipeline, "get_monitoring_summary"):
            console.print("[blue]Analyzing performance metrics...[/blue]")
            monitoring_summary = pipeline.get_monitoring_summary()

            # Display key metrics
            console.print("\n[bold green]Performance Summary[/bold green]")

            if "pipeline_summary" in monitoring_summary:
                pipeline_stats = monitoring_summary["pipeline_summary"]
                console.print(f"Total calls: {pipeline_stats.get('total_calls', 0)}")
                console.print(
                    f"Success rate: {pipeline_stats.get('success_rate', 0):.1%}"
                )
                console.print(
                    f"Average duration: {pipeline_stats.get('avg_duration', 0):.3f}s"
                )

            if "system_metrics" in monitoring_summary:
                system_stats = monitoring_summary["system_metrics"]
                if "cpu" in system_stats:
                    console.print(
                        f"Average CPU usage: {system_stats['cpu']['avg']:.1f}%"
                    )
                if "memory" in system_stats:
                    console.print(
                        f"Average memory usage: {system_stats['memory']['avg']:.1f}%"
                    )

            # Show recommendations
            if "recommendations" in monitoring_summary:
                recommendations = monitoring_summary["recommendations"]
                if recommendations and any(
                    r.get("type") != "info" for r in recommendations
                ):
                    console.print(
                        "\n[bold yellow]Performance Recommendations:[/bold yellow]"
                    )
                    for i, rec in enumerate(recommendations[:5], 1):
                        if rec.get("type") != "info":
                            priority = rec.get("priority", "medium")
                            component = rec.get("component", "unknown")
                            issue = rec.get("issue", "No issue specified")

                            console.print(
                                f"{i}. [{priority.upper()}] {component}: {issue}"
                            )

                            if "recommendations" in rec:
                                for suggestion in rec["recommendations"][:2]:
                                    console.print(f"   • {suggestion}")
                else:
                    console.print("\n[green]✓ No performance issues detected[/green]")

            # Show bottlenecks
            if "bottlenecks" in monitoring_summary:
                bottlenecks = monitoring_summary["bottlenecks"]
                if (
                    "slowest_components" in bottlenecks
                    and bottlenecks["slowest_components"]
                ):
                    console.print("\n[bold blue]Slowest Components:[/bold blue]")
                    for comp_name, avg_time in bottlenecks["slowest_components"][:3]:
                        console.print(f"  • {comp_name}: {avg_time:.3f}s average")

        # Export detailed metrics if requested
        if output:
            console.print(f"[blue]Exporting detailed metrics to {output}...[/blue]")

            if hasattr(pipeline, "export_performance_metrics"):
                pipeline.export_performance_metrics(str(output))
                console.print(f"[green]✓ Metrics exported to {output}[/green]")
            else:
                # Fallback to basic export
                basic_metrics = {
                    "timestamp": time.time(),
                    "total_time": total_time,
                    "sample_size": len(test_asns),
                    "throughput": len(batch_result.results) / total_time,
                    "success_count": len(batch_result.results),
                }

                with open(output, "w") as f:
                    json.dump(basic_metrics, f, indent=2)

                console.print(f"[green]✓ Basic metrics exported to {output}[/green]")

        console.print(
            "\n[green]✓ Performance monitoring completed successfully[/green]"
        )

    except Exception as e:
        console.print(f"[red]✗ Performance monitoring failed: {e}[/red]")
        sys.exit(1)


@model.command("evaluate")
@click.option(
    "--predictions",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Predictions CSV",
)
@click.option(
    "--labels",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="True labels CSV",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Output directory for metrics",
)
@click.option(
    "--taxonomy",
    type=click.Choice(list(_BUILTIN_TAXONOMIES)),
    default="linneaus",
    help="Taxonomy to use (determines expected label columns)",
)
@click.option(
    "--taxonomy-file",
    type=click.Path(exists=True),
    default=None,
    help="Custom taxonomy definitions JSON file",
)
def model_evaluate(predictions, labels, output_dir, taxonomy, taxonomy_file):
    """Evaluate model predictions against ground truth."""
    import pandas as pd

    from ..models.llm.schema_generation import (
        get_toplevel_tag_names,
        load_tags_descriptions,
    )
    from ..models.utils.metrics import get_metrics, save_metrics_for_debugging

    tax_path = _resolve_taxonomy_path(taxonomy, taxonomy_file)
    if tax_path:
        console.print(f"[blue]Using taxonomy: {tax_path}[/blue]")

    console.print("[bold]Evaluating predictions...[/bold]")

    y_pred = pd.read_csv(predictions)
    y_true = pd.read_csv(labels)

    # Align rows on ASN when both files carry it (safer than relying on order)
    if "asn" in y_pred.columns and "asn" in y_true.columns:
        common_asns = set(y_pred["asn"]) & set(y_true["asn"])
        y_pred = (
            y_pred[y_pred["asn"].isin(common_asns)]
            .sort_values("asn")
            .reset_index(drop=True)
        )
        y_true = (
            y_true[y_true["asn"].isin(common_asns)]
            .sort_values("asn")
            .reset_index(drop=True)
        )
        console.print(f"[blue]Aligned on {len(common_asns)} common ASNs[/blue]")

    # Drop ASN columns if present
    for df in [y_pred, y_true]:
        if "asn" in df.columns:
            df.drop(columns=["asn"], inplace=True)

    # Restrict to the taxonomy's category columns present in both files
    taxonomy_columns = get_toplevel_tag_names(load_tags_descriptions(tax_path))
    use_columns = [
        c for c in taxonomy_columns if c in y_true.columns and c in y_pred.columns
    ]
    missing = [c for c in taxonomy_columns if c not in use_columns]
    if missing:
        console.print(
            f"[yellow]⚠ {len(missing)} taxonomy categories missing from the "
            f"input files and excluded: {', '.join(missing[:5])}"
            f"{'...' if len(missing) > 5 else ''}[/yellow]"
        )
    if not use_columns:
        console.print(
            "[red]✗ No taxonomy category columns found in both files. "
            "Check --taxonomy/--taxonomy-file matches the data.[/red]"
        )
        sys.exit(1)
    y_true = y_true[use_columns]
    y_pred = y_pred[use_columns]

    metrics = get_metrics(y_true, y_pred, print_results=False)

    # Display results
    console.print(f"\n[bold]Overall Accuracy:[/bold] {metrics['accuracy']:.4f}")
    console.print(f"[bold]Macro Precision:[/bold] {metrics['macro_precision']:.4f}")
    console.print(f"[bold]Macro Recall:[/bold] {metrics['macro_recall']:.4f}")
    console.print(f"[bold]Macro F1:[/bold] {metrics['macro_f1']:.4f}")
    console.print(f"[bold]Macro Jaccard:[/bold] {metrics['macro_jaccard']:.4f}")

    if metrics.get("mean_pr_auc"):
        console.print(f"[bold]Mean PR AUC:[/bold] {metrics['mean_pr_auc']}")

    # Show per-label metrics
    label_df = metrics["label_metrics"]
    table = Table(title="Per-Label Metrics")
    table.add_column("Label")
    table.add_column("Accuracy", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")

    for _, row in label_df.iterrows():
        table.add_row(
            str(row["Label"]),
            f"{row['Accuracy']:.3f}",
            f"{row['Precision']:.3f}",
            f"{row['Recall']:.3f}",
            f"{row['F1 Score']:.3f}",
        )
    console.print(table)

    # Save if output dir specified
    if output_dir:
        save_metrics_for_debugging(
            metrics, pd.read_csv(predictions), str(output_dir), "evaluation"
        )
        console.print(f"\nMetrics saved to {output_dir}")


def _load_finetune_inputs(
    data_dir: Optional[str],
    features_path: Path,
    taxonomy: str,
    taxonomy_file: Optional[str],
    tags_csv: Optional[str],
):
    """Shared setup for ``fine-tune`` and ``prepare-data``.

    Returns (train_labels, train_features, tags, system_prompt).
    """
    import pandas as pd

    from ..data.alignment import AlignedDataset
    from ..models.llm.schema_generation import (
        filter_taxonomy,
        generate_developer_instructions,
        get_toplevel_tag_names,
        load_tags_descriptions,
    )

    tax_path = _resolve_taxonomy_path(taxonomy, taxonomy_file)
    tags_descriptions = load_tags_descriptions(tax_path)
    all_tags = get_toplevel_tag_names(tags_descriptions)

    if tags_csv:
        tags = [t.strip() for t in tags_csv.split(",") if t.strip()]
        tags_descriptions = filter_taxonomy(tags_descriptions, tags)
        console.print(
            f"[blue]Restricting to {len(tags)} categories: {', '.join(tags)}[/blue]"
        )
    else:
        tags = all_tags

    system_prompt = generate_developer_instructions(tags_descriptions)

    # Labels: the released label matrix matching the taxonomy
    ds = AlignedDataset(data_dir)
    label_level = (
        "toplevel" if taxonomy == "linneaus" and not taxonomy_file else taxonomy
    )
    labels = ds.load_labels(label_level)
    splits = ds.load_splits()
    train_asns = set(splits[splits["split"] == "train"]["asn"])
    train_labels = labels[labels["asn"].isin(train_asns)].copy()

    if features_path.suffix == ".parquet":
        features = pd.read_parquet(features_path)
    else:
        features = pd.read_csv(features_path)
    train_features = features[features["asn"].isin(train_asns)].copy()

    console.print(
        f"[blue]{len(train_labels)} train labels, "
        f"{len(train_features)} train features[/blue]"
    )
    return train_labels, train_features, tags, system_prompt


@model.command("fine-tune")
@click.option(
    "--data-dir",
    type=click.Path(exists=True),
    default=None,
    help="Released data directory (default: data/released/202506)",
)
@click.option(
    "--features",
    "features_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/local/features/llm_input.parquet"),
    help="Features file with asn, name, website, country columns",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/local/finetune"),
    help="Directory for the training JSONL and resulting model ID",
)
@click.option(
    "--model-suffix", default="linneaus", help="Suffix for the fine-tuned model name"
)
@click.option(
    "--base-model",
    default="gpt-4o-mini-2024-07-18",
    help="Base model to fine-tune",
)
@click.option(
    "--tags",
    "tags_csv",
    default=None,
    help=(
        "Comma-separated subset of top-level categories to fine-tune on "
        "(e.g. 'Government,Access,Enterprise,ContentProvider'). "
        "Default: all categories in the taxonomy."
    ),
)
@click.option(
    "--taxonomy",
    type=click.Choice(list(_BUILTIN_TAXONOMIES)),
    default="linneaus",
    help="Taxonomy to use for fine-tuning",
)
@click.option(
    "--taxonomy-file",
    type=click.Path(exists=True),
    default=None,
    help="Custom taxonomy definitions JSON file",
)
@llm_options
def fine_tune(
    data_dir: Optional[str],
    features_path: Path,
    output_dir: Path,
    model_suffix: str,
    base_model: str,
    tags_csv: Optional[str],
    taxonomy: str,
    taxonomy_file: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
) -> None:
    """
    Fine-tune an LLM for top-level AS classification.

    Builds a training JSONL from the released labels and local features,
    uploads it to the OpenAI-compatible endpoint, runs the fine-tuning job,
    and waits for completion. The resulting model ID is printed and saved
    to <output-dir>/finetuned_model_id.txt.

    Examples:
    \b
    # Fine-tune on all 20 linneaus categories
    linneaus model fine-tune --model-suffix "v1"

    \b
    # Cheaper: fine-tune on a 4-category subset
    linneaus model fine-tune --tags "Government,Access,Enterprise,ContentProvider"
    """
    console.print("[bold blue]Starting LLM fine-tuning[/bold blue]")

    try:
        from ..models.llm.client import create_client
        from ..models.llm.finetuning import prepare_training_jsonl, run_fine_tuning

        train_labels, train_features, tags, system_prompt = _load_finetune_inputs(
            data_dir, features_path, taxonomy, taxonomy_file, tags_csv
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        training_file, train_count = prepare_training_jsonl(
            train_labels,
            train_features,
            tags,
            system_prompt,
            output_dir / "training.jsonl",
        )
        console.print(
            f"[blue]{train_count} training examples -> {training_file}[/blue]"
        )

        client = create_client(api_key=api_key, base_url=base_url)
        console.print(
            "[blue]Running fine-tuning job (typically 10-20 minutes)...[/blue]"
        )
        model_id = run_fine_tuning(
            client, training_file, suffix=model_suffix, base_model=base_model
        )

        with open(output_dir / "finetuned_model_id.txt", "w") as f:
            f.write(model_id + "\n")

        console.print(f"[green]✓ Fine-tuned model: {model_id}[/green]")
        console.print(
            "[green]✓ Model ID saved to "
            f"{output_dir / 'finetuned_model_id.txt'}[/green]"
        )

    except Exception as e:
        console.print(f"[red]✗ Fine-tuning failed: {e}[/red]")
        sys.exit(1)


@model.command("prepare-data")
@click.option(
    "--data-dir",
    type=click.Path(exists=True),
    default=None,
    help="Released data directory (default: data/released/202506)",
)
@click.option(
    "--features",
    "features_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/local/features/llm_input.parquet"),
    help="Features file with asn, name, website, country columns",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/local/finetune"),
    help="Output directory for the training JSONL",
)
@click.option(
    "--tags",
    "tags_csv",
    default=None,
    help="Comma-separated subset of top-level categories (default: all)",
)
@click.option(
    "--taxonomy",
    type=click.Choice(list(_BUILTIN_TAXONOMIES)),
    default="linneaus",
    help="Taxonomy to use for data preparation",
)
@click.option(
    "--taxonomy-file",
    type=click.Path(exists=True),
    default=None,
    help="Custom taxonomy definitions JSON file",
)
def prepare_data(
    data_dir: Optional[str],
    features_path: Path,
    output_dir: Path,
    tags_csv: Optional[str],
    taxonomy: str,
    taxonomy_file: Optional[str],
) -> None:
    """
    Prepare labeled data for LLM fine-tuning (JSONL only, no API calls).

    Converts the released labels + local features into the OpenAI
    fine-tuning JSONL format used by `linneaus model fine-tune`.
    """
    try:
        from ..models.llm.finetuning import prepare_training_jsonl

        train_labels, train_features, tags, system_prompt = _load_finetune_inputs(
            data_dir, features_path, taxonomy, taxonomy_file, tags_csv
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        training_file, train_count = prepare_training_jsonl(
            train_labels,
            train_features,
            tags,
            system_prompt,
            output_dir / "training.jsonl",
        )

        console.print("[green]✓ Training data prepared successfully[/green]")
        console.print(f"Training file: {training_file}")
        console.print(f"Total examples: {train_count}")

    except Exception as e:
        console.print(f"[red]✗ Data preparation failed: {e}[/red]")
        sys.exit(1)


@model.command("list-jobs")
@click.option("--active-only", is_flag=True, help="Show only active fine-tuning jobs")
@llm_options
def list_jobs(
    active_only: bool, base_url: Optional[str], api_key: Optional[str]
) -> None:
    """
    List OpenAI fine-tuning jobs.

    Shows the status of fine-tuning jobs managed by the system.
    """
    console.print("[blue]Listing fine-tuning jobs...[/blue]")

    try:
        from ..models.llm.client import create_client
        from ..models.llm.fine_tuning import FineTuningManager

        # Initialize manager
        client = create_client(api_key=api_key, base_url=base_url)
        manager = FineTuningManager(client)

        # Get jobs
        jobs = manager.list_jobs(active_only=active_only)

        if not jobs:
            console.print("[yellow]No fine-tuning jobs found[/yellow]")
            return

        # Display jobs table
        table = Table()
        table.add_column("Job ID", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Model", style="yellow")
        table.add_column("Created", style="magenta")
        table.add_column("Fine-tuned Model", style="blue")

        for job_id, job_data in jobs.items():
            status = job_data.get("status", "unknown")
            model = job_data.get("model", "unknown")
            created = job_data.get("created_at", "unknown")
            fine_tuned = job_data.get("fine_tuned_model", "pending")

            # Truncate long values
            if len(job_id) > 20:
                job_id = job_id[:17] + "..."
            if len(fine_tuned) > 25:
                fine_tuned = fine_tuned[:22] + "..."

            table.add_row(job_id, status, model, created[:10], fine_tuned)

        console.print(table)
        console.print(f"\n[blue]Total jobs: {len(jobs)}[/blue]")

    except Exception as e:
        console.print(f"[red]✗ Failed to list jobs: {e}[/red]")
        sys.exit(1)


@model.command("monitor-job")
@click.argument("job_id")
@click.option(
    "--poll-interval", type=int, default=60, help="Polling interval in seconds"
)
@llm_options
def monitor_job(
    job_id: str, poll_interval: int, base_url: Optional[str], api_key: Optional[str]
) -> None:
    """
    Monitor a fine-tuning job until completion.

    JOB_ID is the OpenAI fine-tuning job identifier.
    """
    console.print(f"[blue]Monitoring fine-tuning job: {job_id}[/blue]")

    try:
        from ..models.llm.client import create_client
        from ..models.llm.fine_tuning import FineTuningManager

        # Initialize manager
        client = create_client(api_key=api_key, base_url=base_url)
        manager = FineTuningManager(client)

        # Monitor job
        console.print(f"[blue]Polling every {poll_interval} seconds...[/blue]")
        result = manager.monitor_job(job_id, poll_interval)

        console.print(f"[green]✓ Job completed: {result['status']}[/green]")

        if result.get("fine_tuned_model"):
            console.print(f"[green]✓ Model: {result['fine_tuned_model']}[/green]")

        if result.get("trained_tokens"):
            console.print(f"Tokens trained: {result['trained_tokens']}")

    except Exception as e:
        console.print(f"[red]✗ Monitoring failed: {e}[/red]")
        sys.exit(1)


@model.command("fine-tune-async")
def fine_tune_async() -> None:
    """Removed. Use `linneaus model fine-tune` instead."""
    raise click.UsageError(
        "'fine-tune-async' was removed in this release. "
        "Use 'linneaus model fine-tune' (optionally with --tags to restrict "
        "categories); it prepares data, runs the job, and waits for completion."
    )


@model.command("predict-svm")
@click.option(
    "--model-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to trained model weights",
)
@click.option(
    "--input-data",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input features CSV",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("predictions.csv"),
    help="Output predictions file",
)
def predict_svm(model_path, input_data, output):
    """Generate predictions using a trained stacking model."""
    import pandas as pd

    from ..models.hybrid.svc_llm_estimator import SVCLLMEstimator

    console.print(f"[bold]Loading model from {model_path}...[/bold]")

    estimator = SVCLLMEstimator(svm_params={"kernel": "rbf", "probability": True})
    estimator.load_weights(str(model_path))

    console.print(f"Loading input data from {input_data}...")
    features_df = pd.read_csv(input_data)

    console.print(f"Generating predictions for {len(features_df)} samples...")
    predictions = estimator.predict(features_df)

    # Save predictions
    pred_df = pd.DataFrame(predictions)
    if "asn" in features_df.columns:
        pred_df.insert(0, "asn", features_df["asn"].values)

    pred_df.to_csv(output, index=False)
    console.print(f"[bold green]Predictions saved to {output}[/bold green]")


@main.command("benchmark")
@click.option(
    "--models",
    default="hybrid,svm-only,llm-only",
    help=(
        "Comma-separated list of models to benchmark"
        " (hybrid,svm-only,llm-only,flat,hierarchical)"
    ),
)
@click.option(
    "--dataset",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Test dataset CSV file with 'asn' column",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Output file for benchmark results",
)
@click.option(
    "--sample-size",
    type=int,
    default=100,
    help="Number of ASNs to sample for benchmarking",
)
def main_benchmark(
    models: str, dataset: Path, output: Optional[Path], sample_size: int
) -> None:
    """
    Benchmark multiple classification approaches.

    Compares the performance of different classification models on a test dataset.
    Useful for evaluating which approach works best for your specific use case.

    Examples:
    \b
    # Compare all approaches
    linneaus benchmark --dataset test_asns.csv --output benchmark_results.json

    \b
    # Compare specific models
    linneaus benchmark --models hybrid,svm-only --dataset test.csv --sample-size 50
    """
    console.print("[blue]Running model benchmark comparison...[/blue]")

    try:
        import json
        import time
        from datetime import datetime

        import pandas as pd

        # Parse model list
        model_list = [m.strip() for m in models.split(",")]
        valid_models = ["hybrid", "svm-only", "llm-only", "flat", "hierarchical"]

        for model in model_list:
            if model not in valid_models:
                console.print(
                    f"[red]✗ Invalid model: {model}."
                    f" Valid options: {valid_models}[/red]"
                )
                sys.exit(1)

        console.print(f"[blue]Benchmarking models: {model_list}[/blue]")

        # Load dataset
        console.print(f"[blue]Loading dataset from {dataset}...[/blue]")
        df = pd.read_csv(dataset)

        if "asn" not in df.columns:
            console.print("[red]✗ Dataset must contain 'asn' column[/red]")
            sys.exit(1)

        # Sample data if needed
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            console.print(f"[blue]Sampled {sample_size} ASNs for benchmarking[/blue]")

        console.print(f"[green]✓ Using {len(df)} ASNs for benchmarking[/green]")

        # Benchmark results
        benchmark_results = {
            "timestamp": datetime.now().isoformat(),
            "dataset": str(dataset),
            "sample_size": len(df),
            "models": {},
        }

        # Run benchmark for each model
        for model_name in model_list:
            console.print(
                f"\n[bold blue]Benchmarking {model_name} approach...[/bold blue]"
            )

            start_time = time.time()

            try:
                # Import required classes (only when needed to avoid import errors)
                from ..models.hybrid.stacking_classifier import HybridASClassifier
                from ..models.svm.svm_models import SVMClassifier

                # Initialize classifier
                classifier = None

                if model_name == "hybrid":
                    classifier = HybridASClassifier(approach="hybrid")
                elif model_name == "hierarchical":
                    classifier = HybridASClassifier(approach="hierarchical")
                elif model_name == "flat":
                    classifier = HybridASClassifier(approach="flat")
                elif model_name == "svm-only":
                    classifier = SVMClassifier(approach="flat")
                elif model_name == "llm-only":
                    console.print(
                        f"[yellow]⚠ {model_name} approach"
                        " not available, skipping[/yellow]"
                    )
                    continue

                # Run prediction
                console.print(f"[blue]Running {model_name} predictions...[/blue]")

                if hasattr(classifier, "predict_unified"):
                    results = classifier.predict_unified(df[["asn"]])
                    num_predictions = len(results)
                    num_tagged = sum(
                        1 for r in results if r.top_level_tags or r.hierarchical_tags
                    )
                else:
                    console.print(
                        f"[yellow]⚠ {model_name} doesn't"
                        " support unified interface,"
                        " skipping[/yellow]"
                    )
                    num_predictions = len(df)
                    num_tagged = 0

                end_time = time.time()
                processing_time = end_time - start_time

                # Store results
                model_results = {
                    "predictions_made": num_predictions,
                    "organizations_tagged": num_tagged,
                    "processing_time_seconds": processing_time,
                    "predictions_per_second": (
                        num_predictions / processing_time if processing_time > 0 else 0
                    ),
                    "success": True,
                    "error": None,
                }

                benchmark_results["models"][model_name] = model_results

                console.print(
                    f"[green]✓ {model_name}:"
                    f" {num_predictions} predictions in"
                    f" {processing_time:.2f}s"
                    f" ({num_predictions/processing_time:.2f}"
                    " pred/s)[/green]"
                )

            except Exception as e:
                end_time = time.time()
                processing_time = end_time - start_time

                console.print(f"[red]✗ {model_name} failed: {e}[/red]")

                benchmark_results["models"][model_name] = {
                    "predictions_made": 0,
                    "organizations_tagged": 0,
                    "processing_time_seconds": processing_time,
                    "predictions_per_second": 0,
                    "success": False,
                    "error": str(e),
                }

        # Display summary
        console.print("\n[bold blue]Benchmark Summary[/bold blue]")
        table = Table()
        table.add_column("Model", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Predictions", style="yellow")
        table.add_column("Time (s)", style="magenta")
        table.add_column("Speed (pred/s)", style="blue")

        for model_name, results in benchmark_results["models"].items():
            status = "✓ Success" if results["success"] else "✗ Failed"
            table.add_row(
                model_name,
                status,
                str(results["predictions_made"]),
                f"{results['processing_time_seconds']:.2f}",
                f"{results['predictions_per_second']:.2f}",
            )

        console.print(table)

        # Save results if output specified
        if output:
            with open(output, "w") as f:
                json.dump(benchmark_results, f, indent=2)
            console.print(f"\n[green]✓ Benchmark results saved to {output}[/green]")

    except ImportError as e:
        console.print(f"[red]✗ Failed to import required modules: {e}[/red]")
        console.print(
            "[yellow]⚠ Some hybrid features may not be available yet[/yellow]"
        )
    except Exception as e:
        console.print(f"[red]✗ Benchmark failed: {e}[/red]")
        sys.exit(1)


@main.command()
def info() -> None:
    """Show information about Linneaus."""
    from .. import __author__, __version__

    console.print(f"[bold blue]Linneaus v{__version__}[/bold blue]")
    console.print(f"Author: {__author__}")
    console.print("\nAI-powered classification of Internet Autonomous Systems")
    console.print("Repository: https://github.com/linneaus-project/linneaus")


if __name__ == "__main__":
    main()
