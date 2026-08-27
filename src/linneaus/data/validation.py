"""
Data validation and quality checks for Linneaus.

This module provides comprehensive validation for downloaded data,
quality metrics calculation, and data integrity checks across all
data sources including ASRank, PeeringDB, ASPOP, and IPinfo.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from ..utils.country_mapping import load_country_rir_registry

logger = logging.getLogger(__name__)
console = Console()


class DataQualityMetrics(BaseModel):
    """Data quality metrics for a dataset."""

    source: str = Field(..., description="Data source name")
    total_records: int = Field(..., description="Total number of records")
    valid_records: int = Field(..., description="Number of valid records")
    invalid_records: int = Field(..., description="Number of invalid records")
    completeness_score: float = Field(..., description="Data completeness score (0-1)")
    consistency_score: float = Field(..., description="Data consistency score (0-1)")
    accuracy_score: float = Field(..., description="Data accuracy score (0-1)")
    overall_quality_score: float = Field(..., description="Overall quality score (0-1)")
    validation_timestamp: datetime = Field(default_factory=datetime.now)

    # Detailed metrics
    missing_required_fields: int = Field(
        0, description="Records missing required fields"
    )
    duplicate_records: int = Field(0, description="Number of duplicate records")
    format_violations: int = Field(0, description="Records with format violations")
    range_violations: int = Field(0, description="Records with out-of-range values")

    # Source-specific metrics
    source_specific_metrics: Dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Result of data validation."""

    is_valid: bool = Field(..., description="Whether data passes validation")
    quality_metrics: DataQualityMetrics = Field(..., description="Quality metrics")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    recommendations: List[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )


class DataValidator:
    """
    Main data validator for all Linneaus data sources.

    This class provides validation methods for each data source and
    calculates comprehensive quality metrics.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the data validator.

        Parameters
        ----------
        config_path : Path, optional
            Path to validation configuration file.
        """
        self.config = self._load_validation_config(config_path)

        # Quality thresholds
        self.quality_thresholds = {
            "completeness_min": 0.8,
            "consistency_min": 0.9,
            "accuracy_min": 0.85,
            "overall_min": 0.8,
        }

    def validate_asrank_data(self, file_path: Path) -> ValidationResult:
        """
        Validate ASRank data file.

        Parameters
        ----------
        file_path : Path
            Path to ASRank CSV file.

        Returns
        -------
        ValidationResult
            Validation results with quality metrics.
        """
        console.print(f"[blue]Validating ASRank data: {file_path}[/blue]")

        errors = []
        warnings = []
        recommendations = []

        try:
            # Load and parse data
            df = pd.read_csv(file_path)
            total_records = len(df)

            if total_records == 0:
                errors.append("ASRank file is empty")
                return self._create_failed_result(
                    "asrank", errors, warnings, recommendations
                )

            # Required fields for ASRank
            required_fields = ["asn", "asnName", "rank", "orgName"]
            optional_fields = [
                "latitude",
                "longitude",
                "country_iso",
                "cone_numberAsns",
            ]

            # Check required fields
            missing_fields = [
                field for field in required_fields if field not in df.columns
            ]
            if missing_fields:
                errors.append(f"Missing required fields: {missing_fields}")

            # Validate ASN format
            invalid_asns = self._validate_asn_format(df.get("asn", pd.Series()))

            # Validate geographic coordinates
            coord_issues = self._validate_coordinates(
                df.get("latitude", pd.Series()), df.get("longitude", pd.Series())
            )

            # Check for duplicates
            duplicate_asns = df["asn"].duplicated().sum() if "asn" in df.columns else 0
            if duplicate_asns > 0:
                warnings.append(f"Found {duplicate_asns} duplicate ASNs")

            # Calculate quality metrics
            valid_records = total_records - len(invalid_asns) - duplicate_asns
            missing_required = sum(
                df[field].isna().sum()
                for field in required_fields
                if field in df.columns
            )

            # ASRank-specific checks
            rank_issues = self._validate_asrank_ranks(df.get("rank", pd.Series()))
            org_name_issues = self._validate_organization_names(
                df.get("orgName", pd.Series())
            )

            # Calculate scores
            completeness_score = self._calculate_completeness_score(
                df, required_fields + optional_fields
            )
            consistency_score = 1.0 - (
                len(invalid_asns) + coord_issues + rank_issues
            ) / max(total_records, 1)
            accuracy_score = 1.0 - org_name_issues / max(total_records, 1)
            overall_score = (
                completeness_score + consistency_score + accuracy_score
            ) / 3

            # Source-specific metrics
            source_metrics = {
                "avg_rank": df["rank"].mean() if "rank" in df.columns else None,
                "unique_countries": (
                    df["country_iso"].nunique() if "country_iso" in df.columns else None
                ),
                "organizations_with_location": df[["latitude", "longitude"]]
                .notna()
                .all(axis=1)
                .sum(),
                "avg_cone_size": (
                    df["cone_numberAsns"].mean()
                    if "cone_numberAsns" in df.columns
                    else None
                ),
            }

            quality_metrics = DataQualityMetrics(
                source="asrank",
                total_records=total_records,
                valid_records=valid_records,
                invalid_records=total_records - valid_records,
                completeness_score=completeness_score,
                consistency_score=consistency_score,
                accuracy_score=accuracy_score,
                overall_quality_score=overall_score,
                missing_required_fields=missing_required,
                duplicate_records=duplicate_asns,
                format_violations=len(invalid_asns),
                range_violations=coord_issues,
                source_specific_metrics=source_metrics,
            )

            # Generate recommendations
            if completeness_score < self.quality_thresholds["completeness_min"]:
                recommendations.append(
                    "Improve data completeness by filling missing required fields"
                )
            if len(invalid_asns) > total_records * 0.01:
                recommendations.append(
                    "Review ASN format validation - high number of invalid ASNs"
                )
            if coord_issues > 0:
                recommendations.append("Validate geographic coordinates for accuracy")

            is_valid = (
                overall_score >= self.quality_thresholds["overall_min"]
                and len(errors) == 0
            )

            return ValidationResult(
                is_valid=is_valid,
                quality_metrics=quality_metrics,
                errors=errors,
                warnings=warnings,
                recommendations=recommendations,
            )

        except Exception as e:
            errors.append(f"Failed to validate ASRank data: {e}")
            return self._create_failed_result(
                "asrank", errors, warnings, recommendations
            )

    def validate_peeringdb_data(self, file_path: Path) -> ValidationResult:
        """
        Validate PeeringDB data file.

        Parameters
        ----------
        file_path : Path
            Path to PeeringDB JSON file.

        Returns
        -------
        ValidationResult
            Validation results with quality metrics.
        """
        console.print(f"[blue]Validating PeeringDB data: {file_path}[/blue]")

        errors = []
        warnings = []
        recommendations = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                errors.append("PeeringDB data should be a JSON object")
                return self._create_failed_result(
                    "peeringdb", errors, warnings, recommendations
                )

            total_records = len(data)
            if total_records == 0:
                errors.append("PeeringDB file is empty")
                return self._create_failed_result(
                    "peeringdb", errors, warnings, recommendations
                )

            # Validate each network record
            valid_records = 0
            invalid_asns = []
            missing_names = 0
            invalid_urls = 0

            for asn_str, network_data in data.items():
                # Validate ASN
                try:
                    asn = int(asn_str)
                    if asn <= 0 or asn > 4294967295:  # 32-bit ASN limit
                        invalid_asns.append(asn_str)
                        continue
                except ValueError:
                    invalid_asns.append(asn_str)
                    continue

                # Check required fields
                if not isinstance(network_data, dict):
                    continue

                if not network_data.get("name"):
                    missing_names += 1

                # Validate URLs if present
                website = network_data.get("website")
                if website and not self._is_valid_url(website):
                    invalid_urls += 1

                valid_records += 1

            # Calculate quality metrics
            required_fields = ["name", "asn"]  # Essential fields
            optional_fields = ["website", "looking_glass", "info_type", "info_scope"]

            # Estimate completeness based on field presence
            total_possible_fields = len(required_fields) + len(optional_fields)
            total_filled_fields = sum(
                1
                for record in data.values()
                if isinstance(record, dict)
                for field in required_fields + optional_fields
                if record.get(field)
            )
            completeness_score = (
                total_filled_fields / (total_records * total_possible_fields)
                if total_records > 0
                else 0
            )

            consistency_score = 1.0 - (len(invalid_asns) + invalid_urls) / max(
                total_records, 1
            )
            accuracy_score = 1.0 - missing_names / max(total_records, 1)
            overall_score = (
                completeness_score + consistency_score + accuracy_score
            ) / 3

            # PeeringDB-specific metrics
            source_metrics = {
                "networks_with_websites": sum(
                    1
                    for record in data.values()
                    if isinstance(record, dict) and record.get("website")
                ),
                "networks_with_looking_glass": sum(
                    1
                    for record in data.values()
                    if isinstance(record, dict) and record.get("looking_glass")
                ),
                "network_types": list(
                    set(
                        record.get("info_type")
                        for record in data.values()
                        if isinstance(record, dict) and record.get("info_type")
                    )
                ),
                "ipv6_enabled_networks": sum(
                    1
                    for record in data.values()
                    if isinstance(record, dict) and record.get("info_ipv6")
                ),
            }

            quality_metrics = DataQualityMetrics(
                source="peeringdb",
                total_records=total_records,
                valid_records=valid_records,
                invalid_records=total_records - valid_records,
                completeness_score=completeness_score,
                consistency_score=consistency_score,
                accuracy_score=accuracy_score,
                overall_quality_score=overall_score,
                missing_required_fields=missing_names,
                duplicate_records=0,  # ASNs are unique keys
                format_violations=len(invalid_asns),
                range_violations=invalid_urls,
                source_specific_metrics=source_metrics,
            )

            # Generate recommendations
            if missing_names > total_records * 0.1:
                recommendations.append(
                    "Many networks missing names - consider data enrichment"
                )
            if invalid_urls > 0:
                recommendations.append("Validate and clean website URLs")
            if completeness_score < 0.5:
                recommendations.append(
                    "PeeringDB data appears sparse - consider refreshing"
                )

            is_valid = (
                overall_score >= self.quality_thresholds["overall_min"]
                and len(errors) == 0
            )

            return ValidationResult(
                is_valid=is_valid,
                quality_metrics=quality_metrics,
                errors=errors,
                warnings=warnings,
                recommendations=recommendations,
            )

        except Exception as e:
            errors.append(f"Failed to validate PeeringDB data: {e}")
            return self._create_failed_result(
                "peeringdb", errors, warnings, recommendations
            )

    def validate_aspop_data(self, file_path: Path) -> ValidationResult:
        """Validate ASPOP data file."""
        console.print(f"[blue]Validating ASPOP data: {file_path}[/blue]")

        errors = []
        warnings = []
        recommendations = []

        try:
            # ASPOP data can be either CSV or JSON
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
                total_records = len(df)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                total_records = len(data)
                # Convert to DataFrame for consistent processing
                df = pd.DataFrame.from_dict(data, orient="index")

            if total_records == 0:
                errors.append("ASPOP file is empty")
                return self._create_failed_result(
                    "aspop", errors, warnings, recommendations
                )

            # Required fields for ASPOP
            required_fields = ["asn", "name", "country", "rir"]
            missing_fields = [
                field for field in required_fields if field not in df.columns
            ]
            if missing_fields:
                errors.append(f"Missing required fields: {missing_fields}")

            # Validate data
            invalid_asns = self._validate_asn_format(df.get("asn", pd.Series()))
            invalid_countries = self._validate_country_codes(
                df.get("country", pd.Series())
            )
            invalid_rirs = self._validate_rir_codes(df.get("rir", pd.Series()))

            # Calculate metrics
            valid_records = (
                total_records
                - len(invalid_asns)
                - len(invalid_countries)
                - len(invalid_rirs)
            )
            missing_required = sum(
                df[field].isna().sum()
                for field in required_fields
                if field in df.columns
            )

            completeness_score = self._calculate_completeness_score(df, required_fields)
            consistency_score = 1.0 - (
                len(invalid_asns) + len(invalid_countries) + len(invalid_rirs)
            ) / max(total_records, 1)
            accuracy_score = 1.0 - missing_required / (
                max(total_records, 1) * len(required_fields)
            )
            overall_score = (
                completeness_score + consistency_score + accuracy_score
            ) / 3

            # ASPOP-specific metrics
            source_metrics = {
                "unique_countries": (
                    df["country"].nunique() if "country" in df.columns else None
                ),
                "rir_distribution": (
                    df["rir"].value_counts().to_dict() if "rir" in df.columns else {}
                ),
                "avg_customer_cone": df.get("customer_cone_asns", pd.Series()).mean(),
                "networks_with_customer_data": (
                    df["customer_cone_asns"].notna().sum()
                    if "customer_cone_asns" in df.columns
                    else 0
                ),
            }

            quality_metrics = DataQualityMetrics(
                source="aspop",
                total_records=total_records,
                valid_records=valid_records,
                invalid_records=total_records - valid_records,
                completeness_score=completeness_score,
                consistency_score=consistency_score,
                accuracy_score=accuracy_score,
                overall_quality_score=overall_score,
                missing_required_fields=missing_required,
                duplicate_records=0,
                format_violations=len(invalid_asns),
                range_violations=len(invalid_countries) + len(invalid_rirs),
                source_specific_metrics=source_metrics,
            )

            is_valid = (
                overall_score >= self.quality_thresholds["overall_min"]
                and len(errors) == 0
            )

            return ValidationResult(
                is_valid=is_valid,
                quality_metrics=quality_metrics,
                errors=errors,
                warnings=warnings,
                recommendations=recommendations,
            )

        except Exception as e:
            errors.append(f"Failed to validate ASPOP data: {e}")
            return self._create_failed_result(
                "aspop", errors, warnings, recommendations
            )

    def validate_ipinfo_data(self, file_path: Path) -> ValidationResult:
        """Validate IPinfo data file."""
        console.print(f"[blue]Validating IPinfo data: {file_path}[/blue]")

        errors = []
        warnings = []
        recommendations = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                batch_data = json.load(f)

            # Handle batch format
            if "data" in batch_data:
                data = batch_data["data"]
            else:
                data = batch_data

            total_records = len(data)
            if total_records == 0:
                errors.append("IPinfo file is empty")
                return self._create_failed_result(
                    "ipinfo", errors, warnings, recommendations
                )

            valid_records = 0
            error_records = 0
            missing_names = 0
            invalid_domains = 0

            for asn_str, record in data.items():
                if isinstance(record, dict) and "error" in record:
                    error_records += 1
                    continue

                # Validate ASN
                try:
                    asn = int(asn_str)
                    if asn <= 0:
                        continue
                except ValueError:
                    continue

                # Check data quality
                if not record.get("name") and not record.get("org"):
                    missing_names += 1

                domain = record.get("domain")
                if domain and not self._is_valid_domain(domain):
                    invalid_domains += 1

                valid_records += 1

            # Calculate quality metrics
            completeness_score = valid_records / max(total_records, 1)
            consistency_score = 1.0 - invalid_domains / max(total_records, 1)
            accuracy_score = 1.0 - (missing_names + error_records) / max(
                total_records, 1
            )
            overall_score = (
                completeness_score + consistency_score + accuracy_score
            ) / 3

            # IPinfo-specific metrics
            source_metrics = {
                "error_rate": error_records / max(total_records, 1),
                "networks_with_domains": sum(
                    1
                    for record in data.values()
                    if isinstance(record, dict) and record.get("domain")
                ),
                "unique_countries": len(
                    set(
                        record.get("country")
                        for record in data.values()
                        if isinstance(record, dict) and record.get("country")
                    )
                ),
                "network_types": list(
                    set(
                        record.get("type")
                        for record in data.values()
                        if isinstance(record, dict) and record.get("type")
                    )
                ),
            }

            quality_metrics = DataQualityMetrics(
                source="ipinfo",
                total_records=total_records,
                valid_records=valid_records,
                invalid_records=error_records,
                completeness_score=completeness_score,
                consistency_score=consistency_score,
                accuracy_score=accuracy_score,
                overall_quality_score=overall_score,
                missing_required_fields=missing_names,
                duplicate_records=0,
                format_violations=invalid_domains,
                range_violations=0,
                source_specific_metrics=source_metrics,
            )

            # Recommendations
            if error_records > total_records * 0.1:
                recommendations.append(
                    "High error rate - consider checking API limits or connectivity"
                )
            if missing_names > total_records * 0.2:
                recommendations.append("Many records missing organization names")

            is_valid = (
                overall_score >= self.quality_thresholds["overall_min"]
                and len(errors) == 0
            )

            return ValidationResult(
                is_valid=is_valid,
                quality_metrics=quality_metrics,
                errors=errors,
                warnings=warnings,
                recommendations=recommendations,
            )

        except Exception as e:
            errors.append(f"Failed to validate IPinfo data: {e}")
            return self._create_failed_result(
                "ipinfo", errors, warnings, recommendations
            )

    def generate_quality_report(
        self, validation_results: List[ValidationResult]
    ) -> str:
        """
        Generate a comprehensive quality report.

        Parameters
        ----------
        validation_results : List[ValidationResult]
            Results from validating multiple data sources.

        Returns
        -------
        str
            Formatted quality report.
        """
        table = Table(title="Data Quality Report")
        table.add_column("Source", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Records", justify="right")
        table.add_column("Completeness", justify="right")
        table.add_column("Consistency", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_column("Overall", justify="right", style="bold")

        for result in validation_results:
            metrics = result.quality_metrics
            status = "✓ PASS" if result.is_valid else "✗ FAIL"
            status_style = "green" if result.is_valid else "red"

            table.add_row(
                metrics.source.upper(),
                f"[{status_style}]{status}[/{status_style}]",
                f"{metrics.valid_records:,}/{metrics.total_records:,}",
                f"{metrics.completeness_score:.2%}",
                f"{metrics.consistency_score:.2%}",
                f"{metrics.accuracy_score:.2%}",
                f"{metrics.overall_quality_score:.2%}",
            )

        console.print(table)

        # Generate detailed report text
        report_lines = [
            "# Data Quality Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
        ]

        for result in validation_results:
            metrics = result.quality_metrics
            report_lines.extend(
                [
                    f"## {metrics.source.upper()} Data Quality",
                    f"- **Status**: {'PASS' if result.is_valid else 'FAIL'}",
                    f"- **Total Records**: {metrics.total_records:,}",
                    f"- **Valid Records**: {metrics.valid_records:,}",
                    f"- **Overall Score**: {metrics.overall_quality_score:.2%}",
                    "",
                ]
            )

            if result.errors:
                report_lines.extend(
                    ["**Errors:**"] + [f"- {error}" for error in result.errors] + [""]
                )

            if result.warnings:
                report_lines.extend(
                    ["**Warnings:**"]
                    + [f"- {warning}" for warning in result.warnings]
                    + [""]
                )

            if result.recommendations:
                report_lines.extend(
                    ["**Recommendations:**"]
                    + [f"- {rec}" for rec in result.recommendations]
                    + [""]
                )

        return "\n".join(report_lines)

    def _validate_asn_format(self, asn_series: pd.Series) -> List[Any]:
        """Validate ASN format and range."""
        invalid_asns = []
        for asn in asn_series.dropna():
            try:
                asn_int = int(asn)
                if asn_int <= 0 or asn_int > 4294967295:  # 32-bit ASN limit
                    invalid_asns.append(asn)
            except (ValueError, TypeError):
                invalid_asns.append(asn)
        return invalid_asns

    def _validate_coordinates(
        self, lat_series: pd.Series, lon_series: pd.Series
    ) -> int:
        """Validate geographic coordinates."""
        issues = 0
        for lat, lon in zip(lat_series.dropna(), lon_series.dropna()):
            try:
                lat_float = float(lat)
                lon_float = float(lon)
                if not (-90 <= lat_float <= 90) or not (-180 <= lon_float <= 180):
                    issues += 1
            except (ValueError, TypeError):
                issues += 1
        return issues

    def _validate_asrank_ranks(self, rank_series: pd.Series) -> int:
        """Validate ASRank rank values."""
        issues = 0
        for rank in rank_series.dropna():
            try:
                rank_int = int(rank)
                if rank_int <= 0:
                    issues += 1
            except (ValueError, TypeError):
                issues += 1
        return issues

    def _validate_organization_names(self, name_series: pd.Series) -> int:
        """Validate organization names."""
        issues = 0
        for name in name_series.dropna():
            if not isinstance(name, str) or len(name.strip()) < 2:
                issues += 1
        return issues

    def _validate_country_codes(self, country_series: pd.Series) -> List[Any]:
        """Validate country codes against official country-RIR registry."""
        try:
            country_registry = load_country_rir_registry()
            valid_codes = set(country_registry.countries.keys())
        except Exception as e:
            logger.warning(
                f"Failed to load country registry, using fallback validation: {e}"
            )
            # Fallback to basic format validation
            valid_codes = None

        invalid_codes = []
        for code in country_series.dropna():
            if not isinstance(code, str) or len(code) != 2:
                invalid_codes.append(code)
            elif valid_codes is not None and code.upper() not in valid_codes:
                invalid_codes.append(code)
        return invalid_codes

    def _validate_rir_codes(self, rir_series: pd.Series) -> List[Any]:
        """Validate RIR codes against official RIR registry."""
        try:
            country_registry = load_country_rir_registry()
            # Get valid RIR codes from the registry
            valid_rirs = set(country_registry.registries.keys())
        except Exception as e:
            logger.warning(
                f"Failed to load country registry, using fallback RIR validation: {e}"
            )
            # Fallback to hardcoded list
            valid_rirs = {"ARIN", "RIPE NCC", "APNIC", "LACNIC", "AFRINIC"}

        invalid_rirs = []
        for rir in rir_series.dropna():
            if not isinstance(rir, str) or rir not in valid_rirs:
                invalid_rirs.append(rir)
        return invalid_rirs

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        url_pattern = re.compile(
            r"^https?://"  # http:// or https://
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
            r"localhost|"  # localhost...
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
            r"(?::\d+)?"  # optional port
            r"(?:/?|[/?]\S+)$",
            re.IGNORECASE,
        )
        return bool(url_pattern.match(url))

    def _is_valid_domain(self, domain: str) -> bool:
        """Validate domain format."""
        domain_pattern = re.compile(
            r"^(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?$", re.IGNORECASE
        )
        return bool(domain_pattern.match(domain))

    def _calculate_completeness_score(
        self, df: pd.DataFrame, fields: List[str]
    ) -> float:
        """Calculate data completeness score."""
        if df.empty or not fields:
            return 0.0

        available_fields = [field for field in fields if field in df.columns]
        if not available_fields:
            return 0.0

        total_possible = len(df) * len(available_fields)
        total_filled = sum(df[field].notna().sum() for field in available_fields)

        return total_filled / total_possible if total_possible > 0 else 0.0

    def _create_failed_result(
        self,
        source: str,
        errors: List[str],
        warnings: List[str],
        recommendations: List[str],
    ) -> ValidationResult:
        """Create a failed validation result."""
        quality_metrics = DataQualityMetrics(
            source=source,
            total_records=0,
            valid_records=0,
            invalid_records=0,
            completeness_score=0.0,
            consistency_score=0.0,
            accuracy_score=0.0,
            overall_quality_score=0.0,
        )

        return ValidationResult(
            is_valid=False,
            quality_metrics=quality_metrics,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
        )

    def _load_validation_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load validation configuration."""
        # Default configuration
        default_config = {
            "quality_thresholds": {
                "completeness_min": 0.8,
                "consistency_min": 0.9,
                "accuracy_min": 0.85,
                "overall_min": 0.8,
            },
            "validation_rules": {
                "asn_max": 4294967295,
                "coordinate_precision": 6,
                "url_timeout": 5,
            },
        }

        if config_path and config_path.exists():
            try:
                with open(config_path, "r") as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load validation config: {e}")

        return default_config


def validate_all_data_sources(
    data_dir: Path, output_report: Optional[Path] = None
) -> List[ValidationResult]:
    """
    Validate all available data sources in a directory.

    Parameters
    ----------
    data_dir : Path
        Directory containing data files.
    output_report : Path, optional
        Path to save validation report.

    Returns
    -------
    List[ValidationResult]
        Validation results for all sources.
    """
    validator = DataValidator()
    results = []

    # Check for each data source
    source_files = {
        "asrank": list(data_dir.glob("as_rank_*.csv")),
        "peeringdb": list(data_dir.glob("peeringdb_*.json")),
        "aspop": list(data_dir.glob("aspop_*.csv"))
        + list(data_dir.glob("aspop_*.json")),
        "ipinfo": list(data_dir.glob("ipinfo_*.json")),
    }

    for source, files in source_files.items():
        if files:
            # Use the most recent file
            latest_file = max(files, key=lambda x: x.stat().st_mtime)

            if source == "asrank":
                result = validator.validate_asrank_data(latest_file)
            elif source == "peeringdb":
                result = validator.validate_peeringdb_data(latest_file)
            elif source == "aspop":
                result = validator.validate_aspop_data(latest_file)
            elif source == "ipinfo":
                result = validator.validate_ipinfo_data(latest_file)

            results.append(result)

    # Generate and optionally save report
    if results:
        report = validator.generate_quality_report(results)

        if output_report:
            with open(output_report, "w", encoding="utf-8") as f:
                f.write(report)
            console.print(f"[green]✓[/green] Quality report saved to {output_report}")

    return results
