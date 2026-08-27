"""
Data processors for converting raw data to structured formats.

This module contains classes for processing raw data files downloaded
from various sources and converting them to the structured formats
used by the classification pipeline.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from rich.console import Console

from ..config import get_config
from .validation import DataValidator

console = Console()


class DataProcessor:
    """
    Main data processor for converting raw data to structured formats.

    This class processes raw data files from various sources and converts
    them to the standardized formats used throughout the application.
    """

    def __init__(self, input_dir: Path, output_dir: Path):
        """
        Initialize the data processor.

        Parameters
        ----------
        input_dir : Path
            Directory containing raw data files.
        output_dir : Path
            Directory for processed data output.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = get_config()

    def process_all(self, validate_output: bool = True) -> Dict[str, Path]:
        """
        Process all available raw data files with optional validation.

        Parameters
        ----------
        validate_output : bool
            Whether to validate processed data files.

        Returns
        -------
        Dict[str, Path]
            Dictionary mapping data source names to processed file paths.
        """
        processed_files = {}
        validator = DataValidator() if validate_output else None

        # Process ASRank data
        asrank_file = self._find_asrank_file()
        if asrank_file:
            processed_files["asrank"] = self.process_asrank(
                asrank_file, validate_output
            )

        # Process PeeringDB data
        peeringdb_file = self._find_peeringdb_file()
        if peeringdb_file:
            processed_files["peeringdb"] = self.process_peeringdb(
                peeringdb_file, validate_output
            )

        # Process ASPOP data
        aspop_file = self._find_aspop_file()
        if aspop_file:
            processed_files["aspop"] = self.process_aspop(aspop_file, validate_output)

        # Process IPinfo data
        ipinfo_file = self._find_ipinfo_file()
        if ipinfo_file:
            processed_files["ipinfo"] = self.process_ipinfo(
                ipinfo_file, validate_output
            )

        # Generate validation report if requested
        if validate_output and validator and processed_files:
            validation_results = []
            for source, file_path in processed_files.items():
                try:
                    if source == "asrank":
                        result = validator.validate_asrank_data(file_path)
                    elif source == "peeringdb":
                        result = validator.validate_peeringdb_data(file_path)
                    elif source == "aspop":
                        result = validator.validate_aspop_data(file_path)
                    elif source == "ipinfo":
                        result = validator.validate_ipinfo_data(file_path)
                    else:
                        continue

                    validation_results.append(result)

                    if not result.is_valid:
                        console.print(
                            f"[yellow]⚠[/yellow] Data quality issues in {source}: {', '.join(result.errors)}"
                        )
                    else:
                        console.print(
                            f"[green]✓[/green] {source} data passed validation (quality: {result.quality_metrics.overall_quality_score:.1%})"
                        )

                except Exception as e:
                    console.print(f"[red]✗[/red] Failed to validate {source}: {e}")

            # Save validation report
            if validation_results:
                report_file = self.output_dir / "data_quality_report.md"
                report = validator.generate_quality_report(validation_results)
                with open(report_file, "w", encoding="utf-8") as f:
                    f.write(report)
                console.print(
                    f"[blue]📊[/blue] Data quality report saved to {report_file}"
                )

        return processed_files

    def process_asrank(self, input_file: Path, validate: bool = False) -> Path:
        """
        Process ASRank JSONL file to CSV format.

        Parameters
        ----------
        input_file : Path
            Path to raw ASRank JSONL file.

        Returns
        -------
        Path
            Path to processed CSV file.
        """
        output_file = self.output_dir / "as_rank_features.csv"

        console.print(f"[blue]Processing ASRank data: {input_file}[/blue]")

        records = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    processed_record = self._flatten_asrank_record(record)
                    records.append(processed_record)

        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False)

        console.print(
            f"[green]✓[/green] Processed {len(records)} ASRank records to {output_file}"
        )
        return output_file

    def process_peeringdb(self, input_file: Path, validate: bool = False) -> Path:
        """
        Process PeeringDB JSON file to structured format.

        Parameters
        ----------
        input_file : Path
            Path to raw PeeringDB JSON file.

        Returns
        -------
        Path
            Path to processed JSON file.
        """
        output_file = self.output_dir / "peeringdb_net.json"

        console.print(f"[blue]Processing PeeringDB data: {input_file}[/blue]")

        with open(input_file, "r", encoding="utf-8") as f:
            peeringdb_dump = json.load(f)

        # Extract network data
        networks = {}
        if "net" in peeringdb_dump and "data" in peeringdb_dump["net"]:
            for net in peeringdb_dump["net"]["data"]:
                asn = str(net.get("asn", ""))
                if asn:
                    networks[asn] = net

        # Save processed data
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(networks, f, indent=2)

        console.print(
            f"[green]✓[/green] Processed {len(networks)} PeeringDB records to {output_file}"
        )
        return output_file

    def process_aspop(self, input_file: Path, validate: bool = False) -> Path:
        """
        Process ASPOP CSV file to structured JSON format.

        Parameters
        ----------
        input_file : Path
            Path to raw ASPOP CSV file.

        Returns
        -------
        Path
            Path to processed JSON file.
        """
        output_file = self.output_dir / "aspop_processed.json"

        console.print(f"[blue]Processing ASPOP data: {input_file}[/blue]")

        aspop_data = {}

        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                asn = row.get("ASN", "").replace("AS", "")
                if asn:
                    aspop_data[asn] = {
                        "name": row.get("Name", ""),
                        "country": row.get("CC", ""),
                        "rir": row.get("RIR", ""),
                        "customer_cone_asns": (
                            int(row.get("Cust_cone_ASN", 0))
                            if row.get("Cust_cone_ASN")
                            else None
                        ),
                        "customer_cone_prefixes": (
                            int(row.get("Cust_cone_pfx", 0))
                            if row.get("Cust_cone_pfx")
                            else None
                        ),
                        "customer_cone_addresses": (
                            int(row.get("Cust_cone_addr", 0))
                            if row.get("Cust_cone_addr")
                            else None
                        ),
                    }

        # Save processed data
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(aspop_data, f, indent=2)

        console.print(
            f"[green]✓[/green] Processed {len(aspop_data)} ASPOP records to {output_file}"
        )
        return output_file

    def process_ipinfo(self, input_file: Path, validate: bool = False) -> Path:
        """
        Process IPinfo JSON file to structured format.

        Parameters
        ----------
        input_file : Path
            Path to raw IPinfo JSON file (batch format).

        Returns
        -------
        Path
            Path to processed JSON file.
        """
        output_file = self.output_dir / "ipinfo_processed.json"

        console.print(f"[blue]Processing IPinfo data: {input_file}[/blue]")

        with open(input_file, "r", encoding="utf-8") as f:
            ipinfo_batch = json.load(f)

        # Extract network data from batch format
        ipinfo_data = {}

        # Handle both direct data format and batch format
        if "data" in ipinfo_batch and isinstance(ipinfo_batch["data"], dict):
            # Batch format from downloader
            for asn_str, data in ipinfo_batch["data"].items():
                if "error" not in data:
                    # Clean up the data for processing
                    processed_record = self._process_ipinfo_record(data)
                    if processed_record:
                        ipinfo_data[asn_str] = processed_record
        elif isinstance(ipinfo_batch, dict):
            # Direct data format
            for asn_str, data in ipinfo_batch.items():
                if isinstance(data, dict) and "error" not in data:
                    processed_record = self._process_ipinfo_record(data)
                    if processed_record:
                        ipinfo_data[asn_str] = processed_record

        # Save processed data
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(ipinfo_data, f, indent=2)

        console.print(
            f"[green]✓[/green] Processed {len(ipinfo_data)} IPinfo records to {output_file}"
        )
        return output_file

    def _process_ipinfo_record(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process individual IPinfo record.

        Parameters
        ----------
        data : Dict[str, Any]
            Raw IPinfo record.

        Returns
        -------
        Optional[Dict[str, Any]]
            Processed record or None if invalid.
        """
        if not data or "error" in data:
            return None

        # Extract ASN from metadata or org field
        asn = None
        if "_ipinfo_metadata" in data and "asn" in data["_ipinfo_metadata"]:
            asn = data["_ipinfo_metadata"]["asn"]
        elif "org" in data:
            # Try to extract ASN from org field (format: "AS15169 Google LLC")
            org_parts = data["org"].split()
            if org_parts and org_parts[0].startswith("AS"):
                try:
                    asn = int(org_parts[0][2:])  # Remove "AS" prefix
                except ValueError:
                    pass

        if not asn:
            return None

        # Build processed record
        processed = {
            "asn": asn,
            "org": data.get("org"),
            "name": data.get("name"),
            "domain": data.get("domain"),
            "country": data.get("country"),
            "country_name": data.get("country_name"),
            "region": data.get("region"),
            "city": data.get("city"),
            "postal": data.get("postal"),
            "timezone": data.get("timezone"),
            "route": data.get("route"),
            "type": data.get("type"),
            "allocation": data.get("allocation"),
            "registry": data.get("registry"),
            "abuse": data.get("abuse"),
            "tech": data.get("tech"),
            "admin": data.get("admin"),
        }

        # Add metadata if available
        if "_ipinfo_metadata" in data:
            processed["downloaded_at"] = data["_ipinfo_metadata"].get("downloaded_at")
            processed["api_source"] = data["_ipinfo_metadata"].get("api_source")

        # Remove None values
        processed = {k: v for k, v in processed.items() if v is not None}

        return processed

    def _find_asrank_file(self) -> Path:
        """Find the most recent ASRank JSONL file."""
        asrank_files = list(self.input_dir.glob("asrank_*.jsonl"))
        return (
            max(asrank_files, key=lambda x: x.stat().st_mtime) if asrank_files else None
        )

    def _find_peeringdb_file(self) -> Path:
        """Find the most recent PeeringDB JSON file."""
        peeringdb_files = list(self.input_dir.glob("peeringdb_*.json"))
        return (
            max(peeringdb_files, key=lambda x: x.stat().st_mtime)
            if peeringdb_files
            else None
        )

    def _find_aspop_file(self) -> Path:
        """Find the most recent ASPOP CSV file."""
        aspop_files = list(self.input_dir.glob("aspop_*.csv"))
        return (
            max(aspop_files, key=lambda x: x.stat().st_mtime) if aspop_files else None
        )

    def _find_ipinfo_file(self) -> Path:
        """Find the most recent IPinfo JSON file."""
        # Look for both batch and individual files
        ipinfo_files = list(self.input_dir.glob("ipinfo_*.json"))
        return (
            max(ipinfo_files, key=lambda x: x.stat().st_mtime) if ipinfo_files else None
        )

    def _flatten_asrank_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten nested ASRank record structure.

        Parameters
        ----------
        record : Dict[str, Any]
            Raw ASRank record from API.

        Returns
        -------
        Dict[str, Any]
            Flattened record suitable for CSV format.
        """
        flat_record = {
            "asn": record.get("asn"),
            "asnName": record.get("asnName"),
            "rank": record.get("rank"),
            "cliqueMember": record.get("cliqueMember"),
            "seen": record.get("seen"),
            "longitude": record.get("longitude"),
            "latitude": record.get("latitude"),
        }

        # Organization info
        org = record.get("organization", {})
        if org:
            flat_record["orgId"] = org.get("orgId")
            flat_record["orgName"] = org.get("orgName")

        # Country info
        country = record.get("country", {})
        if country:
            flat_record["country_iso"] = country.get("iso")
            flat_record["country_name"] = country.get("name")

        # Cone statistics
        cone = record.get("cone", {})
        if cone:
            flat_record["cone_numberAsns"] = cone.get("numberAsns")
            flat_record["cone_numberPrefixes"] = cone.get("numberPrefixes")
            flat_record["cone_numberAddresses"] = cone.get("numberAddresses")

        # Degree statistics
        degree = record.get("asnDegree", {})
        if degree:
            flat_record["degree_provider"] = degree.get("provider")
            flat_record["degree_peer"] = degree.get("peer")
            flat_record["degree_customer"] = degree.get("customer")
            flat_record["degree_total"] = degree.get("total")
            flat_record["degree_transit"] = degree.get("transit")
            flat_record["degree_sibling"] = degree.get("sibling")

        # Announcing statistics
        announcing = record.get("announcing", {})
        if announcing:
            flat_record["announcing_numberPrefixes"] = announcing.get("numberPrefixes")
            flat_record["announcing_numberAddresses"] = announcing.get(
                "numberAddresses"
            )

        return flat_record
