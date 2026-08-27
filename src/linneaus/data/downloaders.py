"""
Data downloaders for external APIs and datasets.

This module contains classes for downloading data from various sources
including ASRank, PeeringDB, APNIC ASPOP, and IPinfo.
"""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import get_config

console = Console()


class BaseDownloader(ABC):
    """Base class for all data downloaders."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the downloader.

        Parameters
        ----------
        output_dir : Path, optional
            Output directory for downloaded data. Uses config default if None.
        """
        self.config = get_config()
        self.output_dir = output_dir or self.config.data.raw_data_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def download(self, date: Optional[str] = None) -> Path:
        """
        Download data from the source.

        Parameters
        ----------
        date : str, optional
            Target date in YYYY-MM-DD format. Uses latest if None.

        Returns
        -------
        Path
            Path to the downloaded data file.
        """

    @abstractmethod
    def get_latest_date(self) -> str:
        """
        Get the latest available data date.

        Returns
        -------
        str
            Latest available date in YYYY-MM-DD format.
        """


class ASRankDownloader(BaseDownloader):
    """
    Downloader for ASRank data via GraphQL API.

    This class downloads AS ranking and metadata from CAIDA's ASRank API
    using GraphQL queries.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize ASRank downloader."""
        super().__init__(output_dir)
        self.api_url = self.config.apis.asrank_url
        self.page_size = 10000

    async def download(self, date: Optional[str] = None) -> Path:
        """
        Download ASRank data.

        Parameters
        ----------
        date : str, optional
            Date is not used for ASRank as it provides current data.

        Returns
        -------
        Path
            Path to the downloaded ASRank JSONL file.
        """
        output_file = (
            self.output_dir / f"asrank_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading ASRank data...", total=None)

            async with aiohttp.ClientSession() as session:
                await self._download_asns(session, output_file, progress, task)

        console.print(f"[green]✓[/green] ASRank data downloaded to {output_file}")
        return output_file

    def get_latest_date(self) -> str:
        """Get current date as ASRank provides real-time data."""
        return datetime.now().strftime("%Y-%m-%d")

    async def _download_asns(
        self,
        session: aiohttp.ClientSession,
        output_file: Path,
        progress: Progress,
        task_id,
    ) -> None:
        """Download ASN data using GraphQL pagination."""
        has_next_page = True
        offset = 0
        total_count = 0

        with open(output_file, "w", encoding="utf-8") as f:
            while has_next_page:
                query = self._build_asn_query(self.page_size, offset)

                async with session.post(
                    self.api_url,
                    json={"query": query},
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status != 200:
                        raise Exception(f"ASRank API request failed: {response.status}")

                    data = await response.json()

                    if "errors" in data:
                        raise Exception(f"ASRank API errors: {data['errors']}")

                    asns_data = data["data"]["asns"]

                    # Write ASN records
                    for edge in asns_data["edges"]:
                        asn_record = edge["node"]
                        f.write(json.dumps(asn_record) + "\n")

                    # Update pagination
                    has_next_page = asns_data["pageInfo"]["hasNextPage"]
                    offset += asns_data["pageInfo"]["first"]
                    total_count = asns_data["totalCount"]

                    progress.update(
                        task_id,
                        description=f"Downloaded {offset}/{total_count} ASNs...",
                    )

    def _build_asn_query(self, page_size: int, offset: int) -> str:
        """Build GraphQL query for ASN data."""
        return f"""
        {{
            asns(first: {page_size}, offset: {offset}) {{
                totalCount
                pageInfo {{
                    first
                    hasNextPage
                }}
                edges {{
                    node {{
                        asn
                        asnName
                        rank
                        organization {{
                            orgId
                            orgName
                        }}
                        cliqueMember
                        seen
                        longitude
                        latitude
                        cone {{
                            numberAsns
                            numberPrefixes
                            numberAddresses
                        }}
                        country {{
                            iso
                            name
                        }}
                        asnDegree {{
                            provider
                            peer
                            customer
                            total
                            transit
                            sibling
                        }}
                        announcing {{
                            numberPrefixes
                            numberAddresses
                        }}
                    }}
                }}
            }}
        }}
        """


class PeeringDBDownloader(BaseDownloader):
    """
    Downloader for PeeringDB data snapshots.

    This class downloads PeeringDB data snapshots from CAIDA's public
    data repository.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize PeeringDB downloader."""
        super().__init__(output_dir)
        self.base_url = self.config.apis.peeringdb_base_url

    async def download(self, date: Optional[str] = None) -> Path:
        """
        Download PeeringDB data snapshot.

        Parameters
        ----------
        date : str, optional
            Target date in YYYY-MM-DD format. Uses latest if None.

        Returns
        -------
        Path
            Path to the downloaded PeeringDB JSON file.
        """
        if date is None:
            date = self.get_latest_date()

        # Find closest available date
        actual_date = await self._find_closest_date(date)

        # Build download URL
        year, month, day = actual_date.split("-")
        filename = f"peeringdb_2_dump_{actual_date.replace('-', '_')}.json"
        url = f"{self.base_url}/{year}/{month:0>2}/{filename}"

        output_file = self.output_dir / filename

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Downloading PeeringDB data for {actual_date}...", total=None
            )

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise Exception(f"PeeringDB download failed: {response.status}")

                    with open(output_file, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)

        console.print(f"[green]✓[/green] PeeringDB data downloaded to {output_file}")
        return output_file

    def get_latest_date(self) -> str:
        """Get the latest available PeeringDB data date."""
        # PeeringDB snapshots are typically available with 1-2 day delay
        return (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")

    async def _find_closest_date(self, target_date: str) -> str:
        """
        Find the closest available PeeringDB snapshot date with enhanced search.

        Parameters
        ----------
        target_date : str
            Target date in YYYY-MM-DD format.

        Returns
        -------
        str
            Closest available date in YYYY-MM-DD format.
        """
        target = datetime.strptime(target_date, "%Y-%m-%d")
        console.print(
            f"[blue]Searching for PeeringDB snapshot closest to {target_date}...[/blue]"
        )

        # Enhanced search: try both backward and forward from target date
        max_search_days = 14  # Extended search window
        candidates = []

        # Create concurrent tasks for checking multiple dates
        check_tasks = []

        # Check target date and surrounding dates
        for days_offset in range(-max_search_days // 2, max_search_days // 2 + 1):
            check_date = target + timedelta(days=days_offset)

            # Don't check future dates beyond today
            if check_date > datetime.now():
                continue

            date_str = check_date.strftime("%Y-%m-%d")
            check_tasks.append(self._check_date_with_info(date_str, abs(days_offset)))

        # Execute all checks concurrently
        results = await asyncio.gather(*check_tasks, return_exceptions=True)

        # Collect available dates with their distance from target
        for result in results:
            if isinstance(result, tuple) and result[1]:  # (date, available, distance)
                candidates.append((result[0], result[2]))  # (date, distance)

        if candidates:
            # Sort by distance from target date (closest first)
            candidates.sort(key=lambda x: x[1])
            closest_date = candidates[0][0]

            console.print(
                f"[green]✓[/green] Found closest PeeringDB snapshot: {closest_date}"
            )
            return closest_date

        # Enhanced fallback: try known good dates
        console.print(
            f"[yellow]⚠[/yellow] No snapshots found near {target_date}, trying fallback dates..."
        )

        fallback_dates = [
            self.get_latest_date(),
            (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
            (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
            (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        ]

        for fallback_date in fallback_dates:
            if await self._check_date_available(fallback_date):
                console.print(
                    f"[green]✓[/green] Using fallback PeeringDB snapshot: {fallback_date}"
                )
                return fallback_date

        # Final fallback
        return self.get_latest_date()

    async def _check_date_with_info(
        self, date: str, distance: int
    ) -> Tuple[str, bool, int]:
        """
        Check if date is available and return with distance info.

        Parameters
        ----------
        date : str
            Date to check in YYYY-MM-DD format.
        distance : int
            Distance in days from target date.

        Returns
        -------
        Tuple[str, bool, int]
            (date, is_available, distance_from_target)
        """
        try:
            available = await self._check_date_available(date)
            return (date, available, distance)
        except Exception:
            return (date, False, distance)

    async def _check_date_available(self, date: str) -> bool:
        """Check if PeeringDB data is available for a specific date with timeout."""
        year, month, day = date.split("-")
        filename = f"peeringdb_2_dump_{date.replace('-', '_')}.json"
        url = f"{self.base_url}/{year}/{month:0>2}/{filename}"

        try:
            timeout = aiohttp.ClientTimeout(total=10)  # Quick check timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.head(url) as response:
                    return response.status == 200
        except Exception:
            return False


class APNICDownloader(BaseDownloader):
    """
    Downloader for APNIC AS Population (ASPOP) data.

    This class downloads AS population statistics from APNIC.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize APNIC downloader."""
        super().__init__(output_dir)
        self.base_url = self.config.apis.apnic_aspop_url

    async def download(self, date: Optional[str] = None) -> Path:
        """
        Download APNIC ASPOP data with enhanced date handling.

        Parameters
        ----------
        date : str, optional
            Target date in YYYY-MM-DD format. Uses latest if None.

        Returns
        -------
        Path
            Path to the downloaded ASPOP CSV file.
        """
        if date is None:
            date = self.get_latest_date()

        # Find closest available date
        actual_date = await self._find_closest_aspop_date(date)

        # Format date for APNIC API (DD/MM/YYYY)
        date_obj = datetime.strptime(actual_date, "%Y-%m-%d")
        apnic_date = date_obj.strftime("%d/%m/%Y")

        # Build API URL with enhanced parameters
        params = {
            "w": "120",  # Width parameter
            "d": apnic_date,
            "f": "c",  # CSV format
        }

        url = f"{self.base_url}?" + "&".join(f"{k}={v}" for k, v in params.items())
        output_file = self.output_dir / f"aspop_{actual_date.replace('-', '')}.csv"

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Downloading ASPOP data for {actual_date}...", total=None
            )

            # Enhanced timeout and retry configuration
            timeout = aiohttp.ClientTimeout(total=60)  # ASPOP can be slow

            async with aiohttp.ClientSession(timeout=timeout) as session:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        async with session.get(url) as response:
                            if response.status == 200:
                                content = await response.text()

                                # Validate content (should be CSV)
                                if self._validate_aspop_content(content):
                                    with open(output_file, "w", encoding="utf-8") as f:
                                        f.write(content)

                                    console.print(
                                        f"[green]✓[/green] ASPOP data downloaded to {output_file}"
                                    )
                                    return output_file
                                else:
                                    raise Exception(
                                        "Downloaded content is not valid CSV data"
                                    )
                            else:
                                raise Exception(
                                    f"ASPOP download failed: {response.status}"
                                )

                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = 2**attempt  # Exponential backoff
                            console.print(
                                f"[yellow]⚠[/yellow] Attempt {attempt + 1} failed: {e}, retrying in {wait_time}s..."
                            )
                            await asyncio.sleep(wait_time)
                        else:
                            raise

        raise Exception("Failed to download ASPOP data after all retries")

    def get_latest_date(self) -> str:
        """Get the latest available ASPOP data date."""
        # ASPOP data is typically available with 1 day delay
        return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    async def _find_closest_aspop_date(self, target_date: str) -> str:
        """
        Find the closest available ASPOP data date.

        ASPOP data is typically available daily, but we should check
        a few days around the target date for availability.

        Parameters
        ----------
        target_date : str
            Target date in YYYY-MM-DD format.

        Returns
        -------
        str
            Closest available date in YYYY-MM-DD format.
        """
        target = datetime.strptime(target_date, "%Y-%m-%d")
        console.print(
            f"[blue]Searching for ASPOP data closest to {target_date}...[/blue]"
        )

        # ASPOP data is usually available daily, so check a smaller window
        max_search_days = 7

        # Try dates in order of preference: target, previous days, then future days
        search_order = [0]  # Start with target date
        for days in range(1, max_search_days):
            search_order.extend([-days, days])  # Add both past and future

        for days_offset in search_order:
            check_date = target + timedelta(days=days_offset)

            # Don't check future dates beyond today
            if check_date > datetime.now():
                continue

            date_str = check_date.strftime("%Y-%m-%d")

            if await self._check_aspop_date_available(date_str):
                if date_str != target_date:
                    console.print(
                        f"[yellow]⚠[/yellow] Target date {target_date} not available, using {date_str}"
                    )
                else:
                    console.print(
                        f"[green]✓[/green] Found ASPOP data for target date: {date_str}"
                    )
                return date_str

        # Fallback to latest available
        console.print(
            f"[yellow]⚠[/yellow] No ASPOP data found near {target_date}, using latest available"
        )
        return self.get_latest_date()

    async def _check_aspop_date_available(self, date: str) -> bool:
        """
        Check if ASPOP data is available for a specific date.

        Parameters
        ----------
        date : str
            Date to check in YYYY-MM-DD format.

        Returns
        -------
        bool
            True if data is available for the date.
        """
        try:
            # Format date for APNIC API
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            apnic_date = date_obj.strftime("%d/%m/%Y")

            # Build test URL
            params = {
                "w": "120",
                "d": apnic_date,
                "f": "c",
            }

            url = f"{self.base_url}?" + "&".join(f"{k}={v}" for k, v in params.items())

            # Quick HEAD request to check availability
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.head(url) as response:
                    if response.status == 200:
                        # For ASPOP, also check if we get a reasonable content-length
                        content_length = response.headers.get("content-length")
                        if (
                            content_length and int(content_length) > 1000
                        ):  # Reasonable data size
                            return True
                    return False

        except Exception:
            return False

    def _validate_aspop_content(self, content: str) -> bool:
        """
        Validate that downloaded content is valid ASPOP CSV data.

        Parameters
        ----------
        content : str
            Downloaded content to validate.

        Returns
        -------
        bool
            True if content appears to be valid ASPOP data.
        """
        try:
            # Basic validation: check for CSV structure and expected headers
            lines = content.strip().split("\n")

            if len(lines) < 10:  # Should have at least some data
                return False

            # Check for expected CSV headers (ASPOP typically has ASN, Name, etc.)
            header_line = lines[0].lower()
            expected_fields = ["asn", "name", "cc", "rir"]

            for field in expected_fields:
                if field not in header_line:
                    return False

            # Check that we have a reasonable number of data rows
            data_rows = [line for line in lines[1:] if line.strip()]
            return len(data_rows) > 100  # Should have many ASNs

        except Exception:
            return False


class IPinfoDownloader(BaseDownloader):
    """
    Downloader for IPinfo AS data.

    This class downloads AS organization data from IPinfo's API,
    supporting both single ASN and batch downloads.
    """

    def __init__(self, output_dir: Optional[Path] = None, token: Optional[str] = None):
        """
        Initialize IPinfo downloader.

        Parameters
        ----------
        output_dir : Path, optional
            Output directory for downloaded data.
        token : str, optional
            IPinfo API token. If None, will use IPINFO_TOKEN environment variable.
        """
        super().__init__(output_dir)
        self.token = token or os.getenv("IPINFO_TOKEN")
        self.base_url = self.config.apis.get("ipinfo_base_url", "https://ipinfo.io")

        # Get rate limiting configuration
        rate_limits = self.config.download.get("rate_limits", {})
        self.rate_limit_delay = rate_limits.get("ipinfo", 1.0)
        self.max_retries = self.config.download.get("max_retries", 3)
        self.timeout = self.config.download.get("timeout", 30)

    async def download(self, date: Optional[str] = None) -> Path:
        """
        Download IPinfo data for a single ASN.

        This method is primarily for API compatibility. For batch processing,
        use download_batch() instead.

        Parameters
        ----------
        date : str, optional
            Date parameter is not used for IPinfo (real-time data).

        Returns
        -------
        Path
            Path to downloaded data file.
        """
        # For API compatibility, return a sample download
        # In practice, use download_asn() or download_batch()
        output_file = (
            self.output_dir / f"ipinfo_sample_{datetime.now().strftime('%Y%m%d')}.json"
        )

        # Create empty file to satisfy interface
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({}, f)

        console.print(
            f"[yellow]⚠[/yellow] IPinfo requires specific ASN(s). Use download_asn() or download_batch()"
        )
        return output_file

    def get_latest_date(self) -> str:
        """Get current date as IPinfo provides real-time data."""
        return datetime.now().strftime("%Y-%m-%d")

    async def download_asn(
        self, asn: int, include_details: bool = True, retry_count: int = 0
    ) -> Dict:
        """
        Download data for a specific ASN from IPinfo.

        Parameters
        ----------
        asn : int
            Autonomous System Number.
        include_details : bool
            Whether to include detailed information.
        retry_count : int
            Current retry attempt count.

        Returns
        -------
        Dict
            IPinfo data for the ASN.
        """
        url = f"{self.base_url}/AS{asn}/json"
        params = {}

        if self.token:
            params["token"] = self.token

        # Create session with timeout
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                # Rate limiting with jitter
                jitter = 0.1 * (1 + retry_count)  # Add jitter for retries
                await asyncio.sleep(self.rate_limit_delay + jitter)

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Add metadata
                        data["_ipinfo_metadata"] = {
                            "asn": asn,
                            "downloaded_at": datetime.now().isoformat(),
                            "api_source": "ipinfo",
                            "retry_count": retry_count,
                        }

                        return data

                    elif response.status == 429:  # Rate limited
                        if retry_count < self.max_retries:
                            backoff_delay = min(
                                5.0 * (2**retry_count), 30.0
                            )  # Exponential backoff
                            console.print(
                                f"[yellow]⚠[/yellow] Rate limited for ASN {asn}, retrying in {backoff_delay:.1f}s..."
                            )
                            await asyncio.sleep(backoff_delay)
                            return await self.download_asn(
                                asn, include_details, retry_count + 1
                            )
                        else:
                            return {
                                "error": "Rate limit exceeded after retries",
                                "asn": asn,
                            }

                    elif response.status in [500, 502, 503, 504]:  # Server errors
                        if retry_count < self.max_retries:
                            backoff_delay = min(2.0 * (2**retry_count), 10.0)
                            console.print(
                                f"[yellow]⚠[/yellow] Server error for ASN {asn}, retrying in {backoff_delay:.1f}s..."
                            )
                            await asyncio.sleep(backoff_delay)
                            return await self.download_asn(
                                asn, include_details, retry_count + 1
                            )
                        else:
                            return {
                                "error": f"Server error {response.status} after retries",
                                "asn": asn,
                            }

                    else:
                        error_text = await response.text()
                        console.print(
                            f"[red]✗[/red] Failed to download ASN {asn}: {response.status} - {error_text}"
                        )
                        return {"error": f"HTTP {response.status}", "asn": asn}

            except asyncio.TimeoutError:
                if retry_count < self.max_retries:
                    console.print(
                        f"[yellow]⚠[/yellow] Timeout for ASN {asn}, retrying..."
                    )
                    await asyncio.sleep(2.0 * (retry_count + 1))
                    return await self.download_asn(
                        asn, include_details, retry_count + 1
                    )
                else:
                    return {"error": "Timeout after retries", "asn": asn}

            except Exception as e:
                if retry_count < self.max_retries:
                    console.print(
                        f"[yellow]⚠[/yellow] Error downloading ASN {asn}: {e}, retrying..."
                    )
                    await asyncio.sleep(1.0 * (retry_count + 1))
                    return await self.download_asn(
                        asn, include_details, retry_count + 1
                    )
                else:
                    console.print(
                        f"[red]✗[/red] Error downloading ASN {asn} after retries: {e}"
                    )
                    return {"error": str(e), "asn": asn}

    async def download_batch(self, asns: List[int], max_concurrent: int = 5) -> Path:
        """
        Download data for multiple ASNs concurrently with enhanced async patterns.

        Parameters
        ----------
        asns : List[int]
            List of Autonomous System Numbers.
        max_concurrent : int
            Maximum concurrent requests.

        Returns
        -------
        Path
            Path to the downloaded batch file.
        """
        output_file = (
            self.output_dir
            / f"ipinfo_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        console.print(f"[blue]Downloading IPinfo data for {len(asns)} ASNs...[/blue]")

        # Use configured max_concurrent or parameter
        max_workers = min(max_concurrent, self.config.download.get("max_concurrent", 5))

        # Enhanced connection configuration
        connector_config = {
            "limit": max_workers * 2,  # Connection pool size
            "limit_per_host": max_workers,
            "ttl_dns_cache": 300,  # DNS cache TTL
            "use_dns_cache": True,
            "keepalive_timeout": 30,
            "enable_cleanup_closed": True,
        }

        timeout_config = aiohttp.ClientTimeout(
            total=self.timeout,
            connect=10,  # Connection timeout
            sock_read=15,  # Socket read timeout
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading IPinfo data...", total=len(asns))

            # Create enhanced semaphore with fair scheduling
            semaphore = asyncio.Semaphore(max_workers)

            # Shared session with connection pooling
            connector = aiohttp.TCPConnector(**connector_config)

            async with aiohttp.ClientSession(
                connector=connector, timeout=timeout_config
            ) as session:

                async def download_with_enhanced_semaphore(
                    asn: int,
                ) -> Tuple[int, Dict]:
                    """Enhanced download function with better resource management."""
                    async with semaphore:
                        try:
                            result = await self._download_asn_with_session(session, asn)
                            progress.advance(task)
                            return asn, result
                        except Exception as e:
                            progress.advance(task)
                            console.print(f"[yellow]⚠[/yellow] Failed ASN {asn}: {e}")
                            return asn, {"error": str(e), "asn": asn}

                # Process in chunks to avoid overwhelming the API
                chunk_size = max_workers * 2
                all_results = []

                for i in range(0, len(asns), chunk_size):
                    chunk_asns = asns[i : i + chunk_size]

                    # Execute chunk downloads concurrently
                    chunk_tasks = [
                        download_with_enhanced_semaphore(asn) for asn in chunk_asns
                    ]
                    chunk_results = await asyncio.gather(
                        *chunk_tasks, return_exceptions=True
                    )

                    all_results.extend(chunk_results)

                    # Brief pause between chunks to be API-friendly
                    if i + chunk_size < len(asns):
                        await asyncio.sleep(0.5)

                # Process all results
                ipinfo_data = {}
                success_count = 0
                error_count = 0
                exception_count = 0

                for result in all_results:
                    if isinstance(result, Exception):
                        exception_count += 1
                        console.print(f"[red]✗[/red] Exception: {result}")
                    else:
                        asn, data = result
                        ipinfo_data[str(asn)] = data

                        if "error" not in data:
                            success_count += 1
                        else:
                            error_count += 1

                # Enhanced metadata with performance metrics
                batch_output = {
                    "metadata": {
                        "source": "ipinfo",
                        "downloaded_at": datetime.now().isoformat(),
                        "total_asns": len(asns),
                        "successful_downloads": success_count,
                        "failed_downloads": error_count,
                        "exception_count": exception_count,
                        "max_concurrent": max_workers,
                        "chunk_size": chunk_size,
                        "connection_pool_size": connector_config["limit"],
                        "success_rate": success_count / len(asns) if asns else 0,
                    },
                    "data": ipinfo_data,
                }

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(batch_output, f, indent=2)

        console.print(
            f"[green]✓[/green] IPinfo batch download completed: "
            f"{success_count} successful, {error_count} failed, {exception_count} exceptions"
        )
        console.print(f"[green]✓[/green] Results saved to {output_file}")

        return output_file

    async def _download_asn_with_session(
        self, session: aiohttp.ClientSession, asn: int, retry_count: int = 0
    ) -> Dict:
        """
        Download data for a specific ASN using a shared session.

        Parameters
        ----------
        session : aiohttp.ClientSession
            Shared HTTP session with connection pooling.
        asn : int
            Autonomous System Number.
        retry_count : int
            Current retry attempt count.

        Returns
        -------
        Dict
            IPinfo data for the ASN.
        """
        url = f"{self.base_url}/AS{asn}/json"
        params = {}

        if self.token:
            params["token"] = self.token

        try:
            # Enhanced rate limiting with adaptive jitter
            base_delay = self.rate_limit_delay
            jitter = base_delay * 0.1 * (1 + retry_count)  # Adaptive jitter
            await asyncio.sleep(base_delay + jitter)

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # Add enhanced metadata
                    data["_ipinfo_metadata"] = {
                        "asn": asn,
                        "downloaded_at": datetime.now().isoformat(),
                        "api_source": "ipinfo",
                        "retry_count": retry_count,
                        "response_time_ms": getattr(response, "_response_time", None),
                        "server": response.headers.get("Server"),
                    }

                    return data

                elif response.status == 429:  # Rate limited
                    if retry_count < self.max_retries:
                        # Enhanced exponential backoff with jitter
                        backoff_delay = min(5.0 * (2**retry_count), 30.0)
                        jitter = (
                            backoff_delay
                            * 0.2
                            * (0.5 + asyncio.get_event_loop().time() % 1)
                        )
                        total_delay = backoff_delay + jitter

                        console.print(
                            f"[yellow]⚠[/yellow] Rate limited for ASN {asn}, retrying in {total_delay:.1f}s..."
                        )
                        await asyncio.sleep(total_delay)
                        return await self._download_asn_with_session(
                            session, asn, retry_count + 1
                        )
                    else:
                        return {
                            "error": "Rate limit exceeded after retries",
                            "asn": asn,
                        }

                elif response.status in [500, 502, 503, 504]:  # Server errors
                    if retry_count < self.max_retries:
                        backoff_delay = min(2.0 * (2**retry_count), 10.0)
                        jitter = backoff_delay * 0.1  # Smaller jitter for server errors

                        console.print(
                            f"[yellow]⚠[/yellow] Server error for ASN {asn}, retrying in {backoff_delay + jitter:.1f}s..."
                        )
                        await asyncio.sleep(backoff_delay + jitter)
                        return await self._download_asn_with_session(
                            session, asn, retry_count + 1
                        )
                    else:
                        return {
                            "error": f"Server error {response.status} after retries",
                            "asn": asn,
                        }

                else:
                    error_text = await response.text()
                    console.print(
                        f"[red]✗[/red] Failed to download ASN {asn}: {response.status} - {error_text}"
                    )
                    return {
                        "error": f"HTTP {response.status}",
                        "asn": asn,
                        "details": error_text,
                    }

        except asyncio.TimeoutError:
            if retry_count < self.max_retries:
                console.print(f"[yellow]⚠[/yellow] Timeout for ASN {asn}, retrying...")
                await asyncio.sleep(2.0 * (retry_count + 1))
                return await self._download_asn_with_session(
                    session, asn, retry_count + 1
                )
            else:
                return {"error": "Timeout after retries", "asn": asn}

        except Exception as e:
            if retry_count < self.max_retries:
                console.print(
                    f"[yellow]⚠[/yellow] Error downloading ASN {asn}: {e}, retrying..."
                )
                await asyncio.sleep(1.0 * (retry_count + 1))
                return await self._download_asn_with_session(
                    session, asn, retry_count + 1
                )
            else:
                console.print(
                    f"[red]✗[/red] Error downloading ASN {asn} after retries: {e}"
                )
                return {"error": str(e), "asn": asn}


async def download_all_sources(
    sources: List[str], date: Optional[str] = None, output_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Download data from multiple sources concurrently.

    Parameters
    ----------
    sources : List[str]
        List of data sources to download. Options: 'asrank', 'peeringdb', 'aspop', 'ipinfo'.
    date : str, optional
        Target date in YYYY-MM-DD format. Uses latest if None.
    output_dir : Path, optional
        Output directory for downloaded data.

    Returns
    -------
    Dict[str, Path]
        Dictionary mapping source names to downloaded file paths.
    """
    downloaders = {
        "asrank": ASRankDownloader(output_dir),
        "peeringdb": PeeringDBDownloader(output_dir),
        "aspop": APNICDownloader(output_dir),
        "ipinfo": IPinfoDownloader(output_dir),
    }

    # Validate sources
    invalid_sources = set(sources) - set(downloaders.keys())
    if invalid_sources:
        raise ValueError(f"Invalid sources: {invalid_sources}")

    # Create download tasks
    tasks = []
    for source in sources:
        downloader = downloaders[source]
        tasks.append(downloader.download(date))

    # Execute downloads concurrently
    console.print(f"[blue]Starting download of {len(sources)} data sources...[/blue]")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    downloaded_files = {}
    for source, result in zip(sources, results):
        if isinstance(result, Exception):
            console.print(f"[red]✗[/red] Failed to download {source}: {result}")
        else:
            downloaded_files[source] = result

    console.print(
        f"[green]Downloaded {len(downloaded_files)}/{len(sources)} sources successfully[/green]"
    )
    return downloaded_files
