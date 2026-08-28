"""
Pydantic data models for Linnaeus.

This module defines all the data structures used throughout the application,
including input data schemas, classification responses, and API data models.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator


class ClassificationTags(str, Enum):
    """Enumeration of all possible classification tags for AS organizations."""

    ACCESS = "Access"
    TRANSIT = "Transit"
    MOBILE = "Mobile"
    SATELLITE = "Satellite"
    CONTENT_PROVIDER = "Content Provider"
    EDUCATIONAL_RESEARCH = "Educational Research"
    GOVERNMENT = "Government"
    IXP = "Internet Exchange Point"
    DNS = "DNS"
    ENERGY_UTILITY = "Energy & Utility"
    ENTERPRISE = "Enterprise"
    FINANCE = "Finance"
    LAW_ENFORCEMENT = "Law Enforcement"
    HEALTH = "Health"
    COOPERATIVES = "Cooperatives"
    TV_RADIO_CULTURAL = "TV/Radio and Cultural Amenities"
    TRANSPORTATION = "Transportation"
    VPNS = "Virtual Private Networks"
    PERSONAL = "Personal"
    COMMUNITY = "Community"


class ASClassification(BaseModel):
    """Classification result for a single AS organization."""

    asn: int = Field(..., description="Autonomous System Number")
    organization_name: str = Field(..., description="Organization name")
    tags: List[ClassificationTags] = Field(
        ..., description="Assigned classification tags"
    )
    confidence_scores: Optional[Dict[str, float]] = Field(
        None, description="Confidence scores per tag"
    )
    model_used: str = Field(..., description="Model used for classification")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Classification timestamp"
    )

    @field_validator("asn")
    @classmethod
    def validate_asn(cls, v):
        """Validate that ASN is a positive integer."""
        if v <= 0:
            raise ValueError("ASN must be a positive integer")
        return v


class BatchClassificationResponse(BaseModel):
    """Response object for batch classification requests."""

    classifications: List[ASClassification] = Field(
        ..., description="Individual classification results"
    )
    batch_id: Optional[str] = Field(None, description="Batch processing ID")
    total_processed: int = Field(
        ..., description="Total number of organizations processed"
    )
    successful: int = Field(..., description="Number of successful classifications")
    failed: int = Field(..., description="Number of failed classifications")
    processing_time_seconds: float = Field(..., description="Total processing time")


class ASRankData(BaseModel):
    """Data structure for ASRank API responses."""

    asn: int = Field(..., description="Autonomous System Number")
    asn_name: Optional[str] = Field(None, description="AS name")
    rank: Optional[int] = Field(None, description="AS rank")
    organization_id: Optional[str] = Field(
        None, validation_alias=AliasChoices("orgId", "organization_id")
    )
    organization_name: Optional[str] = Field(
        None, validation_alias=AliasChoices("orgName", "organization_name")
    )
    clique_member: Optional[bool] = Field(
        None, validation_alias=AliasChoices("cliqueMember", "clique_member")
    )
    seen: Optional[bool] = Field(
        None, description="Whether AS is seen in routing tables"
    )
    longitude: Optional[float] = Field(None, description="Geographic longitude")
    latitude: Optional[float] = Field(None, description="Geographic latitude")
    country_iso: Optional[str] = Field(None, description="Country ISO code")
    country_name: Optional[str] = Field(None, description="Country name")

    # Cone statistics
    cone_asns: Optional[int] = Field(
        None, description="Number of ASNs in customer cone"
    )
    cone_prefixes: Optional[int] = Field(
        None, description="Number of prefixes in customer cone"
    )
    cone_addresses: Optional[int] = Field(
        None, description="Number of addresses in customer cone"
    )

    # Degree statistics
    degree_provider: Optional[int] = Field(None, description="Number of providers")
    degree_peer: Optional[int] = Field(None, description="Number of peers")
    degree_customer: Optional[int] = Field(None, description="Number of customers")
    degree_total: Optional[int] = Field(None, description="Total degree")
    degree_transit: Optional[int] = Field(None, description="Transit degree")
    degree_sibling: Optional[int] = Field(None, description="Sibling degree")

    # Announcing statistics
    announcing_prefixes: Optional[int] = Field(
        None, description="Number of announced prefixes"
    )
    announcing_addresses: Optional[int] = Field(
        None, description="Number of announced addresses"
    )


class PeeringDBData(BaseModel):
    """Data structure for PeeringDB network information."""

    asn: int = Field(..., description="Autonomous System Number")
    name: Optional[str] = Field(None, description="Network name")
    organization: Optional[str] = Field(None, description="Organization name")
    website: Optional[str] = Field(None, description="Organization website")
    looking_glass: Optional[str] = Field(None, description="Looking glass URL")
    route_server: Optional[str] = Field(None, description="Route server information")
    irr_as_set: Optional[str] = Field(None, description="IRR AS-SET")
    info_type: Optional[str] = Field(None, description="Network type information")
    info_scope: Optional[str] = Field(None, description="Network scope")
    info_unicast: Optional[bool] = Field(None, description="Unicast routing")
    info_multicast: Optional[bool] = Field(None, description="Multicast routing")
    info_ipv6: Optional[bool] = Field(None, description="IPv6 support")
    policy_url: Optional[str] = Field(None, description="Routing policy URL")
    policy_general: Optional[str] = Field(None, description="General routing policy")
    policy_locations: Optional[str] = Field(
        None, description="Location-specific policies"
    )
    policy_ratio: Optional[str] = Field(None, description="Traffic ratio policy")
    policy_contracts: Optional[str] = Field(None, description="Contract requirements")


class ASPOPData(BaseModel):
    """Data structure for APNIC AS Population data."""

    asn: int = Field(..., description="Autonomous System Number")
    name: Optional[str] = Field(None, description="AS name")
    country: Optional[str] = Field(None, description="Country code")
    rir: Optional[str] = Field(None, description="Regional Internet Registry")
    customer_cone_asns: Optional[int] = Field(
        None, description="Customer cone ASN count"
    )
    customer_cone_prefixes: Optional[int] = Field(
        None, description="Customer cone prefix count"
    )
    customer_cone_addresses: Optional[int] = Field(
        None, description="Customer cone address count"
    )
    customer_cone_unique: Optional[int] = Field(
        None, description="Unique customer cone count"
    )


class IPinfoData(BaseModel):
    """Data structure for IPinfo AS data."""

    asn: int = Field(..., description="Autonomous System Number")

    # Basic organization information
    org: Optional[str] = Field(None, description="Organization name with ASN prefix")
    name: Optional[str] = Field(None, description="Organization name")
    domain: Optional[str] = Field(None, description="Organization domain")

    # Geographic information
    country: Optional[str] = Field(None, description="Country code")
    country_name: Optional[str] = Field(None, description="Full country name")
    region: Optional[str] = Field(None, description="Region/state")
    city: Optional[str] = Field(None, description="City")
    postal: Optional[str] = Field(None, description="Postal code")
    timezone: Optional[str] = Field(None, description="Timezone")

    # Network information
    route: Optional[str] = Field(None, description="IP route/network range")
    type: Optional[str] = Field(
        None, description="Network type (business, hosting, etc.)"
    )
    allocation: Optional[str] = Field(None, description="IP allocation type")
    registry: Optional[str] = Field(None, description="Regional Internet Registry")

    # Contact information
    abuse: Optional[str] = Field(None, description="Abuse contact email")
    tech: Optional[str] = Field(None, description="Technical contact email")
    admin: Optional[str] = Field(None, description="Administrative contact email")

    # Metadata
    downloaded_at: Optional[str] = Field(None, description="Download timestamp")
    api_source: Optional[str] = Field(None, description="API source identifier")

    @field_validator("asn")
    @classmethod
    def validate_asn(cls, v):
        if v <= 0:
            raise ValueError("ASN must be positive")
        return v


class OrganizationData(BaseModel):
    """Combined organization data from multiple sources."""

    asn: int = Field(..., description="Autonomous System Number")

    # Primary identifiers
    name: Optional[str] = Field(None, description="Primary organization name")
    country: Optional[str] = Field(None, description="Country code or name")
    website: Optional[str] = Field(None, description="Organization website")

    # Source data
    asrank: Optional[ASRankData] = Field(None, description="ASRank data")
    peeringdb: Optional[PeeringDBData] = Field(None, description="PeeringDB data")
    aspop: Optional[ASPOPData] = Field(None, description="ASPOP data")
    ipinfo: Optional[IPinfoData] = Field(None, description="IPinfo data")

    # Derived fields
    organization_type: Optional[str] = Field(
        None, description="Inferred organization type"
    )
    network_size: Optional[str] = Field(
        None, description="Network size category (small/medium/large)"
    )
    geographic_scope: Optional[str] = Field(
        None, description="Geographic scope (local/regional/global)"
    )

    @model_validator(mode="after")
    def extract_derived_fields(self):
        """Extract derived fields from available sources."""
        # Extract primary name if not provided
        if not self.name:
            if self.asrank and self.asrank.organization_name:
                self.name = self.asrank.organization_name
            elif self.peeringdb and self.peeringdb.organization:
                self.name = self.peeringdb.organization
            elif self.ipinfo:
                # Enhanced IPinfo name extraction with fallback
                if self.ipinfo.name:
                    self.name = self.ipinfo.name
                elif self.ipinfo.org:
                    # Clean up org field (remove AS prefix if present)
                    org_name = self.ipinfo.org
                    if org_name.startswith(f"AS{self.asn} "):
                        org_name = org_name[len(f"AS{self.asn} ") :]
                    elif org_name.startswith("AS") and " " in org_name:
                        # Handle cases like "AS1234 Company Name"
                        parts = org_name.split(" ", 1)
                        if len(parts) > 1 and parts[0].startswith("AS"):
                            org_name = parts[1]
                    self.name = org_name
            elif self.aspop and self.aspop.name:
                self.name = self.aspop.name

        # Extract country if not provided
        if not self.country:
            if self.asrank and self.asrank.country_iso:
                self.country = self.asrank.country_iso
            elif self.ipinfo and self.ipinfo.country:
                self.country = self.ipinfo.country
            elif self.aspop and self.aspop.country:
                self.country = self.aspop.country

        # Extract website if not provided
        if not self.website:
            if self.peeringdb and self.peeringdb.website:
                self.website = self.peeringdb.website
            elif self.ipinfo and self.ipinfo.domain:
                # IPinfo provides domain, construct website URL with validation
                domain = self.ipinfo.domain.strip()
                if domain and not domain.startswith(("http://", "https://")):
                    # Remove any protocol prefix if somehow present
                    if domain.startswith("//"):
                        domain = domain[2:]
                    self.website = f"https://{domain}"
                elif domain:
                    self.website = domain

        return self


class ClassificationRequest(BaseModel):
    """Request model for AS classification."""

    organizations: List[OrganizationData] = Field(
        ..., description="Organizations to classify"
    )
    model: Optional[str] = Field(None, description="Model to use for classification")
    batch_size: Optional[int] = Field(None, description="Batch size for processing")
    include_confidence: bool = Field(
        False, description="Include confidence scores in response"
    )

    @field_validator("organizations")
    @classmethod
    def validate_organizations(cls, v):
        """Validate that we have at least one organization to classify."""
        if not v:
            raise ValueError("At least one organization must be provided")
        return v


class DataDownloadRequest(BaseModel):
    """Request model for data download operations."""

    sources: List[str] = Field(
        default=["peeringdb", "asrank", "aspop"], description="Data sources to download"
    )
    date: Optional[str] = Field(
        None, description="Target date for data snapshots (YYYY-MM-DD)"
    )
    output_dir: Optional[str] = Field(
        None, description="Output directory for downloaded data"
    )
    force_refresh: bool = Field(
        False, description="Force refresh even if cached data exists"
    )

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v):
        """Validate that requested sources are supported."""
        supported_sources = {"peeringdb", "asrank", "aspop", "ipinfo"}
        for source in v:
            if source not in supported_sources:
                raise ValueError(f"Unsupported data source: {source}")
        return v

    @field_validator("date")
    @classmethod
    def validate_date_format(cls, v):
        """Validate date format if provided."""
        if v:
            try:
                datetime.strptime(v, "%Y-%m-%d")
            except ValueError:
                raise ValueError("Date must be in YYYY-MM-DD format")
        return v


class RIRInfo(BaseModel):
    """Information about a Regional Internet Registry."""

    name: str = Field(..., description="Full name of the RIR")
    region: str = Field(..., description="Geographic region covered")
    founded: int = Field(..., description="Year founded")
    website: str = Field(..., description="Official website URL")
    coverage: str = Field(..., description="Coverage description")
    countries_count: int = Field(
        ..., description="Number of countries/territories covered"
    )


class CountryInfo(BaseModel):
    """Country information with RIR mapping."""

    name: str = Field(..., description="Full country name")
    rir: str = Field(..., description="Regional Internet Registry code")
    continent: str = Field(..., description="Continent name")


class CountryRIRRegistry(BaseModel):
    """Complete country-RIR registry data structure."""

    metadata: Dict[str, Any] = Field(..., description="Registry metadata")
    registries: Dict[str, RIRInfo] = Field(..., description="RIR information")
    countries: Dict[str, CountryInfo] = Field(
        ..., description="ISO country code to info mappings"
    )

    @field_validator("countries")
    @classmethod
    def validate_iso_codes(cls, v):
        """Validate that all keys are valid ISO 3166-1 alpha-2 codes."""
        for code in v.keys():
            if not (isinstance(code, str) and len(code) == 2 and code.isupper()):
                raise ValueError(f"Invalid ISO country code: {code}")
        return v

    def get_rir_for_country(self, iso_code: str) -> Optional[str]:
        """Get RIR for a given ISO country code."""
        country_info = self.countries.get(iso_code.upper())
        return country_info.rir if country_info else None

    def get_countries_by_rir(self, rir: str) -> List[str]:
        """Get list of country codes for a given RIR."""
        return [code for code, info in self.countries.items() if info.rir == rir]

    def get_countries_by_continent(self, continent: str) -> List[str]:
        """Get list of country codes for a given continent."""
        return [
            code for code, info in self.countries.items() if info.continent == continent
        ]

    def get_country_info(self, iso_code: str) -> Optional[CountryInfo]:
        """Get complete country information."""
        return self.countries.get(iso_code.upper())

    def get_rir_info(self, rir: str) -> Optional[RIRInfo]:
        """Get complete RIR information."""
        return self.registries.get(rir)
