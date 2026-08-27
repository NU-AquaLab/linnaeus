"""
Unified schema system supporting both hierarchical and flat AS classification.

This module provides a comprehensive tag taxonomy that supports:
1. Flat classification (simple top-level categories)
2. Hierarchical classification (detailed sub-categories)
3. Backward compatibility with existing models
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class TopLevelTags(str, Enum):
    """Top-level classification categories (flat system)."""

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
    FINANCIAL = "Financial"
    LAW_ENFORCEMENT = "Law Enforcement"
    HEALTH = "Health"
    COOPERATIVES = "Cooperatives"
    TV_RADIO_CULTURAL = "TV/Radio and Cultural Amenities"
    TRANSPORTATION = "Transportation"
    VPNS = "Virtual Private Networks"
    PERSONAL = "Personal"
    COMMUNITY = "Community"


class HierarchicalTags(str, Enum):
    """Detailed hierarchical classification system."""

    # Access
    ACCESS_LARGE_ISP = "Access Large ISP"
    ACCESS_SMALL_ISP = "Access Small ISP"

    # Transit
    TRANSIT_GLOBAL = "Transit Global"
    TRANSIT_REGIONAL = "Transit Regional"
    TRANSIT_DOMESTIC = "Transit Domestic"

    # Mobile (no sub-categories)
    MOBILE = "Mobile"

    # Satellite (no sub-categories)
    SATELLITE = "Satellite"

    # Content Provider
    CONTENT_PROVIDER_CLOUD = "ContentProvider Cloud"
    CONTENT_PROVIDER_HOSTING = "ContentProvider Hosting"
    CONTENT_PROVIDER_CDN = "ContentProvider CDN"

    # Educational Research
    EDUCATIONAL_RESEARCH_UNIVERSITY = "Educational Research University"
    EDUCATIONAL_RESEARCH_ACADEMIC_BACKBONE = "Educational Research Academic Backbone"
    EDUCATIONAL_RESEARCH_SCHOOLS = "Educational Research Schools"
    EDUCATIONAL_RESEARCH_INSTITUTES = "Educational Research Research Institutes"

    # Government
    GOVERNMENT_FEDERAL_NATIONAL = "Government Federal National"
    GOVERNMENT_STATE_PROVINCE = "Government State Province"
    GOVERNMENT_CITY_COUNTY_MUNICIPALITY = "Government City County Municipality"
    GOVERNMENT_LEGISLATIVE = "Government Legislative"
    GOVERNMENT_JUDICIARY = "Government Judiciary"
    GOVERNMENT_REGULATORS = "Government Regulators"
    GOVERNMENT_AGENCIES = "Government Agencies"
    GOVERNMENT_AGENCIES_SPACE = "Government Agencies Space"
    GOVERNMENT_AGENCIES_CENTRAL_BANKS = "Government Agencies CentralBanks"
    GOVERNMENT_POLITICAL_PARTIES = "Government Political Parties"

    # Internet Exchange Point (no sub-categories)
    IXP = "Internet Exchange Point"

    # DNS
    DNS_ROOTS = "DNS Roots"
    DNS_CCTLD = "DNS ccTLD"
    DNS_ANS = "DNS ANS"

    # Energy & Utility (no sub-categories)
    ENERGY_UTILITY = "Energy & Utility"

    # Enterprise
    ENTERPRISE_ECOMMERCE = "Enterprise E-commerce"
    ENTERPRISE_ENTERTAINMENT = "Enterprise Entertainment"
    ENTERPRISE_INDUSTRIAL_MANUFACTURING = "Enterprise Industrial Manufacturing"
    ENTERPRISE_TECHNOLOGY = "Enterprise Technology"

    # Financial
    FINANCIAL_CENTRAL_BANKS = "Financial CentralBanks"
    FINANCIAL_BANK = "Financial Bank"
    FINANCIAL_CREDIT_UNION = "Financial CreditUnion"
    FINANCIAL_ENTITIES = "Financial Entities"
    FINANCIAL_STOCK_EXCHANGES = "Financial StockExchanges"

    # Law Enforcement (no sub-categories)
    LAW_ENFORCEMENT = "Law Enforcement"

    # Health
    HEALTH_INSURANCES = "Health Insurances"
    HEALTH_HOSPITALS = "Health Hospitals"

    # Cooperatives (no sub-categories)
    COOPERATIVES = "Cooperatives"

    # TV/Radio and Cultural Amenities (no sub-categories)
    TV_RADIO_CULTURAL = "TV/Radio and Cultural Amenities"

    # Transportation
    TRANSPORTATION_TRAINS = "Transportation Trains"
    TRANSPORTATION_SHIPS = "Transportation Ships"
    TRANSPORTATION_BUSES = "Transportation Buses"
    TRANSPORTATION_TRANSIT_AUTHORITY = "Transportation Transit Authority"
    TRANSPORTATION_AIRPORTS = "Transportation Airports"

    # VPNs (no sub-categories)
    VPNS = "Virtual Private Networks"

    # Personal (no sub-categories)
    PERSONAL = "Personal"

    # Community (no sub-categories)
    COMMUNITY = "Community"


class TagHierarchy:
    """Mapping between hierarchical and top-level tags."""

    HIERARCHY_MAP = {
        # Access
        HierarchicalTags.ACCESS_LARGE_ISP: TopLevelTags.ACCESS,
        HierarchicalTags.ACCESS_SMALL_ISP: TopLevelTags.ACCESS,
        # Transit
        HierarchicalTags.TRANSIT_GLOBAL: TopLevelTags.TRANSIT,
        HierarchicalTags.TRANSIT_REGIONAL: TopLevelTags.TRANSIT,
        HierarchicalTags.TRANSIT_DOMESTIC: TopLevelTags.TRANSIT,
        # Content Provider
        HierarchicalTags.CONTENT_PROVIDER_CLOUD: TopLevelTags.CONTENT_PROVIDER,
        HierarchicalTags.CONTENT_PROVIDER_HOSTING: TopLevelTags.CONTENT_PROVIDER,
        HierarchicalTags.CONTENT_PROVIDER_CDN: TopLevelTags.CONTENT_PROVIDER,
        # Educational Research
        HierarchicalTags.EDUCATIONAL_RESEARCH_UNIVERSITY: TopLevelTags.EDUCATIONAL_RESEARCH,
        HierarchicalTags.EDUCATIONAL_RESEARCH_ACADEMIC_BACKBONE: TopLevelTags.EDUCATIONAL_RESEARCH,
        HierarchicalTags.EDUCATIONAL_RESEARCH_SCHOOLS: TopLevelTags.EDUCATIONAL_RESEARCH,
        HierarchicalTags.EDUCATIONAL_RESEARCH_INSTITUTES: TopLevelTags.EDUCATIONAL_RESEARCH,
        # Government
        HierarchicalTags.GOVERNMENT_FEDERAL_NATIONAL: TopLevelTags.GOVERNMENT,
        HierarchicalTags.GOVERNMENT_STATE_PROVINCE: TopLevelTags.GOVERNMENT,
        HierarchicalTags.GOVERNMENT_CITY_COUNTY_MUNICIPALITY: TopLevelTags.GOVERNMENT,
        HierarchicalTags.GOVERNMENT_LEGISLATIVE: TopLevelTags.GOVERNMENT,
        HierarchicalTags.GOVERNMENT_JUDICIARY: TopLevelTags.GOVERNMENT,
        HierarchicalTags.GOVERNMENT_REGULATORS: TopLevelTags.GOVERNMENT,
        HierarchicalTags.GOVERNMENT_AGENCIES: TopLevelTags.GOVERNMENT,
        HierarchicalTags.GOVERNMENT_AGENCIES_SPACE: TopLevelTags.GOVERNMENT,
        HierarchicalTags.GOVERNMENT_AGENCIES_CENTRAL_BANKS: TopLevelTags.GOVERNMENT,
        HierarchicalTags.GOVERNMENT_POLITICAL_PARTIES: TopLevelTags.GOVERNMENT,
        # DNS
        HierarchicalTags.DNS_ROOTS: TopLevelTags.DNS,
        HierarchicalTags.DNS_CCTLD: TopLevelTags.DNS,
        HierarchicalTags.DNS_ANS: TopLevelTags.DNS,
        # Enterprise
        HierarchicalTags.ENTERPRISE_ECOMMERCE: TopLevelTags.ENTERPRISE,
        HierarchicalTags.ENTERPRISE_ENTERTAINMENT: TopLevelTags.ENTERPRISE,
        HierarchicalTags.ENTERPRISE_INDUSTRIAL_MANUFACTURING: TopLevelTags.ENTERPRISE,
        HierarchicalTags.ENTERPRISE_TECHNOLOGY: TopLevelTags.ENTERPRISE,
        # Financial
        HierarchicalTags.FINANCIAL_CENTRAL_BANKS: TopLevelTags.FINANCIAL,
        HierarchicalTags.FINANCIAL_BANK: TopLevelTags.FINANCIAL,
        HierarchicalTags.FINANCIAL_CREDIT_UNION: TopLevelTags.FINANCIAL,
        HierarchicalTags.FINANCIAL_ENTITIES: TopLevelTags.FINANCIAL,
        HierarchicalTags.FINANCIAL_STOCK_EXCHANGES: TopLevelTags.FINANCIAL,
        # Health
        HierarchicalTags.HEALTH_INSURANCES: TopLevelTags.HEALTH,
        HierarchicalTags.HEALTH_HOSPITALS: TopLevelTags.HEALTH,
        # Transportation
        HierarchicalTags.TRANSPORTATION_TRAINS: TopLevelTags.TRANSPORTATION,
        HierarchicalTags.TRANSPORTATION_SHIPS: TopLevelTags.TRANSPORTATION,
        HierarchicalTags.TRANSPORTATION_BUSES: TopLevelTags.TRANSPORTATION,
        HierarchicalTags.TRANSPORTATION_TRANSIT_AUTHORITY: TopLevelTags.TRANSPORTATION,
        HierarchicalTags.TRANSPORTATION_AIRPORTS: TopLevelTags.TRANSPORTATION,
        # Single-level tags (map to themselves)
        HierarchicalTags.MOBILE: TopLevelTags.MOBILE,
        HierarchicalTags.SATELLITE: TopLevelTags.SATELLITE,
        HierarchicalTags.IXP: TopLevelTags.IXP,
        HierarchicalTags.ENERGY_UTILITY: TopLevelTags.ENERGY_UTILITY,
        HierarchicalTags.LAW_ENFORCEMENT: TopLevelTags.LAW_ENFORCEMENT,
        HierarchicalTags.COOPERATIVES: TopLevelTags.COOPERATIVES,
        HierarchicalTags.TV_RADIO_CULTURAL: TopLevelTags.TV_RADIO_CULTURAL,
        HierarchicalTags.VPNS: TopLevelTags.VPNS,
        HierarchicalTags.PERSONAL: TopLevelTags.PERSONAL,
        HierarchicalTags.COMMUNITY: TopLevelTags.COMMUNITY,
    }

    @classmethod
    def get_top_level(cls, hierarchical_tag: HierarchicalTags) -> TopLevelTags:
        """Get the top-level tag for a hierarchical tag."""
        return cls.HIERARCHY_MAP.get(hierarchical_tag)

    @classmethod
    def get_subtags(cls, top_level_tag: TopLevelTags) -> List[HierarchicalTags]:
        """Get all hierarchical tags that belong to a top-level category."""
        return [
            hierarchical
            for hierarchical, top_level in cls.HIERARCHY_MAP.items()
            if top_level == top_level_tag
        ]

    # Aliases for backward compatibility
    @classmethod
    def get_top_level_for_hierarchical(
        cls, hierarchical_tag: HierarchicalTags
    ) -> TopLevelTags:
        """Alias for get_top_level()."""
        return cls.get_top_level(hierarchical_tag)

    @classmethod
    def get_hierarchical_for_top_level(
        cls, top_level_tag: TopLevelTags
    ) -> List[HierarchicalTags]:
        """Alias for get_subtags()."""
        return cls.get_subtags(top_level_tag)

    @classmethod
    def hierarchical_to_flat(
        cls, hierarchical_tags: List[HierarchicalTags]
    ) -> List[TopLevelTags]:
        """Convert hierarchical tags to top-level tags."""
        flat_tags = set()
        for tag in hierarchical_tags:
            top_level = cls.get_top_level(tag)
            if top_level:
                flat_tags.add(top_level)
        return list(flat_tags)

    @classmethod
    def flat_to_hierarchical(
        cls, flat_tags: List[TopLevelTags]
    ) -> List[HierarchicalTags]:
        """Convert flat tags to their possible hierarchical representations."""
        hierarchical_tags = []
        for flat_tag in flat_tags:
            subtags = cls.get_subtags(flat_tag)
            hierarchical_tags.extend(subtags)
        return hierarchical_tags


class UnifiedASClassification(BaseModel):
    """Unified classification result supporting both hierarchical and flat tags."""

    asn: int = Field(..., description="Autonomous System Number")
    organization_name: str = Field(..., description="Organization name")

    # Support both tag systems
    top_level_tags: List[TopLevelTags] = Field(
        default_factory=list, description="Top-level classification tags"
    )
    hierarchical_tags: List[HierarchicalTags] = Field(
        default_factory=list, description="Detailed hierarchical tags"
    )

    # Confidence scores for both systems
    top_level_confidence: Optional[Dict[str, float]] = Field(
        None, description="Confidence scores for top-level tags"
    )
    hierarchical_confidence: Optional[Dict[str, float]] = Field(
        None, description="Confidence scores for hierarchical tags"
    )

    # Model information
    model_used: str = Field(..., description="Model used for classification")
    classification_approach: str = Field(
        ..., description="Classification approach: 'flat', 'hierarchical', or 'hybrid'"
    )
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

    @model_validator(mode="after")
    def sync_tag_systems(self):
        """Ensure consistency between hierarchical and top-level tags."""
        # If only hierarchical tags provided, derive top-level tags
        if self.hierarchical_tags and not self.top_level_tags:
            self.top_level_tags = TagHierarchy.hierarchical_to_flat(
                self.hierarchical_tags
            )

        # If only top-level tags provided, we can't derive specific hierarchical tags
        # but we can validate the classification approach
        if self.top_level_tags and not self.hierarchical_tags:
            if self.classification_approach == "hierarchical":
                # For hierarchical approach, we need hierarchical tags
                raise ValueError(
                    "Hierarchical classification approach requires hierarchical_tags"
                )

        return self

    def get_all_tags(self) -> List[str]:
        """Get all tags as string values for compatibility."""
        all_tags = []
        all_tags.extend([tag.value for tag in self.top_level_tags])
        all_tags.extend([tag.value for tag in self.hierarchical_tags])
        return list(set(all_tags))  # Remove duplicates


class BatchUnifiedClassificationResponse(BaseModel):
    """Response object for batch unified classification requests."""

    classifications: List[UnifiedASClassification] = Field(
        ..., description="Individual classification results"
    )
    batch_id: Optional[str] = Field(None, description="Batch processing ID")
    total_processed: int = Field(
        ..., description="Total number of organizations processed"
    )
    successful: int = Field(..., description="Number of successful classifications")
    failed: int = Field(..., description="Number of failed classifications")
    processing_time_seconds: float = Field(..., description="Total processing time")
    approach_used: str = Field(
        ..., description="Overall approach: 'flat', 'hierarchical', or 'hybrid'"
    )


class ClassificationRequest(BaseModel):
    """Unified request model for AS classification supporting multiple approaches."""

    organizations: List = Field(
        ..., description="Organizations to classify"
    )  # Using List for flexibility with different organization schemas
    approach: str = Field(
        default="hybrid",
        description="Classification approach: 'flat', 'hierarchical', or 'hybrid'",
    )
    model: Optional[str] = Field(None, description="Model to use for classification")
    batch_size: Optional[int] = Field(None, description="Batch size for processing")
    include_confidence: bool = Field(
        False, description="Include confidence scores in response"
    )

    @field_validator("approach")
    @classmethod
    def validate_approach(cls, v):
        """Validate classification approach."""
        if v not in ["flat", "hierarchical", "hybrid"]:
            raise ValueError("Approach must be 'flat', 'hierarchical', or 'hybrid'")
        return v

    @field_validator("organizations")
    @classmethod
    def validate_organizations(cls, v):
        """Validate that we have at least one organization to classify."""
        if not v:
            raise ValueError("At least one organization must be provided")
        return v
