"""
Hierarchical schema system for AS classification.

This module provides Pydantic models for the new hierarchical data format
with TopLevelCategory and SubCategory columns.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TopLevelCategory(str, Enum):
    """Top-level classification categories."""

    ACCESS = "Access"
    TRANSIT = "Transit"
    MOBILE = "Mobile"
    SATELLITE = "Satellite"
    CONTENT_PROVIDER = "ContentProvider"
    EDUCATIONAL_RESEARCH = "EducationalResearch"
    GOVERNMENT = "Government"
    IXP = "IXP"
    DNS = "DNS"
    ENERGY_AND_UTILITY = "EnergyAndUtility"
    ENTERPRISE = "Enterprise"
    FINANCE = "Finance"
    LAW_ENFORCEMENT = "LawEnforcement"
    HEALTH = "Health"
    COOPERATIVES = "Cooperatives"
    TV_RADIO_CULTURAL_AMENITIES = "TvRadioCulturalAmenities"
    TRANSPORTATION = "Transportation"
    VPNS = "VPNs"
    PERSONAL = "Personal"
    COMMUNITY = "Community"


class AccessSubCategory(str, Enum):
    """Subcategories for Access organizations."""

    LARGE_ISP = "LargeISP"
    SMALL_ISP = "SmallISP"


class TransitSubCategory(str, Enum):
    """Subcategories for Transit organizations."""

    GLOBAL = "Global"
    REGIONAL = "Regional"
    DOMESTIC = "Domestic"


class ContentProviderSubCategory(str, Enum):
    """Subcategories for Content Provider organizations."""

    CLOUD = "Cloud"
    HOSTING = "Hosting"
    CDN = "CDN"


class EducationalResearchSubCategory(str, Enum):
    """Subcategories for Educational Research organizations."""

    UNIVERSITY = "University"
    ACADEMIC_BACKBONE = "AcademicBackbone"
    SCHOOLS = "Schools"
    RESEARCH_INSTITUTES = "ResearchInstitutes"


class GovernmentSubCategory(str, Enum):
    """Subcategories for Government organizations."""

    EXECUTIVE = "Executive"
    LEGISLATIVE = "Legislative"
    JUDICIARY = "Judiciary"
    FEDERAL_NATIONAL = "FederalNational"
    STATE_PROVINCE = "StateProvince"
    CITY_COUNTY_MUNICIPALITY = "CityCountyMunicipality"
    REGULATORS = "Regulators"
    AGENCIES = "Agencies"
    AGENCIES_SPACE = "AgenciesSpace"
    AGENCIES_CENTRAL_BANKS = "AgenciesCentralBanks"
    POLITICAL_PARTIES = "PoliticalParties"


class DNSSubCategory(str, Enum):
    """Subcategories for DNS organizations."""

    ROOTS = "Roots"
    CC_TLD = "ccTLD"
    ANS = "ANS"


class EnterpriseSubCategory(str, Enum):
    """Subcategories for Enterprise organizations."""

    ECOMMERCE = "Ecommerce"
    ENTERTAINMENT = "Entertainment"
    INDUSTRIAL_MANUFACTURING = "IndustrialManufacturing"
    TECHNOLOGY = "Technology"


class FinanceSubCategory(str, Enum):
    """Subcategories for Financial organizations."""

    BANK = "Bank"
    CENTRAL_BANKS = "CentralBanks"
    CREDIT_UNION = "CreditUnion"
    ENTITIES = "Entities"
    STOCK_EXCHANGES = "StockExchanges"


class HealthSubCategory(str, Enum):
    """Subcategories for Health organizations."""

    INSURANCES = "Insurances"
    HOSPITALS = "Hospitals"


class TransportationSubCategory(str, Enum):
    """Subcategories for Transportation organizations."""

    TRAINS = "Trains"
    SHIPS = "Ships"
    BUSES = "Buses"
    TRANSIT_AUTHORITY = "TransitAuthority"
    AIRPORTS = "Airports"


class HierarchicalRecord(BaseModel):
    """Model for a single hierarchical classification record."""

    asn: int = Field(..., description="Autonomous System Number", gt=0)
    as_name: str = Field(..., description="AS organization name", min_length=1)
    top_level_category: TopLevelCategory = Field(..., description="Top-level category")
    sub_category: Optional[str] = Field(None, description="Subcategory (if applicable)")

    @field_validator("sub_category")
    @classmethod
    def validate_sub_category(cls, v, info):
        """Validate subcategory based on top-level category."""
        if "top_level_category" not in info.data:
            return v

        category = info.data["top_level_category"]

        # Categories that should not have subcategories
        single_level_categories = {
            TopLevelCategory.MOBILE,
            TopLevelCategory.SATELLITE,
            TopLevelCategory.IXP,
            TopLevelCategory.ENERGY_AND_UTILITY,
            TopLevelCategory.LAW_ENFORCEMENT,
            TopLevelCategory.COOPERATIVES,
            TopLevelCategory.TV_RADIO_CULTURAL_AMENITIES,
            TopLevelCategory.VPNS,
            TopLevelCategory.PERSONAL,
            TopLevelCategory.COMMUNITY,
        }

        if category in single_level_categories:
            if v is not None:
                raise ValueError(
                    f"Category {category.value} should not have subcategory"
                )
            return None

        # Categories that should have subcategories
        if v is None:
            raise ValueError(f"Category {category.value} requires a subcategory")

        # Validate specific subcategory values
        subcategory_validators = {
            TopLevelCategory.ACCESS: AccessSubCategory,
            TopLevelCategory.TRANSIT: TransitSubCategory,
            TopLevelCategory.CONTENT_PROVIDER: ContentProviderSubCategory,
            TopLevelCategory.EDUCATIONAL_RESEARCH: EducationalResearchSubCategory,
            TopLevelCategory.GOVERNMENT: GovernmentSubCategory,
            TopLevelCategory.DNS: DNSSubCategory,
            TopLevelCategory.ENTERPRISE: EnterpriseSubCategory,
            TopLevelCategory.FINANCE: FinanceSubCategory,
            TopLevelCategory.HEALTH: HealthSubCategory,
            TopLevelCategory.TRANSPORTATION: TransportationSubCategory,
        }

        if category in subcategory_validators:
            validator_enum = subcategory_validators[category]
            valid_values = [e.value for e in validator_enum]
            if v not in valid_values:
                raise ValueError(
                    f"Invalid subcategory '{v}' for category {category.value}. Valid: {valid_values}"
                )

        return v

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)


class HierarchicalDataset(BaseModel):
    """Model for a complete hierarchical dataset."""

    records: List[HierarchicalRecord] = Field(
        ..., description="List of hierarchical records"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Dataset metadata"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )

    @field_validator("records")
    @classmethod
    def validate_unique_records(cls, v):
        """Ensure no duplicate records."""
        seen = set()
        for record in v:
            key = (record.asn, record.top_level_category.value, record.sub_category)
            if key in seen:
                raise ValueError(
                    f"Duplicate record: ASN {record.asn}, Category {record.top_level_category.value}, SubCategory {record.sub_category}"
                )
            seen.add(key)
        return v

    def get_asn_categories(self, asn: int) -> List[HierarchicalRecord]:
        """Get all categories for a specific ASN."""
        return [record for record in self.records if record.asn == asn]

    def get_category_counts(self) -> Dict[str, int]:
        """Get counts of each top-level category."""
        counts = {}
        for record in self.records:
            category = record.top_level_category.value
            counts[category] = counts.get(category, 0) + 1
        return counts

    def get_subcategory_counts(self) -> Dict[str, int]:
        """Get counts of each subcategory."""
        counts = {}
        for record in self.records:
            if record.sub_category:
                counts[record.sub_category] = counts.get(record.sub_category, 0) + 1
        return counts

    def to_flat_format(self) -> Dict[int, List[str]]:
        """Convert to flat format (ASN -> list of top-level categories)."""
        flat = {}
        for record in self.records:
            if record.asn not in flat:
                flat[record.asn] = []
            category = record.top_level_category.value
            if category not in flat[record.asn]:
                flat[record.asn].append(category)
        return flat


class DataSplits(BaseModel):
    """Model for train/validation/test data splits."""

    train: HierarchicalDataset = Field(..., description="Training data")
    validation: HierarchicalDataset = Field(..., description="Validation data")
    test: HierarchicalDataset = Field(..., description="Test data")

    split_proportions: Dict[str, float] = Field(
        default={"train": 0.7, "validation": 0.15, "test": 0.15},
        description="Split proportions",
    )
    random_state: int = Field(default=42, description="Random seed used for splitting")

    @model_validator(mode="after")
    def validate_no_overlap(self):
        """Ensure no ASN appears in multiple splits."""
        train_asns = set(record.asn for record in self.train.records)
        val_asns = set(record.asn for record in self.validation.records)
        test_asns = set(record.asn for record in self.test.records)

        overlap_train_val = train_asns & val_asns
        overlap_train_test = train_asns & test_asns
        overlap_val_test = val_asns & test_asns

        if overlap_train_val:
            raise ValueError(
                f"ASNs overlap between train and validation: {list(overlap_train_val)[:5]}..."
            )
        if overlap_train_test:
            raise ValueError(
                f"ASNs overlap between train and test: {list(overlap_train_test)[:5]}..."
            )
        if overlap_val_test:
            raise ValueError(
                f"ASNs overlap between validation and test: {list(overlap_val_test)[:5]}..."
            )

        return self


class CategoryMapping(BaseModel):
    """Model for mapping between hierarchical and flat representations."""

    category: TopLevelCategory = Field(..., description="Top-level category")
    subcategories: List[str] = Field(
        default_factory=list, description="Valid subcategories"
    )
    description: str = Field(..., description="Category description")

    @classmethod
    def get_all_mappings(cls) -> List["CategoryMapping"]:
        """Get all category mappings."""
        mappings = [
            CategoryMapping(
                category=TopLevelCategory.ACCESS,
                subcategories=["LargeISP", "SmallISP"],
                description="Internet Service Providers",
            ),
            CategoryMapping(
                category=TopLevelCategory.TRANSIT,
                subcategories=["Global", "Regional", "Domestic"],
                description="Transit service providers",
            ),
            CategoryMapping(
                category=TopLevelCategory.MOBILE,
                subcategories=[],
                description="Mobile network operators",
            ),
            CategoryMapping(
                category=TopLevelCategory.SATELLITE,
                subcategories=[],
                description="Satellite internet providers",
            ),
            CategoryMapping(
                category=TopLevelCategory.CONTENT_PROVIDER,
                subcategories=["Cloud", "Hosting", "CDN"],
                description="Cloud, hosting, CDN providers",
            ),
            CategoryMapping(
                category=TopLevelCategory.EDUCATIONAL_RESEARCH,
                subcategories=[
                    "University",
                    "AcademicBackbone",
                    "Schools",
                    "ResearchInstitutes",
                ],
                description="Universities, research institutions",
            ),
            CategoryMapping(
                category=TopLevelCategory.GOVERNMENT,
                subcategories=[
                    "Executive",
                    "Legislative",
                    "Judiciary",
                    "FederalNational",
                    "StateProvince",
                    "CityCountyMunicipality",
                    "Regulators",
                    "Agencies",
                    "AgenciesSpace",
                    "AgenciesCentralBanks",
                    "PoliticalParties",
                ],
                description="Government organizations",
            ),
            CategoryMapping(
                category=TopLevelCategory.IXP,
                subcategories=[],
                description="Internet Exchange Points",
            ),
            CategoryMapping(
                category=TopLevelCategory.DNS,
                subcategories=["Roots", "ccTLD", "ANS"],
                description="DNS infrastructure",
            ),
            CategoryMapping(
                category=TopLevelCategory.ENERGY_AND_UTILITY,
                subcategories=[],
                description="Energy and utility companies",
            ),
            CategoryMapping(
                category=TopLevelCategory.ENTERPRISE,
                subcategories=[
                    "Ecommerce",
                    "Entertainment",
                    "IndustrialManufacturing",
                    "Technology",
                ],
                description="Commercial organizations",
            ),
            CategoryMapping(
                category=TopLevelCategory.FINANCE,
                subcategories=[
                    "Bank",
                    "CentralBanks",
                    "CreditUnion",
                    "Entities",
                    "StockExchanges",
                ],
                description="Financial institutions",
            ),
            CategoryMapping(
                category=TopLevelCategory.LAW_ENFORCEMENT,
                subcategories=[],
                description="Police, military, intelligence",
            ),
            CategoryMapping(
                category=TopLevelCategory.HEALTH,
                subcategories=["Insurances", "Hospitals"],
                description="Healthcare organizations",
            ),
            CategoryMapping(
                category=TopLevelCategory.COOPERATIVES,
                subcategories=[],
                description="Cooperative organizations",
            ),
            CategoryMapping(
                category=TopLevelCategory.TV_RADIO_CULTURAL_AMENITIES,
                subcategories=[],
                description="Media and cultural organizations",
            ),
            CategoryMapping(
                category=TopLevelCategory.TRANSPORTATION,
                subcategories=[
                    "Trains",
                    "Ships",
                    "Buses",
                    "TransitAuthority",
                    "Airports",
                ],
                description="Transport infrastructure",
            ),
            CategoryMapping(
                category=TopLevelCategory.VPNS,
                subcategories=[],
                description="VPN service providers",
            ),
            CategoryMapping(
                category=TopLevelCategory.PERSONAL,
                subcategories=[],
                description="Individual networks",
            ),
            CategoryMapping(
                category=TopLevelCategory.COMMUNITY,
                subcategories=[],
                description="Community networks",
            ),
        ]
        return mappings


# Utility functions for working with hierarchical data


def validate_hierarchical_record(
    asn: int, as_name: str, category: str, subcategory: Optional[str] = None
) -> HierarchicalRecord:
    """
    Create and validate a hierarchical record.

    Parameters
    ----------
    asn : int
        Autonomous System Number
    as_name : str
        Organization name
    category : str
        Top-level category
    subcategory : Optional[str]
        Subcategory (if applicable)

    Returns
    -------
    HierarchicalRecord
        Validated hierarchical record
    """
    return HierarchicalRecord(
        asn=asn,
        as_name=as_name,
        top_level_category=TopLevelCategory(category),
        sub_category=subcategory,
    )


def convert_legacy_to_hierarchical(
    legacy_data: Dict[str, Any],
) -> List[HierarchicalRecord]:
    """
    Convert legacy data format to hierarchical records.

    Parameters
    ----------
    legacy_data : Dict[str, Any]
        Legacy data in old format

    Returns
    -------
    List[HierarchicalRecord]
        List of hierarchical records
    """
    # This would be implemented based on the specific legacy format
    # For now, return empty list
    return []


if __name__ == "__main__":
    # Example usage
    records = [
        HierarchicalRecord(
            asn=7922,
            as_name="Comcast",
            top_level_category=TopLevelCategory.ACCESS,
            sub_category="LargeISP",
        ),
        HierarchicalRecord(
            asn=7922,
            as_name="Comcast",
            top_level_category=TopLevelCategory.TRANSIT,
            sub_category="Domestic",
        ),
        HierarchicalRecord(
            asn=7922,
            as_name="Comcast",
            top_level_category=TopLevelCategory.MOBILE,
            sub_category=None,
        ),
        HierarchicalRecord(
            asn=16509,
            as_name="AWS",
            top_level_category=TopLevelCategory.CONTENT_PROVIDER,
            sub_category="Cloud",
        ),
        HierarchicalRecord(
            asn=16509,
            as_name="AWS",
            top_level_category=TopLevelCategory.ENTERPRISE,
            sub_category="Technology",
        ),
    ]

    dataset = HierarchicalDataset(records=records)
    print(f"Dataset with {len(dataset.records)} records")
    print(f"Category counts: {dataset.get_category_counts()}")
    print(f"Subcategory counts: {dataset.get_subcategory_counts()}")
    print(f"ASN 7922 categories: {len(dataset.get_asn_categories(7922))}")
