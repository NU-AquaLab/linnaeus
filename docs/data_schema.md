# Data Schema Documentation

## Overview

This document describes the hierarchical data schema used for Autonomous System (AS) classification in the Linneaus project. The schema has been redesigned to provide clearer separation between top-level categories and their subcategories.

## Schema Evolution

### Previous Schema (Legacy)
The original schema used 46 flat columns representing all possible hierarchical tags:
```csv
ASN,ASName,Access_LargeISP,Access_SmallISP,Transit_Global,Transit_Regional,...
7922,Comcast,1,0,0,0,...
```

**Problems with legacy schema:**
- Mixed granularity levels in the same row
- Unclear hierarchical relationships
- Difficult to extend with new subcategories
- Sparse data representation (mostly zeros)

### New Schema (Current)
The new schema uses a normalized approach with separate records for each classification:

```csv
ASN,ASName,TopLevelCategory,SubCategory
7922,Comcast,Access,LargeISP
7922,Comcast,Transit,Domestic
7922,Comcast,Mobile,null
7922,Comcast,ContentProvider,CDN
```

## Hierarchical Structure

### Top-Level Categories

| Category | Description | Has Subcategories |
|----------|-------------|-------------------|
| Access | Internet Service Providers | ✅ |
| Transit | Transit service providers | ✅ |
| Mobile | Mobile network operators | ❌ |
| Satellite | Satellite internet providers | ❌ |
| ContentProvider | Cloud, hosting, CDN providers | ✅ |
| EducationalResearch | Universities, research institutions | ✅ |
| Government | Government organizations | ✅ |
| IXP | Internet Exchange Points | ❌ |
| DNS | DNS infrastructure | ✅ |
| EnergyAndUtility | Energy and utility companies | ❌ |
| Enterprise | Commercial organizations | ✅ |
| Finance | Financial institutions | ✅ |
| LawEnforcement | Police, military, intelligence | ❌ |
| Health | Healthcare organizations | ✅ |
| Cooperatives | Cooperative organizations | ❌ |
| TvRadioCulturalAmenities | Media and cultural organizations | ❌ |
| Transportation | Transport infrastructure | ✅ |
| VPNs | VPN service providers | ❌ |
| Personal | Individual networks | ❌ |
| Community | Community networks | ❌ |

### Subcategory Definitions

#### Access
- **LargeISP**: Major national/international ISPs (Comcast, Charter, AT&T)
- **SmallISP**: Regional or local ISPs

#### Transit
- **Global**: Tier-1 global transit providers
- **Regional**: Regional transit providers
- **Domestic**: Domestic transit providers

#### ContentProvider
- **Cloud**: Cloud computing platforms (AWS, Azure, Google Cloud)
- **Hosting**: Web hosting companies
- **CDN**: Content delivery networks

#### EducationalResearch
- **University**: Universities and colleges
- **AcademicBackbone**: Academic network backbones (GEANT, Internet2)
- **Schools**: K-12 schools and school districts
- **ResearchInstitutes**: Research institutions and labs

#### Government
- **FederalNational**: Federal/national government agencies
- **StateProvince**: State or provincial government
- **CityCountyMunicipality**: Local government (cities, counties)
- **Legislative**: Legislative bodies (parliaments, congresses)
- **Judiciary**: Courts and judicial systems
- **Regulators**: Regulatory agencies and commissions
- **Agencies**: Specialized government agencies
- **AgenciesSpace**: Space agencies (NASA, ESA)
- **AgenciesCentralBanks**: Central banks
- **PoliticalParties**: Political parties and organizations

#### DNS
- **Roots**: Root DNS servers
- **ccTLD**: Country code top-level domain registries
- **ANS**: Authoritative name servers

#### Enterprise
- **Ecommerce**: E-commerce platforms and online retailers
- **Entertainment**: Entertainment and media companies
- **IndustrialManufacturing**: Industrial and manufacturing companies
- **Technology**: Technology companies with own networks

#### Finance
- **Bank**: Commercial and investment banks
- **CentralBanks**: Central banks (overlaps with Government)
- **CreditUnion**: Credit unions and community banks
- **Entities**: Other financial institutions
- **StockExchanges**: Stock exchanges and trading platforms

#### Health
- **Insurances**: Health insurance companies
- **Hospitals**: Hospitals and healthcare providers

#### Transportation
- **Trains**: Railway companies
- **Ships**: Shipping and maritime companies
- **Buses**: Bus companies and transit systems
- **TransitAuthority**: Public transit authorities
- **Airports**: Airport authorities

## Data Format Specifications

### CSV Format
```csv
ASN,ASName,TopLevelCategory,SubCategory
integer,string,enum,enum|null
```

### Example Data
```csv
ASN,ASName,TopLevelCategory,SubCategory
7922,Comcast,Access,LargeISP
7922,Comcast,Transit,Domestic
7922,Comcast,Mobile,null
7922,Comcast,ContentProvider,CDN
20115,Charter Communications,Access,LargeISP
16509,AWS,ContentProvider,Cloud
16509,AWS,Enterprise,Technology
15169,Google LLC,ContentProvider,Cloud
15169,Google LLC,Enterprise,Technology
```

### Parquet Schema
The Parquet files maintain the same structure with proper typing:
- `ASN`: int64
- `ASName`: string
- `TopLevelCategory`: categorical (enum)
- `SubCategory`: categorical (enum, nullable)

## Data Validation Rules

### Required Fields
- `ASN`: Must be positive integer
- `ASName`: Must be non-empty string
- `TopLevelCategory`: Must be valid enum value

### Optional Fields
- `SubCategory`: Can be null for categories without subcategories

### Business Rules
1. **Single-level categories** must have `SubCategory = null`
2. **Multi-level categories** should have valid subcategory values
3. **ASN uniqueness**: Each ASN can appear multiple times with different category combinations
4. **Category validity**: TopLevelCategory and SubCategory combinations must be valid

### Invalid Examples
```csv
# Invalid: Mobile cannot have subcategory
7922,Comcast,Mobile,SomeSubcategory

# Invalid: Access must have subcategory
7922,Comcast,Access,null

# Invalid: Invalid subcategory for category
7922,Comcast,Government,Cloud
```

## Migration from Legacy Schema

### Conversion Process
1. **Parse legacy columns**: Extract category and subcategory from column names
2. **Create records**: Generate one record per active classification
3. **Validate**: Ensure all combinations are valid
4. **Normalize**: Convert to new format

### Mapping Examples
```
Legacy: Access_LargeISP=1 → New: TopLevelCategory=Access, SubCategory=LargeISP
Legacy: Mobile=1 → New: TopLevelCategory=Mobile, SubCategory=null
Legacy: Government_Legislative=1 → New: TopLevelCategory=Government, SubCategory=Legislative
```

## Usage in ML Pipeline

### Training Data Preparation
1. Load hierarchical data
2. Convert to appropriate format for model:
   - **Hierarchical approach**: Use both category and subcategory
   - **Flat approach**: Use only top-level category

### Inference Output
Model outputs should match the training format:
- Hierarchical: Predict both category and subcategory
- Flat: Predict only top-level category

### Evaluation Metrics
- Calculate metrics at both category and subcategory levels
- Support hierarchical evaluation (partial credit for correct category)

## File Organization

```
data/
├── labeled/
│   ├── consolidated/
│   │   ├── labeled_data.parquet          # Main dataset
│   │   └── labeled_data_legacy.parquet   # Backup of old format
│   ├── splits/
│   │   ├── train.parquet                 # Training set (70%)
│   │   ├── validation.parquet            # Validation set (15%)
│   │   └── test.parquet                  # Test set (15%)
│   └── metadata.json                     # Dataset statistics
└── predictions/                          # Model outputs
    └── {model_name}_{timestamp}/
```

## Benefits of New Schema

1. **Clarity**: Clear separation between category levels
2. **Flexibility**: Easy to add new subcategories without schema changes
3. **Consistency**: Uniform structure across all classifications
4. **Maintainability**: Easier to understand and modify
5. **Extensibility**: Simple to add new hierarchical levels
6. **Storage efficiency**: No sparse matrices with mostly zeros
7. **Query efficiency**: Better for database operations and filtering

## Future Enhancements

1. **Multi-level hierarchy**: Support for deeper hierarchical structures
2. **Weighted classifications**: Support for confidence scores
3. **Temporal data**: Track changes in classifications over time
4. **Provenance**: Track source of each classification
5. **Validation rules**: More sophisticated validation logic
