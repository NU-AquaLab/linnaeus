# AS Classification Categories

This document describes the comprehensive taxonomy used by Linneaus for classifying Internet Autonomous Systems (AS). The system supports both flat (top-level) and hierarchical (detailed) classification approaches.

**Machine-readable definitions**: the authoritative category names and descriptions live in [src/linneaus/resources/taxonomies/linneaus.json](src/linneaus/resources/taxonomies/linneaus.json) — that file drives the LLM prompt and output schema, so keep this document in sync with it. Supplementary reference taxonomies are provided in the same directory: `asdb.json` (Stanford ASdb) and `isic.json` (ISIC Rev.4), with matching benchmark labels in `data/released/202506/labels/{asdb,isic}.csv`. To author your own taxonomy, see [docs/custom_taxonomies.md](docs/custom_taxonomies.md).

## Top-Level Categories (Flat Classification)

The following 20 top-level categories provide broad classification suitable for most use cases. The **Column name** is the canonical machine name used in label files, prediction matrices, and the LLM schema:

| Category | Column name | Description | Example Organizations |
|----------|-------------|-------------|----------------------|
| **Access** | `Access` | Internet service providers serving end users | Local ISPs, Regional ISPs |
| **Transit** | `Transit` | Providers offering IP transit services | Tier 1 carriers, Global backbone providers |
| **Mobile** | `Mobile` | Mobile network operators | Cellular carriers, Mobile MVNOs |
| **Satellite** | `Satellite` | Satellite communication providers | Satellite internet, VSAT providers |
| **Content Provider** | `ContentProvider` | CDNs, hosting providers, cloud services | AWS, Google Cloud, Cloudflare |
| **Educational Research** | `EducationalResearch` | Universities, research institutions | Universities, Academic networks |
| **Government** | `Government` | Government agencies, public sector | Federal agencies, Municipal networks |
| **Internet Exchange Point** | `IXP` | Internet exchange points | IXPs, Peering exchanges |
| **DNS** | `DNS` | Domain name system operators | Root servers, DNS providers |
| **Energy & Utility** | `EnergyAndUtility` | Energy companies, utilities | Power companies, Utility networks |
| **Enterprise** | `Enterprise` | Commercial enterprises, businesses | Corporate networks, Private companies |
| **Finance** | `Finance` | Banks, financial institutions | Banks, Stock exchanges, Credit unions |
| **Law Enforcement** | `LawEnforcement` | Law enforcement agencies | Police networks, Security agencies |
| **Health** | `Health` | Healthcare organizations | Hospitals, Health insurance providers |
| **Cooperatives** | `Cooperatives` | Cooperative organizations | Telecom cooperatives, Community networks |
| **TV/Radio and Cultural Amenities** | `TvRadioCulturalAmenities` | Media, cultural organizations | Broadcasting, Media companies |
| **Transportation** | `Transportation` | Transport companies, airports | Airlines, Shipping companies, Airports |
| **VPNs** | `VPNs` | VPN providers | VPN services, Privacy networks |
| **Personal** | `Personal` | Individual/personal networks | Personal ASNs, Individual users |
| **Community** | `Community` | Community networks, non-profits | Community ISPs, Non-profit networks |

## Hierarchical Categories (Detailed Classification)

The hierarchical system provides more granular classification with specific subcategories:

### Access
- **Access Large ISP** - Major internet service providers with extensive infrastructure
- **Access Small ISP** - Regional or local internet service providers

### Transit
- **Transit Global** - Tier 1 global transit providers with worldwide reach
- **Transit Regional** - Regional transit providers serving specific geographic areas
- **Transit Domestic** - Domestic transit providers within specific countries

### Mobile
- **Mobile** - Mobile network operators (no subcategories)

### Satellite
- **Satellite** - Satellite communication providers (no subcategories)

### Content Provider
- **ContentProvider Cloud** - Cloud computing and infrastructure services
- **ContentProvider Hosting** - Web hosting and managed services
- **ContentProvider CDN** - Content delivery networks and edge computing

### Educational Research
- **Educational Research University** - Universities and higher education institutions
- **Educational Research Academic Backbone** - Academic backbone networks and research networks
- **Educational Research Schools** - Primary and secondary educational institutions
- **Educational Research Research Institutes** - Research institutions and laboratories

### Government
- **Government Executive** - Executive branch agencies and departments
- **Government Legislative** - Legislative bodies and related institutions
- **Government Judiciary** - Judicial branch and court systems
- **Government National** - National government institutions
- **Government State Province** - State and provincial government networks
- **Government Municipal** - Municipal and local government networks
- **Government Agency** - Government agencies and specialized departments

### Internet Exchange Point
- **Internet Exchange Point** - Internet exchange points (no subcategories)

### DNS
- **DNS Roots** - Root DNS servers and operators
- **DNS ccTLD** - Country code top-level domain operators
- **DNS ANS** - Authoritative name servers and DNS providers

### Energy & Utility
- **Energy & Utility** - Energy and utility companies (no subcategories)

### Enterprise
- **Enterprise E-commerce** - E-commerce platforms and online retail
- **Enterprise Entertainment** - Entertainment, gaming, and media companies
- **Enterprise Industrial Manufacturing** - Industrial and manufacturing companies
- **Enterprise Technology** - Technology companies and software providers

### Financial
- **Financial CentralBanks** - Central banks and monetary authorities
- **Financial Bank** - Commercial banks and banking institutions
- **Financial CreditUnion** - Credit unions and cooperative financial institutions
- **Financial Entities** - Other financial service providers
- **Financial StockExchanges** - Stock exchanges and trading platforms

### Law Enforcement
- **Law Enforcement** - Law enforcement agencies (no subcategories)

### Health
- **Health Insurances** - Health insurance providers and managed care organizations
- **Health Hospitals** - Hospitals and healthcare facilities

### Cooperatives
- **Cooperatives** - Cooperative organizations (no subcategories)

### TV/Radio and Cultural Amenities
- **TV/Radio and Cultural Amenities** - Broadcasting and cultural organizations (no subcategories)

### Transportation
- **Transportation** - Transportation companies and logistics (no subcategories)

### Virtual Private Networks
- **Virtual Private Networks** - VPN providers (no subcategories)

### Personal
- **Personal** - Individual/personal networks (no subcategories)

### Community
- **Community** - Community networks and non-profits (no subcategories)

## Using the Classification System

### Flat Classification
Use top-level categories when you need:
- Simple, broad categorization
- Compatibility with existing systems
- Quick overview of AS types
- Statistical analysis at high level

```python
from linneaus.models import HybridASNClassifier

# Use flat approach
classifier = HybridASNClassifier(approach="flat")
```

### Hierarchical Classification
Use hierarchical categories when you need:
- Detailed, granular classification
- Specific subcategory identification
- Research and analysis requiring precision
- Policy and regulatory compliance

```python
from linneaus.models import HybridASNClassifier

# Use hierarchical approach
classifier = HybridASNClassifier(approach="hierarchical")
```

### Hybrid Classification
Combine both approaches for maximum flexibility:

```python
from linneaus.models import HybridASNClassifier

# Use hybrid approach (default)
classifier = HybridASNClassifier(approach="hybrid")
```

## Tag Mapping

The system automatically maps between hierarchical and top-level tags:

- **Hierarchical → Top-level**: Each hierarchical tag maps to exactly one top-level category
- **Top-level → Hierarchical**: Each top-level category may map to multiple hierarchical subcategories

This mapping ensures consistency and allows seamless conversion between classification approaches.

## Data Sources

Classifications are based on multiple authoritative data sources:

- **[ASRank](https://asrank.caida.org/)**: AS ranking and topology data
- **[PeeringDB](https://www.peeringdb.com/)**: Network operator information
- **[APNIC AS Population](https://stats.labs.apnic.net/)**: AS relationship data
- **Manual curation**: Expert validation and refinement

## Validation and Quality

The classification system undergoes continuous validation:

- **Expert review**: Manual validation by network operators and researchers
- **Cross-validation**: Comparison across multiple data sources
- **Community feedback**: Input from the Internet measurement community
- **Regular updates**: Periodic review and taxonomy refinement

For questions about specific classifications or to suggest improvements, please open an issue in the project repository.
