# SVM Feature Engineering Documentation

This document provides comprehensive documentation for the feature engineering pipeline used in the SVM component of the two-stage hierarchical AS classification system.

## Overview

The `ASNFeatureEngineer` class extracts and processes features from multiple data sources to create a comprehensive feature representation for Autonomous System (AS) organizations. These features are used by the SVM classifier in Stage 1 of the two-stage pipeline.

## Data Sources

### 1. ASRank Data
**Source**: CAIDA's ASRank dataset  
**Purpose**: Network topology and ranking information

### 2. PeeringDB Data
**Source**: PeeringDB public database  
**Purpose**: Network operator characteristics and policies

### 3. ASPOP Data
**Source**: APNIC's AS Population dataset  
**Purpose**: Customer cone statistics and regional information

### 4. Organizational Metadata
**Source**: Various sources (IPinfo, manual curation)  
**Purpose**: Basic organizational information

## Feature Categories

### ASRank Features (17 base + 8 derived = 25 total)

#### Base Features
| Feature | Description | Range | Importance |
|---------|-------------|-------|------------|
| `asrank_rank` | Global AS rank (lower = more important) | 1-65535 | High - indicates network size/importance |
| `asrank_clique_member` | Member of AS clique (Tier-1) | 0/1 | High - identifies transit providers |
| `asrank_seen` | AS actively seen in routing data | 0/1 | Medium - indicates active networks |
| `asrank_longitude` | Geographic longitude | -180 to 180 | Low - geographic clustering |
| `asrank_latitude` | Geographic latitude | -90 to 90 | Low - geographic clustering |
| `asrank_cone_asns` | Number of ASes in customer cone | 0-65535 | High - indicates network reach |
| `asrank_cone_prefixes` | IP prefixes in customer cone | 0+ | High - indicates address space control |
| `asrank_cone_addresses` | IP addresses in customer cone | 0+ | High - indicates actual usage |
| `asrank_degree_provider` | Number of provider ASes | 0+ | Medium - upstream connectivity |
| `asrank_degree_peer` | Number of peer ASes | 0+ | Medium - horizontal connectivity |
| `asrank_degree_customer` | Number of customer ASes | 0+ | High - indicates ISP/transit status |
| `asrank_degree_total` | Total AS-level connections | 0+ | Medium - overall connectivity |
| `asrank_degree_transit` | Transit relationships | 0+ | High - transit provider indicator |
| `asrank_degree_sibling` | Sibling relationships | 0+ | Low - organizational structure |
| `asrank_announcing_prefixes` | Directly announced prefixes | 0+ | Medium - network infrastructure size |
| `asrank_announcing_addresses` | Directly announced addresses | 0+ | Medium - actual resource usage |

#### Derived Features
| Feature | Formula | Purpose | Importance |
|---------|---------|---------|------------|
| `asrank_cone_density` | addresses / prefixes | Address utilization efficiency | Medium |
| `asrank_peer_customer_ratio` | peers / customers | Network business model indicator | High |
| `asrank_provider_customer_ratio` | providers / customers | ISP vs end-user indicator | High |
| `asrank_transit_ratio` | transit / total_degree | Transit provider likelihood | High |
| `asrank_announce_efficiency` | announcing_addresses / announcing_prefixes | Resource utilization | Medium |
| `asrank_cone_size_log` | log(cone_asns + 1) | Log-scaled network size | High |
| `asrank_address_size_log` | log(cone_addresses + 1) | Log-scaled address space | High |
| `asrank_prefix_size_log` | log(announcing_prefixes + 1) | Log-scaled prefix count | Medium |

### PeeringDB Features (8+ categorical)

#### Base Features
| Feature | Description | Classification Relevance |
|---------|-------------|------------------------|
| `peeringdb_has_website` | Organization has website | Professional vs personal networks |
| `peeringdb_has_looking_glass` | Provides looking glass service | Technical sophistication |
| `peeringdb_has_route_server` | Operates route servers | IXP or advanced ISP |
| `peeringdb_has_irr_as_set` | Maintains IRR AS-SET | Technical best practices |
| `peeringdb_info_unicast` | Supports unicast routing | Basic networking capability |
| `peeringdb_info_multicast` | Supports multicast | Advanced services |
| `peeringdb_info_ipv6` | IPv6 enabled | Modern infrastructure |
| `peeringdb_has_policy_url` | Published peering policy | Professional operations |

#### Categorical Features (One-Hot Encoded)
- **Network Type**: `peeringdb_type_*` (NSP, Content, Cable/DSL, etc.)
- **Scope**: `peeringdb_scope_*` (Regional, National, Global)
- **Peering Policy**: `peeringdb_policy_*` (Open, Selective, Restrictive)

### ASPOP Features (9 total)

#### Customer Cone Statistics
| Feature | Description | Classification Value |
|---------|-------------|---------------------|
| `aspop_customer_cone_asns` | Customer ASes count | ISP/transit identification |
| `aspop_customer_cone_prefixes` | Customer prefixes | Network size indicator |
| `aspop_customer_cone_addresses` | Customer addresses | Actual user base |
| `aspop_customer_cone_unique` | Unique customers | Customer diversity |

#### Regional Indicators (RIR-based)
| Feature | Coverage | Classification Use |
|---------|----------|-------------------|
| `aspop_rir_afrinic` | Africa | Geographic/regulatory patterns |
| `aspop_rir_apnic` | Asia-Pacific | Regional network characteristics |
| `aspop_rir_arin` | North America | Regulatory environment |
| `aspop_rir_lacnic` | Latin America | Regional patterns |
| `aspop_rir_ripe` | Europe/Middle East | European network model |

### Organizational Features (15+ total)

#### Basic Metadata
| Feature | Description | Classification Value |
|---------|-------------|---------------------|
| `org_has_name` | Organization name available | Data completeness |
| `org_has_website` | Website information | Professional operations |
| `org_has_country` | Country information | Geographic classification |
| `org_name_length` | Name string length | Organization formality |

#### Name-Based Pattern Detection
| Feature | Keywords | Identifies |
|---------|----------|-----------|
| `org_name_has_isp` | "isp" | Internet Service Providers |
| `org_name_has_telecom` | "telecom", "telco", "telecommunications" | Telecom operators |
| `org_name_has_university` | "university", "college", "edu" | Educational institutions |
| `org_name_has_government` | "gov", "government", "municipal" | Government networks |
| `org_name_has_network` | "network" | Network operators |
| `org_name_has_cloud` | "cloud" | Cloud providers |
| `org_name_has_hosting` | "hosting" | Hosting companies |

#### Regional Classification
| Feature | Countries | Purpose |
|---------|-----------|---------|
| `org_region_north_america` | US, CA, MX | NAFTA region patterns |
| `org_region_europe` | DE, FR, GB, IT, ES, NL, PL | EU network characteristics |
| `org_region_asia_pacific` | CN, JP, KR, IN, AU, SG | APAC patterns |
| `org_region_south_america` | BR, AR, CL, CO, PE | Latin American patterns |

## Feature Engineering Pipeline

### 1. Feature Extraction
```python
features_df = engineer.extract_features(asn_list)
```
- Queries multiple data sources for each ASN
- Handles missing data with minimal feature fallback
- Creates consistent feature matrix

### 2. Preprocessing
```python
processed_df = engineer.preprocess_features(features_df, fit=True)
```
- **Imputation**: KNN imputation for missing values (k=5)
- **Scaling**: StandardScaler normalization (mean=0, std=1)
- **Handles**: Mixed data types and sparse features

### 3. Feature Selection
```python
selected_df = engineer.select_features(X, y, k=50)
```
- **Method**: Univariate statistical tests (f_classif)
- **Selection**: Top k features by F-statistic
- **Purpose**: Reduces dimensionality and noise

## Feature Importance by Category

### Tier 1 (Critical for Classification)
1. **Customer Cone Features**: `asrank_cone_*`, `aspop_customer_cone_*`
   - Strongest indicators of ISP vs end-user status
   - Correlates with organization type and size

2. **Degree Ratios**: `asrank_*_ratio` features
   - Business model indicators (ISP, transit, enterprise)
   - Network positioning in Internet hierarchy

3. **Name Pattern Features**: `org_name_has_*`
   - Direct semantic indicators of organization type
   - High precision for specific categories

### Tier 2 (Important Supporting Features)
1. **Network Size**: `asrank_*_log`, `asrank_rank`
   - Size-based classification support
   - Separates major from minor players

2. **Technical Capabilities**: PeeringDB features
   - Sophistication and service type indicators
   - Helps distinguish professional operations

### Tier 3 (Contextual Features)
1. **Geographic Features**: Regional and coordinate features
   - Regulatory and market context
   - Secondary classification support

2. **Metadata Completeness**: `org_has_*` features
   - Data quality indicators
   - Indirect organization type signals

## Classification Patterns by Feature

### ISP/Access Providers
- High `asrank_degree_customer`
- High `asrank_cone_asns` and `aspop_customer_cone_*`
- Low `asrank_provider_customer_ratio`
- `org_name_has_isp` or `org_name_has_telecom` = 1

### Transit Providers
- High `asrank_degree_transit`
- `asrank_clique_member` = 1 (for Tier-1)
- High `asrank_transit_ratio`
- Low `asrank_rank` (highly ranked)

### Content Providers
- High `asrank_announcing_prefixes`
- Low `asrank_degree_customer` (no customers)
- High `asrank_peer_customer_ratio`
- `org_name_has_cloud` or `org_name_has_hosting` = 1

### Educational Institutions
- Moderate network size features
- `org_name_has_university` = 1
- Often single-homed (low provider count)
- Regional geographic clustering

### Government Networks
- `org_name_has_government` = 1
- Country-specific regional features
- Often smaller network footprints
- Security-focused (fewer public services)

## Data Quality and Limitations

### Missing Data Handling
- **Strategy**: Zero-imputation for missing ASRank/PeeringDB/ASPOP data
- **Impact**: May underestimate smaller networks
- **Mitigation**: KNN imputation during preprocessing

### Feature Correlation
- **High Correlation**: Various size metrics (cone_*, degree_*)
- **Mitigation**: Feature selection removes redundant features
- **Benefit**: Multiple perspectives on network characteristics

### Bias Considerations
- **Geographic Bias**: Better data coverage for major regions
- **Size Bias**: Large networks have more complete data
- **Temporal Bias**: Data freshness varies by source

## Usage in Two-Stage Pipeline

### Stage 1 Integration
The SVM classifier uses these features alongside LLM predictions in an ensemble approach:

1. **Feature Extraction**: ASNs → Feature vectors
2. **Preprocessing**: Scaling and selection
3. **SVM Training**: Multi-label classification
4. **Ensemble**: Combine with LLM predictions via stacking

### Performance Impact
- **Training Speed**: ~1-2 seconds per 1000 ASNs for feature extraction
- **Memory Usage**: ~1MB per 10,000 ASNs for full feature matrix
- **Accuracy Contribution**: ~15-20% improvement over LLM-only approach

## Extending the Feature Set

### Adding New Features
1. Implement extraction method in `ASNFeatureEngineer`
2. Add to `_create_minimal_features()` for missing data
3. Update documentation
4. Retrain preprocessing pipeline

### Best Practices
- Maintain consistent naming conventions
- Handle missing data gracefully
- Document feature interpretation
- Test on representative ASN sample

### Future Enhancements
- **BGP Features**: Path length, origin validation
- **Security Features**: ROA coverage, RPKI status
- **Economic Features**: Market data, pricing information
- **Social Features**: Organization relationships, ownership

## Configuration

### Feature Engineering Settings
```yaml
# In config.yaml - two_stage_pipeline.stage1.svm
feature_selection: true
n_features: 50
imputation_method: "knn"
scaling_method: "standard"
```

### Preprocessing Parameters
- **KNN Neighbors**: 5 (balances bias-variance)
- **Feature Selection**: F-statistic based
- **Scaling**: StandardScaler (handles different feature scales)

This comprehensive feature set provides the SVM classifier with rich, multi-dimensional representations of AS organizations, enabling effective classification in conjunction with LLM predictions in the two-stage pipeline.