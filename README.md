# Linnaeus: AI-Powered Autonomous Systems Classification

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Linnaeus is a comprehensive tool for automatically classifying Internet Autonomous Systems (AS) using machine learning and large language models. It provides end-to-end functionality for data collection, model training, and inference to categorize organizations that operate AS networks based on their primary functions and purposes.

## Features

- **🌐 Multi-Source Data Integration**: Automatically downloads and processes data from ASRank, PeeringDB, and APNIC
- **🤖 LLM-Powered Classification**: Uses fine-tuned OpenAI models for high-accuracy classification
- **📊 20+ Organization Categories**: Supports comprehensive taxonomy including ISPs, content providers, government, education, enterprise, and more (see [CATEGORIES.md](CATEGORIES.md))
- **⚡ Async Processing**: Efficient batch processing with configurable concurrency
- **🧪 Scikit-learn Compatible**: Provides familiar `fit()`/`predict()` interface for easy integration
- **📈 Comprehensive Evaluation**: Built-in metrics, visualizations, and model comparison tools
- **🛠️ CLI Interface**: Full command-line interface for all operations
- **📋 Multiple Output Formats**: Export results in JSON, CSV, or Excel formats

## Quick Start

### Installation

#### Using UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a fast Python package manager that we recommend for development and installation:

```bash
# Install UV first
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install Linnaeus
git clone https://github.com/linnaeus-project/linnaeus.git
cd linnaeus
uv sync

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### From Source Only

```bash
# Clone and install from source
git clone https://github.com/linnaeus-project/linnaeus.git
cd linnaeus
pip install -e .
```

### Basic Usage

#### Quick Start Example

```python
# Create sample input file
import pandas as pd
df = pd.DataFrame({
    'asn': [174, 15169, 32934, 20940, 13335]
})
df.to_csv('sample_asns.csv', index=False)
```

```bash
# Classify using two-stage approach (recommended)
linnaeus predict --input sample_asns.csv --output results.json --approach two-stage

# View results
cat results.json
```

Expected output structure:
```json
[
  {
    "asn": 174,
    "organization_name": "Cogent Communications",
    "stage1_predictions": [
      {
        "category": "Transit",
        "confidence": {"value": 0.95, "model_type": "ASSEMBLED"}
      }
    ],
    "stage2_predictions": [
      {
        "subcategory": "Transit Global",
        "confidence": {"value": 0.92, "model_type": "LLM"}
      }
    ],
    "classification_approach": "two-stage"
  }
]
```

#### Python API Examples

**Two-Stage Classification (Recommended)**

```python
import numpy as np
import pandas as pd
from linnaeus.models.two_stage.pipeline import TwoStageClassificationPipeline
from linnaeus.data.access import DataAccessLayer

# Initialize data access and pipeline
data_access = DataAccessLayer()
pipeline = TwoStageClassificationPipeline(data_access=data_access)

# Prepare ASN data
asns = [174, 15169, 32934]

# Get two-stage predictions (no training needed)
results = pipeline.classify_batch(asns)

# Access results
for result in results:
    print(f"ASN {result.asn}: {result.organization_name}")
    if result.stage1_predictions:
        print(f"Top-level: {[p.category.value for p in result.stage1_predictions]}")
    if result.stage2_predictions:
        print(f"Hierarchical: {[p.subcategory for p in result.stage2_predictions]}")
    print(f"Confidence: {result.stage1_predictions[0].confidence.value if result.stage1_predictions else 'N/A'}")
    print("---")
```

**LLM-Only Classification**

```python
from linnaeus.models.llm.inference import HierarchicalBatchInferenceProcessor
from linnaeus.data.access import DataAccessLayer

# Initialize LLM processor with a fine-tuned model
processor = HierarchicalBatchInferenceProcessor(
    model_id="ft:gpt-4o-mini-2024-07-18:your-org:model-id",
    batch_size=10,
    temperature=0.0001
)

# Get organization data
data_access = DataAccessLayer()
organizations = []
for asn in [174, 15169, 32934]:
    org_data = data_access.get_organization_data(asn)
    if org_data:
        # Convert to dict format for LLM processing
        organizations.append({
            'asn': org_data.asn,
            'name': org_data.name,
            'description': org_data.description or f"Autonomous System {org_data.asn}"
        })

# Run classification
results = processor.process_batch(organizations)
print(f"Classified {len(results)} organizations")
```

**Traditional ML with Feature Engineering**

```python
from linnaeus.models.svm import ASNFeatureEngineer, SVMClassifier
import pandas as pd

# Extract features from network topology data
feature_engineer = ASNFeatureEngineer(
    include_asrank=True,
    include_peeringdb=True,
    include_aspop=True
)

# Prepare data
asns_df = pd.DataFrame({'asn': [174, 15169, 32934]})
X_features = feature_engineer.fit_transform(asns_df)

print(f"Extracted {X_features.shape[1]} features")
print(f"Feature names: {feature_engineer.get_feature_names_out()[:5]}...")

# Train SVM classifier
svm_clf = SVMClassifier(approach="flat")
# Note: Training requires labeled data - this is just feature extraction demo
```

#### Command Line Interface

**Basic Classification Commands**

```bash
# Hybrid approach (combines SVM + LLM) - Recommended
linnaeus model predict --input asns.csv --output results.json --approach hybrid

# SVM-only approach (fast, works offline)
linnaeus model predict --input asns.csv --output results.csv --approach svm-only --format csv

# LLM-only approach (requires API key, highest accuracy)
linnaeus model predict --input asns.csv --output results.xlsx --approach llm-only --format excel --model ft:gpt-4o-mini-your-model

# Hierarchical classification (detailed subcategories)
linnaeus model predict --input asns.csv --output detailed_results.json --approach hierarchical

# Flat classification (broad categories only)
linnaeus model predict --input asns.csv --output simple_results.json --approach flat
```

**Benchmarking and Comparison**

```bash
# Compare multiple approaches on test dataset
linnaeus benchmark --dataset test_asns.csv --models hybrid,svm-only,llm-only --output benchmark_results.json

# Quick benchmark with specific sample size
linnaeus benchmark --dataset large_dataset.csv --sample-size 100 --models hybrid,hierarchical

# Benchmark specific models only
linnaeus benchmark --dataset validation_set.csv --models hybrid --output hybrid_performance.json
```

**Data Management**

```bash
# Download fresh data from all sources
linnaeus data download --sources peeringdb,asrank,aspop

# Download specific data sources only
linnaeus data download --sources asrank,peeringdb --date 2024-01-01

# Check current data status and freshness
linnaeus data status

# Force refresh cached data
linnaeus data download --sources peeringdb --force-refresh
```

**Complete Workflow Example**

```bash
# 1. Download latest data
linnaeus data download --sources peeringdb,asrank,aspop

# 2. Check data availability
linnaeus data status

# 3. Create sample input file
echo "asn" > example_asns.csv
echo "174" >> example_asns.csv    # Cogent Communications
echo "15169" >> example_asns.csv  # Google
echo "32934" >> example_asns.csv  # Facebook
echo "20940" >> example_asns.csv  # Akamai
echo "13335" >> example_asns.csv  # Cloudflare

# 4. Run hybrid classification
linnaeus model predict \
    --input example_asns.csv \
    --output classifications.json \
    --approach hybrid \
    --format json

# 5. View results
cat classifications.json | jq '.[] | {asn: .asn, org: .organization_name, tags: .top_level_tags}'

# 6. Benchmark different approaches
linnaeus benchmark \
    --dataset example_asns.csv \
    --models hybrid,svm-only,hierarchical \
    --output performance_comparison.json

# 7. View benchmark results
cat performance_comparison.json | jq '.models'
```


## Classification Categories

Linnaeus supports both **flat** (20 top-level categories) and **hierarchical** (detailed subcategories) classification systems. For the complete taxonomy with detailed subcategories, see **[CATEGORIES.md](CATEGORIES.md)**.

### Top-Level Categories (Flat System)

The following 20 categories provide broad classification suitable for most use cases:

| Category | Description |
|----------|-------------|
| **Access** | Internet service providers serving end users |
| **Transit** | Providers offering IP transit services |
| **Mobile** | Mobile network operators |
| **Satellite** | Satellite communication providers |
| **Content Provider** | CDNs, hosting providers, cloud services |
| **Educational Research** | Universities, research institutions |
| **Government** | Government agencies, public sector |
| **Internet Exchange Point** | Internet exchange points |
| **DNS** | Domain name system operators |
| **Energy & Utility** | Energy companies, utilities |
| **Enterprise** | Commercial enterprises, businesses |
| **Finance** | Banks, financial institutions |
| **Law Enforcement** | Law enforcement agencies |
| **Health** | Healthcare organizations |
| **Cooperatives** | Cooperative organizations |
| **TV/Radio and Cultural** | Media, cultural organizations |
| **Transportation** | Transport companies, airports |
| **Virtual Private Networks** | VPN providers |
| **Personal** | Individual/personal networks |
| **Community** | Community networks, non-profits |

## Taxonomies, Definitions, and Benchmark Labels

### Taxonomy definition files

Category definitions live in editable JSON files packaged with linnaeus:

```
src/linnaeus/resources/taxonomies/linnaeus.json   # default: 20 top-level categories (+ subcategories)
src/linnaeus/resources/taxonomies/asdb.json       # Stanford ASdb taxonomy (17 categories)
src/linnaeus/resources/taxonomies/isic.json       # ISIC Rev.4 taxonomy (20 sections)
src/linnaeus/resources/prompts.yaml               # LLM prompt templates
```

Each taxonomy JSON maps a category name to a description (or to a dict of
subcategory descriptions). The category names and descriptions drive both the
LLM system prompt and the structured-output schema — **edit the JSON to change
what the classifier can predict**; edit `prompts.yaml` to change the prompt
wording. To write your own taxonomy, see
[docs/custom_taxonomies.md](docs/custom_taxonomies.md).

### Choosing a taxonomy on the CLI

```bash
# Classify with the ASdb reference taxonomy
linnaeus model predict --approach llm-only --taxonomy asdb \
    --input asns.csv --output results.csv --format csv

# Or with a custom taxonomy file and a custom model/provider
linnaeus model predict --approach llm-only --taxonomy-file my_taxonomy.json \
    --model my-model --base-url http://localhost:11434/v1 \
    --input asns.csv --output results.csv --format csv

# Evaluate predictions against ASdb ground truth
linnaeus model evaluate --taxonomy asdb \
    --predictions results.csv --labels data/released/202506/labels/asdb.csv

# The standalone script accepts the same taxonomies
python scripts/classify.py -i asns.csv -o results.csv --taxonomy isic
```

### Benchmark labels (ASDB and ISIC)

Manually curated benchmark labels — labels only, no model predictions — are
released for comparing linnaeus against other classification schemes:

| File | ASNs | Categories | Scheme |
|------|------|-----------|--------|
| `data/released/202506/labels/asdb.csv` (+ `.parquet`) | 1,978 | 17 | Stanford ASdb |
| `data/released/202506/labels/isic.csv` (+ `.parquet`) | 2,063 | 20 | ISIC Rev.4 |

Format: one row per ASN (`asn` as a bare integer) with one binary 0/1 column
per category. The matching category definitions are released alongside them as
`data/released/202506/{asdb,isic}_definitions.json`.

## Validating the pipeline

`scripts/test_pipeline_e2e.py` runs the full workflow — training-data
preparation, fine-tuning, inference on the validation split, and comparison
against the stored reference metrics:

```bash
export OPENAI_API_KEY=...   # keep the key in the environment, not in files

# 1. Structural check, zero API calls
python scripts/test_pipeline_e2e.py --dry-run --focus

# 2. Cheap smoke test (~25 baseline predictions)
python scripts/test_pipeline_e2e.py --baseline-only --focus --max-samples 25

# 3. Full restricted validation: fine-tune on 4 categories (~795 examples,
#    ~$3, ~15 min) and evaluate on the 601-ASN validation split
python scripts/test_pipeline_e2e.py --focus --suffix my-validation

# Reuse a fine-tuned model without retraining
python scripts/test_pipeline_e2e.py --focus --skip-finetune --model-id ft:gpt-4o-mini-...
```

`--focus` restricts everything (prompt, schema, training data, metrics) to
Government, Access, Enterprise, and ContentProvider; `--tags` accepts any
comma-separated category subset. For restricted runs the reference metrics are
recomputed from the committed pre-refactor predictions, so the comparison is
apples-to-apples. Expected reference values for `--focus`: baseline macro-F1
**0.545**, fine-tuned macro-F1 **0.732** — a fresh baseline within ±0.05 and a
fine-tuned result ≥ 0.68 mean the pipeline reproduces the pre-refactor
behavior.

## Known Limitations

- `HybridASClassifier`'s stacking LLM path is a placeholder (returns zeros /
  constant 0.5 probabilities) — use the two-stage pipeline or `llm-only`
  approaches instead.
- `AssembledClassifier.predict_proba` derives pseudo-probabilities from hard
  predictions rather than LLM logprobs, so PR AUC is not reported.
- `linnaeus data process` (raw → features) is not yet implemented; use the
  prebuilt features in `data/local/features/` (see STUDENT_GUIDE.md).
- The released split has only 3 `test` rows; the 601-row `val` split is the
  effective evaluation set.
- Without an `organization_name` (or downloaded metadata), the two-stage
  pipeline falls back to `AS<asn>` as the organization name, which degrades
  quality.
- The sub-level subcategory names in `taxonomies/linnaeus.json` predate the
  canonical 48-column sub-level label schema and do not match it one-to-one;
  the top-level names are canonical.

## Architecture

Linnaeus implements a **two-stage hierarchical classification system** that combines the strengths of Large Language Models (LLMs) and traditional machine learning for high-accuracy AS classification.

### 🏗️ System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Two-Stage Pipeline                       │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │     Stage 1     │    │            Stage 2             │ │
│  │  Top-Level      │───▶│        Sublevel             │ │
│  │ Classification  │    │      Classification           │ │
│  │                 │    │                               │ │
│  │  LLM + SVM +    │    │  Category-Specific LLMs +    │ │
│  │  Stacking       │    │  Consistency Validation      │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 🎯 Key Features

- **🔧 Hybrid Approach**: Combines LLM reasoning with feature-engineered SVM
- **📊 Two-Stage Design**: Separates broad categorization from detailed subcategorization
- **⚡ Validated Performance**: 0.79 macro F1 on top-level classification (fine-tuned, 601-ASN validation split)
- **🛡️ Robust**: Comprehensive error handling and graceful degradation
- **🔒 Type Safe**: Pydantic validation throughout the pipeline

### 📋 Architecture Documentation

- **📖 [Complete Architecture Guide](ARCHITECTURE.md)** - Comprehensive system overview with diagrams
- **🔧 [Technical Deep-Dive](docs/architecture_overview.md)** - Detailed technical implementation
- **⚙️ [Implementation Guide](docs/two_stage_pipeline_documentation.md)** - Usage and API reference

### Data Pipeline

```
Raw Data Sources → Data Processing → Feature Engineering → Classification → Results
     ↓                   ↓               ↓                ↓            ↓
┌─ ASRank API ─┐   ┌─ Downloaders ─┐ ┌─ Organization ─┐ ┌─ Fine-tuned ─┐ ┌─ JSON/CSV ─┐
├─ PeeringDB ──┤ → ├─ Processors ──┤→├─ Data Models ─┤→├─ LLM Models ─┤→├─ Excel ─────┤
└─ APNIC ASPOP ┘   └─ Data Access ─┘ └─ Validation ──┘ └─ Evaluation ─┘ └─ Dashboards┘
```

### Model Performance

Measured on the 601-ASN validation split (`data/released/202506/metrics/`):

| | Top-level baseline | Top-level fine-tuned | Sub-level baseline | Sub-level fine-tuned |
|---|---|---|---|---|
| Exact-match accuracy | 0.528 | 0.676 | 0.443 | 0.595 |
| Macro F1 | 0.679 | 0.792 | 0.659 | 0.732 |
| Macro precision | 0.754 | 0.810 | 0.738 | 0.793 |
| Macro recall | 0.633 | 0.779 | 0.634 | 0.714 |

Baseline = `gpt-4o-mini` zero-shot; fine-tuned = `gpt-4o-mini` fine-tuned on the
1,402-ASN training split. Reproduce with `python scripts/test_pipeline_e2e.py`
(see [Validating the pipeline](#validating-the-pipeline)).

## Real-World Examples

### Example 1: Network Research Analysis

```bash
# Analyze major content delivery networks
echo "asn,name" > cdn_analysis.csv
echo "13335,Cloudflare" >> cdn_analysis.csv
echo "20940,Akamai" >> cdn_analysis.csv
echo "16509,Amazon" >> cdn_analysis.csv
echo "15169,Google" >> cdn_analysis.csv

# Run hierarchical classification to get detailed subcategories
linnaeus model predict \
    --input cdn_analysis.csv \
    --output cdn_results.json \
    --approach hierarchical \
    --format json
```

Expected output:
```json
[
  {
    "asn": 13335,
    "organization_name": "Cloudflare, Inc.",
    "top_level_tags": ["Content Provider"],
    "hierarchical_tags": ["ContentProvider CDN"],
    "classification_approach": "hierarchical",
    "confidence_scores": {"Content Provider": 0.98}
  },
  {
    "asn": 20940,
    "organization_name": "Akamai Technologies",
    "top_level_tags": ["Content Provider"],
    "hierarchical_tags": ["ContentProvider CDN"],
    "classification_approach": "hierarchical"
  }
]
```

### Example 2: ISP Market Analysis

```python
import pandas as pd
from linnaeus.models import HybridASNClassifier

# Analyze ISP landscape in a region
regional_isps = pd.DataFrame({
    'asn': [174, 7018, 3320, 1299, 6830, 3356]  # Major transit/access providers
})

# Use hybrid approach for comprehensive analysis
clf = HybridASNClassifier(approach="hybrid")
results = clf.predict_unified(regional_isps)

# Analyze by category
categories = {}
for result in results:
    for tag in result.top_level_tags:
        if tag.value not in categories:
            categories[tag.value] = []
        categories[tag.value].append({
            'asn': result.asn,
            'name': result.organization_name
        })

# Print analysis
for category, orgs in categories.items():
    print(f"\n{category} ({len(orgs)} organizations):")
    for org in orgs:
        print(f"  AS{org['asn']}: {org['name']}")
```

Expected output:
```
Transit (4 organizations):
  AS174: Cogent Communications
  AS7018: AT&T Services
  AS3320: Deutsche Telekom AG
  AS1299: Arelion (formerly Telia)

Access (2 organizations):
  AS6830: Liberty Global
  AS3356: Level 3 Communications
```

### Example 3: Performance Benchmarking

```bash
# Compare all approaches on a test dataset
linnaeus benchmark \
    --dataset test_dataset.csv \
    --models hybrid,svm-only,llm-only,hierarchical,flat \
    --sample-size 50 \
    --output comprehensive_benchmark.json

# View performance comparison
cat comprehensive_benchmark.json | jq '{
  timestamp: .timestamp,
  total_processed: .total_processed,
  models: .models | to_entries | map({
    approach: .key,
    success: .value.success,
    predictions: .value.predictions_made,
    speed: (.value.predictions_per_second | round),
    time: (.value.processing_time_seconds | round)
  })
}'
```

Expected benchmark output:
```json
{
  "timestamp": "2024-01-15T10:30:45",
  "total_processed": 50,
  "models": [
    {
      "approach": "hybrid",
      "success": true,
      "predictions": 50,
      "speed": 12,
      "time": 4
    },
    {
      "approach": "svm-only",
      "success": true,
      "predictions": 50,
      "speed": 25,
      "time": 2
    },
    {
      "approach": "llm-only",
      "success": true,
      "predictions": 50,
      "speed": 8,
      "time": 6
    }
  ]
}
```

### Example 4: Integration with Data Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
from linnaeus.models import HybridASNClassifier

# Load ASN dataset
asns_df = pd.read_csv('large_asn_dataset.csv')  # ASN, country, org_type columns

# Classify using hybrid approach
clf = HybridASNClassifier(approach="flat")  # Use flat for summary analysis
results = clf.predict_unified(asns_df)

# Convert to DataFrame for analysis
results_df = pd.DataFrame([
    {
        'asn': r.asn,
        'organization': r.organization_name,
        'primary_category': r.top_level_tags[0].value if r.top_level_tags else 'Unknown',
        'confidence': max(r.top_level_confidence.values()) if r.top_level_confidence else 0.0
    }
    for r in results
])

# Analysis and visualization
category_counts = results_df['primary_category'].value_counts()
print("Distribution of AS Categories:")
print(category_counts)

# Plot distribution
plt.figure(figsize=(12, 6))
category_counts.plot(kind='bar')
plt.title('Distribution of Autonomous Systems by Category')
plt.xlabel('Category')
plt.ylabel('Number of ASNs')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('asn_category_distribution.png')

# High-confidence predictions only
high_confidence = results_df[results_df['confidence'] >= 0.8]
print(f"\nHigh-confidence predictions: {len(high_confidence)}/{len(results_df)} ({len(high_confidence)/len(results_df)*100:.1f}%)")
```

## Advanced Usage

### Custom Training Pipeline

```python
import asyncio
from pathlib import Path
from linnaeus.models import TrainingPipeline

# Initialize training pipeline
pipeline = TrainingPipeline()

# Train new model
model_id = await pipeline.run_training_pipeline(
    labeled_data_path=Path("training_data.csv"),
    base_model="gpt-4o-mini-2024-07-18",
    model_suffix="v2",
    hyperparameters={
        "n_epochs": 3,
        "batch_size": "auto",
        "learning_rate_multiplier": 0.1
    }
)

print(f"New model trained: {model_id}")
```

### Batch Classification

```python
from linnaeus.models import ClassificationPipeline
from linnaeus.data import DataAccessLayer

# Load organization data
data_layer = DataAccessLayer()
organizations = data_layer.get_bulk_data([174, 15169, 32934])

# Run classification pipeline
pipeline = ClassificationPipeline(model_id="ft:gpt-4o-mini-...")
results = await pipeline.run_pipeline(
    input_data=organizations,
    output_format="excel",
    output_path="classifications.xlsx",
    include_confidence=True
)
```

### Model Evaluation

```python
from linnaeus.models import ClassificationEvaluator
import pandas as pd

# Load ground truth data
ground_truth = pd.read_csv("labeled_data.csv", index_col="asn")

# Evaluate model predictions
evaluator = ClassificationEvaluator()
metrics = evaluator.evaluate_from_files(
    predictions_path=Path("predictions.json"),
    ground_truth_path=Path("ground_truth.csv"),
    include_pr_auc=True
)

# Print detailed metrics
print(f"Overall Accuracy: {metrics['overall_accuracy']:.3f}")
print(f"Macro F1-Score: {metrics['macro_f1']:.3f}")
```

### Feature Engineering

```python
from linnaeus.models import ASNDataTransformer
from sklearn.ensemble import RandomForestClassifier

# Transform ASNs to feature vectors
transformer = ASNDataTransformer(
    include_asrank=True,
    include_peeringdb=True,
    include_aspop=True,
    normalize_features=True
)

X_features = transformer.fit_transform(asns)

# Use with traditional ML models
rf = RandomForestClassifier()
rf.fit(X_features, y_labels)
```

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_ORG_ID=org-your-org-id  # Optional

# Data Configuration
DEFAULT_DATA_DIR=./data
CACHE_EXPIRY_HOURS=24

# Model Configuration
DEFAULT_MODEL=gpt-4o-mini
BATCH_SIZE=10
TEMPERATURE=0.0001
```

### Configuration File

Create `config.yaml`:

```yaml
# Application Environment
environment: production

# OpenAI Settings
openai:
  default_model: "gpt-4o-mini"
  temperature: 0.0001
  batch_size: 10
  max_concurrent_fine_tunes: 3

# Data Sources
apis:
  asrank_url: "https://api.asrank.caida.org/v2/graphql"
  peeringdb_base_url: "https://publicdata.caida.org/datasets/peeringdb"
  apnic_aspop_url: "https://stats.labs.apnic.net/cgi-bin/aspop"

# Logging
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Development

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/linnaeus-project/linnaeus.git
cd linnaeus

# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate

# Install in development mode
uv pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=linnaeus --cov-report=html

# Run specific test categories
pytest tests/test_models/
pytest tests/test_data/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

## Data Sources

Linnaeus integrates data from several authoritative sources:

- **[ASRank](https://asrank.caida.org/)**: AS ranking and connectivity data from CAIDA
- **[PeeringDB](https://www.peeringdb.com/)**: Network operator information and peering policies
- **[APNIC AS Population](https://stats.labs.apnic.net/)**: AS customer cone and address space data
- **[IPinfo](https://ipinfo.io/)**: Optional geolocation and ISP data (premium features)

## Model Details

### Training Data

- **1,978 labeled ASNs** across all 20 top-level categories
  (`data/released/202506/labels/`)
- **Multi-label classification** supporting organizations with multiple functions
- **Hierarchical labeling** with both broad (20 top-level) and specific
  (48 sub-level) categories
- **Fixed splits**: 1,402 train / 601 validation
  (`data/released/202506/splits/assignments.csv`)

### Model Architecture

- **Base Model**: OpenAI GPT-4o-mini (fine-tuned, base
  `gpt-4o-mini-2024-07-18`)
- **Input Format**: ASN, organization name, country, and website per sample
- **Output**: Structured JSON constrained by a Pydantic schema generated from
  the taxonomy definitions
- **Training**: OpenAI default hyperparameters over the 1,402-ASN train split

### Performance Metrics

Top-level fine-tuned model, 601-ASN validation split
(`data/released/202506/metrics/toplevel_finetuned.txt`):

| Metric | Value |
|--------|-------|
| Exact-match Accuracy | 67.6% |
| Macro Precision | 81.0% |
| Macro Recall | 77.9% |
| Macro F1-Score | 79.2% |
| Macro Jaccard | 66.5% |

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Areas for Contribution

- 🏷️ **Labeling**: Help expand and refine our training dataset
- 🧪 **Testing**: Add test cases and improve coverage
- 📚 **Documentation**: Improve documentation and examples
- 🚀 **Features**: Implement new classification categories or data sources
- 🐛 **Bug Fixes**: Fix issues and improve reliability

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Linnaeus in your research, please cite:

```bibtex
@software{linnaeus2026,
  title={Linnaeus: AI-Powered Autonomous Systems Classification},
  author={M. Piotto, I. Schuemer, S.T. Torres, M.G. Beiró, E. Carisimo, F.E. Bustamante},
  year={2026},
  url={https://github.com/linnaeus-project/linnaeus}
}
```

## Acknowledgments

- [CAIDA](https://www.caida.org/) for ASRank data and Internet measurement research
- [PeeringDB](https://www.peeringdb.com/) for network operator data
- [APNIC](https://www.apnic.net/) for AS population statistics
- [OpenAI](https://openai.com/) for fine-tuning capabilities

## Support

- 📖 [Documentation](https://linnaeus.readthedocs.io/)
- 🐛 [Issue Tracker](https://github.com/linnaeus-project/linnaeus/issues)
- 💬 [Discussions](https://github.com/linnaeus-project/linnaeus/discussions)
- 📧 Email: linnaeus@example.com
