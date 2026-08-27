# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Linneaus is a Python package for classifying Internet Autonomous Systems (AS) using a hybrid machine learning approach that combines SVM classifiers with LLM inference (stacking). The system performs multi-label classification across 20 top-level categories with ~40+ subcategories, using data from multiple Internet infrastructure sources.

## Development Environment

### Prerequisites
- Python >= 3.9
- UV package manager
- OpenAI API key (for LLM inference and fine-tuning)

### Installation
```bash
uv pip install -e ".[dev]"
```

### Running Tests
```bash
uv run pytest tests/
```
There are 231 tests across the test suite covering models, CLI, configuration, and data access.

### Environment Variables
- `OPENAI_API_KEY` -- required for LLM inference and fine-tuning
- `IPINFO_TOKEN` -- optional, for IPinfo API premium features

## Package Structure

The package is located at `src/linneaus/` and installed as `linneaus`.

```
src/linneaus/
  cli/              # Click-based CLI (entry point: linneaus.cli.main:main)
  config/           # Configuration management (settings.py, two_stage_config.py)
  data/             # Data access, downloaders, labeled data, validation
  models/
    hybrid/         # HybridASClassifier, SVCLLMEstimator, LLMPredictor, ModernHierarchicalTagger
    svm/            # SVMClassifier, ASNFeatureEngineer, pretrained models
    llm/            # LLM inference, fine-tuning, training, schema generation
    two_stage/      # TwoStageHierarchicalPipeline, assembled/sublevel classifiers, evaluation
    utils/          # Cross-validation, metrics, data adapters, tag reduction
    schemas.py              # Core Pydantic schemas (ClassificationTags, ASClassification)
    hierarchical_schemas.py # TopLevelCategory/SubCategory enums and models
    unified_schemas.py      # TopLevelTags/HierarchicalTags unified system
    two_stage_schemas.py    # Two-stage pipeline schemas
  resources/        # Static data (tags_descriptions.json, country_rir_registry.json)
  utils/            # Utility modules (country_mapping)
```

### Test Structure
```
tests/
  test_cli/         # CLI command tests
  test_config/      # Configuration tests
  test_data/        # Data access layer tests
  test_models/      # Model tests (SVM, hybrid, schemas, two-stage, sklearn interface)
```

## Classification Architecture

### Hybrid Approach (SVM + LLM Stacking)
The core classifier (`HybridASClassifier`) combines:
1. **SVM classifiers** -- trained on engineered features from AS metadata
2. **LLM inference** -- fine-tuned GPT-4o-mini for structured classification
3. **Stacking meta-learner** -- `SVCLLMEstimator` combines SVM and LLM predictions via a meta-SVM

### Two-Stage Hierarchical Pipeline
`TwoStageHierarchicalPipeline` provides a two-pass classification:
- **Stage 1**: Top-level category classification (assembled classifier)
- **Stage 2**: Subcategory classification (sublevel classifier)

### Schema Systems
Two parallel schema systems exist:

1. **Hierarchical schemas** (`hierarchical_schemas.py`): `TopLevelCategory` and per-category `SubCategory` enums (e.g., `AccessSubCategory`, `TransitSubCategory`)
2. **Unified schemas** (`unified_schemas.py`): `TopLevelTags` (20 flat categories) and `HierarchicalTags` (detailed tags like "Access Large ISP", "Transit Global")

### Classification Categories (20 top-level)
Access, Transit, Mobile, Satellite, Content Provider, Educational Research, Government, IXP, DNS, Energy & Utility, Enterprise, Finance, Law Enforcement, Health, Cooperatives, TV/Radio and Cultural Amenities, Transportation, VPNs, Personal, Community

## Data Sources

Data modules in `src/linneaus/data/` pull from four sources:
- **ASRank** (CAIDA): AS ranking, organization metadata, AS relationships via GraphQL API
- **PeeringDB**: Network operator information, facility data, IX participation
- **ASPOP** (APNIC): AS population estimates
- **IPinfo**: Geolocation, ISP identification, organization data

### Labeled Data
1,977 labeled ASNs in `data/labeled_data/labeled_data.csv`, managed by `LabeledDataManager`.

## CLI Reference

The CLI is Click-based, entry point defined in `pyproject.toml` as `linneaus = "linneaus.cli.main:main"`.

### Data Commands
```bash
linneaus data download --sources peeringdb,asrank,aspop,ipinfo
linneaus data download-ipinfo --asn 15169
linneaus data download-ipinfo-batch --asns "15169,7922,8075"
linneaus data validate --source asrank
linneaus data status
```

### Model Commands
```bash
linneaus model train --llm-preds predictions.csv --output-dir models/
linneaus model predict --approach two-stage --input asns.csv --output results.json
linneaus model evaluate --predictions preds.csv --labels labels.csv
linneaus model fine-tune --approach hierarchical --model-suffix "v1"
linneaus model fine-tune-async --approach hierarchical --max-concurrent 10
linneaus model prepare-data --approach hierarchical
linneaus model list-jobs
linneaus model monitor-job JOB_ID
linneaus model predict-svm --model-path model.pkl --input-data features.csv
```

### Two-Stage Pipeline Commands
```bash
linneaus two-stage classify-single --asn 15169 --organization-name "Google LLC"
linneaus two-stage classify-batch --input asns.csv --output results.json --parallel
linneaus two-stage evaluate --predictions results.json --ground-truth labels.csv
linneaus two-stage benchmark --test-dataset test.csv --batch-sizes "1,5,10,20"
linneaus two-stage monitor-performance --test-dataset test.csv --output metrics.json
```

### Other Commands
```bash
linneaus info
linneaus benchmark --models hybrid,svm-only --dataset test.csv --output results.json
```

## Key Implementation Details

### Batch Processing
- LLM inference uses batch processing (default batch size: 10)
- Pydantic models enforce structured output parsing
- Probability extraction from logprobs for confidence scoring
- Async inference support via `aiohttp` for concurrent requests

### Metrics
- Accuracy, precision, recall, F1, Jaccard (macro-averaged)
- PR AUC when prediction probabilities are available
- Per-tag performance analysis
- `get_metrics()`, `get_global_metrics()`, `compare_model_metrics()` in `models/utils/metrics.py`

### Tag Reduction
- `reduce_tags()` and `simplify_tags()` in `models/utils/tag_reduction.py` for consolidating rare or overlapping categories

### Cross-Validation
- `ASNStratifiedKFold` for ASN-aware stratified splitting
- Nested cross-validation support for hyperparameter tuning

### Configuration
Managed via `src/linneaus/config/settings.py` using Pydantic models. Supports YAML config files and environment variable overrides. Key config sections: `OpenAIConfig`, `DataConfig`, `APIConfig`, `LoggingConfig`.

## Build System

Uses Hatchling as the build backend. Defined in `pyproject.toml`:
- Package name: `linneaus`
- Version: `0.1.0`
- Build target: `src/linneaus`
- Tooling: black, isort, mypy, pytest with coverage
