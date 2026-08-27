# Linneaus CLI Usage Guide

## Overview

The Linneaus CLI provides comprehensive command-line access to the AS classification pipeline, including both legacy approaches and the new two-stage hierarchical system.

## Command Structure

```
linneaus [OPTIONS] COMMAND [ARGS]...
```

### Main Command Groups

- `data` - Data management commands
- `model` - Legacy model training and inference commands
- `two-stage` - Two-stage hierarchical pipeline commands (recommended)

## Two-Stage Pipeline Commands

### Quick Start

```bash
# Show available two-stage commands
linneaus two-stage help-commands

# Classify a single organization
linneaus two-stage classify-single --asn 15169 --organization-name "Google LLC"

# Batch processing
linneaus two-stage classify-batch --input asns.csv --output results.json

# Evaluate performance
linneaus two-stage evaluate --predictions results.json --ground-truth labels.csv
```

### Single Classification

Classify a single AS organization:

```bash
# Basic classification
linneaus two-stage classify-single --asn 15169

# With organization name for better accuracy
linneaus two-stage classify-single --asn 15169 --organization-name "Google LLC"

# JSON output with timing information
linneaus two-stage classify-single --asn 7922 --output-format json --include-timing

# Simple text output
linneaus two-stage classify-single --asn 174 --output-format simple
```

**Options:**
- `--asn` - Autonomous System Number (required)
- `--organization-name` - Organization name (optional, improves accuracy)
- `--output-format` - Output format: table (default), json, simple
- `--include-timing` - Include processing time information
- `--config` - Custom configuration file

### Batch Classification

Process multiple organizations from a CSV file:

```bash
# Basic batch processing
linneaus two-stage classify-batch --input asns.csv --output results.json

# Large batch with custom settings
linneaus two-stage classify-batch \
  --input large_dataset.csv \
  --output results.csv \
  --format csv \
  --batch-size 20 \
  --parallel

# Performance analysis with timing
linneaus two-stage classify-batch \
  --input test.csv \
  --output results.json \
  --include-timing \
  --sequential
```

**Options:**
- `--input` - Input CSV file with ASN column (required)
- `--output` - Output file for results (required)
- `--format` - Output format: json (default), csv, excel
- `--batch-size` - Batch size for processing (default: 10)
- `--parallel/--sequential` - Processing mode (default: parallel)
- `--include-timing` - Include processing timing information
- `--config` - Custom configuration file

**Input CSV Format:**
```csv
asn,organization_name
15169,Google LLC
7922,Comcast Cable Communications
174,Cogent Communications
```

### Evaluation

Evaluate predictions against ground truth:

```bash
# Basic evaluation
linneaus two-stage evaluate \
  --predictions results.json \
  --ground-truth labels.csv

# Detailed evaluation with report
linneaus two-stage evaluate \
  --predictions results.json \
  --ground-truth labels.csv \
  --output eval_report.txt \
  --detailed-analysis
```

**Options:**
- `--predictions` - JSON file with predictions (required)
- `--ground-truth` - CSV file with ground truth (required)
- `--output` - Output file for detailed report
- `--detailed-analysis/--summary-only` - Include detailed error analysis

**Ground Truth CSV Format:**
```csv
ASN,TopLevelCategory,SubCategory
15169,Content,Search Engine
7922,Access,Cable
174,Transit,Tier 1
```

### Performance Benchmarking

Test pipeline performance with different configurations:

```bash
# Basic benchmark
linneaus two-stage benchmark --test-dataset test_asns.csv

# Custom benchmark with results
linneaus two-stage benchmark \
  --test-dataset large_test.csv \
  --sample-size 100 \
  --batch-sizes "5,10,25,50" \
  --output benchmark.json
```

**Options:**
- `--test-dataset` - CSV file with ASNs for benchmarking (required)
- `--sample-size` - Number of ASNs to sample (default: 50)
- `--batch-sizes` - Comma-separated batch sizes to test (default: "1,5,10,20")
- `--output` - Output file for benchmark results
- `--config` - Custom configuration file

## Legacy Model Commands

### Basic Prediction

The legacy model predict command now supports the two-stage approach:

```bash
# Two-stage approach (recommended)
linneaus model predict \
  --approach two-stage \
  --input asns.csv \
  --output results.json

# Legacy hybrid approach
linneaus model predict \
  --approach hybrid \
  --input asns.csv \
  --output results.json \
  --format csv

# SVM-only approach
linneaus model predict \
  --approach svm-only \
  --input asns.csv \
  --output results.csv \
  --format csv
```

**Approaches:**
- `two-stage` - Advanced two-stage hierarchical classification (recommended)
- `hybrid` - Legacy hybrid LLM+SVM approach
- `hierarchical` - Legacy hierarchical approach
- `flat` - Legacy flat classification
- `svm-only` - SVM-only classification

## Data Management Commands

### Download Data

Download fresh data from external sources:

```bash
# Download all sources
linneaus data download

# Download specific sources
linneaus data download --sources "peeringdb,asrank"

# Force refresh existing data
linneaus data download --force-refresh
```

### Check Data Status

Check availability and status of data sources:

```bash
linneaus data status
```

## Configuration

### Environment Variables

Set required environment variables in `.env`:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional logging level
LOG_LEVEL=INFO
```

### Configuration File

Customize behavior through `config.yaml`:

```yaml
two_stage_pipeline:
  stage1:
    svm_weight: 0.4
    llm_weight: 0.6
    parallel_batch_size: 10
  stage2:
    confidence_threshold: 0.3
  error_handling:
    enable_graceful_degradation: true
    max_retry_attempts: 3
```

### Custom Configuration

Use a custom configuration file:

```bash
linneaus --config /path/to/custom/config.yaml two-stage classify-single --asn 15169
```

## Output Formats

### JSON Format

Structured output with full prediction details:

```json
{
  "asn": 15169,
  "organization_name": "Google LLC",
  "stage1_predictions": [
    {
      "category": "Content",
      "confidence": 0.92,
      "model_type": "assembled"
    }
  ],
  "stage2_predictions": [
    {
      "parent_category": "Content",
      "subcategory": "Search Engine",
      "confidence": 0.87
    }
  ],
  "overall_confidence": 0.895
}
```

### CSV Format

Flat format suitable for spreadsheet analysis:

```csv
asn,organization_name,stage1_categories,stage2_subcategories,overall_confidence
15169,Google LLC,Content,Content:Search Engine,0.895
7922,Comcast Cable Communications,Access,Access:Cable,0.823
```

### Table Format

Rich formatted output for terminal display:

```
Classification Results for ASN 15169
Organization: Google LLC
Overall Confidence: 0.895

┌─────────────┬────────────┬───────────┐
│ Category    │ Confidence │ Model     │
├─────────────┼────────────┼───────────┤
│ Content     │ 0.920      │ assembled │
└─────────────┴────────────┴───────────┘

┌─────────────┬────────────────┬────────────┐
│ Top-Level   │ Subcategory    │ Confidence │
├─────────────┼────────────────┼────────────┤
│ Content     │ Search Engine  │ 0.870      │
└─────────────┴────────────────┴────────────┘
```

## Performance Considerations

### Batch Size Optimization

- **Small batches (1-5)**: Lower latency, higher overhead
- **Medium batches (10-20)**: Optimal balance for most use cases
- **Large batches (50+)**: Higher throughput, more memory usage

### Parallel vs Sequential Processing

- **Parallel**: Better throughput, uses more resources
- **Sequential**: Lower resource usage, deterministic timing

### Memory Management

For large datasets, process in chunks:

```bash
# Split large CSV into smaller files
split -l 1000 large_dataset.csv chunk_

# Process each chunk
for chunk in chunk_*; do
  linneaus two-stage classify-batch --input $chunk --output results_$chunk.json
done
```

## Troubleshooting

### Common Issues

1. **OpenAI API Rate Limits**
   ```bash
   # Reduce batch size and enable retries
   linneaus two-stage classify-batch --batch-size 5 --input asns.csv --output results.json
   ```

2. **Memory Issues**
   ```bash
   # Use sequential processing for large batches
   linneaus two-stage classify-batch --sequential --batch-size 10 --input asns.csv --output results.json
   ```

3. **Missing Data**
   ```bash
   # Check data availability
   linneaus data status

   # Download missing data
   linneaus data download --force-refresh
   ```

### Error Handling

The two-stage pipeline includes comprehensive error handling:

- Automatic retries for transient failures
- Graceful degradation when components fail
- Detailed error logging and recovery statistics
- Fallback predictions for problematic inputs

### Getting Help

```bash
# General help
linneaus --help

# Command group help
linneaus two-stage --help

# Specific command help
linneaus two-stage classify-single --help

# Show two-stage features overview
linneaus two-stage help-commands
```

## Examples

### Complete Workflow

```bash
# 1. Check data availability
linneaus data status

# 2. Download data if needed
linneaus data download

# 3. Classify organizations
linneaus two-stage classify-batch \
  --input test_organizations.csv \
  --output predictions.json \
  --include-timing

# 4. Evaluate against ground truth
linneaus two-stage evaluate \
  --predictions predictions.json \
  --ground-truth ground_truth.csv \
  --output evaluation_report.txt

# 5. Benchmark performance
linneaus two-stage benchmark \
  --test-dataset test_organizations.csv \
  --output benchmark_results.json
```

### Integration with Scripts

```bash
#!/bin/bash

# Automated classification pipeline
INPUT_FILE="$1"
OUTPUT_DIR="results/$(date +%Y%m%d)"

mkdir -p "$OUTPUT_DIR"

echo "Starting AS classification pipeline..."

# Run classification
linneaus two-stage classify-batch \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_DIR/predictions.json" \
  --format json \
  --include-timing \
  --batch-size 20

echo "Classification completed. Results saved to $OUTPUT_DIR/"

# Generate CSV for analysis
linneaus two-stage classify-batch \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_DIR/predictions.csv" \
  --format csv \
  --batch-size 20

echo "Pipeline completed successfully!"
```

This CLI provides a comprehensive interface to the two-stage hierarchical AS classification system with robust error handling, multiple output formats, and extensive configuration options.
