# Two-Stage Pipeline Quick Reference

## Installation & Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your OpenAI API key
```

## Basic Usage

### Single Prediction
```python
from linnaeus.models.two_stage import TwoStageHierarchicalPipeline

pipeline = TwoStageHierarchicalPipeline()
result = pipeline.predict_single(asn=15169, organization_name="Google LLC")

print(f"Categories: {[p.category.value for p in result.stage1_predictions]}")
print(f"Confidence: {result.overall_confidence:.3f}")
```

### Batch Prediction
```python
organizations = [
    {"asn": 15169, "organization_name": "Google LLC"},
    {"asn": 7922, "organization_name": "Comcast"}
]

batch_result = pipeline.predict_batch(organizations, parallel=True)
print(f"Processed: {len(batch_result.results)} organizations")
```

## Configuration

### Quick Config Changes
```python
from linnaeus.config import load_config

config = load_config()
config.two_stage_pipeline.stage1.svm_weight = 0.3
config.two_stage_pipeline.stage1.llm_weight = 0.7

pipeline = TwoStageHierarchicalPipeline(config=config)
```

### Environment Variables
```bash
# .env file
OPENAI_API_KEY=your_key_here
LOG_LEVEL=INFO
```

## Error Handling

### Custom Error Handler
```python
from linnaeus.models.two_stage.error_handling import ErrorHandler, FallbackStrategy

error_handler = ErrorHandler(
    enable_graceful_degradation=True,
    max_retry_attempts=3,
    fallback_strategy=FallbackStrategy.DEFAULT
)

pipeline = TwoStageHierarchicalPipeline(error_handler=error_handler)
```

## Evaluation

### Quick Evaluation
```python
from linnaeus.models.two_stage.evaluation import HierarchicalEvaluator

evaluator = HierarchicalEvaluator()
results = evaluator.evaluate(predictions, ground_truth_df)

print(f"Stage 1 Accuracy: {results['stage1']['accuracy']:.3f}")
print(f"Stage 2 Accuracy: {results['stage2']['overall_accuracy']:.3f}")
```

## Common Patterns

### Check for Errors
```python
error_summary = error_handler.get_error_summary()
print(f"Total errors: {error_summary['total_errors']}")
```

### Generate Report
```python
report = evaluator.create_performance_report(results, "report.txt")
```

### Benchmark Performance
```python
import time

start = time.time()
result = pipeline.predict_batch(organizations)
elapsed = time.time() - start

print(f"Throughput: {len(result.results) / elapsed:.1f} pred/sec")
```

## Troubleshooting

### Rate Limits
```python
# Reduce batch size
config.two_stage_pipeline.stage1.parallel_batch_size = 5
```

### Memory Issues
```python
# Process in chunks
chunk_size = 100
for i in range(0, len(large_list), chunk_size):
    chunk = large_list[i:i+chunk_size]
    results = pipeline.predict_batch(chunk)
```

### Missing Data
```python
# Check if ASN exists in data
from linnaeus.data.access import DataAccessLayer
data_access = DataAccessLayer()
org_data = data_access.get_organization_data(asn)
```

## Key Classes

- `TwoStageHierarchicalPipeline` - Main pipeline
- `AssembledTopLevelClassifier` - Stage 1 (LLM + SVM)
- `HierarchicalSublevelClassifier` - Stage 2 (LLM only)
- `HierarchicalEvaluator` - Evaluation framework
- `ErrorHandler` - Error management
- `DataSplitManager` - Data management

## Configuration Files

- `config.yaml` - Main configuration
- `.env` - Environment variables (API keys)
- `data/labeled_data/` - Training data location
