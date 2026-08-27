# LLM Fine-tuning Pipeline

## Overview

The LLM fine-tuning pipeline provides a complete workflow for training custom language models to classify Autonomous Systems (AS) using OpenAI's fine-tuning API. This pipeline supports both hierarchical (47 detailed tags) and flat (20 consolidated tags) classification approaches.

## Architecture

### Core Components

1. **Data Management** (`data_preparation.py`)
   - Converts labeled data to OpenAI fine-tuning format
   - Handles train/validation/test splits
   - Validates training data format

2. **Fine-tuning** (`fine_tuning.py`)
   - Manages OpenAI fine-tuning jobs
   - Handles file uploads and job monitoring
   - Tracks costs and usage metrics

3. **Inference** (`inference.py`)
   - Batch processing for classification
   - Probability extraction from logprobs
   - Result aggregation and formatting

4. **Training Pipeline** (`training.py`)
   - Orchestrates the complete workflow
   - Handles configuration management
   - Provides evaluation metrics

5. **Schemas** (`schemas.py`)
   - Pydantic models for all data structures
   - Type safety and validation
   - Legacy compatibility models

## Data Structure

### Input Format
The pipeline expects labeled data in Parquet format with the following structure:
```
data/
├── labeled/
│   ├── consolidated/
│   │   ├── labeled_data.parquet          # 20-tag format
│   │   └── labeled_data_hierarchical.parquet # 47-tag format
│   ├── splits/
│   │   ├── train.parquet                 # Training set (70%)
│   │   ├── validation.parquet            # Validation set (15%)
│   │   └── test.parquet                  # Test set (15%)
│   └── metadata.json                     # Dataset metadata
└── predictions/                          # Model outputs
    └── {model_name}_{timestamp}/
```

### Tag Hierarchies

#### Hierarchical Tags (47 tags)
- **Access**: Large ISP, Small ISP
- **Transit**: Global, Regional, Domestic
- **Government**: Federal/National, State/Province, City/County/Municipality, Legislative, Judiciary, Regulators, Agencies, Agencies Space, Agencies CentralBanks, Political Parties
- **Content Provider**: Cloud, Hosting, CDN
- **Educational Research**: University, Academic Backbone, Schools, Research Institutes
- And many more...

#### Flat Tags (20 tags)
- Access, Transit, Mobile, Satellite, Content Provider
- Educational Research, Government, IXP, DNS, Energy & Utility
- Enterprise, Finance, Law Enforcement, Health, Cooperatives
- TV/Radio and Cultural Amenities, Transportation, VPNs, Personal, Community

## Usage

### CLI Commands

#### Fine-tune a Model
```bash
# Complete fine-tuning pipeline (hierarchical)
linneaus model fine-tune --approach hierarchical --model-suffix "v1"

# Flat classification without waiting
linneaus model fine-tune --approach flat --no-wait

# Custom configuration
linneaus model fine-tune --approach hierarchical --config custom_config.yaml

# Async fine-tuning for improved performance
linneaus model fine-tune-async --approach hierarchical --model-suffix "v1-async"

# High-performance async processing
linneaus model fine-tune-async --approach flat --max-concurrent 10
```

#### Prepare Training Data Only
```bash
# Prepare data for hierarchical classification
linneaus model prepare-data --approach hierarchical

# Specify custom output directory
linneaus model prepare-data --approach flat --output-dir /path/to/output
```

#### Monitor Fine-tuning Jobs
```bash
# List all jobs
linneaus model list-jobs

# List only active jobs
linneaus model list-jobs --active-only

# Monitor specific job
linneaus model monitor-job ftjob-xxx
```

### Programmatic API

#### Basic Usage
```python
from pathlib import Path
from linneaus.models.llm.training import LLMTrainingPipeline

# Initialize pipeline
pipeline = LLMTrainingPipeline(
    config_path=Path("config.yaml"),
    data_dir=Path("data")
)

# Run complete pipeline
results = pipeline.run_complete_pipeline(
    approach="hierarchical",
    model_suffix="v1",
    wait_for_completion=True
)

print(f"Fine-tuned model: {results['fine_tuned_model']}")
print(f"Accuracy: {results['evaluation']['metrics']['accuracy']:.3f}")
```

#### Data Preparation Only
```python
from linneaus.models.llm.data_preparation import LLMDataPreparation
from linneaus.data.labeled_data import LabeledDataManager

# Initialize components
data_manager = LabeledDataManager(Path("data"))
prep = LLMDataPreparation(data_manager)

# Load labeled data
labeled_df = data_manager.load_consolidated_labels()

# Create prompt templates
templates = prep.create_prompt_templates("hierarchical")

# Prepare training data
training_examples = prep.prepare_training_data(
    labeled_df=labeled_df,
    approach="hierarchical",
    system_prompt=templates["system_prompt"],
    message_template=templates["message_template"]
)

# Save training files
train_path, val_path = prep.save_training_files(
    training_examples,
    output_dir=Path("training_data")
)
```

#### Fine-tuning Management
```python
import os
from openai import OpenAI
from linneaus.models.llm.fine_tuning import FineTuningManager

# Initialize
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
manager = FineTuningManager(client)

# Start fine-tuning
job_id = manager.start_fine_tuning_workflow(
    training_file_path=Path("training_data/hierarchical_training_train.jsonl"),
    validation_file_path=Path("training_data/hierarchical_training_validation.jsonl"),
    model="gpt-4o-mini-2024-07-18",
    suffix="hierarchical-v1"
)

# Monitor job
result = manager.monitor_job(job_id)
print(f"Fine-tuned model: {result['fine_tuned_model']}")
```

#### Batch Inference
```python
from linneaus.models.llm.inference import BatchInferenceProcessor
from linneaus.models.llm.schemas import BatchTaggingResponseHierarchical

# Initialize processor
processor = BatchInferenceProcessor(
    client=client,
    model="ft:gpt-4o-mini:your-org:hierarchical-v1",
    response_format=BatchTaggingResponseHierarchical,
    system_prompt="Your classification prompt...",
    batch_size=10
)

# Process organizations
predictions_df, probs_df = processor.process_batches(
    features_df=organizations_df,
    message_template="Classify ASN {asn}: {name}",
    output_columns=hierarchical_tag_names,
    extract_probs=True
)
```

## Configuration

### Config File Structure (`config.yaml`)
```yaml
llm:
  models:
    base_model: "gpt-4o-mini-2024-07-18"
    default_approach: "hierarchical"
    temperature: 0.0001

  fine_tuning:
    train_split: 0.7
    validation_split: 0.15
    test_split: 0.15
    random_state: 42
    max_concurrent_jobs: 3
    daily_limit: 10
    hyperparameters:
      n_epochs: "auto"
      batch_size: "auto"
      learning_rate_multiplier: "auto"

  inference:
    batch_size: 10
    max_retries: 3
    extract_probabilities: false

  prompts:
    system:
      hierarchical: |
        You are an expert in classifying Internet infrastructure...
      flat: |
        You are an expert in classifying Internet infrastructure...

    message_templates:
      basic: |
        Classify this Autonomous System:
        ASN: {asn}
        Organization Name: {name}
        Description: {description}
```

### Environment Variables
```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
IPINFO_TOKEN=...
FINE_TUNING_ENABLED=true
```

## Performance Metrics

### Model Performance (Production)
- **Hierarchical Model**: 75.4% accuracy, 0.701 F1-score
- **Flat Model**: 78.2% accuracy, 0.743 F1-score
- **Processing Speed**: ~10 organizations/batch, ~30 seconds/batch

### Cost Estimates (per 1000 examples)
- **Training**: $10-15 (depends on example length)
- **Inference**: $2-5 (depends on batch size)

## Best Practices

### Data Quality
1. **Consistent Labeling**: Ensure labels are consistent across the dataset
2. **Representative Sampling**: Include diverse organization types
3. **Quality Control**: Validate labels before training

### Model Training
1. **Start Small**: Begin with a small dataset to test the pipeline
2. **Monitor Costs**: Track OpenAI usage to avoid unexpected charges
3. **Version Control**: Use model suffixes to track different versions
4. **Validation**: Always use a held-out validation set

### Production Deployment
1. **Rate Limiting**: Implement proper rate limiting for API calls
2. **Error Handling**: Handle API failures gracefully
3. **Monitoring**: Track model performance over time
4. **Backup Models**: Keep multiple model versions available

## Troubleshooting

### Common Issues

#### "OpenAI API key not found"
```bash
export OPENAI_API_KEY=sk-your-key-here
```

#### "Training data validation failed"
- Check that your CSV has the required columns
- Ensure ASN values are positive integers
- Verify labels are binary (0 or 1)

#### "Fine-tuning job failed"
- Check OpenAI account limits
- Verify training data format
- Review job logs using `linneaus model monitor-job`

#### "Prediction errors"
- Ensure the model exists and is accessible
- Check batch size (reduce if hitting rate limits)
- Verify input data format

### Performance Optimization

#### Reduce Training Time
- Use smaller datasets for initial testing
- Optimize hyperparameters (epochs, batch size)
- Use validation early stopping

#### Improve Accuracy
- Add more diverse training examples
- Use hierarchical approach for detailed classification
- Implement ensemble methods
- Fine-tune prompts and examples

#### Scale Inference
- Increase batch size (up to API limits)
- Implement async processing
- Cache frequent predictions
- Use multiple models for load balancing

## Development

### Testing
```bash
# Run basic data preparation test
python -m pytest tests/test_data_preparation.py

# Test fine-tuning pipeline (requires API key)
python -m pytest tests/test_fine_tuning.py -k "not test_actual_api"

# Full integration test
python -m pytest tests/test_llm_integration.py
```

### Contributing
1. Follow numpy-style docstrings
2. Add type hints to all functions
3. Include error handling and logging
4. Write comprehensive tests
5. Update documentation

### Future Enhancements
- [ ] Async batch processing
- [ ] Multi-model ensemble
- [ ] Real-time inference API
- [ ] Model versioning system
- [ ] A/B testing framework
- [ ] Cost optimization
- [ ] Custom embedding integration
