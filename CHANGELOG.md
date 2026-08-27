# Changelog

All notable changes to Linneaus will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-09

First public release: a complete refactor from research notebooks to a
production-ready Python package.

### Added
- `linneaus` Python package (`src/linneaus/`) with CLI, configuration, data
  access, and model subpackages; installable wheel with all resources bundled
- Two-stage hierarchical classification pipeline (top-level + sub-level) and
  direct LLM classification (`--approach llm-only`) with structured output
- Multi-taxonomy support: builtin `linneaus` (20 top-level categories),
  `asdb` (Stanford ASdb, 17), and `isic` (ISIC Rev.4, 20) taxonomies packaged
  under `linneaus/resources/taxonomies/`; `--taxonomy` / `--taxonomy-file`
  flags on `model predict`, `model evaluate`, `model fine-tune`, and
  `model prepare-data`; custom-taxonomy authoring guide
  (`docs/custom_taxonomies.md`)
- Editable LLM prompt templates in `linneaus/resources/prompts.yaml`
- Released data snapshot `data/released/202506/`: 1,978 labeled ASNs
  (top-level, sub-level, hierarchical), ASDB (1,978) and ISIC (2,063)
  benchmark labels (labels only), train/val splits, reference predictions and
  metrics, and 119,809-ASN complete predictions
- Supported fine-tuning workflow (`linneaus model fine-tune`,
  `linneaus model prepare-data`) built on the validated end-to-end path, with
  `--tags` to restrict training to a category subset
- End-to-end validation script `scripts/test_pipeline_e2e.py` with
  `--focus`/`--tags` restricted runs and subset-aware reference metrics
- Standalone classification script `scripts/classify.py` supporting all
  taxonomies and any OpenAI-compatible provider (`--base-url`)
- Flexible LLM provider/model selection via client factory
  (`LLM_API_KEY`/`LLM_BASE_URL`/`OPENAI_API_KEY` env vars, `--api-key`/
  `--base-url` CLI flags)
- Multi-source data integration (ASRank, PeeringDB, APNIC ASPOP, IPinfo)
- Scikit-learn compatible interface, Docker support, GitHub Actions CI, and a
  313-test suite

### Changed
- Canonicalized the linneaus taxonomy definitions to the 20 canonical
  top-level names used by the label files (`IXP`, `Finance`,
  `TvRadioCulturalAmenities`; added `VPNs` and `Community`)
- `model evaluate` now aligns predictions and labels on `asn` and restricts
  metrics to the selected taxonomy's columns
- Documented model performance with the measured validation metrics
  (top-level fine-tuned: 0.676 exact-match accuracy, 0.792 macro F1)

### Removed
- `linneaus model fine-tune-async` and the non-functional
  `LLMTrainingPipeline` / `LLMDataPreparation` classes (replaced by
  `linneaus.models.llm.finetuning`)
- Legacy notebook-based research layout, notebook data-processing scripts,
  and committed raw data (~2M lines)
