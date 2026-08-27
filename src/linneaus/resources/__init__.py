"""
Package resources for Linneaus.

This module contains static data files and resources used throughout
the Linneaus package, including:

- ``country_rir_registry.json``: Mapping of countries to their Regional
  Internet Registries (RIRs).
- ``taxonomies/``: Taxonomy definition files (``linneaus.json``,
  ``asdb.json``, ``isic.json``) mapping category names to descriptions,
  used for generating LLM prompts and schemas. Edit these files (or pass
  a custom one via ``--taxonomy-file``) to change category definitions.
- ``prompts.yaml``: Editable LLM prompt templates used for classification.
"""
