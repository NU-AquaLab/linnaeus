# Taxonomy definitions have moved

The taxonomy definition files now live inside the package so they ship with
the wheel and are importable from any working directory:

```
src/linnaeus/resources/taxonomies/linnaeus.json   # default Linnaeus taxonomy (20 top-level categories)
src/linnaeus/resources/taxonomies/asdb.json       # Stanford ASdb reference taxonomy
src/linnaeus/resources/taxonomies/isic.json       # ISIC Rev.4 reference taxonomy
src/linnaeus/resources/prompts.yaml               # editable LLM prompt templates
```

- Edit those JSON files to change category definitions (they drive both the
  LLM system prompt and the structured-output schema).
- Use `--taxonomy linnaeus|asdb|isic` on supported CLI commands, or
  `--taxonomy-file path/to/custom.json` for a custom taxonomy.
- See `docs/custom_taxonomies.md` for the file format and a worked example.

Release snapshots of these files are published under
`data/released/<YYYYMM>/` by `scripts/release_benchmark_labels.py`.
