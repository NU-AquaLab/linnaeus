# Student Guide: AS Classification

## Prerequisites

- Python >= 3.9
- An OpenAI API key (get one at <https://platform.openai.com/api-keys>)
- [UV](https://docs.astral.sh/uv/) package manager (recommended) **or** pip

## 1. Clone the repository

```bash
git clone --branch refactor https://github.com/TheMarcosP/Proyecto-Clasificar-Sistemas-Autonomos.git
cd Proyecto-Clasificar-Sistemas-Autonomos
```

## 2. Install

```bash
uv pip install -e ".[dev]"
```

## 3. Set your API key

```bash
export OPENAI_API_KEY="sk-..."
```

Or pass it on the command line with `--api-key`.

## 4. Feature data

Your instructor will provide a zip file (`linneaus_features.zip`).  Unzip it
into the `data/local/` directory:

```bash
unzip linneaus_features.zip -d data/local/
```

This creates `data/local/features/` containing:

| File | Description |
|---|---|
| `llm_input.csv` | ~119K ASNs with columns: `asn`, `name`, `website`, `country`. **Use this as input for classification.** |
| `ipinfo.csv` | IP geolocation and ISP data from IPinfo |
| `peeringdb.csv` | Network operator data from PeeringDB |

Each CSV also has a `.parquet` counterpart for faster loading in notebooks.

## 5. Run classification

Use `llm_input.csv` (or any CSV with an `asn` column) as input:

```bash
python scripts/classify.py \
  --input data/local/features/llm_input.csv \
  --output my_results.csv
```

The script sends ASN metadata to the LLM in batches and writes a binary
classification CSV.

A small sample file is also included at `data/released/202506/sample_input.csv`
for quick testing (5 ASNs, no API key cost concerns).

### Input format

The input CSV must have an `asn` column. Additional columns improve accuracy:

| Column | Required? | Description |
|---|---|---|
| `asn` | Yes | Autonomous System Number |
| `name` (or `organization_name`) | No | Organization name (e.g., "Google LLC") |
| `website` | No | Organization website (e.g., "google.com") |
| `country` | No | Country name (e.g., "United States") |

### CLI options

| Flag | Description |
|---|---|
| `--input`, `-i` | Input CSV (required) |
| `--output`, `-o` | Output CSV (required) |
| `--model`, `-m` | Model name (default: `gpt-4o-mini`) |
| `--batch-size`, `-b` | ASNs per API call (default: 5) |
| `--taxonomy`, `-t` | Builtin taxonomy name (`linneaus`, `asdb`, `isic`) or path to a taxonomy definitions JSON (default: built-in linneaus) |
| `--api-key` | API key (overrides `OPENAI_API_KEY` env var) |
| `--base-url` | Base URL for OpenAI-compatible API |
| `--verbose`, `-v` | Enable debug logging |

## 6. Taxonomies

Three classification taxonomies are available.  Each defines a different set of
categories and has its own ground-truth labels.

### Linneaus (default)

20 top-level categories designed for Internet infrastructure (Access, Transit,
Mobile, Content Provider, etc.).

- Definitions: `data/released/202506/linneaus_definitions.json`
- Ground truth: `data/released/202506/labels/toplevel.csv`

This is the default taxonomy — no `--taxonomy` flag needed.

### ASDB

17 categories based on the Stanford ASdb classification scheme (education,
government, finance, health, etc.).

- Definitions: `data/released/202506/asdb_definitions.json`
- Ground truth: `data/released/202506/labels/asdb.csv`

```bash
python scripts/classify.py \
  --input data/local/features/llm_input.csv \
  --output results_asdb.csv \
  --taxonomy asdb
```

### ISIC

20 categories following the International Standard Industrial Classification
(manufacturing, finance, public administration, etc.).

- Definitions: `data/released/202506/isic_definitions.json`
- Ground truth: `data/released/202506/labels/isic.csv`

```bash
python scripts/classify.py \
  --input data/local/features/llm_input.csv \
  --output results_isic.csv \
  --taxonomy isic
```

## 7. Output format

The output CSV has one row per ASN with binary (0/1) columns for each category
in the selected taxonomy:

```
asn,Access,Transit,Mobile,Satellite,ContentProvider,...,Community
15169,0,0,0,0,1,...,0
```

The column names match the corresponding ground-truth labels file exactly.

## 8. Ground truth and evaluation

Ground-truth labels are in `data/released/202506/labels/`:

| File | Taxonomy | Categories |
|---|---|---|
| `toplevel.csv` | Linneaus | 20 |
| `asdb.csv` | ASDB | 17 |
| `isic.csv` | ISIC | 20 |

Train/test split assignments are in `data/released/202506/splits/assignments.csv`
(columns: `asn`, `split` with values `train` or `test`).

You can compare your predictions against the ground truth using standard
multi-label metrics (accuracy, precision, recall, F1, Jaccard).

## 9. Submit your results

Submit the output CSV to the course platform as instructed.  Make sure:

- The `asn` column matches the assignment input exactly.
- All category columns are present and contain only 0 or 1.
- The column names match the ground-truth file for the taxonomy you used.
