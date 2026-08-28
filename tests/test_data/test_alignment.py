"""Tests for linnaeus.data.alignment module."""

import json
from pathlib import Path

import pandas as pd
import pytest

from linnaeus.data.alignment import (
    CANONICAL_SUBLEVEL,
    CANONICAL_TOPLEVEL,
    COLUMNS_TO_DROP,
    CURRENT_RELEASE,
    LEGACY_TO_CANONICAL,
    AlignedDataset,
    ColumnNormalizer,
)

# ===================================================================
# ColumnNormalizer.normalize_asn
# ===================================================================


class TestNormalizeAsn:
    """Tests for ColumnNormalizer.normalize_asn."""

    def test_int_passthrough(self):
        assert ColumnNormalizer.normalize_asn(7922) == 7922

    def test_float_truncation(self):
        assert ColumnNormalizer.normalize_asn(7922.0) == 7922

    def test_string_strip_as_prefix(self):
        assert ColumnNormalizer.normalize_asn("AS7922") == 7922

    def test_string_lowercase_as_prefix(self):
        assert ColumnNormalizer.normalize_asn("as7922") == 7922

    def test_string_plain_number(self):
        assert ColumnNormalizer.normalize_asn("7922") == 7922

    def test_string_with_whitespace(self):
        assert ColumnNormalizer.normalize_asn("  AS7922  ") == 7922

    def test_zero(self):
        assert ColumnNormalizer.normalize_asn(0) == 0

    def test_large_asn(self):
        assert ColumnNormalizer.normalize_asn("AS4294967295") == 4294967295


# ===================================================================
# ColumnNormalizer.normalize_asn_column
# ===================================================================


class TestNormalizeAsnColumn:
    """Tests for ColumnNormalizer.normalize_asn_column."""

    def test_renames_uppercase_asn(self):
        df = pd.DataFrame({"ASN": [7922, 15169], "x": [1, 2]})
        result = ColumnNormalizer.normalize_asn_column(df)
        assert "asn" in result.columns
        assert "ASN" not in result.columns
        assert list(result["asn"]) == [7922, 15169]

    def test_strips_as_prefix(self):
        df = pd.DataFrame({"asn": ["AS7922", "AS15169"], "x": [1, 2]})
        result = ColumnNormalizer.normalize_asn_column(df)
        assert list(result["asn"]) == [7922, 15169]

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"asn": ["AS7922"]})
        result = ColumnNormalizer.normalize_asn_column(df)
        assert df["asn"].iloc[0] == "AS7922"
        assert result["asn"].iloc[0] == 7922

    def test_mixed_formats(self):
        df = pd.DataFrame({"asn": ["AS100", "200", 300]})
        result = ColumnNormalizer.normalize_asn_column(df)
        assert list(result["asn"]) == [100, 200, 300]


# ===================================================================
# ColumnNormalizer.normalize_category_columns
# ===================================================================


class TestNormalizeCategoryColumns:
    """Tests for ColumnNormalizer.normalize_category_columns."""

    def test_financial_to_finance(self):
        df = pd.DataFrame({"asn": [1], "Financial": [1], "Access": [0]})
        result = ColumnNormalizer.normalize_category_columns(df)
        assert "Finance" in result.columns
        assert "Financial" not in result.columns

    def test_sublevel_renames(self):
        df = pd.DataFrame(
            {
                "asn": [1],
                "Financial_Bank": [1],
                "Enterprise_E_commerce": [1],
                "Enterprise_Industrial_Manufacturing": [0],
                "Government_National": [1],
            }
        )
        result = ColumnNormalizer.normalize_category_columns(df)
        assert "Finance_Bank" in result.columns
        assert "Enterprise_Ecommerce" in result.columns
        assert "Enterprise_IndustrialManufacturing" in result.columns
        assert "Government_FederalNational" in result.columns
        # Legacy names gone
        assert "Financial_Bank" not in result.columns
        assert "Enterprise_E_commerce" not in result.columns

    def test_drops_government_soe(self):
        df = pd.DataFrame({"asn": [1], "Government_SOE": [1], "IXP": [0]})
        result = ColumnNormalizer.normalize_category_columns(df)
        assert "Government_SOE" not in result.columns
        assert "IXP" in result.columns

    def test_passthrough_correct_names(self):
        df = pd.DataFrame({"asn": [1], "Finance": [1], "Access": [0]})
        result = ColumnNormalizer.normalize_category_columns(df)
        assert "Finance" in result.columns
        assert "Access" in result.columns

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"asn": [1], "Financial": [1]})
        result = ColumnNormalizer.normalize_category_columns(df)
        assert "Financial" in df.columns
        assert "Finance" in result.columns


# ===================================================================
# ColumnNormalizer.ensure_columns
# ===================================================================


class TestEnsureColumns:
    """Tests for ColumnNormalizer.ensure_columns."""

    def test_adds_missing_columns(self):
        df = pd.DataFrame({"asn": [1], "Access": [1]})
        result = ColumnNormalizer.ensure_columns(df, ["Access", "Transit"], fill=0)
        assert "Transit" in result.columns
        assert result["Transit"].iloc[0] == 0

    def test_preserves_existing(self):
        df = pd.DataFrame({"asn": [1], "Access": [1], "Transit": [1]})
        result = ColumnNormalizer.ensure_columns(df, ["Access", "Transit"])
        assert result["Access"].iloc[0] == 1
        assert result["Transit"].iloc[0] == 1


# ===================================================================
# Canonical column lists
# ===================================================================


class TestCanonicalLists:
    """Tests for CANONICAL_TOPLEVEL and CANONICAL_SUBLEVEL."""

    def test_toplevel_has_20_categories(self):
        assert len(CANONICAL_TOPLEVEL) == 20

    def test_sublevel_has_48_categories(self):
        assert len(CANONICAL_SUBLEVEL) == 48

    def test_toplevel_no_duplicates(self):
        assert len(CANONICAL_TOPLEVEL) == len(set(CANONICAL_TOPLEVEL))

    def test_sublevel_no_duplicates(self):
        assert len(CANONICAL_SUBLEVEL) == len(set(CANONICAL_SUBLEVEL))

    def test_finance_not_financial_in_toplevel(self):
        assert "Finance" in CANONICAL_TOPLEVEL
        assert "Financial" not in CANONICAL_TOPLEVEL

    def test_finance_not_financial_in_sublevel(self):
        finance_cols = [c for c in CANONICAL_SUBLEVEL if c.startswith("Finance")]
        assert len(finance_cols) == 5
        financial_cols = [c for c in CANONICAL_SUBLEVEL if c.startswith("Financial")]
        assert len(financial_cols) == 0

    def test_vpns_and_community_present(self):
        assert "VPNs" in CANONICAL_TOPLEVEL
        assert "Community" in CANONICAL_TOPLEVEL

    def test_legacy_to_canonical_keys_not_in_canonical(self):
        """Legacy names in the rename map should not appear in canonical lists."""
        for legacy_name in LEGACY_TO_CANONICAL:
            assert legacy_name not in CANONICAL_TOPLEVEL
            assert legacy_name not in CANONICAL_SUBLEVEL

    def test_legacy_to_canonical_values_in_canonical(self):
        """Canonical names in the rename map should appear in canonical lists."""
        for canonical_name in LEGACY_TO_CANONICAL.values():
            assert (
                canonical_name in CANONICAL_TOPLEVEL
                or canonical_name in CANONICAL_SUBLEVEL
            )

    def test_columns_to_drop_not_in_canonical(self):
        for col in COLUMNS_TO_DROP:
            assert col not in CANONICAL_TOPLEVEL
            assert col not in CANONICAL_SUBLEVEL

    def test_current_release_is_string(self):
        assert isinstance(CURRENT_RELEASE, str)
        assert len(CURRENT_RELEASE) == 6  # YYYYMM format


# ===================================================================
# AlignedDataset
# ===================================================================


class TestAlignedDataset:
    """Tests for AlignedDataset."""

    @pytest.fixture
    def aligned_dir(self, tmp_path):
        """Create a minimal aligned data directory for testing."""
        d = tmp_path / "data"

        # labels/
        labels_dir = d / "labels"
        labels_dir.mkdir(parents=True)
        tl = pd.DataFrame(
            {
                "asn": [7922, 15169],
                **{cat: [0, 0] for cat in CANONICAL_TOPLEVEL},
            }
        )
        tl.loc[0, "Access"] = 1
        tl.loc[1, "ContentProvider"] = 1
        tl.to_parquet(labels_dir / "toplevel.parquet", index=False)

        sl = pd.DataFrame(
            {
                "asn": [7922, 15169],
                **{cat: [0, 0] for cat in CANONICAL_SUBLEVEL},
            }
        )
        sl.to_parquet(labels_dir / "sublevel.parquet", index=False)

        hier = pd.DataFrame(
            {
                "asn": [7922, 15169],
                "as_name": ["Comcast", "Google"],
                "top_level_category": ["Access", "ContentProvider"],
                "sub_category": ["LargeISP", "Cloud"],
            }
        )
        hier.to_parquet(labels_dir / "hierarchical.parquet", index=False)

        # features/
        feat_dir = d / "features"
        feat_dir.mkdir()
        ipinfo = pd.DataFrame({"asn": [7922, 15169], "name": ["Comcast", "Google"]})
        ipinfo.to_parquet(feat_dir / "ipinfo.parquet", index=False)

        # predictions/
        pred_tl = d / "predictions" / "toplevel"
        pred_tl.mkdir(parents=True)
        preds = pd.DataFrame(
            {
                "asn": [7922, 15169],
                **{cat: [0.0, 0.0] for cat in CANONICAL_TOPLEVEL},
            }
        )
        preds.to_parquet(pred_tl / "finetuned.parquet", index=False)

        # splits/
        splits_dir = d / "splits"
        splits_dir.mkdir()
        splits = pd.DataFrame(
            {
                "asn": [7922, 15169],
                "split": ["train", "val"],
            }
        )
        splits.to_parquet(splits_dir / "assignments.parquet", index=False)

        # metadata.json
        meta = {"schema_version": "aligned_v1", "toplevel_count": 20}
        with open(d / "metadata.json", "w") as f:
            json.dump(meta, f)

        return d

    def test_load_labels_toplevel(self, aligned_dir):
        ds = AlignedDataset(aligned_dir)
        df = ds.load_labels("toplevel")
        assert "asn" in df.columns
        assert len(df) == 2
        assert pd.api.types.is_integer_dtype(df["asn"])
        cat_cols = [c for c in df.columns if c != "asn"]
        assert len(cat_cols) == 20

    def test_load_labels_hierarchical(self, aligned_dir):
        ds = AlignedDataset(aligned_dir)
        df = ds.load_labels("hierarchical")
        assert set(df.columns) == {
            "asn",
            "as_name",
            "top_level_category",
            "sub_category",
        }

    def test_load_features(self, aligned_dir):
        ds = AlignedDataset(aligned_dir)
        df = ds.load_features("ipinfo")
        assert len(df) == 2
        assert "asn" in df.columns

    def test_load_features_all(self, aligned_dir):
        ds = AlignedDataset(aligned_dir)
        df = ds.load_features("all")
        assert "asn" in df.columns
        assert len(df) == 2

    def test_load_predictions(self, aligned_dir):
        ds = AlignedDataset(aligned_dir)
        df = ds.load_predictions("toplevel", "finetuned")
        assert len(df) == 2
        assert "asn" in df.columns

    def test_load_splits(self, aligned_dir):
        ds = AlignedDataset(aligned_dir)
        df = ds.load_splits()
        assert set(df.columns) == {"asn", "split"}
        assert set(df["split"].unique()) == {"train", "val"}

    def test_get_split(self, aligned_dir):
        ds = AlignedDataset(aligned_dir)
        labels, asns = ds.get_split("train", "toplevel")
        assert len(labels) == 1
        assert labels["asn"].iloc[0] == 7922

    def test_validate_ok(self, aligned_dir):
        # Add test split to assignments so validation passes
        splits = pd.read_parquet(aligned_dir / "splits" / "assignments.parquet")
        extra = pd.DataFrame({"asn": [99999], "split": ["test"]})
        splits = pd.concat([splits, extra], ignore_index=True)
        splits.to_parquet(aligned_dir / "splits" / "assignments.parquet", index=False)

        ds = AlignedDataset(aligned_dir)
        errors = ds.validate()
        assert errors == []

    def test_validate_missing_dir(self, tmp_path):
        ds = AlignedDataset(tmp_path / "nonexistent")
        errors = ds.validate()
        assert len(errors) > 0
        assert any("Missing directory" in e for e in errors)

    def test_load_metadata(self, aligned_dir):
        ds = AlignedDataset(aligned_dir)
        meta = ds.load_metadata()
        assert meta["schema_version"] == "aligned_v1"

    def test_default_path_uses_current_release(self):
        ds = AlignedDataset()
        expected = Path("data") / "released" / CURRENT_RELEASE
        assert ds.data_dir == expected

    # ------------------------------------------------------------------
    # Benchmark labels (ASDB / ISIC)
    # ------------------------------------------------------------------

    def test_load_benchmark_labels_asdb(self, aligned_dir):
        """AlignedDataset.load_labels('asdb') works when asdb.parquet exists."""
        asdb_cols = ["asn", "computerAndInformationTechnology", "educationAndResearch"]
        asdb = pd.DataFrame(
            {"asn": [7922, 15169], **{c: [0, 1] for c in asdb_cols[1:]}}
        )
        asdb.to_parquet(aligned_dir / "labels" / "asdb.parquet", index=False)

        ds = AlignedDataset(aligned_dir)
        df = ds.load_labels("asdb")
        assert "asn" in df.columns
        assert len(df) == 2

    def test_load_benchmark_labels_isic(self, aligned_dir):
        """AlignedDataset.load_labels('isic') works when isic.parquet exists."""
        isic_cols = ["asn", "information_and_communication", "education"]
        isic = pd.DataFrame(
            {"asn": [7922, 15169], **{c: [1, 0] for c in isic_cols[1:]}}
        )
        isic.to_parquet(aligned_dir / "labels" / "isic.parquet", index=False)

        ds = AlignedDataset(aligned_dir)
        df = ds.load_labels("isic")
        assert "asn" in df.columns
        assert len(df) == 2

    def test_validate_with_benchmark_labels(self, aligned_dir):
        """validate() checks benchmark labels when present (no errors for well-formed)."""
        asdb = pd.DataFrame({"asn": [7922], "cat1": [1]})
        asdb.to_parquet(aligned_dir / "labels" / "asdb.parquet", index=False)

        # Add test split so core validation passes
        splits = pd.read_parquet(aligned_dir / "splits" / "assignments.parquet")
        extra = pd.DataFrame({"asn": [99999], "split": ["test"]})
        splits = pd.concat([splits, extra], ignore_index=True)
        splits.to_parquet(aligned_dir / "splits" / "assignments.parquet", index=False)

        ds = AlignedDataset(aligned_dir)
        errors = ds.validate()
        assert errors == []

    def test_validate_benchmark_labels_bad_asn(self, aligned_dir):
        """validate() flags benchmark labels with non-integer asn."""
        asdb = pd.DataFrame({"asn": ["AS7922"], "cat1": [1]})
        asdb.to_parquet(aligned_dir / "labels" / "asdb.parquet", index=False)

        ds = AlignedDataset(aligned_dir)
        errors = ds.validate()
        assert any("asdb.parquet" in e and "not integer" in e for e in errors)

    # ------------------------------------------------------------------
    # Category definitions
    # ------------------------------------------------------------------

    def test_load_category_definitions_from_release(self, aligned_dir):
        """load_category_definitions() reads from release dir when file exists."""
        defs = {"Cat1": "Description 1", "Cat2": "Description 2"}
        with open(aligned_dir / "linnaeus_definitions.json", "w") as f:
            json.dump(defs, f)

        ds = AlignedDataset(aligned_dir)
        result = ds.load_category_definitions("linnaeus")
        assert result == defs

    def test_load_category_definitions_fallback(self, aligned_dir):
        """load_category_definitions() falls back to package resource."""
        ds = AlignedDataset(aligned_dir)
        result = ds.load_category_definitions()
        # Should load the default linnaeus taxonomy from package resources
        assert "Access" in result
        assert "Transit" in result
