"""Tests for taxonomy loading and prompt/schema generation."""

import pytest

from linnaeus.data.alignment import CANONICAL_TOPLEVEL
from linnaeus.models.llm.schema_generation import (
    BUILTIN_TAXONOMIES,
    generate_developer_instructions,
    generate_schema,
    get_builtin_taxonomy_path,
    load_prompt_templates,
    load_tags_descriptions,
)


class TestLoadTagsDescriptions:
    def test_default_matches_canonical_toplevel(self):
        """The bundled linnaeus taxonomy keys must equal the canonical 20 names."""
        descriptions = load_tags_descriptions()
        assert list(descriptions.keys()) == CANONICAL_TOPLEVEL

    @pytest.mark.parametrize("name", BUILTIN_TAXONOMIES)
    def test_builtin_taxonomies_load(self, name):
        descriptions = load_tags_descriptions(name)
        assert isinstance(descriptions, dict)
        assert len(descriptions) > 0
        for value in descriptions.values():
            assert isinstance(value, (str, dict))

    def test_builtin_path_exists(self):
        for name in BUILTIN_TAXONOMIES:
            assert get_builtin_taxonomy_path(name).exists()

    def test_unknown_builtin_raises(self):
        with pytest.raises(ValueError, match="Unknown builtin taxonomy"):
            get_builtin_taxonomy_path("nope")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_tags_descriptions(tmp_path / "missing.json")

    def test_explicit_path(self, tmp_path):
        custom = tmp_path / "custom.json"
        custom.write_text('{"Foo": "A foo network.", "Bar": {"Baz": "A baz."}}')
        descriptions = load_tags_descriptions(custom)
        assert set(descriptions.keys()) == {"Foo", "Bar"}


class TestPromptGeneration:
    def test_templates_load(self):
        templates = load_prompt_templates()
        assert "toplevel_instructions" in templates
        assert "sublevel_instructions" in templates
        assert "{categories_block}" in templates["toplevel_instructions"]
        assert "{category}" in templates["sublevel_instructions"]
        assert "{tags_block}" in templates["sublevel_instructions"]

    def test_toplevel_prompt_snapshot(self):
        """The rendered top-level prompt keeps its established structure."""
        prompt = generate_developer_instructions(load_tags_descriptions())
        assert prompt.startswith(
            "You are an expert at classifying Autonomous Systems (AS) on the Internet."
        )
        assert prompt.endswith(
            "Respond with a list of all applicable tags for the given organization."
        )
        # every canonical category is described in the prompt
        for category in CANONICAL_TOPLEVEL:
            assert f"- {category}:" in prompt or f"- {category}: " in prompt

    def test_sublevel_prompt_snapshot(self):
        prompt = generate_developer_instructions(
            load_tags_descriptions(), category="Access"
        )
        assert prompt.startswith(
            "You are a specialized classification agent focused on "
            "analyzing Access organizations."
        )
        assert "Large ISP:" in prompt
        assert "Small ISP:" in prompt

    def test_sublevel_unknown_category_raises(self):
        with pytest.raises(KeyError):
            generate_developer_instructions(load_tags_descriptions(), category="Nope")


class TestSchemaGeneration:
    def test_schema_from_canonical_tags(self):
        schema = generate_schema(load_tags_descriptions(), tags=CANONICAL_TOPLEVEL)
        assert set(schema.Tags.__members__.keys()) == set(CANONICAL_TOPLEVEL)

    def test_schema_from_custom_taxonomy(self, tmp_path):
        custom = tmp_path / "custom.json"
        custom.write_text('{"Foo": "A foo.", "Bar": "A bar."}')
        descriptions = load_tags_descriptions(custom)
        schema = generate_schema(descriptions, tags=list(descriptions.keys()))
        assert set(schema.Tags.__members__.keys()) == {"Foo", "Bar"}
