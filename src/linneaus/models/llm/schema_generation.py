"""
Schema generation and developer instructions for LLM-based AS classification.

This module provides functions to dynamically generate:
- Developer instructions (system prompts) from tag descriptions
- Pydantic response schemas for OpenAI structured output
- Tag description loading from the bundled resource file

These functions support both top-level and sub-level classification workflows,
replacing the notebook-embedded logic from the reference implementation.
"""

import json
import logging
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import List, Optional, Type, Union

from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)

#: Names of the taxonomies bundled with the package under
#: ``linnaeus/resources/taxonomies/``.
BUILTIN_TAXONOMIES = ("linnaeus", "asdb", "isic")


def get_builtin_taxonomy_path(name: str) -> Path:
    """
    Return the filesystem path of a bundled taxonomy definitions file.

    Parameters
    ----------
    name : str
        One of :data:`BUILTIN_TAXONOMIES`.

    Raises
    ------
    ValueError
        If *name* is not a builtin taxonomy.
    """
    if name not in BUILTIN_TAXONOMIES:
        raise ValueError(
            f"Unknown builtin taxonomy '{name}'. "
            f"Available: {', '.join(BUILTIN_TAXONOMIES)}"
        )
    return Path(
        str(resources.files("linnaeus.resources") / "taxonomies" / f"{name}.json")
    )


def load_tags_descriptions(path: Optional[Union[str, Path]] = None) -> dict:
    """
    Load taxonomy tag descriptions from a JSON file.

    *path* may be a builtin taxonomy name (``"linnaeus"``, ``"asdb"``,
    ``"isic"``) or a path to a custom taxonomy JSON file.  If ``None``,
    the default linnaeus taxonomy bundled with the package is used.

    Parameters
    ----------
    path : str or Path, optional
        Builtin taxonomy name or explicit path to a taxonomy definitions
        JSON file.  If ``None``, the default linnaeus taxonomy is loaded
        from the package resources.

    Returns
    -------
    dict
        Dictionary mapping tag names to either a description string
        (for tags without subcategories) or a dict of subcategory names
        to description strings.

    Raises
    ------
    FileNotFoundError
        If the resolved file does not exist.
    """
    if path is None:
        tags_path = get_builtin_taxonomy_path("linnaeus")
    elif isinstance(path, str) and path in BUILTIN_TAXONOMIES:
        tags_path = get_builtin_taxonomy_path(path)
    else:
        tags_path = Path(path)

    if not tags_path.exists():
        raise FileNotFoundError(f"Taxonomy definitions not found at {tags_path}.")

    with open(tags_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_toplevel_tag_names(tags_descriptions: dict) -> List[str]:
    """
    Return the sanitized top-level category names of a taxonomy, in order.

    Spaces and ``/`` in category names are replaced with underscores so the
    names are safe to use as schema enum members and output column names.
    """
    return [k.replace(" ", "_").replace("/", "_") for k in tags_descriptions]


def filter_taxonomy(tags_descriptions: dict, tags: List[str]) -> dict:
    """
    Restrict a taxonomy dict to the categories named in *tags*.

    Matches on sanitized top-level names. Useful for building a system
    prompt that only describes the categories a restricted schema can emit.

    Raises
    ------
    ValueError
        If any name in *tags* does not match a taxonomy category.
    """
    by_safe_name = {
        key.replace(" ", "_").replace("/", "_"): key for key in tags_descriptions
    }
    unknown = [t for t in tags if t not in by_safe_name]
    if unknown:
        raise ValueError(
            f"Unknown categories {unknown}. " f"Available: {sorted(by_safe_name)}"
        )
    return {by_safe_name[t]: tags_descriptions[by_safe_name[t]] for t in tags}


def generate_developer_instructions(
    tags_descriptions: dict,
    category: Optional[str] = None,
) -> str:
    """
    Generate a system prompt / developer instructions string from tag descriptions.

    When ``category`` is None, this generates instructions for top-level
    classification across all categories. When ``category`` is specified,
    it generates instructions focused on the sub-level tags within that
    single category.

    Parameters
    ----------
    tags_descriptions : dict
        Tag descriptions dictionary as loaded by :func:`load_tags_descriptions`.
        Keys are top-level category names; values are either a description
        string or a dict mapping subcategory names to descriptions.
    category : str, optional
        If provided, generate instructions only for this category's
        subcategories. If None, generate instructions for all categories.

    Returns
    -------
    str
        Formatted developer instructions string suitable for use as a
        system prompt in LLM classification requests.

    Examples
    --------
    >>> descriptions = load_tags_descriptions()
    >>> instructions = generate_developer_instructions(descriptions)
    >>> print(instructions[:60])
    You are an expert at classifying Autonomous Systems (AS) on
    """
    if category is not None:
        return _generate_sublevel_instructions(tags_descriptions, category)

    return _generate_toplevel_instructions(tags_descriptions)


def load_prompt_templates() -> dict:
    """
    Load the LLM prompt templates bundled with the package.

    The templates live in ``linnaeus/resources/prompts.yaml`` and can be
    edited there to customize the system prompts without touching code.

    Returns
    -------
    dict
        Mapping with keys ``toplevel_instructions`` and
        ``sublevel_instructions``.
    """
    import yaml

    prompts_path = Path(str(resources.files("linnaeus.resources") / "prompts.yaml"))
    with open(prompts_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _generate_toplevel_instructions(tags_descriptions: dict) -> str:
    """Generate top-level classification instructions covering all categories."""
    lines: List[str] = []

    for tag_name, tag_value in tags_descriptions.items():
        if isinstance(tag_value, str):
            # Simple tag without subcategories
            lines.append(f"- {tag_name}: {tag_value}")
        elif isinstance(tag_value, dict):
            # Tag with subcategories
            lines.append(f"- {tag_name}:")
            for subtag_name, subtag_desc in tag_value.items():
                lines.append(f"  - {subtag_name}: {subtag_desc}")

    categories_text = "\n".join(lines)

    template = load_prompt_templates()["toplevel_instructions"]
    return template.replace("{categories_block}", categories_text)


def _generate_sublevel_instructions(tags_descriptions: dict, category: str) -> str:
    """Generate sub-level classification instructions for a single category."""
    if category not in tags_descriptions:
        raise KeyError(
            f"Category '{category}' not found in tags_descriptions. "
            f"Available categories: {list(tags_descriptions.keys())}"
        )

    tag_value = tags_descriptions[category]

    if isinstance(tag_value, str):
        raise ValueError(
            f"Category '{category}' does not have subcategories. "
            "Sub-level instructions require a category with subcategory entries."
        )

    tag_lines: List[str] = []
    for subtag_name, subtag_desc in tag_value.items():
        tag_lines.append(f"{subtag_name}: {subtag_desc}")

    tags_text = "\n".join(tag_lines)

    template = load_prompt_templates()["sublevel_instructions"]
    return template.replace("{category}", category).replace("{tags_block}", tags_text)


def generate_schema(
    tags_descriptions: dict,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Type[BaseModel]:
    """
    Generate a Pydantic response model for OpenAI structured output.

    This creates a dynamic Pydantic model with an inner Tags enum representing
    the valid classification values. The generated model follows the
    ``BatchTaggingResponse`` pattern used by the OpenAI structured output API.

    Parameters
    ----------
    tags_descriptions : dict
        Tag descriptions dictionary as loaded by :func:`load_tags_descriptions`.
    category : str, optional
        If provided, generate a schema only for this category's subcategories.
        The tag names will be prefixed with ``"{category}_"``
        (e.g. ``"Access_LargeISP"``).
        If None and ``tags`` is also None, generates a schema for all flat
        top-level tag names.
    tags : list of str, optional
        Explicit list of tag names to use. If provided, ``category`` is used
        only as a label for the schema class name. This allows full control
        over which tags appear in the enum.

    Returns
    -------
    Type[BaseModel]
        A dynamically created Pydantic model class with the following attributes:

        - ``Tags``: An Enum class with the valid tag values.
        - ``responses``: A list of inner response models, each containing a
          ``tags`` field with ``List[Tags]``.

    Examples
    --------
    >>> descriptions = load_tags_descriptions()
    >>> schema = generate_schema(descriptions, category="Access")
    >>> print(schema.Tags.__members__.keys())
    dict_keys(['Access_Large_ISP', 'Access_Small_ISP'])

    >>> schema = generate_schema(descriptions)
    >>> # Returns a schema with all flat top-level tags
    """
    if tags is not None:
        tag_names = tags
        schema_label = category or "All"
    elif category is not None:
        tag_names = _get_category_tag_names(tags_descriptions, category)
        schema_label = category
    else:
        tag_names = _get_flat_tag_names(tags_descriptions)
        schema_label = "All"

    return _build_batch_response_model(schema_label, tag_names)


def get_flat_tag_names(tags_descriptions: dict) -> List[str]:
    """
    Extract flat tag names from the tags_descriptions dictionary.

    For categories without subcategories, the tag name is the category name
    itself. For categories with subcategories, the tag names are formed by
    joining the category and subcategory with an underscore, replacing spaces
    with underscores (e.g. ``"Access_Large_ISP"``).

    Parameters
    ----------
    tags_descriptions : dict
        Tag descriptions dictionary as loaded by :func:`load_tags_descriptions`.

    Returns
    -------
    list of str
        Sorted list of flat tag name strings.
    """
    return _get_flat_tag_names(tags_descriptions)


def _get_flat_tag_names(tags_descriptions: dict) -> List[str]:
    """Extract all flat tag names from the descriptions dictionary."""
    tag_names: List[str] = []

    for tag_name, tag_value in tags_descriptions.items():
        if isinstance(tag_value, str):
            # Simple tag: use the category name directly
            safe_name = tag_name.replace(" ", "_").replace("/", "_")
            tag_names.append(safe_name)
        elif isinstance(tag_value, dict):
            # Category with subcategories
            safe_category = tag_name.replace(" ", "_").replace("/", "_")
            for subtag_name in tag_value:
                safe_subtag = subtag_name.replace(" ", "_").replace("/", "_")
                tag_names.append(f"{safe_category}_{safe_subtag}")

    return sorted(tag_names)


def _get_category_tag_names(tags_descriptions: dict, category: str) -> List[str]:
    """Extract tag names for a specific category, prefixed with the category name."""
    if category not in tags_descriptions:
        raise KeyError(
            f"Category '{category}' not found in tags_descriptions. "
            f"Available categories: {list(tags_descriptions.keys())}"
        )

    tag_value = tags_descriptions[category]

    if isinstance(tag_value, str):
        # Single tag, no subcategories
        safe_name = category.replace(" ", "_").replace("/", "_")
        return [safe_name]

    tag_names: List[str] = []
    safe_category = category.replace(" ", "_").replace("/", "_")
    for subtag_name in tag_value:
        safe_subtag = subtag_name.replace(" ", "_").replace("/", "_")
        tag_names.append(f"{safe_category}_{safe_subtag}")

    return sorted(tag_names)


def _build_batch_response_model(label: str, tag_names: List[str]) -> Type[BaseModel]:
    """
    Build a BatchCategoryTagsResponse Pydantic model with a Tags enum.

    Parameters
    ----------
    label : str
        Label for the schema (used in class names).
    tag_names : list of str
        List of tag name strings to use as enum members.

    Returns
    -------
    Type[BaseModel]
        Dynamically created Pydantic model.
    """
    # Create enum members: name -> display value (spaces instead of underscores)
    enum_members = {tag: tag.replace("_", " ") for tag in tag_names}
    tags_enum = Enum("Tags", enum_members, type=str)  # type: ignore[misc]
    tags_enum.__doc__ = f"Enum class for {label} classification tags"

    # Inner response model: one organization's tags
    category_response = create_model(
        f"{label}TagsResponse",
        tags=(
            List[tags_enum],  # type: ignore[valid-type]
            Field(..., description=f"List of {label} classification tags"),
        ),
    )
    category_response.__doc__ = (
        f"Classification tags response for a single {label} organization."
    )

    # Outer batch response model: list of per-organization responses
    batch_response = create_model(
        f"Batch{label}TagsResponse",
        responses=(
            List[category_response],  # type: ignore[valid-type]
            Field(
                ..., description=f"List of {label} tag responses for each organization"
            ),
        ),
    )
    batch_response.__doc__ = (
        f"Batch response containing {label} classification tags "
        "for multiple organizations."
    )

    # Attach the enum and inner model for convenient access
    batch_response.Tags = tags_enum  # type: ignore[attr-defined]
    batch_response.TagsResponse = category_response  # type: ignore[attr-defined]

    return batch_response
