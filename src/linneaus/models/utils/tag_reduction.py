"""
Tag reduction and simplification utilities for AS classification labels.

This module provides functions to reduce and simplify multi-label tag DataFrames.
Tags follow a hierarchical naming convention where sub-tags are prefixed by their
main category (e.g., "Access_ISP", "Access_Mobile"). These utilities collapse
sub-tags into their main categories or filter to specific category prefixes.
"""

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def simplify_tags(tags: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse multi-column boolean tag indicators into a single binary flag per main category.

    Given a DataFrame where each column is a sub-tag of the form ``"Main_Sub"``,
    this function:

    1. Extracts the set of unique main categories (everything before the first
       underscore).
    2. For each main category, computes the row-wise maximum across its sub-tag
       columns.
    3. Returns a new DataFrame with one column per main category, containing 0/1
       flags.

    Parameters
    ----------
    tags : pd.DataFrame
        Boolean (0/1) DataFrame where columns follow the ``"Main_Sub"`` naming
        convention.

    Returns
    -------
    pd.DataFrame
        DataFrame with one column per main category and 0/1 values.
    """
    main_categories: pd.Index = tags.columns.str.split("_").str[0].unique()
    logger.debug(
        "Found %d main categories: %s", len(main_categories), list(main_categories)
    )

    simplified_tags = pd.DataFrame(index=tags.index)
    for main_category in main_categories:
        columns = tags.columns[tags.columns.str.startswith(main_category)]
        simplified_tags[main_category] = tags[columns].max(axis=1)

    return simplified_tags


def reduce_tags(tags: pd.DataFrame, select_tags: str) -> pd.DataFrame:
    """
    Filter and optionally simplify a raw tag DataFrame.

    Processing steps:

    * Drops the ``"ASName / Org."`` metadata column if present.
    * Fills ``NaN`` values with ``0``.
    * Normalises the ``"ASN"`` column name to ``"asn"`` and casts it to ``int``.
    * Returns one of three variants depending on *select_tags*:

      - ``"all_tags"``: the cleaned original tag DataFrame.
      - A specific main-category prefix (e.g. ``"Access"``): only columns
        starting with that prefix, plus the ``"asn"`` column.
      - ``"reduce_tags"``: collapsed categories via :func:`simplify_tags`,
        grouped by ASN and binarised.

    Parameters
    ----------
    tags : pd.DataFrame
        Raw tag DataFrame, typically loaded directly from a labelled-data CSV.
    select_tags : str
        One of ``"all_tags"``, ``"reduce_tags"``, or a main-category prefix
        present in the column names.

    Returns
    -------
    pd.DataFrame
        Filtered / simplified tag DataFrame.

    Raises
    ------
    ValueError
        If *select_tags* is not a recognised option.
    """
    tags = tags.copy()

    # Drop non-tag metadata columns
    if "ASName / Org." in tags.columns:
        tags = tags.drop(columns=["ASName / Org."])

    tags.fillna(0, inplace=True)
    tags.columns = tags.columns.str.replace("ASN", "asn")
    tags["asn"] = tags["asn"].astype(int)

    logger.info(
        "reduce_tags called with select_tags='%s' on DataFrame with %d rows and %d columns",
        select_tags,
        len(tags),
        len(tags.columns),
    )

    if select_tags == "all_tags":
        return tags

    prefixes: List[str] = tags.columns.str.split("_").str[0].unique().tolist()

    if select_tags in prefixes:
        columns = tags.columns[tags.columns.str.startswith(select_tags)]
        logger.info(
            "Filtering to %d columns for prefix '%s'", len(columns), select_tags
        )
        return tags[["asn"]].join(tags[columns])

    if select_tags == "reduce_tags":
        simplified_tags = simplify_tags(tags)
        simplified_tags = simplified_tags.groupby("asn").sum().reset_index()
        for col in simplified_tags.columns:
            if col != "asn":
                simplified_tags[col] = simplified_tags[col].apply(
                    lambda x: 1 if x > 0 else 0
                )
        logger.info(
            "Reduced tags to %d main categories for %d unique ASNs",
            len(simplified_tags.columns) - 1,
            len(simplified_tags),
        )
        return simplified_tags

    raise ValueError(
        f"Invalid select_tags '{select_tags}'. "
        f"Must be 'all_tags', 'reduce_tags', or one of {prefixes}."
    )
