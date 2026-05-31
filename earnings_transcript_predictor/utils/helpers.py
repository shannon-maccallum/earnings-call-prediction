"""
helpers.py — shared utility functions.
"""

import re


def clean_transcript(text: str) -> str:
    """
    Basic transcript cleaning.
    - Strips operator/moderator lines like "OPERATOR: Please hold."
    - Collapses multiple whitespace into single spaces.
    """
    # Remove common boilerplate lines
    text = re.sub(r"(?i)^operator:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def quarter_to_date_range(quarter: str) -> tuple[str, str]:
    """
    Convert a quarter string like "2024Q1" to approximate date bounds.
    Useful for filtering Yahoo Finance data.

    Returns (start_date, end_date) as "YYYY-MM-DD" strings.
    """
    year, q = int(quarter[:4]), int(quarter[5])
    starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
    ends   = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    return f"{year}-{starts[q]}", f"{year}-{ends[q]}"
