"""
Normalizes raw contract text extracted from PDFs.

PDF extraction commonly introduces:
- de-hyphenated words broken across line ends ("consider-\\nation")
- repeated page headers/footers ("Page 3 of 12", confidentiality stamps)
- excessive whitespace / stray control characters
- inconsistent unicode (curly quotes, non-breaking spaces)

This module cleans those artifacts so downstream LLM prompts see
clean, consistent text (which also reduces token count/cost).
"""

import re
import unicodedata

# Matches common page-footer/header noise seen in CUAD contracts
_PAGE_NUMBER_PATTERN = re.compile(r"\n?\s*Page\s+\d+\s+of\s+\d+\s*\n?", re.IGNORECASE)
_LONE_PAGE_NUM_PATTERN = re.compile(r"\n\s*\d{1,4}\s*\n")
_MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")
_MULTI_SPACE_PATTERN = re.compile(r"[ \t]{2,}")
_HYPHEN_LINEBREAK_PATTERN = re.compile(r"(\w)-\n(\w)")


def normalize_unicode(text: str) -> str:
    """Normalize unicode (e.g. curly quotes -> straight quotes) and strip control chars."""
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-",
        "\xa0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def remove_page_artifacts(text: str) -> str:
    """Strip 'Page X of Y' footers and lone page-number lines."""
    text = _PAGE_NUMBER_PATTERN.sub("\n", text)
    text = _LONE_PAGE_NUM_PATTERN.sub("\n", text)
    return text


def fix_hyphenated_linebreaks(text: str) -> str:
    """Rejoin words split across a line break by a hyphen, e.g. 'termin-\\nation' -> 'termination'."""
    return _HYPHEN_LINEBREAK_PATTERN.sub(r"\1\2", text)


def collapse_whitespace(text: str) -> str:
    """Collapse repeated blank lines and repeated spaces/tabs."""
    text = _MULTI_NEWLINE_PATTERN.sub("\n\n", text)
    text = _MULTI_SPACE_PATTERN.sub(" ", text)
    return text.strip()


def normalize_text(raw_text: str) -> str:
    """
    Full normalization pipeline applied to a contract's raw extracted text.
    Order matters: unicode fixes first, then structural cleanup, then whitespace.
    """
    text = normalize_unicode(raw_text)
    text = fix_hyphenated_linebreaks(text)
    text = remove_page_artifacts(text)
    text = collapse_whitespace(text)
    return text
