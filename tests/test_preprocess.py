"""
Basic unit tests for the text normalization pipeline.
Run with: pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import (
    normalize_unicode,
    remove_page_artifacts,
    fix_hyphenated_linebreaks,
    collapse_whitespace,
    normalize_text,
)


def test_normalize_unicode_curly_quotes():
    text = "This is \u201cconfidential\u201d information."
    result = normalize_unicode(text)
    assert '"confidential"' in result


def test_remove_page_artifacts():
    text = "Some clause text.\nPage 3 of 12\nMore clause text."
    result = remove_page_artifacts(text)
    assert "Page 3 of 12" not in result


def test_fix_hyphenated_linebreaks():
    text = "The term of this agree-\nment shall be five years."
    result = fix_hyphenated_linebreaks(text)
    assert "agreement" in result
    assert "agree-\nment" not in result


def test_collapse_whitespace():
    text = "Line one.\n\n\n\nLine two.     Extra   spaces."
    result = collapse_whitespace(text)
    assert "\n\n\n" not in result
    assert "   " not in result


def test_normalize_text_end_to_end():
    text = "This Agreement (the \u201cAgreement\u201d) is entered into as of\nPage 1 of 5\nJanuary 1, 2024."
    result = normalize_text(text)
    assert "Page 1 of 5" not in result
    assert '"Agreement"' in result


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
