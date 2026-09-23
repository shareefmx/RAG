"""Unit tests for TextCleaner."""

from app.ingestion.text_cleaner import TextCleaner


def test_clean_empty_and_whitespace():
    cleaner = TextCleaner()
    assert cleaner.clean("") == ""
    assert cleaner.clean("    \n\n\t   ") == ""


def test_fix_hyphenation():
    cleaner = TextCleaner()
    raw = "This paper presents a new ap-\nproach to docu-\nment retrieval."
    cleaned = cleaner.clean(raw)
    assert "new approach" in cleaned
    assert "document retrieval" in cleaned


def test_normalize_whitespace_and_newlines():
    cleaner = TextCleaner()
    raw = "Paragraph 1 with   multiple    spaces.\n\n\n\n\nParagraph 2 after excessive blank lines."
    cleaned = cleaner.clean(raw)
    assert cleaned == "Paragraph 1 with multiple spaces.\n\nParagraph 2 after excessive blank lines."


def test_strip_unwanted_control_characters():
    cleaner = TextCleaner()
    raw = "Text with \x00 null byte and \x08 backspace and \x0c form feed.\nNext line."
    cleaned = cleaner.clean(raw)
    assert "\x00" not in cleaned
    assert "\x08" not in cleaned
    assert "Text with null byte and backspace and\nform feed.\nNext line." == cleaned


def test_preserve_equations_numbers_and_punctuation():
    cleaner = TextCleaner()
    raw = (
        "Equation (1): f(x) = 2.5 * x^2 + 10.3\n\n"
        "Model accuracy was 94.8% on N=10,000 samples (p < 0.001)."
    )
    cleaned = cleaner.clean(raw)
    assert "f(x) = 2.5 * x^2 + 10.3" in cleaned
    assert "94.8%" in cleaned
    assert "N=10,000" in cleaned
    assert "(p < 0.001)" in cleaned
