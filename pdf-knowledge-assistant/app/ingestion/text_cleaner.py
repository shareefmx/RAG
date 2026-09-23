"""Text Cleaning and Normalization Pipeline.

Normalizes whitespace, fixes line-break hyphenation, strips non-printable control
characters, and cleans PDF extraction artifacts while strictly preserving equations,
technical terminology, numbers, and paragraph structure.
"""

import re
import unicodedata


class TextCleaner:
    """Normalizes and cleans raw text extracted from PDF documents."""

    def __init__(self, fix_hyphenation: bool = True, normalize_unicode: bool = True):
        self.fix_hyphenation = fix_hyphenation
        self.normalize_unicode = normalize_unicode

    def clean(self, text: str) -> str:
        """Cleans and normalizes the input text string.

        Args:
            text: Raw extracted PDF text.

        Returns:
            Normalized, clean text ready for chunking and embedding.
        """
        if not text:
            return ""

        # 1. Unicode Normalization (NFKC form combines separate accent marks into canonical glyphs)
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        # 2. Replace non-standard whitespace (non-breaking spaces, zero-width spaces, ideographic spaces)
        text = text.replace("\u00a0", " ")  # non-breaking space
        text = text.replace("\u200b", "")   # zero-width space
        text = text.replace("\ufeff", "")   # byte order mark (BOM)
        text = text.replace("\x0c", "\n")   # form feed (page break)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Strip non-printable control characters, preserving newlines (\n) and tabs (\t)
        text = "".join(
            ch for ch in text
            if ch in ("\n", "\t") or (unicodedata.category(ch)[0] != "C" and ord(ch) >= 32)
        )

        # 4. Fix line-break hyphenation (e.g., "retrie-\nval" -> "retrieval")
        # Only joins if hyphen is between lowercase letters across a single newline
        if self.fix_hyphenation:
            text = re.sub(r"([a-zA-Z]{2,})-\n([a-zA-Z]{2,})", r"\1\2", text)

        # 5. Fix fused words commonly produced by PDF extraction bounding box joins
        # e.g., ALLCAPS followed by TitleCase: "EDUCATIONNew" -> "EDUCATION New"
        text = re.sub(r"\b([A-Z]{2,})([A-Z][a-z])", r"\1 \2", text)
        # e.g., Numbers followed by Capitalized word: "2024SKILLS" -> "2024 SKILLS", "10.0SKILLS" -> "10.0 SKILLS"
        text = re.sub(r"(\d+(?:\.\d+)?)([A-Z][a-z]+)", r"\1 \2", text)
        # e.g., lowercase followed by ALLCAPS heading: "DeveloperSUMMARY" -> "Developer SUMMARY"
        text = re.sub(r"([a-z])([A-Z]{2,}\b)", r"\1 \2", text)

        # 6. Normalize horizontal whitespace within lines (tabs/multiple spaces -> single space)
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            # Collapse multiple spaces or tabs into a single space
            collapsed_line = re.sub(r"[ \t]+", " ", line).strip()
            cleaned_lines.append(collapsed_line)

        text = "\n".join(cleaned_lines)

        # 7. Normalize paragraph breaks (collapse 3+ consecutive newlines into 2)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 8. Strip extraneous leading and trailing whitespace
        return text.strip()

