"""Metadata extraction and document section detection heuristics."""

import re
from typing import Optional


COMMON_SECTION_PATTERNS = [
    r"^(?:abstract|executive\s+summary)\b",
    r"^(?:(?:\d+\.?)+\s+)?(?:introduction|background|overview)\b",
    r"^(?:(?:\d+\.?)+\s+)?(?:related\s+work|literature\s+review)\b",
    r"^(?:(?:\d+\.?)+\s+)?(?:methodology|methods|experimental\s+setup|architecture|design)\b",
    r"^(?:(?:\d+\.?)+\s+)?(?:results|findings|evaluation|experiments)\b",
    r"^(?:(?:\d+\.?)+\s+)?(?:discussion|analysis)\b",
    r"^(?:(?:\d+\.?)+\s+)?(?:conclusion|conclusions|future\s+work)\b",
    r"^(?:(?:\d+\.?)+\s+)?(?:references|bibliography|acknowledgements)\b",
]

COMPILED_SECTION_REGEX = [
    re.compile(pattern, re.IGNORECASE) for pattern in COMMON_SECTION_PATTERNS
]


class SectionDetector:
    """Detects structural headings and section names from PDF page text."""

    @classmethod
    def detect_heading(cls, line: str) -> Optional[str]:
        """Checks if a single line resembles a section header.

        A candidate section header:
        - Is not excessively long (< 80 characters).
        - Matches standard academic/technical section names or is in Title/UPPERCASE format.
        - Does not end with terminal punctuation like a period or comma.
        """
        line = line.strip()
        if not line or len(line) > 80:
            return None

        # Discard lines ending with standard sentence punctuation
        if line.endswith((".", ",", ";", ":", "?", "!")):
            return None

        # Check against common known section regexes
        for regex in COMPILED_SECTION_REGEX:
            if regex.search(line):
                return line

        # Check for numbered heading format, e.g., "1. Problem Formulation" or "2.3 System Architecture"
        numbered_match = re.match(r"^\d+(?:\.\d+)*\s+[A-Z][a-zA-Z0-9\s\-_]{2,50}$", line)
        if numbered_match:
            return line

        # Check for standalone uppercase title, e.g. "SYSTEM OVERVIEW"
        if line.isupper() and len(line.split()) <= 6 and len(line) >= 4:
            return line.title()

        return None

    @classmethod
    def extract_sections_from_page(cls, text: str, default_section: str = "General") -> str:
        """Finds the most prominent section heading on a page, or returns the default."""
        for line in text.split("\n"):
            detected = cls.detect_heading(line)
            if detected:
                return detected
        return default_section

