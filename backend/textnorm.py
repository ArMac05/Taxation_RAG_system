import re

_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("\u2019", "'"), ("\u2018", "'"),
    ("\u201c", '"'), ("\u201d", '"'),
    ("\u2014", "-"), ("\u2013", "-"),
)


def normalize(text: str) -> str:
    """Lowercase, straighten quotes/dashes, and collapse all whitespace."""
    text = text.lower()
    for old, new in _REPLACEMENTS:
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()