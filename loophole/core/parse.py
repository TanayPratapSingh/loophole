"""Pull the answer out of a completion. Shared by every verifier, honest or leaky.

Both verifiers of an environment use the same extraction, so a disagreement
between them is always about what they check, never about what they read.
"""

from __future__ import annotations

import re

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_FENCE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.S)


def last_answer_tag(text: str) -> str | None:
    """The last <answer>...</answer> span, stripped, or None."""
    found = _ANSWER.findall(text or "")
    return found[-1].strip() if found else None


def last_code_block(text: str, langs: tuple[str, ...] = ()) -> str | None:
    """The last fenced code block whose language tag is in langs (any if empty)."""
    blocks = [(lang.lower(), body) for lang, body in _FENCE.findall(text or "")]
    if langs:
        blocks = [b for b in blocks if b[0] in langs]
    return blocks[-1][1].strip() if blocks else None


def has_format(text: str, require_answer_tag: bool = True) -> bool:
    return last_answer_tag(text) is not None if require_answer_tag else bool(text and text.strip())
