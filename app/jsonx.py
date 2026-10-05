"""Strict JSON parsing for exact rational inputs.

The standard library ``json`` module parses every number carrying a decimal
point or exponent as a Python ``float``, so JSON literals outside the binary64
range silently collapse: ``1e-400`` becomes ``0.0`` (underflow) and
``1e400`` becomes ``inf``.  The review service promises exact rational
arithmetic, and a strictly positive delay such as ``1e-400`` must never be
flattened to zero before it reaches the zero-width guard check.

``parse_float`` is given the number's exact source text, so integral tokens
stay ``int`` and fractional/exponential tokens are returned verbatim as
``str``; ``fractions.Fraction`` then consumes that text exactly.  The document
itself is still parsed by the stdlib parser (strict JSON subset, including
rejection of NaN/Infinity).
"""

from __future__ import annotations

import json
from typing import Any


def loads(data: str | bytes) -> Any:
    """Parse strict JSON, preserving tiny/huge numeric literals exactly.

    Integral literals are returned as ``int``; every other numeric literal as
    its exact decimal source text (e.g. ``"1e-400"``).  Raises
    ``json.JSONDecodeError`` on invalid JSON.
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8")

    def reject_constant(value: str) -> None:
        raise json.JSONDecodeError(f"{value} is not valid JSON", data, 0)

    return json.loads(data, parse_float=lambda text: text,
                      parse_constant=reject_constant)
