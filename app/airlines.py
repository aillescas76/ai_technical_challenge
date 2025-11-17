"""Utilities for normalizing airline names across the stack."""

from __future__ import annotations

import re

_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

_CANONICAL_DISPLAY = {
    "americanairlines": "American Airlines",
    "deltaairlines": "Delta Air Lines",
    "unitedairlines": "United Airlines",
    "skyfly": "SkyFly",
}

_AIRLINE_ALIAS_TO_KEY = {
    "aa": "americanairlines",
    "americanair": "americanairlines",
    "american": "americanairlines",
    "americanairline": "americanairlines",
    "delta": "deltaairlines",
    "dl": "deltaairlines",
    "deltaair": "deltaairlines",
    "deltaairline": "deltaairlines",
    "united": "unitedairlines",
    "ua": "unitedairlines",
    "unitedair": "unitedairlines",
    "unitedairline": "unitedairlines",
}


def _coerce_key(name: str) -> str:
    """Convert free-form airline text into a lowercase alpha-numeric key."""
    if not name:
        return ""
    spaced = _CAMEL_BOUNDARY.sub(" ", name.strip())
    lowered = spaced.lower()
    collapsed = _NON_ALNUM.sub("", lowered)
    return collapsed


def normalize_airline_key(name: str) -> str:
    """Return a canonical comparison key for the provided airline label."""
    raw_key = _coerce_key(name)
    if not raw_key:
        return ""
    return _AIRLINE_ALIAS_TO_KEY.get(raw_key, raw_key)


def canonical_airline_name(name: str) -> str:
    """Return a human-friendly airline name with known casing."""
    key = normalize_airline_key(name)
    if not key:
        return ""
    display = _CANONICAL_DISPLAY.get(key)
    if display:
        return display
    stripped = name.strip()
    return stripped or key
