"""Constraints supplied WITH a process (by the partner or reviewer), not produced by a model."""
from enum import Enum

from .common import Strict


class HumanOversight(str, Enum):
    MANDATORY = "mandatory"      # every stage needs a human check (review_after or stronger)
    RECOMMENDED = "recommended"  # prefer a human check; 'none' must be justified
    UNSPECIFIED = "unspecified"  # no constraint: choose by risk


class Constraints(Strict):
    human_oversight: HumanOversight = HumanOversight.UNSPECIFIED
    notes: list[str] = []  # free-text constraints, e.g. "no changes during a validated run"
