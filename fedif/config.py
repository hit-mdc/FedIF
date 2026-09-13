"""Public session metadata and numerical bounds. No private data belong here."""

from dataclasses import asdict, dataclass
import json
import math
from numbers import Integral
from pathlib import Path
import re


def positive_int(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class Session:
    """Identical public manifest at all parties; a fresh id for every fit."""

    session_id: str
    party_sizes: tuple
    classes: tuple

    def __post_init__(self):
        object.__setattr__(self, "party_sizes", tuple(self.party_sizes))
        object.__setattr__(self, "classes", tuple(self.classes))
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", self.session_id):
            raise ValueError("session_id must contain 1-80 letters, digits, '_' or '-'")
        if len(self.party_sizes) < 2:
            raise ValueError("Semi training requires at least two parties")
        for size in self.party_sizes:
            positive_int("party size", size, 0)
        if sum(self.party_sizes) == 0:
            raise ValueError("The global training set cannot be empty")
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError("classes must be a nonempty ordered set")
        if not all(isinstance(c, str) for c in self.classes) and not all(
            isinstance(c, (int, float)) and not isinstance(c, bool)
            and math.isfinite(c) for c in self.classes
        ):
            raise ValueError("classes must be all strings or all finite numbers")

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class TrainingSpec:
    party_sizes: tuple
    n_classes: int
    n_features: int
    n_estimators: int
    height: int
    candidates: int
    fraction_bits: int = 16
    feature_bits: int = 48
    entropy_fraction_bits: int = 24
    entropy_bits: int = 64
    threshold_bits: int = 32
    security: int = 40
    ICS: bool = False
    ESA: bool = False
    m: int = 5

    def __post_init__(self):
        object.__setattr__(self, "party_sizes", tuple(self.party_sizes))
        for name in ("n_classes", "n_features", "n_estimators", "candidates",
                     "fraction_bits", "feature_bits", "entropy_fraction_bits",
                     "entropy_bits", "threshold_bits"):
            positive_int(name, getattr(self, name))
        positive_int("height", self.height, 0)
        if type(self.ICS) is not bool:
            raise ValueError("ICS must be a boolean")
        if type(self.ESA) is not bool:
            raise ValueError("ESA must be a boolean")
        if self.ESA:
            positive_int("m", self.m)
            if self.m > self.candidates:
                raise ValueError(f"ESA requires m <= q; got m={self.m}, q={self.candidates}. "
                                 "Decrease m or increase the candidate feature count.")
        if self.security != 40:
            raise ValueError("FedIF uses statistical security kappa=40")
        if self.candidates > self.n_features:
            raise ValueError("Candidate count exceeds K")
        if not self.party_sizes or sum(self.party_sizes) <= 0:
            raise ValueError("M must be positive")
        for size in self.party_sizes:
            positive_int("party size", size, 0)
        if not 1 <= self.fraction_bits < self.feature_bits <= 52:
            raise ValueError("Require 1 <= fraction_bits < feature_bits <= 52")
        if not 1 <= self.threshold_bits <= 48:
            raise ValueError("threshold_bits must be in [1, 48]")
        if not 1 <= self.entropy_fraction_bits < self.entropy_bits <= 64:
            raise ValueError("Invalid entropy precision")
        # Includes room for signed comparisons and entropy arithmetic.
        bound = self.samples * max(1, math.log2(self.samples))
        if bound >= 2 ** (self.entropy_bits - self.entropy_fraction_bits - 3):
            raise ValueError("M log2(M) exceeds the configured entropy range")
        if self.height > 30:
            raise ValueError("height > 30 exceeds the supported public node indexing")

    @property
    def samples(self):
        return sum(self.party_sizes)

    @property
    def integer_bits(self):
        return max(128, 2 * self.entropy_bits + 2,
                   self.feature_bits + self.threshold_bits + 2)

    @property
    def node_count(self):
        return 2 ** (self.height + 1) - 1

    def to_dict(self):
        return asdict(self)

