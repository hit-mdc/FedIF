"""Public full-tree representation; integer thresholds preserve exact routing."""

from dataclasses import dataclass
import numpy as np


@dataclass
class RDT:
    feature: np.ndarray
    threshold: np.ndarray
    majority: np.ndarray
    height: int
    split_rule: str = "lt"

    def validate(self, K, C, feature_bits=48):
        if self.split_rule not in ("lt", "le"):
            raise ValueError("Unsupported split rule")
        size = 2 ** (self.height + 1) - 1
        inner = 2 ** self.height - 1
        if any(a.shape != (size,) for a in (self.feature, self.threshold, self.majority)):
            raise ValueError("Incomplete RDT")
        if ((self.feature[:inner] < 0) | (self.feature[:inner] >= K)).any():
            raise ValueError("Invalid split feature")
        if ((self.majority[inner:] < 0) | (self.majority[inner:] >= C)).any():
            raise ValueError("Invalid leaf class")
        limit = 2 ** (feature_bits - 1)
        if ((self.threshold[:inner] < -limit) | (self.threshold[:inner] >= limit)).any():
            raise ValueError("Invalid split threshold")
        if (self.feature[inner:] != -1).any() or (self.majority[:inner] != -1).any():
            raise ValueError("Invalid full-tree structure")
        return self

    def predict_encoded(self, Z):
        nodes = np.zeros(len(Z), dtype=np.int64)
        rows = np.arange(len(Z))
        for _ in range(self.height):
            values = Z[rows, self.feature[nodes]]
            right = (values >= self.threshold[nodes] if self.split_rule == "lt"
                     else values > self.threshold[nodes])
            nodes = 2 * nodes + 1 + right.astype(np.int64)
        return self.majority[nodes]

    def to_dict(self):
        return {"height": self.height, "split_rule": self.split_rule, "feature": self.feature.tolist(),
                "threshold": self.threshold.tolist(), "majority": self.majority.tolist()}

    @classmethod
    def from_dict(cls, value, default_rule="lt"):
        return cls(*(np.asarray(value[k], dtype=np.int64)
                     for k in ("feature", "threshold", "majority")), height=value["height"],
                   split_rule=value.get("split_rule", default_rule))


def parse_model(text, spec):
    """Read the public tree output; reject incomplete training results."""
    trees = [RDT(np.full(spec.node_count, -1, dtype=np.int64),
                 np.zeros(spec.node_count, dtype=np.int64),
                 np.full(spec.node_count, -1, dtype=np.int64), spec.height)
             for _ in range(spec.n_estimators)]
    seen = set()
    started = finished = False
    for line in text.splitlines():
        if not line.startswith("FEDIF_"):
            continue
        parts = line.split()
        if parts == ["FEDIF_BEGIN"]:
            if started:
                raise ValueError("Duplicate model header")
            started = True
        elif parts == ["FEDIF_END"]:
            if not started or finished:
                raise ValueError("Invalid model footer")
            finished = True
        elif parts[0] == "FEDIF_ICS_QUERY":
            if not spec.ICS or not started or finished or len(parts) != 4:
                raise ValueError("Unexpected ICS request")
            t, v, k = map(int, parts[1:])
            if not (0 <= t < spec.n_estimators and 0 <= v < 2 ** spec.height - 1
                    and 0 <= k < spec.n_features):
                raise ValueError("Invalid ICS request")
        elif parts[0] in ("FEDIF_SPLIT", "FEDIF_LEAF"):
            if not started or finished:
                raise ValueError("Model node outside output boundaries")
            expected = 5 if parts[0] == "FEDIF_SPLIT" else 4
            if len(parts) != expected:
                raise ValueError("Malformed model node")
            t, v = map(int, parts[1:3])
            if not 0 <= t < len(trees) or not 0 <= v < spec.node_count or (t, v) in seen:
                raise ValueError("Invalid or duplicate model node")
            seen.add((t, v))
            if parts[0] == "FEDIF_SPLIT":
                trees[t].feature[v], trees[t].threshold[v] = map(int, parts[3:])
            else:
                trees[t].majority[v] = int(parts[3])
        else:
            raise ValueError("Unexpected model record")
    if not started or not finished or len(seen) != spec.n_estimators * spec.node_count:
        raise ValueError("Missing model output; no partial model is accepted")
    return [tree.validate(spec.n_features, spec.n_classes, spec.feature_bits) for tree in trees]
