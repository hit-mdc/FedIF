"""Party-local fixed-size tables for ImprovedCountSamples.

Only this party's encoded features, labels and public splits are used here.
Dummy points have zero weight. Nothing in this module learns the secret threshold.
"""

import secrets
import numpy as np


def build_count_table(values, labels, active, n_classes, feature_bits=48):
    """Return M_i sorted points and (M_i+1, 2*C) strict-left count rows.

    Row j counts x < points[j], including duplicate-point groups atomically;
    the final row counts all active samples on the left. Dummy generation and
    table dimensions depend on the public M_i, never the active node size.
    """
    values, labels = np.asarray(values), np.asarray(labels)
    active = np.asarray(active, dtype=bool)
    if values.ndim != 1 or labels.shape != values.shape or active.shape != values.shape:
        raise ValueError("ICS requires equally sized feature, label and membership vectors")
    if values.dtype.kind not in "iu" or labels.dtype.kind not in "iu":
        raise ValueError("ICS uses integer-encoded features and labels")
    if n_classes < 1 or not 1 <= feature_bits <= 52:
        raise ValueError("Invalid ICS class count or feature width")
    limit = 1 << (feature_bits - 1)
    if np.any(values < -limit) or np.any(values >= limit):
        raise ValueError("ICS feature outside the declared encoded range")
    if np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("ICS label outside global class vocabulary")
    size = len(values)
    # Generate M_i dummy values even when all samples are active. Sorting remains
    # local, not constant-time; the secure query never accesses a variable length.
    dummy = np.fromiter((secrets.randbits(feature_bits) - limit for _ in range(size)),
                        dtype=np.int64, count=size)
    points = np.where(active, values, dummy)
    order = np.argsort(points, kind="stable")
    points = points[order]
    weights = (labels[order, None] == np.arange(n_classes)) & active[order, None]
    prefix = np.zeros((size + 1, n_classes), dtype=np.int64)
    np.cumsum(weights, axis=0, out=prefix[1:])
    first_equal = np.searchsorted(points, points, side="left")
    left = np.empty_like(prefix)
    left[:size], left[size] = prefix[first_equal], prefix[size]
    counts = np.concatenate((left, prefix[size] - left), axis=1)
    return points, counts


class LocalCountTables:
    """Track the public DFS transcript and answer only valid candidate requests."""

    def __init__(self, Z, y, spec, party_id):
        self.Z, self.y = np.asarray(Z), np.asarray(y)
        self.spec = spec
        self.size = spec.party_sizes[party_id]
        if self.Z.shape != (self.size, spec.n_features) or self.y.shape != (self.size,):
            raise ValueError("ICS local data disagree with public dimensions")
        self.tree = -1
        self.splits = {}
        self.node = None
        self.selected = set()
        self.completed = True
        self.queries = 0

    def _mask(self, node):
        path = []
        while node:
            parent = (node - 1) // 2
            path.append((parent, node == 2 * parent + 1))
            node = parent
        mask = np.ones(self.size, dtype=bool)
        for parent, goes_left in reversed(path):
            if parent not in self.splits:
                raise ValueError("ICS request arrived before an ancestor split")
            feature, threshold = self.splits[parent]
            left = self.Z[:, feature] < threshold
            mask &= left if goes_left else ~left
        return mask

    def query(self, tree, node, feature):
        if not (0 <= tree < self.spec.n_estimators and
                0 <= node < (1 << self.spec.height) - 1 and
                0 <= feature < self.spec.n_features):
            raise ValueError("Invalid public ICS query coordinates")
        if tree != self.tree:
            if tree != self.tree + 1 or node != 0 or not self.completed:
                raise ValueError("Invalid ICS tree transition")
            self.tree, self.splits, self.node = tree, {}, None
        if node != self.node:
            if not self.completed or node in self.splits:
                raise ValueError("Unexpected ICS node transition")
            self.active = self._mask(node)
            self.node, self.selected, self.completed = node, set(), False
        if self.completed or feature in self.selected or len(self.selected) >= self.spec.candidates:
            raise ValueError("Repeated or excessive ICS candidate request")
        self.selected.add(feature)
        self.queries += 1
        return build_count_table(self.Z[:, feature], self.y, self.active,
                                 self.spec.n_classes, self.spec.feature_bits)

    def record_split(self, tree, node, feature, threshold):
        limit = 1 << (self.spec.feature_bits - 1)
        if (tree != self.tree or node != self.node or self.completed
                or len(self.selected) != self.spec.candidates or feature not in self.selected
                or not -limit <= threshold < limit):
            raise ValueError("Public split disagrees with preceding ICS requests")
        self.splits[node] = feature, threshold
        self.completed = True

    def finish(self):
        expected = self.spec.n_estimators * ((1 << self.spec.height) - 1) * self.spec.candidates
        if not self.completed or self.queries != expected:
            raise ValueError("Incomplete ICS interaction transcript")
