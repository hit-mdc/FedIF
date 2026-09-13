"""FedIF secure protocols. All data-dependent choices use secret selection.

This module imports the existing MP-SPDZ Compiler API and is compiled by
compile.py. It is not an in-process Python simulation of MPC.
"""

from fractions import Fraction
from math import isqrt
from Compiler.types import sint, sfix, regint, Array, Matrix, MemValue
from Compiler.library import (for_range, tree_reduce, print_ln,
                              get_program, start_timer, stop_timer, break_point)
from Compiler import mpc_math


def _sum(values):
    return tree_reduce(lambda a, b: a + b, values)


class NodeData:
    """Fixed-size secret features, one-hot labels and a node membership vector."""

    def __init__(self, features, labels, mask, spec, tree=0):
        self.features, self.labels, self.mask, self.spec = features, labels, mask, spec
        self.tree = tree


def GenerateThreshold(D_i, k):
    """Return a secret encoded threshold for the node view D_i."""
    x, active = D_i.features[k][:], D_i.mask

    def merge(left, right):
        av, lo, hi = left
        bv, blo, bhi = right
        use_lo = bv * ((1 - av) + av * (blo < lo))
        use_hi = bv * ((1 - av) + av * (bhi > hi))
        return av + bv - av * bv, use_lo.if_else(blo, lo), use_hi.if_else(bhi, hi)

    valid, lo, hi = tree_reduce(merge, [(active[i], x[i], x[i])
                                       for i in range(D_i.spec["M"])])
    lo, hi = valid.if_else(lo, 0), valid.if_else(hi, 0)
    bits = D_i.spec["threshold_bits"]
    random = sint.get_random_int(bits)
    # Exact unsigned floor, not probabilistic fixed-point truncation. Therefore
    # lo < tau <= hi for nonconstant nodes under strict-left routing.
    product = (hi - lo) * random
    decomposition = product.bit_decompose(D_i.spec["feature_bits"] + bits)
    offset = _sum([bit * (2 ** j) for j, bit in enumerate(decomposition[bits:])])
    return lo + offset + (hi > lo)


def CountSamples(D_i, v, k, tau):
    """Return secret left/right per-class counts; v identifies the node view."""
    left = D_i.mask * (D_i.features[k][:] < tau)
    right = D_i.mask - left
    lc = [_sum(left * D_i.labels[c][:]) for c in range(D_i.spec["n_classes"])]
    rc = [_sum(right * D_i.labels[c][:]) for c in range(D_i.spec["n_classes"])]
    return lc, rc


def LinearSearch(points, tau, sentinel):
    """Secret one-hot lower_bound: first point >= tau, including overflow."""
    found = sint(0)
    result = []
    for point in [*points, sint(sentinel)]:
        before = point < tau
        hit = (1 - before) * (1 - found)
        result.append(hit)
        found = found + hit
    return result


def SecureQuery(points, counts, tau, size, n_classes, feature_bits):
    """Two-level oblivious HST selection, with row-major 2C count sets."""
    width = 2 * n_classes
    if size == 0:  # M_i is public; an empty NODE with M_i>0 never skips work.
        return [sint(0) for _ in range(width)]
    L = isqrt(size - 1) + 1
    B = L
    sentinel = 1 << (feature_bits - 1)  # exceeds every legal feature and threshold

    def index(segment, column):
        return min(segment * L + column, size - 1)

    maxima = [points[index(segment, L - 1)] for segment in range(B)]
    selected = LinearSearch(maxima, tau, sentinel)
    top_points = [
        _sum([selected[a] * points[index(a, j)] for a in range(B)])
        + selected[B] * sentinel for j in range(L)]
    top_counts = [[
        _sum([selected[a] * counts[index(a, j) * width + c] for a in range(B)])
        + selected[B] * counts[size * width + c]
        for c in range(width)] for j in range(L)]
    inside = LinearSearch(top_points, tau, sentinel)
    # A selected segment has max >= tau; the overflow segment is all sentinel.
    # Thus inside[L] is necessarily zero, even for repeated points and padding.
    return [_sum([inside[j] * top_counts[j][c] for j in range(L)]) for c in range(width)]


def ImprovedCountSamples(D_i, v, k, tau):
    """Query each party's fixed-size local table and sum its secret 2C counts."""
    C = D_i.spec["n_classes"]
    # Prevent input merging/scheduling ahead of the public host request.
    break_point("ics-request")
    print_ln("FEDIF_ICS_QUERY %s %s %s", D_i.tree, v, k)
    break_point("ics-input")
    result = [sint(0) for _ in range(2 * C)]
    for party, size in enumerate(D_i.spec["party_sizes"]):
        if size:
            points = sint.get_input_from(party, size=size)
            counts = sint.get_input_from(party, size=(size + 1) * 2 * C)
            queried = SecureQuery(points, counts, tau, size, C, D_i.spec["feature_bits"])
            result = [result[c] + queried[c] for c in range(2 * C)]
    return result[:C], result[C:]


def _xlogx(count):
    positive = count > 0
    safe = positive.if_else(count, 1)
    # Safe log argument is selected BEFORE evaluation, including for empty nodes.
    return sfix(count) * mpc_math.log2_fx(sfix(safe))


def _entropy_mass(counts):
    return _xlogx(_sum(counts)) - _sum([_xlogx(count) for count in counts])


class Candidates:
    def __init__(self, q, C):
        self.features = Array(q, sint)
        self.thresholds = Array(q, sint)
        self.left, self.right = Matrix(q, C, sint), Matrix(q, C, sint)
        self.q, self.C = q, C


class ScoredCandidates:
    """Secret (feature, threshold, score) tuples; no values are opened here."""

    def __init__(self, candidates):
        self.features, self.thresholds = candidates.features, candidates.thresholds
        self.scores = Array(candidates.q, sfix)
        self.q = candidates.q


def EvaluateSplit(candidates):
    """Evaluate all candidates with the original entropy IG formula."""
    parent = [candidates.left[0][c] + candidates.right[0][c]
              for c in range(candidates.C)]
    total = _sum(parent)
    parent_mass = _entropy_mass(parent)
    denominator = sfix((total > 0).if_else(total, 1))
    inverse = sfix(1) / denominator
    result = ScoredCandidates(candidates)

    @for_range(candidates.q)
    def evaluate(j):
        lc = [candidates.left[j][c] for c in range(candidates.C)]
        rc = [candidates.right[j][c] for c in range(candidates.C)]
        valid = (_sum(lc) > 0) * (_sum(rc) > 0)
        ig = (parent_mass - _entropy_mass(lc) - _entropy_mass(rc)) * inverse
        # True IG is nonnegative; suppress negative fixed-point roundoff.
        ig = (ig < 0).if_else(sfix(0), ig)
        result.scores[j] = valid.if_else(ig, sfix(-1))

    return result


def _argmax(scored, available):
    """Secret index of maximum score, breaking ties by feature index."""
    best_score = MemValue(sfix(-3))
    best_k, best_index = MemValue(sint(0)), MemValue(sint(0))

    @for_range(scored.q)
    def compare(j):
        score = available[j].if_else(scored.scores[j], sfix(-2))
        k = scored.features[j]
        choose = (score > best_score.read()) + (score == best_score.read()) * (k < best_k.read())
        best_score.write(choose.if_else(score, best_score.read()))
        best_k.write(choose.if_else(k, best_k.read()))
        best_index.write(choose.if_else(sint(j), best_index.read()))

    return best_index.read()


def EvaluateSplitApprox(candidates, m):
    """Secret Taylor-proxy top-m selection followed by original IG evaluation."""
    if not 1 <= m <= candidates.q:
        raise ValueError(f"ESA requires 1 <= m <= q; got m={m}, q={candidates.q}")
    proxy = ScoredCandidates(candidates)
    parent = [candidates.left[0][c] + candidates.right[0][c]
              for c in range(candidates.C)]
    total = _sum(parent)
    safe_total = sfix((total > 0).if_else(total, 1))
    inverse_total = sfix(1) / safe_total
    p_parent = [sfix(count) * inverse_total for count in parent]
    # V/a is 1/p but avoids dividing by a rounded-to-zero probability.
    inverse_p = [sfix(total) / sfix((count > 0).if_else(count, 1)) for count in parent]

    @for_range(candidates.q)
    def approximate(j):
        left = _sum([candidates.left[j][c] for c in range(candidates.C)])
        right = _sum([candidates.right[j][c] for c in range(candidates.C)])
        valid = (left > 0) * (right > 0)
        inverse_left = sfix(1) / sfix((left > 0).if_else(left, 1))
        ratio = sfix(left) / sfix((right > 0).if_else(right, 1))
        terms = []
        for c in range(candidates.C):
            difference = sfix(candidates.left[j][c]) * inverse_left - p_parent[c]
            term = difference * difference * inverse_p[c]
            terms.append((parent[c] > 0).if_else(term, sfix(0)))
        score = sfix(0.5) * ratio * _sum(terms)
        proxy.scores[j] = valid.if_else(score, sfix(-1))

    retained = Candidates(m, candidates.C)
    available = Array(candidates.q, sint)
    available.assign_all(1)

    @for_range(m)
    def retain(rank):
        index = _argmax(proxy, available)
        retained.features[rank], retained.thresholds[rank] = 0, 0
        for c in range(candidates.C):
            retained.left[rank][c], retained.right[rank][c] = 0, 0

        # Scan all entries: the secret index is never used as a memory address.
        @for_range(candidates.q)
        def select(j):
            hit = index == j
            available[j] = available[j] * (1 - hit)
            retained.features[rank] += hit * candidates.features[j]
            retained.thresholds[rank] += hit * candidates.thresholds[j]
            for c in range(candidates.C):
                retained.left[rank][c] += hit * candidates.left[j][c]
                retained.right[rank][c] += hit * candidates.right[j][c]

    return EvaluateSplit(retained)



def FindBestSplit(D_i, v, r):
    """Sample floor(r*K) candidates; None denotes the exact sqrt default."""
    K, C = D_i.spec["n_features"], D_i.spec["n_classes"]
    q = max(1, isqrt(K) if r is None else int(r * K))
    if not 1 <= q <= K or q != D_i.spec["candidates"]:
        raise ValueError("Feature fraction disagrees with the public candidate budget")
    candidates = Candidates(q, C)
    permutation = Array(K, regint)

    @for_range(K)
    def initialize(k):
        permutation[k] = k

    @for_range(q)
    def candidate(j):
        # Joint public randomness for features; threshold coins remain secret.
        # 60-bit modulo sampling has total variation <= K/2^60 per draw.
        coin = regint(sint.get_random_int(60).reveal())
        index = j + coin % (K - j)
        k = permutation[index]
        old = permutation[j]
        permutation[index], permutation[j] = old, k
        tau = GenerateThreshold(D_i, k)
        if D_i.spec.get("ICS", False):  # public, compile-time protocol selection
            lc, rc = ImprovedCountSamples(D_i, v, k, tau)
        else:
            lc, rc = CountSamples(D_i, v, k, tau)
        candidates.features[j], candidates.thresholds[j] = k, tau
        for c in range(C):
            candidates.left[j][c], candidates.right[j][c] = lc[c], rc[c]

    scored = (EvaluateSplitApprox(candidates, D_i.spec["m"])
              if D_i.spec.get("ESA", False) else EvaluateSplit(candidates))
    available = Array(scored.q, sint)
    available.assign_all(1)
    best = _argmax(scored, available)
    best_k, best_tau = MemValue(sint(0)), MemValue(sint(0))

    @for_range(scored.q)
    def select_best(j):
        hit = best == j
        best_k.write(best_k.read() + hit * scored.features[j])
        best_tau.write(best_tau.read() + hit * scored.thresholds[j])

    return regint(best_k.read().reveal()), regint(best_tau.read().reveal())


def MajorityClass(D_i, v, fallback):
    """Secret majority, with secret ancestor fallback; caller reveals leaves only."""
    counts = [_sum(D_i.mask * D_i.labels[c][:]) for c in range(D_i.spec["n_classes"])]
    best, label = counts[0], sint(0)
    for c in range(1, len(counts)):
        choose = counts[c] > best
        label, best = choose.if_else(c, label), choose.if_else(counts[c], best)
    return (_sum(counts) > 0).if_else(label, fallback)


def build_forest(spec):
    program = get_program()
    program.set_security(spec["security"])
    sfix.set_precision(spec["entropy_fraction_bits"], spec["entropy_bits"])
    sfix.round_nearest = True
    spec = dict(spec, M=sum(spec["party_sizes"]))
    M, K, C, H = spec["M"], spec["n_features"], spec["n_classes"], spec["height"]
    features, labels = Matrix(K, M, sint), Matrix(C, M, sint)
    offset = 0
    for party, size in enumerate(spec["party_sizes"]):
        if size:
            @for_range(K)
            def input_feature(k):
                values = sint.get_input_from(party, size=size)
                features[k].assign_vector(values, base=offset)

            @for_range(C)
            def input_label(c):
                values = sint.get_input_from(party, size=size)
                labels[c].assign_vector(values, base=offset)
        offset += size

    # A fixed DFS schedule uses O(M*H) masks rather than O(M*2^H).
    masks = Matrix(H + 1, M, sint)
    fallback = Array(H + 1, sint)
    node_ids, split_features, split_thresholds = (Array(H + 1, regint) for _ in range(3))
    print_ln("FEDIF_BEGIN")
    start_timer(1)

    @for_range(spec["n_estimators"])
    def train_tree(t):
        masks[0].assign_all(1)
        fallback[0], node_ids[0] = 0, 0

        def visit(depth):
            data = NodeData(features, labels, masks[depth][:], spec, t)
            majority = MajorityClass(data, node_ids[depth], fallback[depth])
            if depth == H:  # public compile-time condition
                print_ln("FEDIF_LEAF %s %s %s", t, node_ids[depth], majority.reveal())
                return
            k, tau = FindBestSplit(data, node_ids[depth], Fraction(spec["candidates"], K))
            split_features[depth], split_thresholds[depth] = k, tau
            # Store once outside child loop; no nonleaf majority is opened.
            fallback[depth + 1] = majority
            print_ln("FEDIF_SPLIT %s %s %s %s", t, node_ids[depth], k, tau)

            @for_range(2)
            def children(branch):
                left = features[split_features[depth]][:] < split_thresholds[depth]
                selected = left * (1 - branch) + (1 - left) * branch
                masks[depth + 1].assign_vector(masks[depth][:] * selected)
                node_ids[depth + 1] = 2 * node_ids[depth] + 1 + branch
                visit(depth + 1)

        visit(0)

    stop_timer(1)
    print_ln("FEDIF_END")
