"""Dataset preparation is separate from the federated training boundary."""

import numpy as np
from pathlib import Path
from .config import positive_int


def stratified_partition(X, y, n_parties, random_state=0):
    """Return disjoint local datasets; per-class counts differ by at most one."""
    positive_int("n_parties", n_parties)
    X, y = np.asarray(X), np.asarray(y)
    if y.ndim != 1 or len(X) != len(y) or not len(y):
        raise ValueError("Expected a nonempty dataset and one label per sample")
    rng = np.random.default_rng(random_state)
    indices = [[] for _ in range(n_parties)]
    offset = 0
    for label in np.unique(y):
        members = np.flatnonzero(y == label)
        rng.shuffle(members)
        # Rotate recipients to distribute remainders across total party sizes.
        for j, row in enumerate(members):
            indices[(offset + j) % n_parties].append(int(row))
        offset = (offset + len(members)) % n_parties
    result = []
    for rows in indices:
        rng.shuffle(rows)  # do not expose class blocks through row ordering
        rows = np.asarray(rows, dtype=np.int64)
        result.append((X[rows], y[rows]))
    return result


def load_ts(path):
    """Read a UCR/UEA .ts classification file into (samples, N), labels.

    No implicit imputation, padding, normalization or test-label encoding.
    The supported FedIF subset is finite, equal-length, univariate series.
    """
    try:
        from aeon.datasets import load_from_ts_file
    except ImportError as exc:
        raise ImportError("Install fedif[datasets] to load .ts datasets") from exc
    path = Path(path)
    if path.suffix.lower() != ".ts":
        raise ValueError("Expected a UCR .ts file, not .txt or .arff")
    if not path.is_file():
        raise FileNotFoundError(path)
    X, y, metadata = load_from_ts_file(str(path), return_meta_data=True)
    if metadata.get("targetlabel") or not metadata.get("classlabel", False):
        raise ValueError("FedIF requires classification labels in the .ts header")
    if metadata.get("timestamps", False):
        raise ValueError("Timestamped .ts series are unsupported")
    if isinstance(X, list):
        raise ValueError("FedIF requires equal-length .ts series; no padding is applied")
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or X.shape[1] != 1:
        raise ValueError("FedIF requires univariate equal-length .ts data")
    if not len(X) or X.shape[2] < 3 or not np.isfinite(X).all():
        raise ValueError("Require nonempty finite .ts series with N >= 3; no imputation is applied")
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("Expected one classification label per .ts sample")
    return np.ascontiguousarray(X[:, 0, :]), y


def load_ucr_dataset(input_path, dataset):
    """Load the official TRAIN/TEST split without recombining or normalizing it."""
    if Path(dataset).name != dataset or dataset in (".", ".."):
        raise ValueError("dataset must be a plain directory name")
    root = Path(input_path) / dataset
    X_train, y_train = load_ts(root / f"{dataset}_TRAIN.ts")
    X_test, y_test = load_ts(root / f"{dataset}_TEST.ts")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("TRAIN and TEST sequence lengths differ")
    if not set(y_test.tolist()) <= set(y_train.tolist()):
        raise ValueError("TEST contains a class absent from TRAIN")
    return X_train, y_train, X_test, y_test
