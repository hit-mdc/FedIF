"""Adapter for the bundled, unmodified QUANT implementation."""

import importlib
import numpy as np
import torch


def as_series(X):
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    values = np.asarray(X, dtype=np.float32)
    if values.ndim == 3 and values.shape[1] == 1:
        values = values[:, 0, :]
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("X must have shape (samples, N) or (samples, 1, N), N >= 3")
    if not np.isfinite(values).all():
        raise ValueError("Time series must be finite; missing values are unsupported")
    return np.ascontiguousarray(values)


class QuantFeatures:
    def __init__(self, depth=6, div=4, module="fedif.quant"):
        try:
            external = importlib.import_module(module)
            Quant = external.Quant
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                f"Cannot import Quant from {module!r}. The bundled default is "
                "'fedif.quant'; custom modules must expose a Quant class."
            ) from exc
        self.quant = Quant(depth=depth, div=div)
        self.module = module
        self.parameters = {"module": module, "depth": depth, "div": div}

    def initialize(self, length):
        # Original QUANT has no learned parameters. A public dummy initializes
        # interval metadata, including for parties with no local samples.
        self.length = length
        with torch.no_grad():
            z = self.quant.fit_transform(torch.zeros((1, 1, length)), None)
        self.n_features = z.shape[1]
        return self

    def transform(self, X):
        values = as_series(X)
        if values.shape[1] != self.length:
            raise ValueError("Sequence length differs from the fitted model")
        if len(values) == 0:
            return np.empty((0, self.n_features), dtype=np.float64)
        with torch.no_grad():
            z = self.quant.transform(torch.from_numpy(values).unsqueeze(1))
        result = z.detach().cpu().numpy().astype(np.float64)
        if result.shape != (len(values), self.n_features) or not np.isfinite(result).all():
            raise ValueError("QUANT returned invalid features")
        return result


def encode_features(z, fraction_bits, feature_bits):
    scaled = np.asarray(z, dtype=np.float64) * (2 ** fraction_bits)
    if not np.isfinite(scaled).all():
        raise ValueError("Non-finite features")
    rounded = np.rint(scaled)  # explicit ties-to-even, used at training AND prediction
    limit = 2 ** (feature_bits - 1)
    if (rounded < -limit).any() or (rounded >= limit).any():
        raise ValueError("QUANT feature outside fixed-point range; adjust precision")
    return rounded.astype(np.int64)
