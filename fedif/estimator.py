"""Public sklearn-style interface. fit is a collective operation across parties."""

import json
import math
from pathlib import Path
import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .config import Session, TrainingSpec, positive_int
from .features import QuantFeatures, as_series, encode_features
from .model import RDT


class FedIF(ClassifierMixin, BaseEstimator):
    """Federated interval forest with local QUANT and secure RDT training.

    Parameters
    ----------
    backend : MPSpdzBackend
        Connection to this party's already deployed MP-SPDZ runtime.
    session : Session
        Shared public party sizes, ordered classes and unique training id.
    n_estimators : int, default=200
    max_depth : int or None
        None uses ceil(log2(M)); every tree is full to this depth.
    r : float or None
        Feature fraction. None selects floor(sqrt(K)) features per node.
    ICS : bool, default=False
        Use ImprovedCountSamples with local tables and secure HST queries.
    ESA : bool, default=False
        Preselect m candidates with the Taylor proxy before entropy evaluation.
    m : int, default=5
        Retained candidates; ESA requires 1 <= m <= the candidate count q.
    depth, div : int
        Original QUANT parameters (distinct from RDT max_depth).
    quant_module : str
        Importable module containing the original Quant class.

    Notes
    -----
    fit(X_i, y_i) must be invoked by every party for the same session.
    Prediction and model loading require no MPC backend.
    """

    def __init__(self, *, backend=None, session=None, n_estimators=200,
                 max_depth=None, r=None, depth=6, div=4, quant_module="fedif.quant",
                 fraction_bits=16, feature_bits=48, entropy_fraction_bits=24,
                 entropy_bits=64, threshold_bits=32, ICS=False, ESA=False, m=5):
        self.backend = backend
        self.session = session
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.r = r
        self.depth = depth
        self.div = div
        self.quant_module = quant_module
        self.fraction_bits = fraction_bits
        self.feature_bits = feature_bits
        self.entropy_fraction_bits = entropy_fraction_bits
        self.entropy_bits = entropy_bits
        self.threshold_bits = threshold_bits
        self.ICS = ICS
        self.ESA = ESA
        self.m = m

    def fit(self, X, y):
        if self.backend is None or not isinstance(self.session, Session):
            raise ValueError("fit requires backend and a shared Session")
        # A failed refit must not leave a previously fitted classifier usable.
        for key in list(vars(self)):
            if key.endswith("_"):
                delattr(self, key)
        positive_int("n_estimators", self.n_estimators)
        positive_int("depth", self.depth)
        positive_int("div", self.div)
        if type(self.ICS) is not bool:
            raise ValueError("ICS must be a boolean")
        if self.max_depth is not None:
            positive_int("max_depth", self.max_depth, 0)
        if self.r is not None and (isinstance(self.r, bool) or not np.isscalar(self.r)
                                   or not 0 < float(self.r) <= 1):
            raise ValueError("r must be a feature fraction in (0, 1], or None for sqrt")
        party = positive_int("party_id", self.backend.party_id, 0)
        if party >= len(self.session.party_sizes):
            raise ValueError("party_id outside session")
        X = as_series(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must have one label per local sample")
        if len(X) != self.session.party_sizes[party]:
            raise ValueError("Local sample count differs from public M_i")
        classes = np.asarray(self.session.classes)
        mapping = {label: c for c, label in enumerate(self.session.classes)}
        try:
            encoded_y = np.asarray([mapping[label] for label in y.tolist()], dtype=np.int64)
        except (KeyError, TypeError) as exc:
            raise ValueError("Local label missing from shared classes") from exc
        started = time.monotonic()
        transform = QuantFeatures(self.depth, self.div, self.quant_module).initialize(X.shape[1])
        K = transform.n_features
        q = max(1, math.isqrt(K) if self.r is None else math.floor(float(self.r) * K))
        M = sum(self.session.party_sizes)
        H = (M - 1).bit_length() if self.max_depth is None else int(self.max_depth)
        spec = TrainingSpec(self.session.party_sizes, len(classes), K, int(self.n_estimators), H, q,
                            self.fraction_bits, self.feature_bits, self.entropy_fraction_bits,
                            self.entropy_bits, self.threshold_bits, ICS=self.ICS, ESA=self.ESA, m=self.m)
        Z = encode_features(transform.transform(X), self.fraction_bits, self.feature_bits)
        transform_seconds = time.monotonic() - started
        parameters = {**transform.parameters, "length": X.shape[1]}
        trees, metrics = self.backend.train(Z, encoded_y, spec, self.session)
        if len(trees) != self.n_estimators:
            raise ValueError("Backend returned the wrong number of RDTs")
        for tree in trees:
            if tree.height != H or tree.split_rule != "lt":
                raise ValueError("Backend returned incorrect tree height or split rule")
            tree.validate(K, len(classes), self.feature_bits)
        self._transform = transform
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        self.n_transformed_features_ = K
        self.max_depth_ = H
        self.max_features_ = q
        self.r_ = q / K
        self.spec_ = spec
        self.estimators_ = trees
        self.metrics_ = {"quant_seconds": transform_seconds, **metrics}
        self.feature_parameters_ = parameters
        self.model_ = self._model_dict()
        return self

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        # Use fitted numerical settings, even if estimator parameters are changed.
        Z = encode_features(self._transform.transform(X), self.spec_.fraction_bits,
                            self.spec_.feature_bits)
        votes = np.zeros((len(Z), len(self.classes_)), dtype=np.int64)
        rows = np.arange(len(Z))
        for tree in self.estimators_:
            votes[rows, tree.predict_encoded(Z)] += 1
        return self.classes_[votes.argmax(axis=1)]

    def _model_dict(self):
        return {"format": "FedIF", "version": 2,
                "split_rule": self.estimators_[0].split_rule,
                "classes": self.classes_.tolist(), "features": self.feature_parameters_,
                "spec": self.spec_.to_dict(),
                "trees": [tree.to_dict() for tree in self.estimators_]}

    def save(self, path):
        check_is_fitted(self, "estimators_")
        Path(path).write_text(json.dumps(self._model_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path, *, quant_module=None):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("format") != "FedIF" or value.get("version") not in (1, 2):
            raise ValueError("Unsupported FedIF model format")
        rule = "le" if value["version"] == 1 else value.get("split_rule")
        if rule not in ("lt", "le"):
            raise ValueError("Missing or invalid model split rule")
        spec = TrainingSpec(**value["spec"])
        sig = {key: value["features"][key] for key in ("module", "depth", "div", "length")}
        module = quant_module or sig["module"]
        model = cls(n_estimators=spec.n_estimators, max_depth=spec.height,
                    r=spec.candidates / spec.n_features, depth=sig["depth"], div=sig["div"],
                    quant_module=module, fraction_bits=spec.fraction_bits,
                    feature_bits=spec.feature_bits, entropy_fraction_bits=spec.entropy_fraction_bits,
                    entropy_bits=spec.entropy_bits, threshold_bits=spec.threshold_bits, ICS=spec.ICS, ESA=spec.ESA, m=spec.m)
        transform = QuantFeatures(model.depth, model.div, module).initialize(sig["length"])
        if transform.n_features != spec.n_features:
            raise ValueError("QUANT feature count mismatch")
        if len(value["classes"]) != spec.n_classes or len(set(value["classes"])) != spec.n_classes:
            raise ValueError("Invalid model classes")
        trees = [RDT.from_dict(tree, default_rule=rule).validate(spec.n_features, spec.n_classes, spec.feature_bits)
                 for tree in value["trees"]]
        if len(trees) != spec.n_estimators or any(t.height != spec.height or t.split_rule != rule for t in trees):
            raise ValueError("Invalid forest dimensions")
        model._transform = transform
        model.classes_ = np.asarray(value["classes"])
        model.n_features_in_ = sig["length"]
        model.n_transformed_features_ = spec.n_features
        model.max_depth_, model.max_features_ = spec.height, spec.candidates
        model.r_ = spec.candidates / spec.n_features
        model.spec_, model.estimators_ = spec, trees
        model.feature_parameters_, model.metrics_ = sig, {}
        model.model_ = model._model_dict()
        return model
