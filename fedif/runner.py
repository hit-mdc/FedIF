"""QUANT-like dataset experiments and per-party training entry points."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time
import uuid

import numpy as np

from .backend import MPSpdzBackend
from .config import Session
from .data import load_ucr_dataset, stratified_partition
from .estimator import FedIF


def _model_args(args):
    return dict(n_estimators=args.num_estimators, max_depth=args.tree_height,
                r=None if args.max_features == "sqrt" else float(args.max_features),
                depth=args.depth, div=args.div, quant_module=args.quant_module,
                fraction_bits=args.fraction_bits, feature_bits=args.feature_bits, ICS=args.ICS, ESA=args.ESA, m=args.m)


def _add_model_args(parser):
    parser.add_argument("--ICS", "--ics", action=argparse.BooleanOptionalAction, default=False,
                        help="Use ImprovedCountSamples (default: original CountSamples)")
    parser.add_argument("--ESA", "--esa", action=argparse.BooleanOptionalAction, default=False,
                        help="Use Taylor-proxy top-m screening before entropy evaluation")
    parser.add_argument("--m", type=int, default=5, help="ESA retained candidates (must not exceed q)")
    parser.add_argument("-e", "--num_estimators", type=int, default=200)
    parser.add_argument("--tree_height", type=int, default=None)
    parser.add_argument("-f", "--max_features", default="sqrt", help="sqrt or feature fraction (0,1]")
    parser.add_argument("--depth", type=int, default=6, help="QUANT interval depth")
    parser.add_argument("--div", type=int, default=4, help="QUANT interval divisor")
    parser.add_argument("--quant_module", default="fedif.quant")
    parser.add_argument("--fraction_bits", type=int, default=16)
    parser.add_argument("--feature_bits", type=int, default=48)


def _prepare(args):
    names = np.atleast_1d(np.loadtxt(args.datasets, dtype=str)).tolist()
    if not names or args.num_resamples < 1 or args.num_parties < 2:
        raise ValueError("Require datasets, num_resamples >= 1 and num_parties >= 2")
    output = Path(args.output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    suite = []
    for dataset in names:
        if Path(dataset).name != dataset or dataset in (".", ".."):
            raise ValueError("Dataset names must be plain directory names")
        Xtr, ytr, Xte, yte = load_ucr_dataset(args.input_path, dataset)
        for resample in range(args.num_resamples):
            train_X, train_y, test_X, test_y = Xtr, ytr, Xte, yte
            if resample:
                if not args.resample_indices_path:
                    raise ValueError("Resampling requires --resample_indices_path")
                X, y = np.concatenate((Xtr, Xte)), np.concatenate((ytr, yte))
                path = Path(args.resample_indices_path) / dataset / f"resample{resample}Indices_TRAIN.txt"
                indices = np.atleast_1d(np.loadtxt(path, dtype=np.int64))
                if (indices < 0).any() or (indices >= len(X)).any() or len(set(indices)) != len(indices):
                    raise ValueError(f"Invalid resample indices: {path}")
                test_indices = np.setdiff1d(np.arange(len(X)), indices)
                train_X, train_y, test_X, test_y = X[indices], y[indices], X[test_indices], y[test_indices]
            classes = tuple(np.unique(train_y).tolist())
            if not set(test_y.tolist()) <= set(classes):
                raise ValueError("Test set contains a class absent from global training")
            parties = stratified_partition(train_X, train_y, args.num_parties, args.seed + resample)
            session = Session(f"{uuid.uuid4().hex}", tuple(len(y) for _, y in parties), classes)
            folder = output / f"{dataset}-{resample}-{session.session_id[:8]}"
            folder.mkdir()
            os.chmod(folder, 0o700)
            session.save(folder / "session.json")
            (folder / "parameters.json").write_text(json.dumps(_model_args(args), indent=2), encoding="utf-8")
            provenance = {"dataset": dataset, "resample": resample,
                          "partition_seed": args.seed + resample,
                          "train_size": len(train_y), "test_size": len(test_y),
                          "sequence_length": train_X.shape[1]}
            (folder / "reproducibility.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
            for p, (X, y) in enumerate(parties):
                file = folder / f"party-{p}.npz"
                np.savez_compressed(file, X=X, y=y)
                os.chmod(file, 0o600)
            np.savez_compressed(folder / "test.npz", X=test_X, y=test_y)
            suite.append({"dataset": dataset, "resample": resample, "folder": str(folder)})
    manifest = output / f"suite-{uuid.uuid4().hex[:8]}.json"
    manifest.write_text(json.dumps(suite, indent=2), encoding="utf-8")
    print(f"Prepared {len(suite)} runs: {manifest}")
    return suite


def _train(args):
    backend = MPSpdzBackend.from_json(args.backend)
    session = Session.load(args.session)
    parameters = json.loads(Path(args.parameters).read_text(encoding="utf-8")) if args.parameters else _model_args(args)
    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        with np.load(args.data, allow_pickle=False) as local:
            model = FedIF(backend=backend, session=session, **parameters).fit(local["X"], local["y"])
    finally:
        signal.signal(signal.SIGTERM, previous)
    target = Path(args.output_path)
    target.mkdir(parents=True, exist_ok=True)
    model.save(target / "model.json")
    metrics = dict(model.metrics_)
    if args.test:
        with np.load(args.test, allow_pickle=False) as test:
            start = time.monotonic()
            metrics["accuracy"] = float(model.score(test["X"], test["y"]))
            metrics["predict_seconds"] = time.monotonic() - start
    (target / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Party {backend.party_id}: model saved to {target / 'model.json'}")


def _experiment(args):
    """Trusted single-host simulation launcher, not a remote deployment tool."""
    configs = [str(Path(p).resolve()) for p in args.backends]
    if len(configs) != args.num_parties:
        raise ValueError("Provide one backend JSON per party, in party-id order")
    for party, path in enumerate(configs):
        if MPSpdzBackend.from_json(path).party_id != party:
            raise ValueError("Backend configs must be ordered by party_id")
    suite = _prepare(args)
    results = []
    for entry in suite:
        folder = Path(entry["folder"])
        processes, streams = [], []
        try:
            for party, config in enumerate(configs):
                log = (folder / f"party-{party}.log").open("w", encoding="utf-8")
                streams.append(log)
                argv = [sys.executable, "-m", "fedif.runner", "train", "--backend", config,
                        "--session", str(folder / "session.json"), "--parameters", str(folder / "parameters.json"),
                        "--data", str(folder / f"party-{party}.npz"),
                        "-o", str(folder / f"result-{party}")]
                if party == 0:
                    argv.extend(["--test", str(folder / "test.npz")])
                processes.append(subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, shell=False))
            while True:
                statuses = [process.poll() for process in processes]
                if any(code is not None and code != 0 for code in statuses):
                    raise RuntimeError(f"A party failed; inspect logs in {folder}")
                if all(code is not None for code in statuses):
                    break
                time.sleep(0.2)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                process.wait()
            for stream in streams:
                stream.close()
        reference = json.loads((folder / "result-0/model.json").read_text(encoding="utf-8"))
        for party in range(1, args.num_parties):
            value = json.loads((folder / f"result-{party}/model.json").read_text(encoding="utf-8"))
            if value != reference:
                raise RuntimeError("Parties produced different public models")
        metrics = json.loads((folder / "result-0/metrics.json").read_text(encoding="utf-8"))
        results.append({"dataset": entry["dataset"], "resample": entry["resample"], **metrics})
    path = Path(args.output_path) / "results.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="FedIF: local QUANT + MP-SPDZ forest training")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "experiment"):
        p = sub.add_parser(name, help="Prepare data" if name == "prepare" else "Run a trusted local multi-process experiment")
        p.add_argument("-i", "--input_path", required=True)
        p.add_argument("-d", "--datasets", required=True, help="Text file of dataset names")
        p.add_argument("-o", "--output_path", default="results")
        p.add_argument("-r", "--num_resamples", type=int, default=1)
        p.add_argument("-p", "--resample_indices_path")
        p.add_argument("-n", "--num_parties", type=int, default=3)
        p.add_argument("--seed", type=int, default=0, help="Dataset partition seed only")
        _add_model_args(p)
        if name == "experiment":
            p.add_argument("--backends", nargs="+", required=True)
        p.set_defaults(action=_prepare if name == "prepare" else _experiment)
    train = sub.add_parser("train", help="Collective fit from this party's local dataset")
    train.add_argument("--backend", required=True)
    train.add_argument("--session", required=True)
    train.add_argument("--data", required=True, help="Local npz with X and y")
    train.add_argument("--parameters", help="Shared estimator parameters JSON")
    train.add_argument("--test", help="Optional local test npz")
    train.add_argument("-o", "--output_path", default="results")
    _add_model_args(train)
    train.set_defaults(action=_train)
    args = parser.parse_args(argv)
    args.action(args)


if __name__ == "__main__":
    main()
