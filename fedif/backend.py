"""Launch existing MP-SPDZ binaries without exposing protocol details to fit()."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from .config import positive_int
from .model import parse_model
from .streaming import run_ics_online


@dataclass
class MPSpdzBackend:
    """Run Semi offline preprocessing and online training in a party environment.

    Each party owns a separate runtime directory. The installed MP-SPDZ must
    use relative Programs/, Player-Data/ and TLS paths (upstream defaults).
    network_args are argv tokens shared by offline and online executables.
    """

    mp_spdz_path: str
    party_id: int
    work_dir: str = "sessions"
    network_args: tuple = ()
    python_executable: str = sys.executable
    timeout: float = 86400
    keep_private_inputs: bool = False

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def train(self, Z, y, spec, session):
        positive_int("party_id", self.party_id, 0)
        if self.party_id >= len(session.party_sizes):
            raise ValueError("party_id outside session")
        root = Path(self.mp_spdz_path).resolve()
        for filename in ("compile.py", "semi-offline.x", "semi-party.x"):
            if not (root / filename).is_file():
                raise FileNotFoundError(f"Missing MP-SPDZ dependency: {root / filename}")
        for arg in self.network_args:
            if not isinstance(arg, str):
                raise ValueError("network_args must be a list/tuple of argv strings")
        # These flags are owned by FedIF and may not silently change the protocol.
        forbidden = {"-IF", "-OF", "-F", "-L", "-p", "-N", "-S", "-I",
                     "--input-file", "--output-file", "--player", "--security",
                     "--file-preprocessing", "--live-preprocessing", "--interactive"}
        if any(arg.split("=")[0] in forbidden for arg in self.network_args):
            raise ValueError("network_args may only configure the existing network")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        name = "fedif"
        work = Path(self.work_dir).resolve() / f"{session.session_id}-P{self.party_id}"
        # Never reuse a session directory or its offline material, even on failure.
        work.mkdir(parents=True, exist_ok=False)
        os.chmod(work, 0o700)
        for folder in ("Programs/Source", "Programs/Bytecode", "Programs/Schedules",
                       "Programs/Public-Input", "Player-Data"):
            (work / folder).mkdir(parents=True, exist_ok=True)
        # Only TLS identity files, never triples, input files or MAC key material.
        source_tls = root / "Player-Data"
        for pattern in ("P*.pem", "P*.key", "P*.crt", "*.0"):
            for source in source_tls.glob(pattern):
                if source.suffix == ".key" and source.stem != f"P{self.party_id}":
                    continue
                target = work / "Player-Data" / source.name
                shutil.copyfile(source, target)
                os.chmod(target, 0o600)
        (work / "parameters.json").write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
        source_path = work / "Programs" / "Source" / (name + ".mpc")
        source_path.write_text(self._source() + "\n\nbuild_forest(" + repr(spec.to_dict())
                               + ")\n", encoding="utf-8")
        input_prefix = work / "Player-Data" / "Input"
        input_file = Path(str(input_prefix) + f"-P{self.party_id}-0")
        metrics = {}
        try:
            self._write_inputs(input_file, Z, y, spec.n_classes)
            metrics["compile_seconds"] = self._run(
                [self.python_executable, str(root / "compile.py"), "-M", "-F", str(spec.integer_bits), name],
                work, "compile")
            common = ["-N", str(len(session.party_sizes)), "-p", str(self.party_id),
                      *self.network_args, name]
            metrics["offline_seconds"] = self._run(
                [str(root / "semi-offline.x"), *common], work, "offline")
            if spec.ICS:
                metrics.update(run_ics_online(
                    [str(root / "semi-party.x"), "-F", "-I", "-OF", ".", *common],
                    work, input_file, Z, y, spec, self.party_id, self.timeout))
            else:
                metrics["online_seconds"] = self._run(
                    [str(root / "semi-party.x"), "-F", "-IF", str(input_prefix),
                     "-OF", ".", *common], work, "online")
            metrics["ICS"] = spec.ICS
            metrics["ESA"] = spec.ESA
            metrics["m"] = spec.m
            output = (work / "online.log").read_text(encoding="utf-8", errors="replace")
            trees = parse_model(output, spec)
            (work / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            return trees, {**metrics, "run_dir": str(work)}
        finally:
            if not self.keep_private_inputs:
                input_file.unlink(missing_ok=True)

    @staticmethod
    def _source():
        return Path(__file__).with_name("mpc").joinpath("protocols.py").read_text(encoding="utf-8")

    @staticmethod
    def _write_inputs(path, Z, y, C):
        # Column-major features then column-major one-hot labels, matching MPC.
        with path.open("x", encoding="ascii") as stream:
            os.chmod(path, 0o600)
            for column in np.asarray(Z).T:
                stream.write(" ".join(str(int(value)) for value in column) + "\n")
            for c in range(C):
                stream.write(" ".join(str(int(value == c)) for value in y) + "\n")

    def _run(self, argv, cwd, phase):
        start = time.monotonic()
        log_path = cwd / f"{phase}.log"
        try:
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(argv, cwd=cwd, stdout=log,
                                           stderr=subprocess.STDOUT, shell=False)
                try:
                    code = process.wait(timeout=self.timeout)
                    if code:
                        raise subprocess.CalledProcessError(code, argv)
                finally:
                    # Also clean up on KeyboardInterrupt/SystemExit from the CLI.
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError(f"MP-SPDZ {phase} failed; inspect {log_path}. "
                               "The session is consumed; retry with a fresh session_id.") from exc
        return time.monotonic() - start
