<div align="center">

# FedIF
### Interval-Based Time Series Classification over Data Federation


[Overview](#overview) · [Setup](#environment-setup) · [Protocols](#protocols) · [UCR Experiments](#ucr-experiments) · [Python API](#python-interface) 

</div>

## Overview

FedIF is a federated `interval-based` framework for time series classification, allowing multiple data owners to build a shared interval-based classifier while retaining their time series locally. Each participant independently extracts interval-based features (e.g., QUANT); the parties then train an ensemble of randomized decision trees (RDTs) through secure multi-party computation (MPC). Training produces a public forest that supports local prediction through a scikit-learn-style interface.

<p align="center">
  <img src="https://github.com/hit-mdc/FedTSC-FedIF/blob/main/docs/FedIF-framework.jpg" width="50%">
</p>
<!---
You may also be interested in:
- [FedST](https://link.springer.com/article/10.1007/s00778-024-00865-w), a *secure* and *interpretable* federated time series classification framework by collaboratively searching for time series `shapelets` through MPC;
- [FedDict](https://ieeexplore.ieee.org/document/10836844), a practical framework built on bag-of-words style `dictionary-based` features for *privacy-preserving*, *efficient*, and *interpretable* time series classification;
- [FedTSC](https://www.vldb.org/pvldb/vol15/p3686-wang.pdf), a *secure* and *interpretable* federated time series classification system, which incorporates MPC protocols in the backends for decentralized feature extraction and classifier training while providing Sklearn-style APIs for easy use and deployment. 
-->

## Environment setup

### MP-SPDZ

Configure a Linux MP-SPDZ environment for each party, either in Docker
or on the participating hosts. Use the official guides below:

| Official guide | How to use it for FedIF |
|:--|:--|
| [Requirements](https://mp-spdz.readthedocs.io/en/latest/readme.html#requirements) and [Compilation](https://mp-spdz.readthedocs.io/en/latest/readme.html#compilation) | Install the dependencies and build for the target host. |
| [Docker setup](https://mp-spdz.readthedocs.io/en/latest/readme.html#tl-dr-docker) | Follow the upstream container workflow, building `semi-party.x` and `semi-offline.x` for FedIF. |
| [Getting Started](https://mp-spdz.readthedocs.io/en/latest/readme.html#secret-sharing) | Select Semi over a prime field and follow the execution and networking conventions for the participating machines. |
| [External working directories](https://mp-spdz.readthedocs.io/en/latest/readme.html#compiling-and-running-programs-from-external-directories) | Refer to the directory and SSL setup conventions used by the adapter. |
| [Preprocessing as required](https://mp-spdz.readthedocs.io/en/latest/readme.html#preprocessing-as-required) | Configure genuine preprocessing for the separate offline and online phases. |

FedIF uses MP-SPDZ's Semi protocol over a prime field as its unified semi-honest
backend. The adapter uses `compile.py`, `semi-offline.x`, and `semi-party.x -F`
for compilation, offline preprocessing, and online training, respectively.
Prepare both Semi binaries in the MP-SPDZ installation and party identities in
`Player-Data/`. Prepare identities and connectivity
for *three parties, numbered 0, 1, and 2*, for the examples below. Retain the
upstream relative `Programs/` and `Player-Data/` conventions. Set the installation
path, Python executable, and network arguments in each example backend JSON.
FedIF then handles compilation, offline preprocessing, and online training.

### Python dependencies

Use Python 3.12 and the pinned dependencies for reproducible experiments:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-repro.txt
python -m pip install -e '.[datasets]'
```

Install the same environment on every party. The original QUANT module is bundled
in [`fedif/quant.py`](fedif/quant.py) and imported directly for interval-based feature extraction locally; the MPC program imports
the installed MP-SPDZ `Compiler` API.

## Protocols

The basic RDT training protocol directly extends centralized algorithm using MPC. Identifying its efficiency bottlenecks in the `CountSamples` and `EvaluateSplit` stages, FedIF further proposes two acceleration methods to boost the efficiency while guaranteeing the security and classification accuracy:

- `ImprovedCountSamples` locally computes all possible sample counts for each split, and then finds and aggregates the matching ones through a hierarchical search tree-based secure query protocol.  
- `EvaluateSplitApprox` retains the top-*m* split candidates using an MPC-friendly proxy metric of information gain (IG). The MPC-expensive exact IG is computed only on these retained candidates to find the best split at each non-leaf node. 

Different protocol variants are configurable as follows: 

| Configuration | Sample counting | Split evaluation |
|:--|:--|:--|
| `ICS=False, ESA=False` | `CountSamples` in the basic protocol | `EvaluateSplit` in the basic protocol |
| `ICS=True, ESA=False` | `ImprovedCountSamples` | `EvaluateSplit` in the basic protocol |
| `ICS=False, ESA=True` | `CountSamples` in the basic protocol | `EvaluateSplitApprox` |
| `ICS=True, ESA=True` | `ImprovedCountSamples` | `EvaluateSplitApprox` |


## Configuration

| Parameter | Default | Role |
|:--|:--:|:--|
| `--num_parties` / `-n` | 3 | Number of collaborating parties |
| `n_estimators` / `-e` | 200 | Number of RDTs |
| `max_depth` / `--tree_height` | `ceil(log2(M))` | Full tree height, root at depth 0 |
| `r` / `-f` | `None` / `sqrt` | Number or ratio of features. Default `floor(sqrt(K))` features; a ratio uses `max(1,floor(r*K))` |
| `ICS` / `--ICS` | `False` | Enable `ImproveCountSamples` |
| `ESA` / `--ESA` | `False` | Enable `EvaluateSplitApprox` |
| `m` / `--m` | 5 | Number of splits retained by ESA |

## UCR experiments

### Dataset preparation

Download the univariate `.ts` files from the
[UCR/UEA Time Series Classification Archive](https://www.timeseriesclassification.com/dataset.php)
and arrange them as follows:

```text
/data/UCR/
└── GunPoint/
    ├── GunPoint_TRAIN.ts
    └── GunPoint_TEST.ts
```

List the datasets to evaluate in a text file, one name per line.
[`examples/datasets.txt`](examples/datasets.txt) starts with GunPoint.
The loader uses aeon and accepts equal-length series.
The official training set is partitioned approximately equally within each
class across the parties; evaluation uses the official test set.

### Three-party local experiment

When all three party runtimes are accessible from the same Python environment,
edit `examples/backend-local-party0.json`, `backend-local-party1.json`, and
`backend-local-party2.json` to match the installation. Run:

```bash
python run.py experiment -i /data/UCR -d examples/datasets.txt \
  -o results/default -n 3 --seed 0 --num_estimators 200 --max_features sqrt \
  --backends examples/backend-local-party0.json \
             examples/backend-local-party1.json \
             examples/backend-local-party2.json
```

To study the proposed ICS and ESA protocols, append `--ICS` and `--ESA --m 5`, respectively.
For a short initial run, use `-e 1 --tree_height 2`.
The launcher starts all participants, evaluates the public model at party 0,
and writes aggregate experiment records to `results.csv`.

### Three parties in existing containers or on separate hosts

Prepare the inputs once:

```bash
python run.py prepare -i /data/UCR -d examples/datasets.txt \
  -o results/prepared -n 3 --seed 0 --num_estimators 200 --max_features sqrt
```

The printed suite file identifies the generated run directory. Make
`session.json` and `parameters.json` available to every party, and provide
`party-i.npz` to party `i`. In the commands below, `/data/run` refers to that
directory as mounted or copied into each environment.

Configure `examples/backend-party0.json`, `backend-party1.json`, and
`backend-party2.json`. The example hostname `party0` must resolve to the first
party from every participant. Start these commands concurrently in their
respective environments:

```bash
# Party 0
python run.py train --backend examples/backend-party0.json \
  --session /data/run/session.json --parameters /data/run/parameters.json \
  --data /data/run/party-0.npz --test /data/run/test.npz -o results/party0

# Party 1
python run.py train --backend examples/backend-party1.json \
  --session /data/run/session.json --parameters /data/run/parameters.json \
  --data /data/run/party-1.npz -o results/party1

# Party 2
python run.py train --backend examples/backend-party2.json \
  --session /data/run/session.json --parameters /data/run/parameters.json \
  --data /data/run/party-2.npz -o results/party2
```

Every party exports the public forest. The party receiving `--test` also reports
test accuracy. To select ICS or ESA in this workflow, pass their flags during
`prepare`; the resulting `parameters.json` supplies the shared training settings.

### Changing the number of parties

The runner defaults to `n=3` and supports any configured `n >= 2`.

1. Prepare MP-SPDZ identities and network connectivity for party IDs
   `0, ..., n-1`, following the official guides above.
2. Create one backend JSON per party, setting its `party_id`, runtime path,
   working directory, and network arguments for the chosen environment.
3. Run `prepare` or `experiment` with `-n n`. For `experiment`, supply exactly
   `n` backend files in party-ID order. For separate environments, launch one
   `train` command per party using its matching input file.

The prepared session records all `M_i` values. FedIF derives the MPC party count
from this session and passes it to MP-SPDZ automatically. Regenerate the prepared
run when changing `n`, and use identical model settings and class ordering across
participants. For example, a four-party run uses `-n 4` and backend files for
parties 0 through 3.


### Repeated resamples

The official train/test split is used by default (Resample 0). To evaluate supplied resamples, add
`-r 30 -p /data/resamples` to `prepare` or `experiment`. The index
files use the following layout:

```text
/data/resamples/GunPoint/resample1Indices_TRAIN.txt
/data/resamples/GunPoint/resample2Indices_TRAIN.txt
```

Each file contains zero-based training indices into the concatenation of the
original TRAIN and TEST sets; the remaining indices form the test set.
Stratified party allocation uses `seed + resample`.

## Python interface

FedIF offers Sklearn-style APIs. Each party calls `fit` on its local data with the common public session:

```python
import numpy as np
from fedif import FedIF, MPSpdzBackend, Session

backend = MPSpdzBackend.from_json("examples/backend-party0.json")
session = Session.load("/data/run/session.json")

with np.load("/data/run/party-0.npz", allow_pickle=False) as local:
    model = FedIF(backend=backend, session=session, ICS=True, ESA=True, m=5)
    model.fit(local["X"], local["y"])

model.save("model.json")
model = FedIF.load("model.json")
with np.load("/data/run/test.npz", allow_pickle=False) as test:
    predictions = model.predict(test["X"])
    accuracy = model.score(test["X"], test["y"])
```

Inputs have shape `(M_i, N)` or `(M_i, 1, N)`, with `N >= 3`.
Training is collective; model loading and prediction are local.

## Code organization

| Source | Research component |
|:--|:--|
| [`fedif/quant.py`](fedif/quant.py) | Interval-based feature extraction based on QUANT |
| [`fedif/mpc/protocols.py`](fedif/mpc/protocols.py) | Protocols for RDT training |
| [`fedif/ics.py`](fedif/ics.py) | Local splitting points and sample counts |
| [`fedif/estimator.py`](fedif/estimator.py) | FedIF training and inference interface |
| [`fedif/backend.py`](fedif/backend.py) | MP-SPDZ execution |
| [`fedif/runner.py`](fedif/runner.py) | UCR preparation and experimental workflow |

## References and license

FedIF builds on [QUANT](https://github.com/angus924/quant),
[MP-SPDZ](https://github.com/data61/MP-SPDZ), and the
[UCR/UEA archive](https://www.timeseriesclassification.com/dataset.php).
Please acknowledge these resources when reporting experiments:

- Angus Dempster, Daniel F. Schmidt, and Geoffrey I. Webb.
  *QUANT: A Minimalist Interval Method for Time Series Classification.*
  Data Mining and Knowledge Discovery, 2024.
- Marcel Keller. *MP-SPDZ: A Versatile Framework for Multi-Party Computation.*
  ACM CCS, 2020. [DOI](https://doi.org/10.1145/3372297.3417872).
- Yanping Chen, Eamonn Keogh, Bing Hu, Nurjahan Begum, Anthony Bagnall,
Abdullah Mueen, and Gustavo Batista. 2015. The UCR Time Series Classification
Archive.

Released under [GNU GPL v3](LICENSE).
