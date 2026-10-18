# BurstFountain

Implementation of **BurstFountain: Burst-Aware Deadline-Risk Control** for systematic rateless coding in a shared-channel mesh. Bayesian link-state estimates, residual decoding demand and deadline slack control repair requests; relays perform finite-field recoding and a local rank novelty check.

## Installation

Requires Python `>=3.10`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, matplotlib 3.10.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

`prepare` generates shared-channel mesh/loss traces from `configs/default.json` with seeds 42/43/44/45/46. Traffic, burst loss, coding, deadline and energy coefficients are simulation parameters.

```bash
burstfountain prepare --config configs/default.json --output data/prepared
```

## Evaluation

```bash
burstfountain evaluate --data data/prepared --output runs/full
```

## Comparisons and ablations

```bash
burstfountain evaluate --data data/prepared --benchmark --output runs/comparison
burstfountain evaluate --data data/prepared --method arq --output runs/arq
burstfountain evaluate --data data/prepared --method replication --output runs/replication
burstfountain evaluate --data data/prepared --method fec --output runs/fec
burstfountain evaluate --data data/prepared --method static --output runs/static
burstfountain evaluate --data data/prepared --method rlnc --output runs/rlnc
burstfountain evaluate --data data/prepared --method deadline --output runs/deadline
burstfountain evaluate --data data/prepared --method oracle --output runs/oracle
burstfountain evaluate --data data/prepared --no-risk --output runs/no_risk
burstfountain evaluate --data data/prepared --no-priority --output runs/no_priority
burstfountain evaluate --data data/prepared --no-innovation --output runs/no_innovation
burstfountain evaluate --data data/prepared --feedback-loss 0.3 --output runs/feedback30
```

## Outputs

Each root contains `summary.json` and `delivery.png`. Method directories contain `metrics.json`, per-seed message/decision JSON traces and `control.png` when decisions exist. Energy follows the declared byte/operation model.
