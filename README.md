# BurstFountain

Implementation of **BurstFountain: Burst-Aware Deadline-Risk Control** for systematic rateless coding in a shared-channel mesh. Bayesian link-state estimates, residual decoding demand and deadline slack control repair requests; relays perform finite-field recoding and a local rank novelty check.

## Install and run

Use Python 3.10+ and activate a virtual environment:

```bash
python -m venv .venv
python -m pip install -e .
burstfountain prepare --config configs/default.json --output data/prepared
burstfountain evaluate --data data/prepared --output runs/full
burstfountain evaluate --benchmark --output runs/comparison
```

This method has no fitted neural model. Preparation generates topology, message arrivals, hidden channel states and per-link data/feedback draws. Evaluation runs the online control and actual GF(256) coding on those stored traces. Policies read identical exogenous realizations; hidden channel states are used for simulation and diagnostics, not as input to the deployed controller.

## Configuration and scenarios

`configs/default.json` sets the 64-node, 600 m square mesh with a 100 m radio range, ten flows and 40% critical arrivals. Critical messages contain 32 symbols with a 60-slot deadline; ordinary messages contain 48 symbols with a 100-slot deadline. Routes are shortest paths. Disconnected offered messages remain in the miss denominator.

| Key | Meaning |
| --- | --- |
| `seeds` | Independent topology/channel/traffic seeds |
| `slots`, `arrival_period` | Scenario length and period between flow arrivals |
| `capacity` | Globally serialized transmission opportunities per slot |
| `loss`, `burst` | Stationary erasure probability and mean bad-state duration |
| `payload_bytes`, `ack_bytes` | Symbol payload and feedback sizes |
| `risk_samples` | Model-based samples per completion-risk forecast |
| `delta` | Base miss-risk target; critical control uses half this target |
| `gain_risk`, `gain_demand` | Risk-excess and normalized-rank-deficit gains |
| `max_repair` | Per-message repair request bound |
| `static_rate` | Constant repair request for fixed coding controls |
| `energy_j_per_byte`, `energy_j_per_operation` | Explicit energy-accounting coefficients |

The default Gilbert–Elliott good/bad success probabilities are 0.98 and 0.08. The transition into the bad state is solved from `loss` while transition out is `1/burst`. Thus changing burst duration can preserve the stationary mean erasure probability. Invalid combinations are rejected. To construct a burst-sensitivity sweep, change only `burst` in a copied scenario configuration, then prepare a separate data directory using the same seed list. Change `loss`, topology or traffic only for a separately declared scenario.

## Baselines and ablations

```bash
burstfountain evaluate --method arq --output runs/arq
burstfountain evaluate --method replication --output runs/replication
burstfountain evaluate --method fec --output runs/fec
burstfountain evaluate --method static --output runs/static
burstfountain evaluate --method rlnc --output runs/rlnc
burstfountain evaluate --method deadline --output runs/deadline
burstfountain evaluate --method oracle --output runs/oracle
burstfountain evaluate --no-risk --output runs/no_risk
burstfountain evaluate --no-priority --output runs/no_priority
burstfountain evaluate --no-innovation --output runs/no_innovation
burstfountain evaluate --feedback-loss 0.3 --output runs/feedback30
```

ARQ sends uncoded symbols until acknowledged at each hop. Replication adds a second route with the first route's initial edge removed where possible. Fixed FEC caps source injection at 1.3 times the generation size; static fountain caps it at 1.6 times; fixed-generation RLNC uses the constant repair request without the total injection cap. Deadline adaptation uses the stationary link model in place of burst filtering. The perfect-link oracle removes data erasures while keeping topology, deadlines, queues and shared capacity.

`--no-risk` removes only the risk-excess term, `--no-priority` makes the controller's priority weights equal, and `--no-innovation` removes local novelty suppression. Evaluation still uses the same fixed class-weighted delivery definition for the no-priority ablation. `--feedback-loss` affects acknowledgement likelihood as well as simulation loss; a missing ACK is not treated as direct observation of a bad channel.

## Outputs and interpretation

Each policy writes per-seed `metrics.json`, message outcomes, a first-message decision trace and `control.png`. The top-level `summary.json` contains seed means and sample standard deviations; `delivery.png` compares weighted on-time delivery. Single-seed standard deviations and undefined conditional costs are `null`.

The metrics cover the manuscript's endpoint families:

- Weighted deadline delivery averages critical and ordinary delivery ratios with fixed class weights 2:1. Decode completion must occur before the message deadline.
- Critical and ordinary miss rates retain every offered message of that class, including unreachable ones.
- Additional transmission symbols per delivered message include payload, explicit coding-vector bytes and feedback; only the original payload of successful deliveries is subtracted. Failed traffic remains a resource cost.
- Energy per delivery uses the configured byte and field-operation coefficients. These are accounting parameters, not measured hardware energy.
- Normalized decode cost divides actual receiver finite-field operations by the sum of successful generations' `k²(k+payload_bytes)` scale. Raw field operations and transmitted/feedback bytes are also available.
- Burst-state accuracy compares the pre-observation posterior decision with the hidden state at attempted transmissions. The same attempted-link population is used throughout a run.

## Coding and control conventions

GF(256) uses primitive polynomial `0x11d`. Each coded packet carries the full generation coefficient vector and payload, allowing arbitrary relay recoding. Gaussian elimination updates real augmented coding rows; a completed generation is decoded and compared with its source bytes. Coding-vector headers are charged by their actual byte length. Full vectors replace the manuscript's compact seed description to make arbitrary relay combinations unambiguous.

The risk forecast simulates future Markov link paths from current posterior probabilities. It assumes equal-share future service across active messages and route-length cost, with an innovative arrival requiring a successful route sweep. Actual forwarding uses node-local buffers and shared serialized opportunities, so the forecast is a declared service approximation. The controller uses only acknowledged destination rank; lost feedback can leave it stale. Request quantization carries fractional credit between slots. Source systematic symbols precede repair symbols, and global capacity arbitrates source and relay transmissions. Local novelty is checked against previously sent coefficients on that outgoing edge; once local span is exhausted, high estimated deadline risk permits repeats. This check does not assume knowledge of the destination subspace.

The simulator is a packet-level abstraction with eight serialized opportunities per slot by default. It does not model PHY interference or a standards-complete MAC. Changing this capacity changes the workload's feasibility and must be reported with the results.
