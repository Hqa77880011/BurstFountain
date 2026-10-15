import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .simulation import generate, run


def prepare(args):
    config = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for seed in config["seeds"]:
        np.savez_compressed(out / f"trace_{seed}.npz", **generate(config, seed))
    (out / "config.json").write_text(json.dumps(config, indent=2))


def evaluate(args):
    root = Path(args.data)
    config = json.loads((root / "config.json").read_text())
    paths = [root / f"trace_{seed}.npz" for seed in config["seeds"]]
    methods = ["arq", "replication", "fec", "static", "rlnc", "deadline", "full", "oracle"] if args.benchmark else [args.method]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    aggregate = {}
    for method in methods:
        rows = []
        target = out / method
        target.mkdir(exist_ok=True)
        for path in paths:
            trace = np.load(path, allow_pickle=False)
            metrics, decisions, messages = run(trace, config, method, args.feedback_loss, args.no_risk, args.no_priority, args.no_innovation)
            rows.append({"seed": int(trace["seed"]), **metrics})
            (target / f"{path.stem}_messages.json").write_text(json.dumps(messages, indent=2))
            (target / f"{path.stem}_decisions.json").write_text(json.dumps(decisions, indent=2))
        (target / "metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
        aggregate[method] = {}
        for key in rows[0]:
            if key == "seed":
                continue
            values = [row[key] for row in rows if row[key] is not None]
            aggregate[method][key] = {"mean": float(np.mean(values)) if values else None,
                                      "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                      "seeds_with_value": len(values)}
        if decisions:
            fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
            axes[0].plot([r["slot"] for r in decisions], [r["risk"] for r in decisions])
            axes[0].set_ylabel("Predicted miss risk")
            axes[1].step([r["slot"] for r in decisions], [r["repair_request"] for r in decisions])
            axes[1].set(xlabel="Slot", ylabel="Repair request")
            fig.tight_layout()
            fig.savefig(target / "control.png", dpi=180)
            plt.close(fig)
    (out / "summary.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(aggregate), [aggregate[m]["weighted_deadline_delivery"]["mean"] or 0 for m in aggregate])
    ax.set(ylabel="Weighted deadline delivery ratio")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out / "delivery.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--output", default="data/prepared")
    p.set_defaults(function=prepare)
    p = commands.add_parser("evaluate")
    p.add_argument("--data", default="data/prepared")
    p.add_argument("--output", default="runs/evaluate")
    p.add_argument("--method", choices=["full", "arq", "replication", "fec", "static", "rlnc", "deadline", "oracle"], default="full")
    p.add_argument("--feedback-loss", type=float, default=0.1)
    p.add_argument("--benchmark", action="store_true")
    for flag in ["no-risk", "no-priority", "no-innovation"]:
        p.add_argument("--" + flag, action="store_true")
    p.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)
