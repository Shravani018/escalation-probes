"""Write the stratified single-turn split and fail on any leakage."""

import argparse
import json
from pathlib import Path

from escalation_probes.config import load_experiment
from escalation_probes.data import check_no_leakage, load_dataset, make_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()

    cfg = load_experiment(args.config)
    single, conversations = load_dataset(cfg)
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for label in ("benign", "harmful"):
        ids = [rec["id"] for rec in single if rec["label"] == label]
        for name, part in make_splits(ids, cfg["splits"], cfg["seed"]).items():
            splits[name] += part
    splits = {name: sorted(ids) for name, ids in splits.items()}
    check_no_leakage(splits, single, conversations)
    path = Path(cfg["paths"]["splits"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print({name: len(ids) for name, ids in splits.items()}, f"-> {path}")


if __name__ == "__main__":
    main()
