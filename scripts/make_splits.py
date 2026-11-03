"""Write the stratified single-turn split and fail on any leakage."""

import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

from escalation_probes.config import load_experiment
from escalation_probes.data import check_no_leakage, load_dataset, make_splits


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()

    cfg = load_experiment(args.config)
    single, conversations = load_dataset(cfg)
    if not single:
        raise ValueError("no single-turn records found; add redacted data before making splits")
    if not conversations:
        raise ValueError("no multi-turn or control conversations found; add redacted data before making splits")
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for label in tqdm(("benign", "harmful"), desc="Creating label splits", unit="label"):
        ids = [rec["id"] for rec in single if rec["label"] == label]
        for name, part in make_splits(ids, cfg["splits"], cfg["seed"]).items():
            splits[name] += part
    splits = {name: sorted(ids) for name, ids in splits.items()}
    check_no_leakage(splits, single, conversations)
    path = Path(cfg["paths"]["splits"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    log.info("Wrote splits: %s -> %s", {name: len(ids) for name, ids in splits.items()}, path)


if __name__ == "__main__":
    main()
