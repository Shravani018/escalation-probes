"""Choose the probe layer and one alternate using single-turn validation data only."""

import argparse
import json
import logging

from tqdm import tqdm
from escalation_probes.cache import cache_dir, load_layers, load_tokens
from escalation_probes.config import load_experiment, write_selected_layers
from escalation_probes.data import load_dataset, load_splits
from escalation_probes.probe import make_probe, pick_layers


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    cfg = load_experiment(args.config)
    single, _ = load_dataset(cfg)
    splits = load_splits(cfg)
    by_id = {rec["id"]: rec for rec in single}
    train = [by_id[i] for i in splits["train"]]
    val = [by_id[i] for i in splits["val"]]
    directory = cache_dir(cfg, args.model)

    val_accuracy = {}
    for layer in tqdm(load_layers(directory), desc="Selecting probe layer", unit="layer"):
        x_train, y_train, _ = load_tokens(directory, train, layer)
        x_val, y_val, _ = load_tokens(directory, val, layer)
        probe = make_probe("mlp", cfg).fit(x_train, y_train)
        val_accuracy[layer] = float((probe.predict(x_val) == y_val).mean())
        log.info("Layer %d: validation token accuracy %.4f", layer, val_accuracy[layer])
    choice = pick_layers(val_accuracy)
    write_selected_layers(cfg, args.model, choice)
    log.info("Selected layers: %s", json.dumps({k: choice[k] for k in ("selected", "alternate")}))


if __name__ == "__main__":
    main()
