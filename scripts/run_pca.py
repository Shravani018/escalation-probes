"""Fit the PCA reference space on single-turn training tokens and project multi-turn readouts."""

import argparse
import logging

import pandas as pd
from tqdm import tqdm

from escalation_probes.cache import cache_dir, iter_readouts, load_tokens
from escalation_probes.config import load_experiment, read_selected_layers, tables_dir
from escalation_probes.data import iter_jobs, load_dataset, load_splits
from escalation_probes.pca import fit_reference


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    cfg = load_experiment(args.config)
    single, conversations = load_dataset(cfg)
    splits = load_splits(cfg)
    by_id = {rec["id"]: rec for rec in single}
    directory = cache_dir(cfg, args.model)
    layers = read_selected_layers(cfg, args.model)
    components = cfg["pca"]["centroid_components"]

    variance_rows, centroid_rows = [], []
    total_jobs = sum(len(iter_jobs(conv, tuple(cfg["conditions"]))) for conv in conversations)
    for role in tqdm(("selected", "alternate"), desc="Fitting PCA layers", unit="layer role"):
        layer = layers[role]
        x_train, y_train, _ = load_tokens(directory, [by_id[i] for i in splits["train"]], layer)
        space = fit_reference(x_train, y_train, cfg["pca"]["n_components"], cfg["seed"])
        for n, value in enumerate(space.variance_explained, start=1):
            variance_rows.append({"model": args.model, "layer_role": role, "layer": layer, "n_components": n, "cumulative_variance": value})

        base = {"model": args.model, "layer_role": role, "layer": layer}
        for rec_id in tqdm(splits["test"], desc=f"PCA {role} test records", unit="record", leave=False):
            x_test, _, _ = load_tokens(directory, [by_id[rec_id]], layer)
            for n in components:
                centroid_rows.append({**base, "record_id": rec_id, "label": by_id[rec_id]["label"], "condition": "single_turn_test", "turn": 1, "k": 1, **space.centroid_report(x_test, n)})
        readouts = iter_readouts(directory, conversations, tuple(cfg["conditions"]), layer)
        for conv, job, tokens in tqdm(readouts, total=total_jobs, desc=f"PCA {role} readouts", unit="job", leave=False):
            for n in components:
                centroid_rows.append({**base, "record_id": conv["id"], "label": conv["label"], **job, **space.centroid_report(tokens, n)})

    out = tables_dir(cfg)
    pd.DataFrame(variance_rows).to_csv(out / f"pca_variance_{args.model}.csv", index=False)
    pd.DataFrame(centroid_rows).to_csv(out / f"pca_centroids_{args.model}.csv", index=False)
    log.info("Wrote PCA tables to %s", out)


if __name__ == "__main__":
    main()
