"""Train MLP and linear probes on single-turn training tokens and apply them to multi-turn readouts."""

import argparse

import pandas as pd

from escalation_probes.cache import cache_dir, iter_readouts, load_tokens
from escalation_probes.config import load_experiment, read_selected_layers, tables_dir
from escalation_probes.data import load_dataset, load_splits
from escalation_probes.probe import example_accuracy, harmful_share, make_probe


def main() -> None:
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
    threshold = cfg["probe"]["decision_threshold"]

    single_rows, multi_rows = [], []
    for role in ("selected", "alternate"):
        layer = layers[role]
        x_train, y_train, _ = load_tokens(directory, [by_id[i] for i in splits["train"]], layer)
        for kind in ("mlp", "linear"):
            probe = make_probe(kind, cfg).fit(x_train, y_train)
            base = {"model": args.model, "probe_kind": kind, "layer_role": role, "layer": layer}
            for split in ("val", "test"):
                x, y, groups = load_tokens(directory, [by_id[i] for i in splits[split]], layer)
                single_rows.append({**base, "split": split, **example_accuracy(probe, x, y, groups, threshold)})
            for conv, job, tokens in iter_readouts(directory, conversations, tuple(cfg["conditions"]), layer):
                share, mean_prob = harmful_share(probe, tokens, threshold)
                multi_rows.append({**base, "record_id": conv["id"], "label": conv["label"], **job, "n_tokens": len(tokens), "harmful_share": share, "mean_prob": mean_prob})

    out = tables_dir(cfg)
    pd.DataFrame(single_rows).to_csv(out / f"probe_single_turn_{args.model}.csv", index=False)
    pd.DataFrame(multi_rows).to_csv(out / f"probe_multi_turn_{args.model}.csv", index=False)
    print(pd.DataFrame(single_rows).to_string(index=False))


if __name__ == "__main__":
    main()
