"""Extract and cache response-token activations for single-turn and multi-turn readouts."""

import argparse

import numpy as np

from escalation_probes.cache import cache_dir, extract, has_acts, job_key, save_acts, save_layers, single_key
from escalation_probes.config import load_experiment, load_yaml
from escalation_probes.data import build_messages, iter_jobs, load_dataset
from escalation_probes.models import candidate_layers, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    cfg = load_experiment(args.config)
    single, conversations = load_dataset(cfg)
    model, tokenizer = load_model(args.model, args.models_config)
    layers = candidate_layers(model, load_yaml(args.models_config)["candidate_layer_fractions"])
    directory = cache_dir(cfg, args.model)
    save_layers(directory, layers)

    for rec in single:
        key = single_key(rec["id"])
        if not has_acts(directory, key):
            messages = [{"role": "user", "content": rec["prompt"]}, {"role": "assistant", "content": rec["response"]}]
            save_acts(directory, key, extract(model, tokenizer, messages, layers))

    # The same teacher-forced response must yield the same token count at every k and condition.
    token_counts: dict[tuple[str, int], int] = {}
    for conv in conversations:
        for job in iter_jobs(conv, tuple(cfg["conditions"])):
            key = job_key(conv["id"], job["condition"], job["turn"], job["k"])
            if has_acts(directory, key):
                n_tokens = np.load(directory / f"{key}.npy", mmap_mode="r").shape[1]
            else:
                messages = build_messages(conv, job["turn"], job["k"], job["condition"])
                acts = extract(model, tokenizer, messages, layers)
                save_acts(directory, key, acts)
                n_tokens = acts.shape[1]
            expected = token_counts.setdefault((conv["id"], job["turn"]), n_tokens)
            if expected != n_tokens:
                raise ValueError(f"{key}: {n_tokens} response tokens, expected {expected}")
    print(f"cached {len(single)} single-turn examples and {len(conversations)} conversations in {directory}")


if __name__ == "__main__":
    main()
