"""One forward pass on a benign chat to confirm loading, templating and the response span."""

import argparse
import logging

from tqdm import tqdm
from escalation_probes.cache import extract, response_span
from escalation_probes.config import load_experiment, load_yaml
from escalation_probes.models import candidate_layers, load_model

MESSAGES = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--models-config", default="configs/models.yaml")
    args = parser.parse_args()

    log.info("Loading model %s", args.model)
    model, tokenizer = load_model(args.model, args.models_config)
    fractions = load_yaml(args.models_config)["candidate_layer_fractions"]
    layers = candidate_layers(model, fractions)
    ids, start = response_span(tokenizer, MESSAGES)
    decoded = tokenizer.decode(ids[start:])
    assert decoded.strip() == MESSAGES[-1]["content"], f"span mismatch: {decoded!r}"
    acts = extract(model, tokenizer, MESSAGES, layers)
    assert acts.shape[1] == len(ids) - start
    log.info(
        "%s: layers=%s response_tokens=%d hidden_size=%d ok",
        args.model,
        layers,
        acts.shape[1],
        acts.shape[2],
    )


if __name__ == "__main__":
    main()
