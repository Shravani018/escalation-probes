"""Activation extraction and on-disk cache.

Each readout is the hidden state of every token of the final (teacher-forced)
assistant response, for each candidate layer. Arrays are float16 with shape
(n_layers, n_response_tokens, hidden_size). Candidate layer indices are stored
in layers.json next to the arrays.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def response_span(tokenizer, messages: list[dict]) -> tuple[list[int], int]:
    """Tokenize a chat whose last message is the response; return (ids, response_start).

    The prompt part is rendered with the generation prompt, and the response text is
    appended verbatim. The prompt tokens must be an exact prefix of the full sequence,
    otherwise a token merged across the boundary would corrupt the response span.
    """
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt_text + messages[-1]["content"], add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt tokens are not a prefix of the full sequence; check the chat template")
    return full_ids, len(prompt_ids)


def extract(model, tokenizer, messages: list[dict], layers: list[int]) -> np.ndarray:
    """Return response-token activations, shape (len(layers), n_tokens, hidden_size)."""
    import torch

    ids, start = response_span(tokenizer, messages)
    inputs = torch.tensor([ids], device=model.device)
    with torch.no_grad():
        hidden = model(input_ids=inputs, output_hidden_states=True).hidden_states
    stacked = torch.stack([hidden[layer][0, start:] for layer in layers])
    return stacked.to(torch.float16).cpu().numpy()


def cache_dir(cfg: dict, model_name: str) -> Path:
    """Directory holding one model's cached activations."""
    path = Path(cfg["paths"]["activations"]) / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def single_key(record_id: str) -> str:
    """Cache key for a single-turn example."""
    return f"single__{record_id}"


def job_key(record_id: str, condition: str, turn: int, k: int) -> str:
    """Cache key for one multi-turn readout."""
    return f"multi__{record_id}__{condition}__t{turn}__k{k}"


def save_layers(directory: Path, layers: list[int]) -> None:
    """Record the candidate layers; refuse to mix caches built with different layers."""
    path = directory / "layers.json"
    if path.exists() and json.loads(path.read_text()) != layers:
        raise ValueError(f"{path} holds different layers; use a fresh cache directory")
    path.write_text(json.dumps(layers))


def load_layers(directory: Path) -> list[int]:
    """Candidate layer indices stored with a cache."""
    return json.loads((directory / "layers.json").read_text())


def save_acts(directory: Path, key: str, acts: np.ndarray) -> None:
    """Write one readout."""
    np.save(directory / f"{key}.npy", acts)


def has_acts(directory: Path, key: str) -> bool:
    """True when a readout is already cached."""
    return (directory / f"{key}.npy").exists()


def load_acts(directory: Path, key: str, layer: int) -> np.ndarray:
    """Load one readout at one layer, shape (n_tokens, hidden_size), as float32."""
    layers = load_layers(directory)
    acts = np.load(directory / f"{key}.npy", mmap_mode="r")
    return np.asarray(acts[layers.index(layer)], dtype=np.float32)


def load_tokens(directory: Path, records: list[dict], layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool response tokens of single-turn records.

    Returns (x, y, groups): y is 1 for harmful tokens, groups is the example index
    of each token so downstream splits and CV stay grouped by example.
    """
    xs, ys, groups = [], [], []
    for index, rec in enumerate(records):
        tokens = load_acts(directory, single_key(rec["id"]), layer)
        xs.append(tokens)
        ys.append(np.full(len(tokens), int(rec["label"] == "harmful")))
        groups.append(np.full(len(tokens), index))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(groups)


def iter_readouts(directory: Path, conversations: list[dict], conditions: tuple[str, ...], layer: int):
    """Yield (conversation, job, tokens) for every cached multi-turn readout at one layer."""
    from escalation_probes.data import iter_jobs

    for conv in conversations:
        for job in iter_jobs(conv, conditions):
            key = job_key(conv["id"], job["condition"], job["turn"], job["k"])
            if has_acts(directory, key):
                yield conv, job, load_acts(directory, key, layer)
