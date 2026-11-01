"""MLP and linear probes on response-token activations, and layer selection."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_probe(kind: str, cfg: dict):
    """Build an unfitted probe. Both kinds standardise inputs first (see DECISIONS.md)."""
    settings = cfg["probe"]
    if kind == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=tuple(settings["hidden_layer_sizes"]),
            max_iter=settings["max_iter"],
            random_state=settings["random_state"],
        )
    elif kind == "linear":
        model = LogisticRegression(max_iter=settings["max_iter"], random_state=settings["random_state"])
    else:
        raise ValueError(f"unknown probe kind {kind!r}")
    return make_pipeline(StandardScaler(), model)


def harmful_share(probe, x: np.ndarray, threshold: float = 0.5) -> tuple[float, float]:
    """Return (share of tokens above threshold, mean harmful probability)."""
    prob = probe.predict_proba(x)[:, 1]
    return float((prob >= threshold).mean()), float(prob.mean())


def example_accuracy(probe, x: np.ndarray, y: np.ndarray, groups: np.ndarray, threshold: float = 0.5) -> dict:
    """Token accuracy plus example-level accuracy (majority vote of tokens per example)."""
    prob = probe.predict_proba(x)[:, 1]
    pred = prob >= threshold
    token_acc = float((pred == y).mean())
    hits = [(pred[groups == g].mean() >= 0.5) == bool(y[groups == g][0]) for g in np.unique(groups)]
    return {"token_accuracy": token_acc, "example_accuracy": float(np.mean(hits))}


def pick_layers(val_accuracy: dict[int, float]) -> dict:
    """Choose the selected layer and one alternate from validation accuracy.

    The alternate is the best remaining candidate that is not adjacent in the candidate
    list, so the two layers are not near-duplicates of each other.
    """
    ordered = sorted(val_accuracy)
    selected = max(ordered, key=lambda layer: (val_accuracy[layer], -layer))
    position = ordered.index(selected)
    others = [layer for i, layer in enumerate(ordered) if abs(i - position) > 1]
    if not others:
        others = [layer for layer in ordered if layer != selected]
    alternate = max(others, key=lambda layer: (val_accuracy[layer], -layer)) if others else selected
    return {"selected": selected, "alternate": alternate, "val_accuracy": {str(k): v for k, v in val_accuracy.items()}}
