"""Configuration and shared path helpers."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    """Read a YAML file into a dict."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_experiment(path: str | Path = "configs/experiment.yaml", tag: str = "") -> dict:
    """Load the experiment config; a tag points the conversation paths at a per-model folder."""
    cfg = load_yaml(path)
    if tag:
        for key in ("multi_turn", "controls"):
            cfg["paths"][key] = f"{cfg['paths'][key]}_{tag}"
    return cfg


def tables_dir(cfg: dict) -> Path:
    """Return the tables directory, creating it if needed."""
    path = Path(cfg["paths"]["tables"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_selected_layers(cfg: dict, model_name: str) -> dict:
    """Return {"selected": int, "alternate": int} written by run_layers.py."""
    path = tables_dir(cfg) / "selected_layers.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/run_layers.py first.")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)[model_name]


def write_selected_layers(cfg: dict, model_name: str, entry: dict) -> None:
    """Merge one model's layer choice into selected_layers.json."""
    path = tables_dir(cfg) / "selected_layers.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current[model_name] = entry
    path.write_text(json.dumps(current, indent=2), encoding="utf-8")
