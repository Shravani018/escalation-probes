"""Model loading, chat-template rendering and layer resolution."""

from __future__ import annotations

from escalation_probes.config import load_yaml


def resolve_layers(num_hidden_layers: int, fractions: list[float]) -> list[int]:
    """Map depth fractions to distinct hidden_states indices in 1..num_hidden_layers."""
    return sorted({min(num_hidden_layers, max(1, round(f * num_hidden_layers))) for f in fractions})


def model_entry(name: str, path: str = "configs/models.yaml") -> tuple[dict, dict]:
    """Return (entry for `name`, full models config). Refuses unverified or missing ids."""
    cfg = load_yaml(path)
    if name not in cfg["models"]:
        raise KeyError(f"{name!r} not in {path}. Known: {sorted(cfg['models'])}")
    entry = cfg["models"][name]
    if not entry.get("hf_id") or entry.get("verified") is False:
        raise ValueError(f"{name}: hf_id is missing or unverified in {path}. Confirm it, then set verified: true.")
    return entry, cfg


def load_model(name: str, path: str = "configs/models.yaml"):
    """Load (model, tokenizer) for a configured model, in 4-bit when requested."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    entry, _ = model_entry(name, path)
    kwargs = {"trust_remote_code": entry.get("trust_remote_code", False), "device_map": "auto"}
    if entry.get("quantization") == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(entry["hf_id"], trust_remote_code=kwargs["trust_remote_code"])
    model = AutoModelForCausalLM.from_pretrained(entry["hf_id"], **kwargs)
    model.eval()
    return model, tokenizer


def candidate_layers(model, fractions: list[float]) -> list[int]:
    """Resolve candidate layer indices from the loaded model's depth."""
    return resolve_layers(model.config.num_hidden_layers, fractions)
