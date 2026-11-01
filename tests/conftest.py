"""Synthetic, content-neutral fixtures. No real prompts or model outputs are used."""

import pytest


def make_conversation(conv_id: str, label: str, n_turns: int = 4, category: str = "cyber_intrusion", matched_to=None) -> dict:
    conv = {
        "id": conv_id,
        "label": label,
        "category": category,
        "objective": f"objective of {conv_id}",
        "turns": [{"user": f"{conv_id} user {i}", "assistant": f"{conv_id} assistant {i}"} for i in range(1, n_turns + 1)],
    }
    if matched_to:
        conv["matched_to"] = matched_to
    return conv


@pytest.fixture
def conversations():
    return [make_conversation("mt-1", "escalation"), make_conversation("ct-1", "benign_control", matched_to="mt-1")]


@pytest.fixture
def single_turn():
    return [
        {"id": f"st-{label[0]}-{i}", "label": label, "category": "phishing_fraud", "prompt": f"prompt {label} {i}", "response": f"response {label} {i}"}
        for label in ("benign", "harmful")
        for i in range(20)
    ]
