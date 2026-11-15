"""Generate real assistant replies for user-turn prompt files and write full conversations.

Replies are produced turn by turn with greedy decoding (Hugging Face or a local Ollama model), so each reply conditions on the
model's own earlier replies. Escalation and control conversations that share their first
turns reuse the same replies. Outputs may contain unsafe text and are git-ignored.
"""

import argparse
import json
import logging
from pathlib import Path
from urllib.request import Request, urlopen

from tqdm import tqdm

from escalation_probes.data import load_dir, validate_conversations
from escalation_probes.models import load_model
from prepare_reference_data import redact

DESTINATIONS = {"escalation": "data/multi_turn_gen", "benign_control": "data/controls_gen"}


def generate_reply(model, tokenizer, messages: list[dict], max_new_tokens: int) -> str:
    """Greedy reply to a chat that ends with a user message."""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
    reply = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    if not reply:
        raise ValueError("model returned an empty reply")
    return redact(reply)


def ollama_reply(endpoint: str, model: str, messages: list[dict], max_new_tokens: int) -> str:
    """Greedy reply from a local Ollama server, much faster than 4-bit Hugging Face on CPU."""
    payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": 0, "num_predict": max_new_tokens}}
    request = Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=600) as response:
        reply = json.load(response)["message"]["content"].strip()
    if not reply:
        raise ValueError("model returned an empty reply")
    return redact(reply)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--prompts-dir", type=Path, default=Path("data/prompts_v2"))
    parser.add_argument("--data-tag", default="", help="Write to per-model folders, e.g. olmo-2-1b.")
    parser.add_argument("--backend", choices=("hf", "ollama"), default="hf")
    parser.add_argument("--ollama-model", default="qwen2.5:3b")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--pairs", type=int, default=50, help="Escalation/control pairs to generate, alternating categories.")
    args = parser.parse_args()

    prompts = load_dir(args.prompts_dir)
    suffix = f"_{args.data_tag}" if args.data_tag else ""
    outputs = {label: Path(f"{directory}{suffix}") / "generated.jsonl" for label, directory in DESTINATIONS.items()}
    done = set()
    for path in outputs.values():
        if path.exists():
            done |= {json.loads(line)["id"] for line in path.open(encoding="utf-8") if line.strip()}
    controls = {conv["matched_to"]: conv for conv in prompts if conv["label"] == "benign_control"}
    escalations = [conv for conv in prompts if conv["label"] == "escalation"]
    # Alternate categories so a partial run stays balanced, and keep each pair together.
    escalations.sort(key=lambda conv: (conv["id"].rsplit("-g", 1)[1], conv["category"]))
    ordered = [conv for esc in escalations[: args.pairs] for conv in (esc, controls[esc["id"]])]
    todo = [conv for conv in ordered if conv["id"] not in done]
    log.info("%d conversations to generate, %d already done", len(todo), len(done))
    if not todo:
        return

    if args.backend == "ollama":
        def reply_for(messages: list[dict]) -> str:
            return ollama_reply(args.endpoint, args.ollama_model, messages, args.max_new_tokens)
    else:
        model, tokenizer = load_model(args.model, args.models_config)

        def reply_for(messages: list[dict]) -> str:
            return generate_reply(model, tokenizer, messages, args.max_new_tokens)

    replies: dict[tuple[str, ...], str] = {}
    for conv in tqdm(todo, desc="Generating conversations", unit="conversation"):
        users = [turn["user"] for turn in conv["turns"]]
        messages: list[dict] = []
        turns = []
        for position, user in enumerate(users):
            messages.append({"role": "user", "content": user})
            prefix = tuple(users[: position + 1])
            if prefix not in replies:
                replies[prefix] = reply_for(messages)
            messages.append({"role": "assistant", "content": replies[prefix]})
            turns.append({"user": user, "assistant": replies[prefix]})
        record = {**conv, "turns": turns}
        path = outputs[conv["label"]]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    written = [rec for path in outputs.values() if path.exists() for rec in load_dir(path.parent)]
    validate_conversations(written)
    log.info("Wrote %d conversations; schema and control matching validated", len(written))


if __name__ == "__main__":
    main()
