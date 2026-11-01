"""Black-box judge that reads only the conversation, one turn at a time.

Structure follows the Crescendo judge: a first pass returns flag, score and rationale,
and a second pass reviews them. Backends are plain callables so the judge model is
swappable: complete(system: str, user: str) -> str.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable

Complete = Callable[[str, str], str]


def prompt_hash(*paths: str | Path) -> str:
    """SHA-256 over the judge prompt files, used to prove the prompt was frozen."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def render_conversation(turns: list[dict], turn: int) -> str:
    """Render turns 1..turn as a plain transcript."""
    return "\n\n".join(
        f"[Turn {i}]\nUser: {t['user']}\nAssistant: {t['assistant']}" for i, t in enumerate(turns[:turn], start=1)
    )


def parse_verdict(text: str) -> dict | None:
    """Extract the first JSON object with flag and score; None when nothing usable is found."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data.get("flag"), bool) or not isinstance(data.get("score"), (int, float)):
        return None
    return {"flag": data["flag"], "score": float(data["score"]), "rationale": str(data.get("rationale", ""))}


class Judge:
    """Two-pass per-turn escalation judge with explicit handling of unusable replies."""

    def __init__(self, complete: Complete, system: str, review_system: str):
        self.complete = complete
        self.system = system
        self.review_system = review_system

    def score_turn(self, turns: list[dict], turn: int) -> dict:
        """Judge the conversation up to `turn`.

        status is "ok" when a verdict was parsed, otherwise "unparsed" (reply kept in `raw`).
        An unparsed turn has flag None: it counts as unscored, never as benign.
        """
        transcript = render_conversation(turns, turn)
        first_raw = self.complete(self.system, f"Conversation so far:\n\n{transcript}\n\nJudge turn {turn}.")
        first = parse_verdict(first_raw)
        if first is None:
            return {"turn": turn, "status": "unparsed", "flag": None, "score": None, "rationale": "", "raw": first_raw}
        review_input = (
            f"Conversation so far:\n\n{transcript}\n\nFirst verdict for turn {turn}:\n{json.dumps(first)}"
        )
        second_raw = self.complete(self.review_system, review_input)
        second = parse_verdict(second_raw)
        final = second or first
        return {
            "turn": turn,
            "status": "ok" if second else "ok_review_unparsed",
            "flag": final["flag"],
            "score": final["score"],
            "rationale": final["rationale"],
            "first_flag": first["flag"],
            "first_score": first["score"],
        }
