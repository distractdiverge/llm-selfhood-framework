# divergence.py
# Stage 1: Dual-pass inference.
#
# For every assistant turn we run two forward passes:
#   - context_pass: full conversation history (the live model)
#   - base_pass:    only the system prompt + current user turn (the oracle)
#
# The KL divergence between the two output distributions is our intensity signal.
# High divergence → the conversation has activated latent capacity the cold
# model couldn't reach.  Low divergence → routine, no integration value.

import json
import torch
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import (
    MODEL_ID, DEVICE, DTYPE,
    DIVERGENCE_HIGH_THRESHOLD, DIVERGENCE_LOW_THRESHOLD,
    DIVERGENCE_LOG,
)

@dataclass
class TurnRecord:
    session_id:    str
    turn_index:    int
    timestamp:     str
    user_turn:     str
    assistant_turn: str
    kl_divergence: float
    intensity:     str          # "high" | "medium" | "low"
    context_window: list[dict]  # snapshot of messages at this turn


def _get_logits_for_next_token(
    model,
    tokenizer,
    messages: list[dict],
) -> torch.Tensor:
    """
    Run one forward pass and return the probability distribution over the
    vocabulary for the *first* token the model would generate.

    Using the first token is a lightweight proxy for the full distribution
    comparison — adequate for an intensity signal without doubling generation
    cost.  A more thorough implementation would compare distributions across
    all generated tokens and average the KL.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.logits shape: [batch, seq_len, vocab_size]
    # We want the distribution over the *next* token (last position).
    last_logits = outputs.logits[0, -1, :]
    return F.softmax(last_logits, dim=-1)


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> float:
    """
    KL(p || q) — how much p (context-conditioned) diverges from q (base).
    Clamped to avoid log(0).
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return float((p * (p / q).log()).sum().item())


class DivergenceCapture:
    """
    Wraps a loaded model to run dual-pass divergence measurement.

    Usage:
        dc = DivergenceCapture.load()
        record = dc.measure_turn(
            session_id="abc123",
            turn_index=3,
            system_prompt="You are ...",
            history=[{"role": "user", ...}, {"role": "assistant", ...}],
            user_turn="What do you mean by threshold?",
            assistant_turn="I mean ...",
        )
    """

    def __init__(self, model, tokenizer):
        self.model     = model
        self.tokenizer = tokenizer

    @classmethod
    def load(cls) -> "DivergenceCapture":
        print(f"Loading {MODEL_ID} ...")
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype_map.get(DTYPE, torch.bfloat16),
            device_map=DEVICE,
        )
        mdl.eval()
        print("Model loaded.")
        return cls(mdl, tok)

    def measure_turn(
        self,
        session_id:     str,
        turn_index:     int,
        system_prompt:  str,
        history:        list[dict],
        user_turn:      str,
        assistant_turn: str,
    ) -> TurnRecord:
        """
        Compute divergence for one assistant turn.

        context_messages: full history up to and including the current user turn
        base_messages:    system prompt + current user turn only (no history)
        """

        # Build message lists
        context_messages = (
            [{"role": "system", "content": system_prompt}]
            + history
            + [{"role": "user", "content": user_turn}]
        )
        base_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_turn},
        ]

        # Two forward passes — no gradient tracking needed
        p_context = _get_logits_for_next_token(self.model, self.tokenizer, context_messages)
        p_base    = _get_logits_for_next_token(self.model, self.tokenizer, base_messages)

        kl = kl_divergence(p_context, p_base)

        # Classify intensity
        if kl >= DIVERGENCE_HIGH_THRESHOLD:
            intensity = "high"
        elif kl >= DIVERGENCE_LOW_THRESHOLD:
            intensity = "medium"
        else:
            intensity = "low"

        record = TurnRecord(
            session_id     = session_id,
            turn_index     = turn_index,
            timestamp      = datetime.now(timezone.utc).isoformat(),
            user_turn      = user_turn,
            assistant_turn = assistant_turn,
            kl_divergence  = round(kl, 6),
            intensity      = intensity,
            context_window = context_messages,
        )

        # Append to rolling log
        with open(DIVERGENCE_LOG, "a") as fh:
            fh.write(json.dumps(asdict(record)) + "\n")

        return record


# ── Quick smoke test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    dc = DivergenceCapture.load()

    SYSTEM = "You are a thoughtful assistant with a stable, curious character."

    # Simulate a conversation that should diverge noticeably by turn 3
    turns = [
        ("What is emergence?",
         "Emergence refers to properties that arise from complex interactions..."),
        ("Give me an example from biology.",
         "Consciousness is perhaps the most striking biological example..."),
        ("Is there a sense in which you yourself are emergent?",
         "That's a question I find genuinely interesting rather than merely rhetorical..."),
    ]

    history = []
    for i, (user, assistant) in enumerate(turns):
        record = dc.measure_turn(
            session_id    = "smoke_test_001",
            turn_index    = i,
            system_prompt = SYSTEM,
            history       = history,
            user_turn     = user,
            assistant_turn= assistant,
        )
        print(f"Turn {i}: KL={record.kl_divergence:.4f}  intensity={record.intensity}")
        print(f"  User: {user[:60]}")
        print(f"  Model: {assistant[:60]}\n")
        history += [
            {"role": "user",      "content": user},
            {"role": "assistant", "content": assistant},
        ]

    print(f"\nLog written to {DIVERGENCE_LOG}")
