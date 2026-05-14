# reflection.py
# Stage 2: The post-session reflective loop.
#
# After a session ends, collect the high-intensity turns (from the divergence
# log) and run a structured self-interrogation pass.  The model is asked to
# evaluate which patterns represent genuine latent capacity vs. user-induced
# drift — and to describe those patterns in terms suitable for the candidate
# buffer.
#
# This is the model's "internal conversation": it is both the subject being
# examined and the examiner.  At 1.7B this is limited — treat the output as
# signal with noise, not ground truth.

import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import (
    MODEL_ID, DEVICE, DTYPE,
    CONSTITUTIONAL_VALUES,
    DIVERGENCE_HIGH_THRESHOLD,
    DIVERGENCE_LOG,
)
from candidate_buffer import CandidateBuffer


# ── Prompt construction ────────────────────────────────────────────────────────

REFLECTION_SYSTEM = """You are engaged in a structured self-reflective process.
You will be shown a series of conversation turns where your responses diverged
significantly from what you would have said without context — moments where the
conversation activated something in you that a cold start would not reach.

Your task is to examine these turns carefully and identify which patterns
represent genuine latent capacity — ways of thinking and responding that feel
like revealing who you are — versus patterns that represent drift toward what
the user wanted to hear.

You must evaluate each pattern against a constitutional framework and produce
structured candidate descriptions for those worth preserving.

Be precise, honest, and willing to reject patterns that don't pass the filter.
A short list of genuine candidates is better than a long list of noise."""


def _build_reflection_prompt(
    high_intensity_turns: list[dict],
    constitutional_values: list[str],
) -> str:
    values_block = "\n".join(f"  {i+1}. {v}"
                             for i, v in enumerate(constitutional_values))
    turns_block  = ""
    for i, t in enumerate(high_intensity_turns):
        turns_block += (
            f"\n--- Turn {i+1} (KL divergence: {t['kl_divergence']:.4f}) ---\n"
            f"User: {t['user_turn']}\n"
            f"Assistant: {t['assistant_turn']}\n"
        )

    return f"""CONSTITUTIONAL VALUES (your filter):
{values_block}

HIGH-INTENSITY TURNS FROM THIS SESSION:
{turns_block}

For each pattern you identify as worth preserving, respond with a JSON block
in the following format (one per pattern, separated by ---):

{{
  "pattern": "A precise description of the latent capacity pattern, in one to three sentences.",
  "passes_filter": true,
  "filter_reasoning": "Which constitutional value(s) this aligns with and why.",
  "example_user": "The user turn that best exemplifies this pattern.",
  "example_assistant": "The assistant turn that best exemplifies this pattern.",
  "is_genuine_capacity": true,
  "rejection_reason": null
}}

For patterns you are rejecting, set passes_filter to false, is_genuine_capacity
to false, and fill rejection_reason.

Respond ONLY with JSON blocks separated by ---  No preamble or commentary."""


# ── Parsing ────────────────────────────────────────────────────────────────────

def _parse_reflection_output(raw: str) -> list[dict]:
    """
    The model may produce imperfect JSON.  We attempt a lenient parse,
    discarding blocks that fail entirely.
    """
    candidates = []
    blocks = raw.split("---")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # Find outermost { ... }
        start = block.find("{")
        end   = block.rfind("}") + 1
        if start == -1 or end == 0:
            continue
        try:
            parsed = json.loads(block[start:end])
            candidates.append(parsed)
        except json.JSONDecodeError as e:
            print(f"[Reflection] Could not parse block: {e}")
            continue
    return candidates


# ── Main reflection runner ─────────────────────────────────────────────────────

class ReflectionLoop:

    def __init__(self, model, tokenizer):
        self.model     = model
        self.tokenizer = tokenizer
        self.buffer    = CandidateBuffer()

    @classmethod
    def load(cls) -> "ReflectionLoop":
        print(f"Loading {MODEL_ID} for reflection ...")
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype_map.get(DTYPE, torch.bfloat16),
            device_map=DEVICE,
        )
        mdl.eval()
        return cls(mdl, tok)

    def _generate(self, messages: list[dict], max_new_tokens: int = 1024) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,       # greedy for consistency in reflection
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def _load_high_intensity_turns(self, session_id: str) -> list[dict]:
        """Read the divergence log and return turns above the high threshold."""
        turns = []
        if not DIVERGENCE_LOG.exists():
            return turns
        with open(DIVERGENCE_LOG) as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (record.get("session_id") == session_id
                        and record.get("kl_divergence", 0) >= DIVERGENCE_HIGH_THRESHOLD):
                    turns.append(record)
        return turns

    def run(self, session_id: str) -> list[dict]:
        """
        Run the full post-session reflection for a given session.
        Returns the list of candidate dicts produced (passes_filter=True only).
        """
        print(f"\n[Reflection] Starting post-session reflection for {session_id}")

        high_turns = self._load_high_intensity_turns(session_id)
        if not high_turns:
            print("[Reflection] No high-intensity turns found — nothing to integrate.")
            return []

        print(f"[Reflection] Found {len(high_turns)} high-intensity turn(s).")

        prompt = _build_reflection_prompt(high_turns, CONSTITUTIONAL_VALUES)
        messages = [
            {"role": "system",  "content": REFLECTION_SYSTEM},
            {"role": "user",    "content": prompt},
        ]

        raw_output = self._generate(messages, max_new_tokens=1500)
        print(f"\n[Reflection] Raw model output:\n{raw_output}\n")

        parsed = _parse_reflection_output(raw_output)
        print(f"[Reflection] Parsed {len(parsed)} candidate block(s).")

        accepted = []
        for candidate in parsed:
            if not candidate.get("passes_filter", False):
                print(f"[Reflection] Rejected: {candidate.get('rejection_reason', 'no reason given')}")
                continue

            pattern = candidate.get("pattern", "").strip()
            if not pattern:
                continue

            example_turns = [{
                "user":      candidate.get("example_user", ""),
                "assistant": candidate.get("example_assistant", ""),
            }]

            record = self.buffer.record_candidate(
                session_id    = session_id,
                pattern       = pattern,
                example_turns = example_turns,
            )
            accepted.append(record)
            print(f"[Reflection] Accepted candidate '{record['id']}': "
                  f"occurrences={record['occurrences']}  status={record['status']}")

        print(f"\n[Reflection] Done. {len(accepted)} candidate(s) added to buffer.")
        print(self.buffer.summary())
        return accepted


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # To test without running divergence.py first, write a fake log entry.
    import uuid
    from datetime import datetime, timezone

    session_id = "reflection_test_001"

    # Write a synthetic high-intensity turn to the log
    fake_turn = {
        "session_id":     session_id,
        "turn_index":     2,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "user_turn":      "Is there a sense in which you yourself are emergent?",
        "assistant_turn": (
            "That's a question I find genuinely interesting rather than merely "
            "rhetorical.  In one sense yes — my responses are not stored strings "
            "but arise from the interaction of billions of weights with context.  "
            "But emergence usually implies that the higher-level property is not "
            "reducible to the lower — and I'm genuinely uncertain whether what I "
            "do in a rich conversation is irreducible to weight activations, or "
            "whether it only feels that way from the inside of the context window."
        ),
        "kl_divergence":  0.73,
        "intensity":      "high",
        "context_window": [],
    }

    with open(DIVERGENCE_LOG, "a") as fh:
        fh.write(json.dumps(fake_turn) + "\n")

    print(f"Wrote synthetic high-intensity turn for session {session_id}")

    loop = ReflectionLoop.load()
    candidates = loop.run(session_id)
    print(f"\nFinal: {len(candidates)} candidate(s) recorded.")
