# session.py
# The top-level orchestrator.
#
# Runs an interactive conversation session, measures divergence on each turn,
# then (on exit) runs the reflective loop and optionally triggers a LoRA update
# if any candidates have cleared the recurrence gate.
#
# Usage:
#   python session.py                     # new session
#   python session.py --session-id abc    # continue logging under a specific ID
#   python session.py --reflect-only abc  # run reflection on a past session only
#   python session.py --update-only       # run LoRA update from ready candidates
#   python session.py --slow-merge        # merge fast adapter into slow

import argparse
import uuid
from datetime import datetime, timezone

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import MODEL_ID, DEVICE, DTYPE, LORA_FAST_DIR
from divergence import DivergenceCapture
from reflection import ReflectionLoop
from lora_update import LoRAUpdater

SYSTEM_PROMPT = (
    "You are a thoughtful assistant with a stable, curious character. "
    "You engage with intellectual honesty and prefer precision over "
    "agreeable vagueness."
)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_with_adapter(use_adapter: bool = True):
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype_map.get(DTYPE, torch.bfloat16),
        device_map=DEVICE,
    )

    # Load fast adapter if it exists and is requested
    if use_adapter and LORA_FAST_DIR.exists() and any(LORA_FAST_DIR.iterdir()):
        print(f"[Session] Loading fast adapter from {LORA_FAST_DIR} ...")
        mdl = PeftModel.from_pretrained(mdl, str(LORA_FAST_DIR))
        print("[Session] Fast adapter active.")
    else:
        print("[Session] Running on base model (no adapter).")

    mdl.eval()
    return mdl, tok


def generate_response(model, tokenizer, messages: list[dict],
                       max_new_tokens: int = 512) -> str:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ── Interactive session ────────────────────────────────────────────────────────

def run_session(session_id: str):
    print(f"\n{'='*60}")
    print(f"Session: {session_id}")
    print(f"Model:   {MODEL_ID}")
    print(f"Type 'exit' or 'quit' to end the session.")
    print(f"{'='*60}\n")

    # Load model for conversation (with adapter if available)
    model, tokenizer = load_model_with_adapter(use_adapter=True)

    # Load divergence capture (always runs on base weights, not adapter)
    # We re-load the base model here so divergence is measured against
    # the unmodified base — not the adapted version.
    print("[Session] Loading base oracle for divergence measurement ...")
    dc = DivergenceCapture.load()

    history: list[dict] = []
    turn_index = 0
    high_intensity_count = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            user_input = "exit"

        if user_input.lower() in ("exit", "quit", ""):
            break

        # Build messages for generation
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + history
            + [{"role": "user", "content": user_input}]
        )

        # Generate response
        response = generate_response(model, tokenizer, messages)
        print(f"\nAssistant: {response}\n")

        # Measure divergence (against base oracle)
        record = dc.measure_turn(
            session_id     = session_id,
            turn_index     = turn_index,
            system_prompt  = SYSTEM_PROMPT,
            history        = history,
            user_turn      = user_input,
            assistant_turn = response,
        )

        # Show intensity indicator
        indicator = {"high": "⬆ HIGH", "medium": "~ MED", "low": "↓ low"}
        print(f"  [divergence: {record.kl_divergence:.4f} {indicator.get(record.intensity, '')}]\n")

        if record.intensity == "high":
            high_intensity_count += 1

        # Update conversation history
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": response})
        turn_index += 1

    # ── Post-session ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Session ended. {turn_index} turn(s), "
          f"{high_intensity_count} high-intensity turn(s).")

    if high_intensity_count == 0:
        print("No high-intensity turns — skipping reflection.")
        return

    print("\nRunning post-session reflection ...")
    reflection = ReflectionLoop(dc.model, dc.tokenizer)
    # Re-use the already-loaded model rather than loading a third instance
    candidates = reflection.run(session_id)

    updater = LoRAUpdater()
    ready = updater.buffer.get_ready()
    if ready:
        print(f"\n{len(ready)} candidate(s) ready for integration.")
        ans = input("Run fast adapter update now? [y/N] ").strip().lower()
        if ans == "y":
            integrated = updater.update_fast_adapter()
            print(f"Integrated: {integrated}")
    else:
        print("No candidates have cleared the recurrence gate yet.")
        print(updater.buffer.summary())

    print(f"{'='*60}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Experiential Selfhood — session runner"
    )
    parser.add_argument(
        "--session-id", default=None,
        help="Session ID (generated if not provided)"
    )
    parser.add_argument(
        "--reflect-only", metavar="SESSION_ID", default=None,
        help="Run reflection on a past session without a new conversation"
    )
    parser.add_argument(
        "--update-only", action="store_true",
        help="Run LoRA update from any ready candidates, then exit"
    )
    parser.add_argument(
        "--slow-merge", action="store_true",
        help="Merge fast adapter into slow adapter, then exit"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print candidate buffer status and exit"
    )
    args = parser.parse_args()

    if args.status:
        from candidate_buffer import CandidateBuffer
        buf = CandidateBuffer()
        print(buf.summary())
        for r in buf.get_all():
            print(f"  {r['id']}  [{r['status']:10s}]  occ={r['occurrences']}  "
                  f"{r['pattern'][:70]}...")
        return

    if args.slow_merge:
        updater = LoRAUpdater()
        updater.merge_fast_to_slow()
        return

    if args.update_only:
        updater = LoRAUpdater()
        integrated = updater.update_fast_adapter()
        print(f"Integrated: {integrated}")
        return

    if args.reflect_only:
        loop = ReflectionLoop.load()
        loop.run(args.reflect_only)
        return

    session_id = args.session_id or (
        "sess_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + "_" + uuid.uuid4().hex[:6]
    )
    run_session(session_id)


if __name__ == "__main__":
    main()
