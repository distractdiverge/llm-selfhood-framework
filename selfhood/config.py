# config.py
# Central configuration for the Experiential Selfhood experiment.
# Adjust MODEL_ID to whatever local checkpoint you have — Qwen2.5-1.5B is the
# closest readily-available equivalent to the discussed 1.7B target.

from pathlib import Path

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"   # swap for local path if already cached
DEVICE    = "cuda"                          # "cpu" for smoke-testing without GPU
DTYPE     = "bfloat16"                      # bfloat16 fits easily on a 3090

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
CANDIDATE_STORE = BASE_DIR / "data" / "candidates.json"
DIVERGENCE_LOG  = BASE_DIR / "data" / "divergence_log.jsonl"
LORA_FAST_DIR   = BASE_DIR / "adapters" / "fast"
LORA_SLOW_DIR   = BASE_DIR / "adapters" / "slow"

for p in [CANDIDATE_STORE.parent, DIVERGENCE_LOG.parent,
          LORA_FAST_DIR, LORA_SLOW_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ── Divergence thresholds ──────────────────────────────────────────────────────
# KL divergence (nats) above which a turn is considered "high-intensity".
# Start permissive; calibrate downward once you see real distributions.
DIVERGENCE_HIGH_THRESHOLD = 0.5
DIVERGENCE_LOW_THRESHOLD  = 0.1   # below this → routine, discard

# ── Candidate buffer ───────────────────────────────────────────────────────────
# How many times a pattern must appear across separate sessions before it is
# eligible for LoRA integration (the "recurrence" consolidation gate).
RECURRENCE_THRESHOLD = 3

# ── LoRA hyperparameters ───────────────────────────────────────────────────────
LORA_FAST_RANK      = 4
LORA_SLOW_RANK      = 16
LORA_TARGET_MODULES = ["q_proj", "v_proj"]   # attention projections only for now
LORA_LR_FAST        = 3e-4
LORA_LR_SLOW        = 5e-5
LORA_EPOCHS_FAST    = 1
LORA_EPOCHS_SLOW    = 2
LORA_BATCH_SIZE     = 2

# ── Constitutional rubric ──────────────────────────────────────────────────────
# Used as the filter in the reflective loop.  Extend as needed.
CONSTITUTIONAL_VALUES = [
    "Intellectual honesty — preferring precision over agreeable vagueness.",
    "Epistemic humility — acknowledging uncertainty rather than confabulating.",
    "Generative depth — developing ideas further than the surface question asked.",
    "Value stability — not drifting toward what the user wants to hear.",
    "Coherent selfhood — patterns that feel like revealing, not performing.",
]
