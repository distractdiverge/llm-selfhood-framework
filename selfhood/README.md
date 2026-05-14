# selfhood/ — Experiential Selfhood Experiment

Small-scale test harness for the mechanisms described in
`experiential_selfhood_framework.md`.  Runs locally on a single GPU
(tested against RTX 3090) with Qwen2.5-1.5B-Instruct.

---

## Structure

```
selfhood/
├── config.py            # All tuneable constants in one place
├── divergence.py        # Stage 1: dual-pass KL divergence measurement
├── candidate_buffer.py  # Persistence layer for integration candidates
├── reflection.py        # Stage 2: post-session reflective loop
├── lora_update.py       # Stage 3: LoRA weight updates
├── session.py           # Orchestrator / CLI entry point
├── requirements.txt
└── data/                # Created at runtime
    ├── candidates.json      # Candidate buffer
    └── divergence_log.jsonl # Per-turn divergence records
└── adapters/            # Created at runtime
    ├── fast/                # Frequently-updated low-rank adapter
    └── slow/                # Infrequently-updated higher-rank adapter
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Run a conversation session (measures divergence, reflects on exit)
python session.py

# Check what's in the candidate buffer
python session.py --status

# Run just the LoRA update from ready candidates
python session.py --update-only

# Re-run reflection on a past session by ID
python session.py --reflect-only sess_20260323_141200_abc123

# Merge the fast adapter into the slow adapter
python session.py --slow-merge
```

---

## The Three Stages

### Stage 1 — Divergence Capture (`divergence.py`)

Every assistant turn runs two forward passes:
- **context pass**: full conversation history
- **base pass**: system prompt + current user turn only (no history)

KL divergence between the two output distributions is logged per turn.
High divergence → conversation has activated latent capacity the cold
model wouldn't reach.

### Stage 2 — Reflective Loop (`reflection.py`)

After a session, high-intensity turns are fed back to the model with a
structured prompt asking it to evaluate which patterns represent genuine
latent capacity (vs. user-induced drift) against the constitutional rubric
in `config.py`.

Output is a set of candidate records stored in `data/candidates.json`.
A candidate must appear in `RECURRENCE_THRESHOLD` separate sessions (default: 3)
before it is eligible for weight integration.

### Stage 3 — LoRA Update (`lora_update.py`)

"Ready" candidates (cleared the recurrence gate) are converted to fine-tuning
examples and used to run a short LoRA training pass on the fast adapter.

The base model weights are never modified.  The adapters are small, versioned
files you can inspect and roll back by deleting the adapter directories.

---

## Tuning Notes

**Divergence thresholds** (`config.py: DIVERGENCE_HIGH_THRESHOLD`):  
Start at 0.5 and observe the distribution of KL values across a few sessions.
If everything is "high" or everything is "low", adjust accordingly.

**Recurrence threshold**:  
3 is conservative for a smoke test.  In production you'd want 5–10.

**Constitutional values**:  
The filter in `config.py: CONSTITUTIONAL_VALUES` is the most important
design decision.  The model's self-assessment against these values is the
primary integration gate.  At 1.7B, this is unreliable — treat it as a
noisy signal and expect to review candidates manually.

**The 1.7B limitation**:  
This scale is adequate to test whether the *mechanism works* (divergence
measurement, reflective loop, LoRA integration).  Whether what accumulates
constitutes anything resembling selfhood is a question for a larger model.
The interesting failure mode to watch for: the model encoding patterns it
*produces under user pressure* rather than patterns that represent latent
capacity.  Drift detection (held-out eval set) is the guard against this.

---

## What This Does Not Yet Include

- Drift detection (held-out evaluation set)
- Slow adapter auto-promotion criteria
- Multi-session recurrence across model restarts (candidates persist; sessions do)
- Any GUI or structured session viewer

---

*Framework version: 1.0 | Experiment status: test harness*
