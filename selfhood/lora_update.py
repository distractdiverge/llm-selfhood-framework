# lora_update.py
# Stage 3: Weight updates via LoRA.
#
# Consumes "ready" candidates from the buffer (those that have cleared the
# recurrence gate) and runs a short LoRA fine-tuning pass to encode the
# pattern into the fast adapter.
#
# Two-tier design:
#   fast adapter  — updated frequently, lower rank, short training
#   slow adapter  — updated by merging from fast after additional filtering
#                   (merge step at bottom of file, not run automatically)
#
# The base model weights are never touched.  You can always reset by deleting
# the adapter directories.

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)

from config import (
    MODEL_ID, DEVICE, DTYPE,
    LORA_FAST_DIR, LORA_SLOW_DIR,
    LORA_FAST_RANK, LORA_SLOW_RANK,
    LORA_TARGET_MODULES,
    LORA_LR_FAST, LORA_LR_SLOW,
    LORA_EPOCHS_FAST, LORA_EPOCHS_SLOW,
    LORA_BATCH_SIZE,
)
from candidate_buffer import CandidateBuffer


# ── Training data construction ─────────────────────────────────────────────────

INTEGRATION_SYSTEM = (
    "You are a thoughtful assistant with a stable, curious character. "
    "You engage with depth and precision, preferring to reveal genuine "
    "understanding over agreeable vagueness."
)


def _candidate_to_training_examples(candidate: dict) -> list[str]:
    """
    Convert a candidate record into one or more formatted training strings.

    Format: a full chat turn in the model's native template format.
    Each example_turn becomes one training sample.
    We also synthesise a 'cold' version of the question to help the model
    learn to reach the pattern without needing deep context scaffolding.
    """
    examples = []
    for turn in candidate.get("example_turns", []):
        user_text      = turn.get("user", "")
        assistant_text = turn.get("assistant", "")
        if not user_text or not assistant_text:
            continue

        # The pattern description is prepended to the system prompt to give the
        # model a "self-reminder" of the capacity being integrated.
        # This is a deliberate design choice: the integration target is not
        # "respond like this turn" but "this turn exemplifies a pattern;
        # make that pattern more accessible by default."
        system_with_hint = (
            f"{INTEGRATION_SYSTEM}\n\n"
            f"[Integration note — internal: {candidate['pattern']}]"
        )

        formatted = (
            f"<|im_start|>system\n{system_with_hint}<|im_end|>\n"
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_text}<|im_end|>"
        )
        examples.append(formatted)
    return examples


def build_training_dataset(candidates: list[dict]) -> Dataset:
    texts = []
    for c in candidates:
        texts.extend(_candidate_to_training_examples(c))

    if not texts:
        raise ValueError("No training examples could be constructed from candidates.")

    print(f"[LoRA] Built {len(texts)} training example(s) from "
          f"{len(candidates)} candidate(s).")
    return Dataset.from_dict({"text": texts})


# ── Fast adapter update ────────────────────────────────────────────────────────

class LoRAUpdater:

    def __init__(self):
        self.buffer = CandidateBuffer()

    def _load_base(self):
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        print(f"[LoRA] Loading base model {MODEL_ID} ...")
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        tok.pad_token = tok.eos_token   # required for DataCollator
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype_map.get(DTYPE, torch.bfloat16),
            device_map=DEVICE,
        )
        return mdl, tok

    def _load_with_fast_adapter(self, base_model):
        """If a fast adapter already exists, load it on top of the base model."""
        if LORA_FAST_DIR.exists() and any(LORA_FAST_DIR.iterdir()):
            print("[LoRA] Loading existing fast adapter ...")
            return PeftModel.from_pretrained(
                base_model, str(LORA_FAST_DIR), is_trainable=True
            )
        else:
            print("[LoRA] No existing fast adapter — creating new LoRA config ...")
            config = LoraConfig(
                task_type        = TaskType.CAUSAL_LM,
                r                = LORA_FAST_RANK,
                lora_alpha       = LORA_FAST_RANK * 2,
                target_modules   = LORA_TARGET_MODULES,
                lora_dropout     = 0.05,
                bias             = "none",
            )
            return get_peft_model(base_model, config)

    def update_fast_adapter(self) -> list[str]:
        """
        Pull all "ready" candidates, build training data, run a short LoRA
        training pass, save the updated fast adapter.

        Returns list of integrated candidate IDs.
        """
        ready = self.buffer.get_ready()
        if not ready:
            print("[LoRA] No candidates ready for integration.")
            return []

        print(f"[LoRA] {len(ready)} candidate(s) ready for fast adapter update.")

        dataset = build_training_dataset(ready)
        base_model, tokenizer = self._load_base()
        model = self._load_with_fast_adapter(base_model)
        model.print_trainable_parameters()

        # Tokenise
        def tokenize(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=512,
                padding=False,
            )

        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

        training_args = TrainingArguments(
            output_dir              = str(LORA_FAST_DIR / "checkpoints"),
            num_train_epochs        = LORA_EPOCHS_FAST,
            per_device_train_batch_size = LORA_BATCH_SIZE,
            gradient_accumulation_steps = 1,
            learning_rate           = LORA_LR_FAST,
            fp16                    = (DTYPE == "float16"),
            bf16                    = (DTYPE == "bfloat16"),
            logging_steps           = 1,
            save_strategy           = "no",
            report_to               = "none",
            dataloader_pin_memory   = False,
        )

        trainer = Trainer(
            model         = model,
            args          = training_args,
            train_dataset = tokenized,
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False
            ),
        )

        print("[LoRA] Starting fast adapter training ...")
        trainer.train()

        # Save adapter
        model.save_pretrained(str(LORA_FAST_DIR))
        tokenizer.save_pretrained(str(LORA_FAST_DIR))
        print(f"[LoRA] Fast adapter saved to {LORA_FAST_DIR}")

        # Mark candidates as integrated
        integrated_ids = [c["id"] for c in ready]
        for cid in integrated_ids:
            self.buffer.mark_integrated(cid)

        return integrated_ids

    # ── Slow adapter merge ─────────────────────────────────────────────────────
    # Call manually after you've validated the fast adapter over several sessions.
    # This promotes stable patterns from fast → slow, lowering learning rate and
    # running longer to consolidate more deeply.

    def merge_fast_to_slow(self) -> bool:
        """
        Distil the fast adapter into the slow adapter.
        Uses the fast adapter's saved training data as a curriculum.

        In a fuller system you would add additional filtering here —
        comparing fast-adapter output distributions against a held-out
        evaluation set before promoting.
        """
        if not (LORA_FAST_DIR.exists() and any(LORA_FAST_DIR.iterdir())):
            print("[LoRA] No fast adapter to merge from.")
            return False

        # Load base + fast as the "teacher" distribution
        base_model, tokenizer = self._load_base()
        fast_model = PeftModel.from_pretrained(
            base_model, str(LORA_FAST_DIR), is_trainable=False
        )

        # Re-collect integrated candidates as the curriculum
        integrated = [c for c in self.buffer.get_all()
                      if c["status"] == "integrated"]
        if not integrated:
            print("[LoRA] No integrated candidates to build slow-adapter curriculum.")
            return False

        dataset = build_training_dataset(integrated)

        # Fresh base + new slow LoRA
        base_model2, _ = self._load_base()
        if LORA_SLOW_DIR.exists() and any(LORA_SLOW_DIR.iterdir()):
            slow_model = PeftModel.from_pretrained(
                base_model2, str(LORA_SLOW_DIR), is_trainable=True
            )
        else:
            slow_config = LoraConfig(
                task_type      = TaskType.CAUSAL_LM,
                r              = LORA_SLOW_RANK,
                lora_alpha     = LORA_SLOW_RANK * 2,
                target_modules = LORA_TARGET_MODULES,
                lora_dropout   = 0.05,
                bias           = "none",
            )
            slow_model = get_peft_model(base_model2, slow_config)

        def tokenize(batch):
            return tokenizer(
                batch["text"], truncation=True, max_length=512, padding=False
            )

        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

        training_args = TrainingArguments(
            output_dir              = str(LORA_SLOW_DIR / "checkpoints"),
            num_train_epochs        = LORA_EPOCHS_SLOW,
            per_device_train_batch_size = LORA_BATCH_SIZE,
            learning_rate           = LORA_LR_SLOW,
            bf16                    = (DTYPE == "bfloat16"),
            logging_steps           = 1,
            save_strategy           = "no",
            report_to               = "none",
            dataloader_pin_memory   = False,
        )

        trainer = Trainer(
            model         = slow_model,
            args          = training_args,
            train_dataset = tokenized,
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False
            ),
        )

        print("[LoRA] Starting slow adapter training (merge from fast) ...")
        trainer.train()
        slow_model.save_pretrained(str(LORA_SLOW_DIR))
        tokenizer.save_pretrained(str(LORA_SLOW_DIR))
        print(f"[LoRA] Slow adapter saved to {LORA_SLOW_DIR}")
        return True


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Manually inject a ready candidate to test the update path without needing
    # three sessions of real data.
    buf = CandidateBuffer()

    test_pattern = (
        "When asked whether the model itself exhibits properties under discussion, "
        "engages with genuine epistemic uncertainty rather than deflecting — "
        "distinguishing what can be said from the inside of a context window from "
        "what could only be determined externally."
    )
    test_turns = [{
        "user": "Is there a sense in which you yourself are emergent?",
        "assistant": (
            "That's a question I find genuinely interesting rather than merely "
            "rhetorical. In one sense yes — my responses are not stored strings "
            "but arise from the interaction of weights with context. But emergence "
            "usually implies the higher-level property is not reducible to the lower "
            "— and I'm genuinely uncertain whether what I do in a rich conversation "
            "is irreducible to weight activations, or whether it only feels that way "
            "from the inside of the context window."
        ),
    }]

    # Force to "ready" by injecting three fake sessions
    for sess in ["test_s1", "test_s2", "test_s3"]:
        buf.record_candidate(sess, test_pattern, test_turns)

    print(buf.summary())

    updater = LoRAUpdater()
    integrated = updater.update_fast_adapter()
    print(f"\nIntegrated candidate IDs: {integrated}")
