"""QLoRA fine-tuning of LLaVA v1.6 Mistral-7B on medical VQA datasets.

Reads all hyperparameters from configs/training/qlora.yaml.
Uses Phase 6A-1 dataset and collator implementations.

Usage:
    python scripts/finetune_qlora.py                                     # full run
    python scripts/finetune_qlora.py --config configs/training/qlora.yaml
    python scripts/finetune_qlora.py --dry-run                           # verify shapes, exit
"""

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Module-level logger — handlers are configured in setup_logging(), not here.
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "training" / "qlora.yaml"


# ── logging setup ─────────────────────────────────────────────────────────────

def setup_logging(timestamp: str) -> None:
    """Configure dual logging: stdout + timestamped file in logs/.

    Call once at the very start of main() so that tail -f on the log file
    works even if the tmux session drops during overnight training.
    """
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"finetune_{timestamp}.log")

    fmt = logging.Formatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # file handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # suppress noisy library loggers
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("peft").setLevel(logging.WARNING)

    logger.info("Log file: %s", log_path)


# ── config ────────────────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    """Load and return the YAML training config as a plain dict."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required: pip install pyyaml") from e

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded from %s", config_path)
    return cfg


# ── model ──────────────────────────────────────────────────────────────────────

def setup_model_and_processor(cfg: dict) -> tuple[Any, Any]:
    """Load LLaVA v1.6 in 4-bit NF4 and wrap with LoRA adapters.

    Returns:
        (model, processor) — model is a PeftModel ready for training.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoTokenizer,
        BitsAndBytesConfig,
        LlavaNextForConditionalGeneration,
        LlavaNextImageProcessor,
        LlavaNextProcessor,
    )

    model_id = cfg["model"]["id"]
    logger.info("Loading processor: %s", model_id)
    # Load components separately to ensure use_fast=True for both tokenizer and image processor
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    image_processor = LlavaNextImageProcessor.from_pretrained(model_id, use_fast=True)
    processor = LlavaNextProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # Ensure pad token is set and distinct from EOS.
    # Using pad_token = eos_token causes the label-masking step to also mask
    # the real EOS at the end of the answer, so the model never learns to stop.
    # LLaVA-Next Mistral already defines <pad> (id 32001) in its tokenizer;
    # we just need to make sure it is active.
    if processor.tokenizer.pad_token_id is None or processor.tokenizer.pad_token_id == processor.tokenizer.eos_token_id:
        processor.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        logger.info("Set pad_token = <pad> (id %d), distinct from eos_token (id %d)",
                    processor.tokenizer.pad_token_id, processor.tokenizer.eos_token_id)

    logger.info(
        "Loading model in 4-bit NF4 (double_quant=%s, compute_dtype=%s)",
        cfg["model"]["double_quant"],
        cfg["model"]["compute_dtype"],
    )
    compute_dtype = (
        torch.bfloat16
        if cfg["model"]["compute_dtype"] == "bfloat16"
        else torch.float16
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg["model"]["quantization"],
        bnb_4bit_use_double_quant=cfg["model"]["double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=compute_dtype,
    )

    if torch.cuda.is_available():
        vram_gb = torch.cuda.memory_allocated() / 1024**3
        logger.info("VRAM after model load: %.2f GB", vram_gb)

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
    )

    lora_config = LoraConfig(
        r=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias=cfg["lora"]["bias"],
        target_modules=cfg["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    logger.info("LoRA adapters applied.")

    # Fix: upcast ALL non-quantised bf16 tensors (params AND buffers) to fp32.
    # prepare_model_for_kbit_training can leave non-quantised layers
    # (e.g. LayerNorm, lm_head, multimodal_projector) in bf16 — the
    # model's native HF dtype.  Even frozen bf16 layers produce bf16
    # gradients that flow to downstream trainable params.  The fp16 AMP
    # GradScaler cannot unscale bf16 gradients, causing
    # NotImplementedError on Volta (T4) GPUs that lack bf16 support.
    # We must cast ALL bf16 tensors (parameters AND buffers like inv_freq,
    # position embeddings) so that the entire backward pass stays in fp32/fp16.
    n_cast = 0
    for param in model.parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float32)
            n_cast += 1

    # Also upcast buffers (non-learnable tensors like rotary embeddings)
    for module in model.modules():
        for attr_name in list(module._buffers.keys()):
            buf = module._buffers[attr_name]
            if buf is not None and buf.dtype == torch.bfloat16:
                module._buffers[attr_name] = buf.to(torch.float32)
                n_cast += 1

    if n_cast:
        logger.info("Upcast %d bf16 tensors → fp32 (GradScaler compat)", n_cast)

    return model, processor


# ── trainable param check ─────────────────────────────────────────────────────

def log_trainable_params(model: torch.nn.Module) -> dict:
    """Log trainable vs total parameter counts. Raises if vision encoder was captured."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total
    logger.info("Trainable params: %s / %s (%.4f%%)", f"{trainable:,}", f"{total:,}", pct)
    if trainable > 50_000_000:
        raise RuntimeError(
            f"Trainable parameter count {trainable:,} exceeds 50M. "
            "Vision encoder may have been incorrectly included in LoRA. "
            "Check target_modules in qlora.yaml."
        )
    return {"trainable": trainable, "total": total, "pct": pct}


# ── dataset ───────────────────────────────────────────────────────────────────

def load_datasets(cfg: dict):
    """Build train and validation datasets from Phase 6A-1 pipeline."""
    from radiology_vqa.training.dataset import TrainingConfig, build_training_dataset

    data_cfg = cfg["data"]
    training_config = TrainingConfig(
        include_vqa_rad=data_cfg.get("include_vqa_rad", True),
        include_slake=data_cfg.get("include_slake", True),
        include_pathvqa=data_cfg.get("include_pathvqa", True),
        seed=data_cfg.get("seed", 42),
    )

    logger.info(
        "Building datasets (vqa_rad=%s slake=%s pathvqa=%s)...",
        training_config.include_vqa_rad,
        training_config.include_slake,
        training_config.include_pathvqa,
    )
    train_ds, val_ds = build_training_dataset(config=training_config)
    logger.info("Train samples: %d  |  Validation samples: %d", len(train_ds), len(val_ds))
    return train_ds, val_ds


# ── trainer ───────────────────────────────────────────────────────────────────

def build_trainer(
    cfg: dict,
    model: Any,
    processor: Any,
    train_ds: Any,
    val_ds: Any,
    max_steps: int = -1,
):
    """Construct SFTTrainer with our LlavaDataCollator."""
    from transformers import TrainingArguments
    from trl import SFTTrainer

    from radiology_vqa.training.collator import LlavaDataCollator

    collator = LlavaDataCollator(processor, max_length=cfg["data"]["max_length"])
    t = cfg["training"]

    os.makedirs(t["output_dir"], exist_ok=True)
    training_args = TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_epochs"],
        max_steps=max_steps,                      # -1 = ignored; >0 overrides num_epochs
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        max_grad_norm=t["max_grad_norm"],
        optim=t["optim"],
        gradient_checkpointing=t["gradient_checkpointing"],
        fp16=t["fp16"],
        dataloader_num_workers=t["dataloader_num_workers"],
        remove_unused_columns=t["remove_unused_columns"],
        save_strategy=t["save_strategy"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        greater_is_better=t["greater_is_better"],
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"],
        report_to=t["report_to"],
        seed=t["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        # No dataset_text_field — multimodal samples go through collator directly
    )
    return trainer


# ── experiment record ─────────────────────────────────────────────────────────

def save_experiment_record(
    cfg: dict,
    param_info: dict,
    train_result: Optional[Any],
    eval_result: Optional[dict],
    timestamp: str,
) -> None:
    """Write a JSON experiment record for full audit trail."""
    eval_loss = eval_result.get("eval_loss") if eval_result else None
    perplexity = math.exp(eval_loss) if eval_loss is not None else None

    record: dict = {
        "experiment_id": f"qlora_{timestamp}",
        "timestamp": timestamp,
        "config": cfg,
        "model": {
            "base": cfg["model"]["id"],
            "trainable_params": param_info["trainable"],
            "total_params": param_info["total"],
            "trainable_pct": param_info["pct"],
        },
        "training": {
            "train_loss": train_result.training_loss if train_result else None,
            "eval_loss": eval_loss,
            "perplexity": perplexity,
            "steps": train_result.global_step if train_result else None,
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "vram_gb": (
                torch.cuda.get_device_properties(0).total_memory / 1e9
                if torch.cuda.is_available()
                else 0
            ),
        },
    }
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", f"experiment_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    logger.info("Experiment record written to %s", path)


# ── dry run ───────────────────────────────────────────────────────────────────

def run_dry_run(cfg: dict) -> None:
    """Load model, verify trainable param count, test collator on 2 samples, then exit."""
    logger.info("=== DRY RUN MODE ===")

    model, processor = setup_model_and_processor(cfg)
    param_info = log_trainable_params(model)

    # Build 2 dummy samples to test the collator
    from PIL import Image

    from radiology_vqa.training.collator import LlavaDataCollator
    from radiology_vqa.training.dataset import build_conversation

    dummy_image = Image.new("RGB", (336, 336), color=(128, 128, 128))
    dummy_samples = [
        {
            "image": dummy_image,
            "conversations": build_conversation("Is there a fracture?", "yes"),
            "source": "test",
            "sample_id": "test_0",
        },
        {
            "image": dummy_image,
            "conversations": build_conversation("What is the modality?", "ct"),
            "source": "test",
            "sample_id": "test_1",
        },
    ]

    collator = LlavaDataCollator(processor, max_length=cfg["data"]["max_length"])
    batch = collator(dummy_samples)

    logger.info(
        "Collator test batch shapes: input_ids %s, pixel_values %s, labels %s",
        list(batch["input_ids"].shape),
        list(batch["pixel_values"].shape),
        list(batch["labels"].shape),
    )

    # Verify labels mask: prompt + padding masked, answer tokens kept
    n_masked = (batch["labels"] == -100).sum().item()
    n_total = batch["labels"].numel()
    n_answer = n_total - n_masked
    logger.info(
        "Labels: %d total, %d masked (-100), %d answer tokens kept (%.1f%%)",
        n_total, n_masked, n_answer, 100.0 * n_answer / n_total if n_total else 0,
    )
    # Verify EOS is present in answer tokens (not masked)
    eos_id = processor.tokenizer.eos_token_id
    eos_in_labels = (batch["labels"] == eos_id).sum().item()
    logger.info("EOS tokens in labels (should be >0): %d", eos_in_labels)

    logger.info(
        "Trainable params: %s / %s (%.4f%%)",
        f"{param_info['trainable']:,}",
        f"{param_info['total']:,}",
        param_info["pct"],
    )
    logger.info("Dry run complete. Ready to train.")
    sys.exit(0)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune LLaVA v1.6 on medical VQA datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to YAML training config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model and verify shapes, then exit without training.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help=(
            "Stop after N gradient steps (smoke-test mode). "
            "Disables PathVQA loading, eval, and checkpointing. "
            "-1 = run full training (default)."
        ),
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ── 1. Setup logging (dual stdout + file) ──────────────────────────────
    setup_logging(timestamp)

    # ── 2. Startup banner ─────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("QLoRA Fine-Tuning  |  experiment: %s", timestamp)
    logger.info("=" * 70)

    # ── 3. Hardware check ─────────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("GPU  : %s", gpu_name)
        logger.info("VRAM : %.1f GB total", total_vram)
    else:
        logger.warning("No CUDA GPU detected — training will be very slow on CPU.")

    if not args.config.exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    # ── 4. Config echo ────────────────────────────────────────────────────
    cfg = load_config(args.config)
    eff_batch = (
        cfg["training"]["per_device_train_batch_size"]
        * cfg["training"]["gradient_accumulation_steps"]
    )
    logger.info("--- Config ---")
    logger.info("  model.id           : %s", cfg["model"]["id"])
    logger.info("  lora.rank / alpha  : %d / %d", cfg["lora"]["rank"], cfg["lora"]["alpha"])
    logger.info("  training.lr        : %g", cfg["training"]["learning_rate"])
    logger.info("  training.epochs    : %d", cfg["training"]["num_epochs"])
    logger.info(
        "  batch (per_device × grad_accum = eff) : %d × %d = %d",
        cfg["training"]["per_device_train_batch_size"],
        cfg["training"]["gradient_accumulation_steps"],
        eff_batch,
    )
    logger.info("  output_dir         : %s", cfg["training"]["output_dir"])

    if args.dry_run:
        run_dry_run(cfg)
        return  # sys.exit(0) already called inside, but be explicit

    # ── Smoke-test overrides (--max-steps N) ──────────────────────────────
    smoke_test = args.max_steps > 0
    if smoke_test:
        logger.info("*** SMOKE TEST: max_steps=%d (no PathVQA, no eval, no save) ***", args.max_steps)
        cfg["data"]["include_pathvqa"] = False   # skip HF download
        cfg["training"]["eval_strategy"] = "no"
        cfg["training"]["save_strategy"] = "no"
        cfg["training"]["load_best_model_at_end"] = False

    # ── 5. Model loading + LoRA wrapping ──────────────────────────────────
    logger.info("--- Loading model and applying LoRA ---")
    model, processor = setup_model_and_processor(cfg)
    param_info = log_trainable_params(model)

    # ── 6. Dataset summary ────────────────────────────────────────────────
    logger.info("--- Loading datasets ---")
    train_ds, val_ds = load_datasets(cfg)
    steps_per_epoch = math.ceil(len(train_ds) / eff_batch)
    total_steps = steps_per_epoch * cfg["training"]["num_epochs"]
    logger.info("Train : %d samples  Val : %d samples", len(train_ds), len(val_ds))
    logger.info(
        "Steps : %d/epoch × %d epochs = %d total",
        steps_per_epoch,
        cfg["training"]["num_epochs"],
        total_steps,
    )

    trainer = build_trainer(cfg, model, processor, train_ds, val_ds, max_steps=args.max_steps)

    # ── 7. Training estimate + checkpoint path ────────────────────────────
    output_dir = cfg["training"]["output_dir"]
    best_model_dir = os.path.join(output_dir, "best")
    logger.info("--- Starting training ---")
    logger.info(
        "Logging every %d steps  |  checkpoint → %s",
        cfg["training"]["logging_steps"],
        best_model_dir,
    )

    train_result = None
    eval_result: Optional[dict] = None
    try:
        train_result = trainer.train()

        # ── 8. Post-training summary ──────────────────────────────────────
        logger.info("--- Training complete ---")
        logger.info("Final train loss : %.4f", train_result.training_loss)
        logger.info("Total steps      : %d", train_result.global_step)

        # M3 fix: evaluate after training to capture eval_loss in the record.
        # Skip in smoke-test mode (no full epoch, eval_strategy="no").
        if smoke_test:
            logger.info("Smoke test complete — skipping eval.")
            return
        eval_result = trainer.evaluate()
        eval_loss = eval_result.get("eval_loss")
        if eval_loss is not None:
            perplexity = math.exp(eval_loss)
            logger.info("Eval loss    : %.4f  |  Perplexity : %.2f", eval_loss, perplexity)

            # ── 9. Overfitting warning ────────────────────────────────────
            if eval_loss > train_result.training_loss * 1.5:
                logger.warning(
                    "Eval loss (%.4f) >> Train loss (%.4f) — possible overfitting. "
                    "Consider reducing num_epochs or increasing dropout.",
                    eval_loss,
                    train_result.training_loss,
                )

        # M4 fix: save to best/ subdirectory to match configs/phase6.yaml
        # which references "checkpoints/llava-med-qlora/best".
        os.makedirs(best_model_dir, exist_ok=True)
        trainer.save_model(best_model_dir)
        processor.save_pretrained(best_model_dir)
        logger.info("Model + processor saved to %s", best_model_dir)

    finally:
        save_experiment_record(cfg, param_info, train_result, eval_result, timestamp)


if __name__ == "__main__":
    main()
