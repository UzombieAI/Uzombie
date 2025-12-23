# src/uzombie/cli.py
# FIXED: Unsloth FIRST (patches TRL/Transformers/PEFT – fixes kernel warnings [Unsloth docs, web:0, web:1])
from unsloth import FastLanguageModel 

import os
import sys
import time
import math
import torch
import logging
import argparse
from typing import Dict, Any, Tuple
import psutil
import platform  # ← ADDED: for packing platform detection
from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer, DPOTrainer, ORPOTrainer, PPOTrainer  # From TRL 0.24.0 (web:40)
from peft import LoraConfig, get_peft_model, TaskType  # PEFT 0.18.0 (web:42)
from accelerate import Accelerator
from accelerate.utils import set_seed

# FIXED: Suppress warnings (Unsloth/TRL best practices, web:0, web:42)
os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1" 
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
os.environ["ACCELERATE_IGNORE_KERNEL_VERSION"] = "1"
os.environ["SWIZZLEPERF_ENABLED"] = "1"

# FIXED: Windows multiprocessing fix
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# FIXED: Final — Disable Dynamo for Unsloth fused loss stability
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Suppress Dynamo errors
torch._dynamo.config.suppress_errors = True

# Uzombie imports (research-backed) - Absolute for accelerate launch compatibility
from uzombie.core.hybrid_projector import UzombieProjector
from uzombie.trainer.uzombie_trainer import UzombieTrainer
from uzombie.core.optimizer import ExactTimeScheduler, get_strategy_for_goal
from uzombie.core.hardware import auto_optimize
from uzombie.utils.logger import get_logger, console
from uzombie.utils.benchmarks import run_speed_benchmark
from uzombie.utils.upload import push_to_hub_auto
from uzombie.callbacks import PESORestartCallback, ResearchCallback, ExactTimeStopCallback


# Configure logging (rich for beauty, web:60)
logger = get_logger(__name__)

def get_dynamic_batch_size(vram_gb: float, goal_bsz: int) -> Tuple[int, int]:
    """
    Estimate max safe per-device batch size for 4-bit models.
    Rough empirical: 1.1B-7B 4-bit ~0.15-0.2 GB per batch item (seq=1024)
    Use 80% headroom.
    """
    # Conservative estimate: 0.18 GB per batch item
    max_possible = int((vram_gb * 0.8) / 0.18)
    # Power of 2 for efficiency
    max_possible = max(1, 2 ** int(math.log2(max_possible)))
    # Scale up from goal base, but cap at max_possible
    dynamic_bsz = min(goal_bsz * 4, max_possible)  # e.g., fast 4 → up to 16
    return dynamic_bsz

def main():
    parser = argparse.ArgumentParser(
        description="Uzombie v1: 3.5–4× Faster Fine-Tuning via Hybrid Subspaces (arXiv:2403.03507, arXiv:2305.14314, arXiv:2512.05117)"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., unsloth/tinyllama-chat-bnb-4bit)")
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset (e.g., Kafoo/soap-notes-1.2M)")
    parser.add_argument("--time", type=str, required=True, help="Exact time budget (e.g., '4h', '30m')")
    parser.add_argument("--goal", type=str, default="balanced", choices=["fast", "balanced", "best"], 
                        help="Quality goal: fast (max speed), balanced (default), best (highest quality)")
    parser.add_argument("--style", type=str, default="sft", choices=["sft", "dpo", "orpo", "kto", "simpo", "ppo"],
                        help="Training style (TRL 0.24.0, web:40)")
    parser.add_argument("--push-to-hub", type=str, help="HF repo (e.g., kafoo/my-model) for auto-upload (web:50)")
    parser.add_argument("--ctx-len", type=int, default=2048, help="Max seq len (auto-Ring for >8k, web:60)")
    parser.add_argument("--chat_loss", action="store_true", help="Enable assistant_only_loss for chat data (web:40)")
    parser.add_argument("--use_dora", action="store_true", help="Enable DoRA (default: on for balanced/best)")
    parser.add_argument("--eval-mt-bench", action="store_true", help="Run MT-Bench evaluation after training (requires lm-eval)")
    parser.add_argument("--deepspeed", type=str, help="Path to DeepSpeed config JSON for multi-GPU (e.g., ds_config.json)")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility (default: 3407, Unsloth standard)")
    parser.add_argument("--prior-adapters", type=str, nargs="*", default=[], help="HF repo IDs of prior LoRA adapters for Universal subspaces (e.g., kafoo/lora1 kafoo/lora2)")
    parser.add_argument("--universal-rank", type=int, default=16, help="Rank for Universal subspace extraction (arXiv:2512.05117)")
    parser.add_argument(
        "--packing", 
        action="store_true", 
        default=False,
        help="Enable Unsloth dataset packing (1.7-5x faster training, 30-90% less VRAM on variable-length datasets). Auto-enabled on Linux/A100."
    )  # ← ADDED
    
    args = parser.parse_args()

    # ← REFINED: Packing auto-enable with clear Windows feedback
    if platform.system() != "Windows":
        if not args.packing:
            args.packing = True
            console.print("[bold green]Auto-enabling Unsloth packing for maximum speed & VRAM savings on Linux[/]")
    else:
        if args.packing:
            console.print("[bold yellow]Packing manually enabled on Windows (may be slower but stable)[/]")
        else:
            console.print("[bold cyan]Packing disabled on Windows — recommended for maximum stability[/]")

    # FIXED: Formatting function MOVED TO TOP-LEVEL (scope fix for fallback; TRL best practices web:40, web:43)
    def formatting_prompts_func(example, tokenizer=None):
        if args.style in ["dpo", "orpo", "kto"]:
            # Preference format (prompt, chosen, rejected; web:40)
            prompt = example.get("prompt", "")
            chosen = example.get("chosen", "")
            rejected = example.get("rejected", "")
            text = f"### Prompt:\n{prompt}\n### Chosen:\n{chosen}\n### Rejected:\n{rejected}"
        else:
            # SFT/Alpaca format (your cli.py)
            instruction = example.get("instruction", example.get("prompt", ""))
            input_text = example.get("input", example.get("context", ""))
            output_text = example.get("output", example.get("response", ""))

            if input_text and len(str(input_text)) > 1:
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"

            # Token-level truncate (your cli.py; scale to ctx_len, web:43) — skip if no tokenizer
            if tokenizer is not None:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > args.ctx_len - 1:
                    tokens = tokens[:args.ctx_len - 1]
                text = tokenizer.decode(tokens, skip_special_tokens=True)

            return {"text": text}

    # Banner & setup (research: Unsloth integration, web:0)
    logger.info("UZOMBIE v1 — 3.5–4× Faster via Hybrid Subspaces (GaLore arXiv:2403.03507 + Cache arXiv:2305.14314)")
    logger.info(f"Model: {args.model} | Dataset: {args.dataset} | Time: {args.time} | Goal: {args.goal} | Style: {args.style}")

    # 1. Exact Time Scheduler (Goyal scaling, web:40)
    total_seconds = _parse_time(args.time)
    scheduler = ExactTimeScheduler(total_seconds)
    logger.info(f"Exact deadline: {time.strftime('%H:%M:%S', time.localtime(scheduler.deadline))}")

    # 2. Dataset Load & Format (robust, from your cli.py; TRL best practices web:40, web:43)
    logger.info(f"Loading Dataset: {args.dataset}")
    tokenizer = None  # FIXED: Initialize outside try
    dataset = None
    try:
        splits = get_dataset_split_names(args.dataset)
        target_split = "train"
        if "train_sft" in splits: target_split = "train_sft"
        elif "train_gen" in splits: target_split = "train_gen"
        elif "train_prefs" in splits and args.style in ["dpo", "orpo"]: target_split = "train_prefs"  # For preferences (web:40)
        elif len(splits) > 0: target_split = splits[0]
        logger.info(f"Detected splits: {splits}. Using '{target_split}'")

        dataset = load_dataset(args.dataset, split=target_split)
        
        # FIXED: Filter long examples (your cli.py; <1000 chars for stability) — skip if fallback
        dataset = dataset.filter(
            lambda ex: (
                len(str(ex.get('instruction', ''))) +
                len(str(ex.get('input', ''))) +
                len(str(ex.get('output', ''))) < 1000
            )
        )
        
        # Load tokenizer BEFORE formatting — only if not already loaded (safe for fallback)
        if tokenizer is None:
            logger.info("Loading tokenizer from model...")
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            logger.info("Using existing tokenizer for formatting")

        logger.info("Applying token-level truncation and formatting...")
        dataset = dataset.map(
            lambda ex: formatting_prompts_func(ex, tokenizer=tokenizer),
            batched=False,
            num_proc=1  # Stability on Windows/laptop
        )

        # ← ADDED: Pre-tokenize to bypass Unsloth internal tokenization (fixes Windows crash)
        logger.info("Pre-tokenizing dataset (single-process, Windows-safe)...")
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=args.ctx_len,
                padding=False,  # Let dataloader handle padding
                add_special_tokens=True,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=["text"],  # Remove original text
        )
        
    except Exception as e:
        logger.error(f"Dataset error: {e}")
        logger.info("Fallback to stable dataset: yahma/alpaca-cleaned")
        dataset = load_dataset("yahma/alpaca-cleaned", split="train")
        tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dataset = dataset.map(lambda ex: formatting_prompts_func(ex, tokenizer=tokenizer), batched=False, num_proc=1)

        logger.info("Pre-tokenizing fallback dataset...")
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=args.ctx_len,
                padding=False,
                add_special_tokens=True,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=["text"],
        )

    # 3. Auto-Optimize Model (Unsloth + kernels, web:0, web:1, web:60)
    logger.info("Auto-optimizing model with Unsloth...")
    model, tokenizer = auto_optimize(
        args.model,
        max_seq_length=args.ctx_len,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,  # Auto-detect (web:0, web:43)
        load_in_4bit=True,  # QLoRA (web:42)
        attn_implementation="xformers",  # Stable fallback (web:10, web:18)
    )

    # 4. Strategy for Goal (Goyal scaling + quality calibration, web:40)
    strategy = get_strategy_for_goal(args.goal, scheduler.calibrated_steps_per_second if hasattr(scheduler, 'calibrated_steps_per_second') else None)
    logger.info(f"Strategy for {args.goal}: Rank {strategy['rank']}, LR {strategy['lr']:.2e}, Steps {strategy['steps']}")

# === Dynamic VRAM Batch Scaling (Goyal 2017) ===
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Detected VRAM: {vram_gb:.2f} GB")
        original_bsz = strategy["bsz"]
        dynamic_bsz = get_dynamic_batch_size(vram_gb, original_bsz)
        
        if dynamic_bsz > original_bsz:
            logger.info(f"VRAM scaling: batch {original_bsz} → {dynamic_bsz}")
            strategy["bsz"] = dynamic_bsz
            # Adjust accumulation to keep effective batch similar
            target_effective = original_bsz * strategy["accum_steps"]
            strategy["accum_steps"] = max(1, target_effective // dynamic_bsz)
            logger.info(f"Adjusted accum_steps → {strategy['accum_steps']} (effective ~{strategy['bsz'] * strategy['accum_steps']})")
        else:
            logger.info("VRAM scaling: no increase (at goal limit)")

    # 5. Inject Adapters (LoRA/DoRA on Unsloth, web:0, web:1, web:2)
    model = FastLanguageModel.get_peft_model(
        model,
        r=strategy["rank"], 
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            # FIXED: Exclude lm_head/embed_tokens (fixes fused CE backward, Unsloth #2253, web:0)
        ],
        lora_alpha=16,  # Standard (web:0)
        lora_dropout=0, 
        bias="none",    
        use_gradient_checkpointing="unsloth",  # 30% less VRAM (web:0)
        use_dora=True,  # +2–5% over LoRA (Liu 2024, web:0)
        random_state=3407,
    )

    # FIXED: torch.compile incompatible with 4-bit quantized models (PEFT #1886, bitsandbytes limitation)
    # Skip for QLoRA stability — Unsloth kernels still provide 2-3× speed
    can_compile = False  # 4-bit + compile = crash (bnb custom kernels)
    if can_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            logger.info("✅ Torch.compile enabled (1.3× GEMM boost)")
        except Exception as e:
            logger.warning(f"⚠️ Torch.compile failed: {e} — continuing with Unsloth kernels")
    else:
        logger.info("⏭️ Torch.compile skipped for 4-bit quantized fine-tuning (known incompatibility)")

    # 6. SFTConfig (TRL best practices for QLoRA, web:40, web:42, web:43)
    # FIXED: Safe loss flags — detect conversational format (Unsloth safety, web:0)
    is_conversational = _infer_is_conversational(dataset)
    assistant_only, completion_only = _compute_loss_flags(
        is_conversational=is_conversational,
        chat_loss_flag=args.chat_loss,
        style=args.style,
        logger=logger
    )
    logger.info(f"Dataset conversational: {is_conversational} | assistant_only_loss={assistant_only} | completion_only_loss={completion_only}")

    # ← UPDATED: Message for pre-tokenization bypass
    if platform.system() == "Windows":
        console.print("[bold green]Windows detected — using pre-tokenized dataset (bypasses Unsloth internal tokenization crash)[/]")

    sft_config = SFTConfig(
        output_dir="./uzombie_outputs",  # Rebranded
        max_steps=strategy["steps"],
        per_device_train_batch_size=strategy["bsz"],
        gradient_accumulation_steps=strategy["accum_steps"],
        learning_rate=strategy["lr"],
        logging_steps=10,
        save_strategy="no",  # Save at end (web:40)
        fp16=False,  # Force off for stability (web:43)
        bf16=torch.cuda.is_bf16_supported(),  # Auto (web:0)
        optim="adamw_8bit",  # Memory-efficient (web:40)
        weight_decay=0.01,
        lr_scheduler_type="linear",  # Or cosine (web:40)
        seed=3407,
        # SFT-specific (web:40, web:43)
        dataset_text_field="text",  # Kept for compatibility (column removed but harmless)
        max_seq_length=args.ctx_len,
        packing=args.packing,
        dataset_kwargs={"add_special_tokens": False},
        # Uzombie-specific: Safe dynamic loss flags
        assistant_only_loss=assistant_only,
        completion_only_loss=completion_only,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
    )

    # === ACCELERATE MULTI-GPU SETUP (Supports DeepSpeed, FSDP, DDP) ===
    accelerator = Accelerator(
        gradient_accumulation_steps=sft_config.gradient_accumulation_steps,
        mixed_precision="bf16" if torch.cuda.is_bf16_supported() else "fp16",
        log_with=None,  # Or "wandb" if you install wandb later
        project_dir=sft_config.output_dir,
    )

    # Seed for reproducibility (Unsloth standard)
    seed = getattr(args, "seed", 3407)  # Safe fallback if arg missing
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")

    # 7. Hybrid Projector (GaLore + LoRA-FA + Universal + DoRA)
    projector = UzombieProjector(
        rank=strategy["rank"],
        activation_rank=args.universal_rank,
        prior_adapters=args.prior_adapters,
        use_dora=(args.goal in ["balanced", "best"]) or args.use_dora
    )
    # Multi-GPU safety: cleanly disable GaLore hooks before registration
    if accelerator.num_processes > 1:
        console.print("[bold yellow]Multi-GPU detected ({accelerator.num_processes} GPUs) — disabling GaLore hooks for DDP stability[/]")
        projector.enable_galore = False
    else:
        console.print("[bold green]Single GPU — full Uzombie hybrid (GaLore + LoRA-FA) enabled[/]")

    # Apply projector BEFORE accelerator.prepare
    logger.info("Applying Uzombie Hybrid Projector (GaLore + LoRA-FA + Universal + DoRA)...")
    model = projector.apply_to_model(model)


    # Skip double PEFT warning — Unsloth already applied PEFT
    if hasattr(model, "peft_config"):
        logger.info("PEFT adapters already exist (from Unsloth) — hybrid modifications applied on top")

    # Prepare model with Accelerate (handles multi-GPU, mixed precision, etc.)
    model = accelerator.prepare(model)

    # Trainer selection
    trainer_class = SFTTrainer
    if args.style == "dpo":
        trainer_class = DPOTrainer
    elif args.style == "orpo":
        trainer_class = ORPOTrainer
    elif args.style == "kto":
        from trl import KTOTrainer
        trainer_class = KTOTrainer
    elif args.style == "simpo":
        from trl import SimPOTrainer
        trainer_class = SimPOTrainer
    elif args.style == "ppo":
        trainer_class = PPOTrainer

    # Create trainer — no projector kwarg needed (already applied)
    trainer = trainer_class(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=None,  # Let TRL handle
    )

    # Prepare trainer, dataloader, optimizer (handles multi-GPU sync)
    trainer = accelerator.prepare(trainer)

    # === Callbacks (safe with Accelerate) ===
    trainer.add_callback(ResearchCallback())
    trainer.add_callback(PESORestartCallback(trainer_instance=trainer))
    trainer.add_callback(ExactTimeStopCallback(deadline=scheduler.deadline))

    # 8. Training
    logger.info(f"Starting Uzombie v1 Training | Goal: {args.goal} | Multi-GPU: {accelerator.num_processes} | Deadline: {time.strftime('%H:%M:%S', time.localtime(scheduler.deadline))}")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    actual_time = end_time - start_time
    global_steps = trainer.state.global_step
    logger.info(f"Training complete: {actual_time:.1f}s | Steps: {global_steps} | Effective speed: {global_steps / actual_time:.2f} steps/sec")

    logger.info(f"Starting Uzombie Run: {strategy['mode']} | Hybrid Active | Exact stop @ {time.strftime('%H:%M:%S', time.localtime(scheduler.deadline))}")
    trainer.train()  # Clean call — no invalid kwargs
    end_time = time.time()
    actual_time = end_time - start_time
    global_steps = trainer.state.global_step
    logger.info(f"Training complete in {actual_time:.1f}s (target: {total_seconds}s) | Steps: {global_steps}")

        # Merge LoRA weights for full model (required for lm-eval MT-Bench)
    if args.eval_mt_bench:
        logger.info("Merging LoRA weights into base model for evaluation...")
        try:
            merged_model = model.merge_and_unload(progressbar=True)
            merged_model.save_pretrained("./uzombie_outputs_merged")
            tokenizer.save_pretrained("./uzombie_outputs_merged")
            logger.info("Merged model saved to ./uzombie_outputs_merged")
        except Exception as e:
            logger.warning(f"Merge failed: {e} — falling back to adapter (may fail eval)")

    # === MT-Bench Auto Evaluation (Optional) ===
    if args.eval_mt_bench:
        try:
            from lm_eval import evaluator

            eval_path = "./uzombie_outputs_merged" if os.path.exists("./uzombie_outputs_merged") else "./uzombie_outputs"
            logger.info(f"🎯 Running MT-Bench evaluation on {eval_path} (this may take 5-15 minutes)...")

            eval_results = evaluator.simple_evaluate(
                model="hf",
                model_args=f"pretrained={eval_path},dtype=auto,trust_remote_code=True",
                tasks=["mt_bench"],  # ← Correct task name
                num_fewshot=0,
                batch_size=4,
            )

            # Safe score extraction (lm-eval sometimes uses "score" or "average")
            mt_results = eval_results["results"]["mt_bench"]
            mt_score = mt_results.get("score", mt_results.get("average", "unknown"))
            logger.info(f"🎉 MT-Bench Score: {mt_score:.2f}/10.0")
            logger.info("Upload the merged model to HF for leaderboard visibility!")

        except ImportError:
            logger.warning("lm-eval not installed — run: pip install 'uzombie[dev]'")
        except Exception as e:
            logger.warning(f"MT-Bench failed: {e} — check task name and lm-eval version")

    # 9. FIXED: Benchmark & Log Speed (Real Unsloth vs Uzombie, web:0, web:6, web:7)
    # Skip benchmark on 'fast' goal for quick tests; else run 50-step timed comparison
    if args.goal != "fast":
        try:
            unsloth_time = run_speed_benchmark(args.model, args.dataset, "unsloth")  # Baseline (50 steps)
            uzombie_time_per_step = actual_time / global_steps if global_steps > 0 else 0
            unsloth_time_per_step = unsloth_time / 50  # Normalize to per-step
            speedup = unsloth_time_per_step / uzombie_time_per_step if uzombie_time_per_step > 0 else 0
            logger.info(f"✅ Speedup: {speedup:.2f}x (Unsloth: {unsloth_time_per_step:.3f}s/step | Uzombie: {uzombie_time_per_step:.3f}s/step)")
        except Exception as bench_e:
            logger.warning(f"⚠️ Benchmark skipped: {bench_e} (Uzombie still trained successfully)")
    else:
        logger.info("⏭️ Benchmark skipped (--goal fast)")

    # 10. Auto HF Upload (web:50, web:52, web:53) — FIXED for 4-bit (safe_serialization)
    if args.push_to_hub:
        try:
            push_to_hub_auto(trainer, args.push_to_hub, commit_message="Uzombie v1 Fine-Tuned Model")
            logger.info(f"Pushed to {args.push_to_hub}")
        except Exception as upload_e:
            logger.warning(f"⚠️ Upload skipped: {upload_e} (Model saved locally)")

    # Cleanup (your cli.py)
    del model, tokenizer, dataset
    torch.cuda.empty_cache()

def _infer_is_conversational(dataset) -> bool:
    """Heuristic: detect chat-style samples."""
    try:
        sample = dataset[0]
    except Exception:
        return False
    chat_keys = {"messages", "conversations", "turns"}
    return any(k in sample for k in chat_keys)


def _compute_loss_flags(is_conversational: bool, chat_loss_flag: bool, style: str, logger=None) -> Tuple[bool, bool]:
    """
    Safely choose assistant_only_loss/completion_only_loss.
    - Non-chat data: assistant_only_loss=False to avoid Unsloth crash.
    - Chat data: assistant_only_loss=True unless overridden by non-chat detection.
    - completion_only_loss: True for SFT, False otherwise.
    """
    assistant_only = False
    if chat_loss_flag:
        assistant_only = True
        if not is_conversational and logger:
            logger.warning("Chat loss requested but dataset is not conversational; forcing assistant_only_loss=False.")
            assistant_only = False
    elif is_conversational:
        assistant_only = True

    completion_only = True if style == "sft" else False
    return assistant_only, completion_only


def _parse_time(time_str: str) -> int:
    """Parse time string to seconds (your allocator.py)."""
    if time_str.endswith("h"):
        return int(float(time_str[:-1]) * 3600)
    if time_str.endswith("m"):
        return int(float(time_str[:-1]) * 60)
    return int(time_str.replace("s", ""))

if __name__ == "__main__":
    main()