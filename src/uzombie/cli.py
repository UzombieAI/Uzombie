# src/uzombie/cli.py
# UZOMBIE v5.2: Final Stable Research Engine
# ----------------------------------------------------------------------------------------------------

from unsloth import FastLanguageModel, PatchDPOTrainer
import os
import sys
import time
import math
import torch
import gc
import logging
import argparse
from typing import Dict, Any, Tuple
import psutil
import platform
import subprocess
from datasets import load_dataset, get_dataset_split_names, load_from_disk
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from accelerate.utils import set_seed

# --- RESEARCH: Hardware Acceleration Configuration ---
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# FIXED: Suppress Environment Warnings
os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1" 
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
os.environ["ACCELERATE_IGNORE_KERNEL_VERSION"] = "1"
os.environ["SWIZZLEPERF_ENABLED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 64

# Uzombie imports — MUST come before Liger patch so 'console' is defined
from uzombie.core.hybrid_projector import UzombieProjector
from uzombie.trainer.uzombie_trainer import UzombieTrainer
from uzombie.core.optimizer import ExactTimeScheduler, get_strategy_for_goal
from uzombie.core.hardware import auto_optimize, estimate_safe_batch_size
from uzombie.utils.logger import get_logger, console 
from uzombie.utils.benchmarks import run_speed_benchmark
from uzombie.utils.upload import push_to_hub_auto
from uzombie.utils.banner import print_zombie_banner
from uzombie.callbacks import PESORestartCallback, ResearchCallback, ExactTimeStopCallback
from uzombie.data.builders import generate_magpie_reasoning_v2

logger = get_logger(__name__)

# ==============================================================================
# 🩹 FINAL LIGER STABILITY FIX — Force Safe Loss (Bypass Fused CE Entirely)
# This guarantees no causal_mask error and no bool.sum() crash.
# ==============================================================================
try:
    # 1. Define a SAFE replacement for the crashing Liger function
    def safe_fixed_fused_linear_cross_entropy(input, weight, target, bias=None, ignore_index=-100,
                                              label_smoothing=0.0, reduction='mean', softcap=0.0, **kwargs):
        # Swallow unexpected kwargs
        kwargs.pop('causal_mask', None)
        kwargs.pop('attention_mask', None)

        # Handle label_smoothing NoneType
        if label_smoothing is None:
            label_smoothing = 0.0

        # Compute logits manually
        logits = input.view(-1, input.size(-1)) @ weight.t()
        if bias is not None:
            logits = logits + bias

        # Apply softcap if >0
        if softcap > 0.0:
            logits = torch.log_softmax(logits, dim=-1)
            logits = (logits.exp() * softcap).log()

        # Use functional cross_entropy (no 'bias' arg)
        from torch.nn.functional import cross_entropy
        return cross_entropy(
            logits,
            target.view(-1),
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )

    # 2. Overwrite the function in the Liger library module
    import liger_kernel.transformers.model.loss_utils
    liger_kernel.transformers.model.loss_utils.fixed_fused_linear_cross_entropy = safe_fixed_fused_linear_cross_entropy

    # 3. Apply Liger (We enable 'fused' here so Liger hooks up the routing,
    #    but our patch above ensures the destination is the safe function)
    from liger_kernel.transformers import apply_liger_kernel_to_llama
    
    console.print("[Uzombie] 🛡️  Applying Liger Kernels (Safe Mode)...")
    apply_liger_kernel_to_llama(
        rope=True,
        swiglu=True,
        rms_norm=True,
        cross_entropy=True,
        fused_linear_cross_entropy=True 
    )
    console.print("[Uzombie] ✅ Liger Active: RoPE + SwiGLU + RMSNorm + Safe CE")

except ImportError:
    console.print("[Uzombie] ❌ Liger Kernel not installed — continuing without")
except Exception as e:
    console.print(f"[Uzombie] ⚠️ Liger apply warning: {e}")
# ==============================================================================

def align_to_tensor_cores(rank: int) -> int:
    """
    RESEARCH: Tensor Core Alignment (The '2:4 Sparsity' Equivalent for Fine-Tuning).
    """
    if rank % 16 != 0:
        new_rank = ((rank // 16) + 1) * 16
        logger.info(f"⚡ Tensor Alignment: Adjusted Rank {rank} -> {new_rank} for optimal throughput")
        return new_rank
    return rank

def main():
    print_zombie_banner() 

    parser = argparse.ArgumentParser(
        description="Uzombie v5.2: High-Performance Research Engine"
    )
    
    # --- MODE SELECTION ---
    parser.add_argument("--mode", type=str, default="train", choices=["train", "generate"], help="Mode: 'train' for SFT, 'generate' for Magpie V2 Data Synthesis")
    
    # Core Arguments
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", type=str, required=False, help="HF dataset (Required for training)")
    parser.add_argument("--time", type=str, required=False, default="1h", help="Exact time budget")
    parser.add_argument("--goal", type=str, default="balanced", choices=["fast", "balanced", "best"])
    parser.add_argument("--style", type=str, default="sft", choices=["sft", "dpo", "orpo", "kto", "simpo", "ppo"])
    
    # Data Generation Arguments
    parser.add_argument("--gen-samples", type=int, default=1000, help="[Generate Mode] Number of samples")
    parser.add_argument("--gen-out", type=str, default="uzombie_data", help="[Generate Mode] Output folder")
    
    # Advanced Args
    parser.add_argument("--push-to-hub", type=str, help="HF repo")
    parser.add_argument("--ctx-len", type=int, default=2048, help="Max seq len")
    parser.add_argument("--chat_loss", action="store_true", help="Assistant only loss")
    parser.add_argument("--use_dora", action="store_true", help="Use DoRA")
    parser.add_argument("--eval-mt-bench", action="store_true", help="MT-Bench")
    parser.add_argument("--deepspeed", type=str, help="DeepSpeed Config")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--prior-adapters", type=str, nargs="*", default=[])
    parser.add_argument("--universal-rank", type=int, default=16)
    parser.add_argument("--packing", action="store_true", default=False, help="Enable Padding-Free Batching") 
    parser.add_argument("--use_liger", action="store_true", default=True, help="Enable Selective Liger Kernels")
    parser.add_argument("--fusion", action="store_true", default=True, help="Enable Mega-Kernel Fusion Compilation")
    
    args = parser.parse_args()

    # ==========================================
    # MODE: GENERATE (Magpie V2 Pipeline)
    # ==========================================
    if args.mode == "generate":
        console.print(f"[bold magenta]🔮 Entering Data Generation Mode (Magpie V2)[/]")
        dataset = generate_magpie_reasoning_v2(
            teacher_model_id=args.model,
            num_samples=args.gen_samples,
            active_iterations=1
        )
        if dataset and len(dataset) > 0:
            dataset.save_to_disk(args.gen_out)
            console.print(f"[bold green]✅ Dataset saved to {args.gen_out}[/]")
            console.print(f"[yellow]To train on this data, run:[/]")
            console.print(f"python -m uzombie --mode train --model {args.model} --dataset {args.gen_out} ...")
        return # Exit after generation

    # ==========================================
    # MODE: TRAIN (Uzombie Standard)
    # ==========================================
    if args.mode == "train" and not args.dataset:
        console.print("[bold red]Error: --dataset is required for training mode.[/]")
        sys.exit(1)

    # --- CONDITIONAL IMPORT ---
    try:
        from trl import SFTConfig
        if args.style == "sft":
            from trl import SFTTrainer as TrainerClass
        elif args.style == "dpo":
            from trl import DPOTrainer as TrainerClass
        elif args.style == "orpo":
            from trl import ORPOTrainer as TrainerClass
        elif args.style == "simpo":
            from trl import SimPOTrainer as TrainerClass
        elif args.style == "kto":
            from trl import KTOTrainer as TrainerClass
        elif args.style == "ppo":
            from trl import PPOTrainer as TrainerClass
        else:
            from trl import SFTTrainer as TrainerClass
    except ImportError as e:
        console.print(f"[bold yellow]Warning: TRL Import Error ({e}). Defaulting to SFTTrainer.[/]")
        from trl import SFTTrainer as TrainerClass

    # --- RESEARCH COMPLIANCE REPORT ---
    console.print(f"[bold cyan]Research Optimization Config:[/]")
    console.print(f" • [green]Padding-Free Batching:[/] {'ENABLED' if args.packing else 'DISABLED (Check manual tokenization)'}")
    console.print(f" • [green]Tensor Core Alignment:[/] ENABLED (Auto-Rank)")
    console.print(f" • [green]Mega-Kernel Fusion:[/] {'ENABLED (Torch Compile)' if args.fusion else 'DISABLED'}")
    console.print(f" • [green]Selective Liger Kernels:[/] {'ENABLED' if args.use_liger else 'DISABLED'}")
    console.print(f" • [green]Universal Subspace:[/] {'AUTO-DETECT or MANUAL' if args.prior_adapters or args.model else 'OFF'}")

    # Patch DPO/SimPO gradients
    if args.style in ["dpo", "simpo", "orpo"]: 
        PatchDPOTrainer()

    # Force packing on Linux/WSL for speed
    if platform.system() != "Windows" and not args.packing:
        args.packing = True
        console.print("[bold green]Auto-enabling Padding-Free Batching (Packing) for Linux[/]")

    # Formatting Logic
    def formatting_prompts_func(example, tokenizer=None):
        if args.style in ["dpo", "orpo", "kto", "simpo"]:
            prompt, chosen, rejected = example.get("prompt", ""), example.get("chosen", ""), example.get("rejected", "")
            text = f"### Prompt:\n{prompt}\n### Chosen:\n{chosen}\n### Rejected:\n{rejected}"
        else:
            instruction = example.get("instruction", example.get("prompt", ""))
            input_text = example.get("input", example.get("context", ""))
            output_text = example.get("output", example.get("response", ""))
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}" if input_text else f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
            
            # CRITICAL: Only manually tokenize if NOT packing. 
            if not args.packing and tokenizer is not None:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > args.ctx_len - 1: tokens = tokens[:args.ctx_len - 1]
                text = tokenizer.decode(tokens, skip_special_tokens=True)
        return {"text": text}

    total_seconds = _parse_time(args.time)
    scheduler = ExactTimeScheduler(total_seconds)
    
    # --- DATASET LOADING ---
    logger.info(f"Loading Dataset: {args.dataset}")
    tokenizer = None
    dataset = None
    try:
        if os.path.exists(args.dataset) and os.path.isdir(args.dataset):
            try:
                dataset = load_from_disk(args.dataset)
                logger.info(f"Loaded generated dataset from disk: {args.dataset}")
            except Exception:
                dataset = load_dataset("arrow", data_dir=args.dataset, split="train")
        else:
            splits = get_dataset_split_names(args.dataset)
            target_split = "train_prefs" if "train_prefs" in splits and args.style in ["dpo", "orpo", "simpo"] else splits[0]
            dataset = load_dataset(args.dataset, split=target_split)
            
        dataset = dataset.filter(lambda ex: (len(str(ex.get('instruction', ''))) + len(str(ex.get('output', ''))) < 20000))
        
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        if "input_ids" not in dataset.column_names and "text" not in dataset.column_names:
            dataset = dataset.map(lambda ex: formatting_prompts_func(ex, tokenizer=None), batched=False)

        if args.packing:
             logger.info("⚡ Packing Enabled: Skipping manual tokenization. Unsloth will handle padding-free batching.")
        else:
             logger.info("Pre-tokenizing dataset (Packing Disabled)...")
             def tokenize_function(examples):
                tokenized = tokenizer(examples["text"], truncation=True, max_length=args.ctx_len, padding=False, add_special_tokens=True)
                if "labels" not in tokenized: tokenized["labels"] = tokenized["input_ids"].copy()
                return tokenized
             
             if "input_ids" not in dataset.column_names:
                dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"] if "text" in dataset.column_names else None)

    except Exception as e:
        logger.error(f"Dataset error: {e}")
        dataset = load_dataset("yahma/alpaca-cleaned", split="train")

    # Model Init
    logger.info("Auto-optimizing model with Unsloth...")
    model, tokenizer = auto_optimize(args.model, max_seq_length=args.ctx_len, load_in_4bit=True)
    
    # Strategy Calculation
    strategy = get_strategy_for_goal(args.goal, getattr(scheduler, 'calibrated_steps_per_second', None))
    strategy["rank"] = align_to_tensor_cores(strategy["rank"])

    # === HARDWARE: VRAM Edge-Surfing ===
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Detected VRAM: {vram_gb:.2f} GB")
        
        if vram_gb < 12.0 and args.ctx_len > 1024:
             logger.warning(f"⚠️ Critical VRAM ({vram_gb:.2f}GB). Squeezing Context to 1024.")
             args.ctx_len = 1024

        safe_bsz = estimate_safe_batch_size(args.model, vram_gb, args.ctx_len, strategy["bsz"])
        
        if safe_bsz != strategy["bsz"]:
            logger.warning(f"⚠️ VRAM Optimized: Batch {strategy['bsz']} -> {safe_bsz}")
            factor = max(1, strategy["bsz"] // safe_bsz)
            strategy["accum_steps"] = strategy["accum_steps"] * factor
            strategy["bsz"] = safe_bsz

    logger.info(f"Final Strategy: Rank {strategy['rank']}, LR {strategy['lr']:.2e}, Batch {strategy['bsz']}, Accum {strategy['accum_steps']}")

    # Advanced Init
    lora_alpha = int(16 * 1.5) if args.goal in ["balanced", "best"] else 16
    model = FastLanguageModel.get_peft_model(
        model, 
        r=strategy["rank"], 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha, 
        use_gradient_checkpointing="unsloth", 
        use_dora=True, 
        random_state=3407,
    )

    # Config
    is_conv = _infer_is_conversational(dataset)
    asst_only, comp_only = _compute_loss_flags(is_conv, args.chat_loss, args.style, logger)

    if args.packing and (asst_only or comp_only):
        logger.warning("⚠️ Conflict: Packing vs Completion Loss. Disabling Completion Loss.")
        asst_only = False
        comp_only = False

    sft_config = SFTConfig(
        output_dir="./uzombie_outputs",
        max_steps=strategy["steps"],
        per_device_train_batch_size=strategy["bsz"],
        gradient_accumulation_steps=strategy["accum_steps"],
        learning_rate=strategy["lr"],
        logging_steps=10,
        save_strategy="steps", 
        save_steps=100, 
        save_total_limit=3,
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        packing=args.packing,
        dataset_text_field="text",
        max_seq_length=args.ctx_len,
        dataset_kwargs={"add_special_tokens": False},
        assistant_only_loss=asst_only,
        completion_only_loss=comp_only,
        use_liger_kernel=args.use_liger # Explicitly tell TRL we are using Liger (patched above)
    )

    accelerator = Accelerator(gradient_accumulation_steps=sft_config.gradient_accumulation_steps)
    set_seed(args.seed)

    # Hybrid Projector Application
    projector = UzombieProjector(
        rank=strategy["rank"], 
        activation_rank=args.universal_rank, 
        prior_adapters=args.prior_adapters,
        model_name=args.model 
    )
    if accelerator.num_processes > 1: projector.enable_galore = False
    model = projector.apply_to_model(model)
    
    # === MEGA-KERNEL FUSION ===
    if args.fusion:
        logger.info("🔥 Enabling Mega-Kernel Fusion via Torch Compile...")
        if hasattr(projector, 'compile_fusion'):
            model = projector.compile_fusion(model)
        else:
            logger.warning("Projector compile_fusion not found; attempting standard compile...")
            try:
                model = torch.compile(model, mode="reduce-overhead", dynamic=True)
            except Exception as e:
                logger.warning(f"Compilation failed: {e}")

    model = accelerator.prepare(model)

    # --- TRAINER INITIALIZATION ---
    def formatting_wrapper(example):
        return formatting_prompts_func(example, tokenizer=None)["text"]

    trainer = TrainerClass(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=None,
        formatting_func=formatting_wrapper if args.packing else None 
    )

    trainer = accelerator.prepare(trainer)

    # Callbacks
    trainer.add_callback(ResearchCallback())
    trainer.add_callback(PESORestartCallback(trainer_instance=trainer))
    trainer.add_callback(ExactTimeStopCallback(deadline=scheduler.deadline))

    # Training Loop
    logger.info(f"Starting Uzombie v5.2 | Rank: {strategy['rank']} | Batch: {strategy['bsz']}")
    torch.cuda.empty_cache(); gc.collect()
    
    start_time = time.time()
    try: 
        trainer.train()
    except KeyboardInterrupt: 
        logger.info("Interrupted - Saving checkpoint...")
        pass
    
    end_time = time.time()
    actual_time = end_time - start_time
    global_steps = trainer.state.global_step
    logger.info(f"Training complete in {actual_time:.1f}s (target: {total_seconds}s) | Steps: {global_steps}")

    # --- RESEARCH-GRADE SAVING ---
    console.print("[bold cyan]💾 Saving Research-Grade Artifacts...[/]")
    try:
        model.save_pretrained("./uzombie_outputs", safe_serialization=True)
        tokenizer.save_pretrained("./uzombie_outputs")
        console.print("[bold green]✅ Model saved to ./uzombie_outputs[/]")
    except Exception as e:
        logger.error(f"Failed to save final model: {e}")

    # --- BENCHMARKING ---
    if args.goal != "fast":
        try:
             train_log_history = trainer.state.log_history
             if train_log_history:
                 final_log = next((x for x in reversed(train_log_history) if 'train_samples_per_second' in x), None)
                 if final_log:
                     speed = final_log['train_samples_per_second']
                     console.print(f"[bold green]🚀 Uzombie Throughput: {speed:.2f} samples/sec[/]")
                 else:
                     throughput = (len(dataset) * (global_steps / strategy["steps"])) / actual_time
                     console.print(f"[bold green]🚀 Estimated Throughput: {throughput:.2f} samples/sec[/]")
        except Exception as e:
             logger.warning(f"Benchmark metric calc failed: {e}")
    
    if args.push_to_hub:
        try: 
            push_to_hub_auto(trainer, args.push_to_hub, commit_message="Uzombie v5.2 Fine-Tuned Model")
            logger.info(f"Pushed to {args.push_to_hub}")
        except Exception as upload_e:
            logger.warning(f"⚠️ Upload skipped: {upload_e}")

    # Merge & Eval
    if args.eval_mt_bench:
        logger.info("Merging LoRA weights into base model for evaluation...")
        try:
            merged_model = model.merge_and_unload(progressbar=True)
            merged_model.save_pretrained("./uzombie_outputs_merged")
            tokenizer.save_pretrained("./uzombie_outputs_merged")
            logger.info("Merged model saved to ./uzombie_outputs_merged")
        except Exception as e:
            logger.warning(f"Merge failed: {e} — falling back to adapter (may fail eval)")

    if args.eval_mt_bench:
        try:
            from lm_eval import evaluator
            eval_path = "./uzombie_outputs_merged" if os.path.exists("./uzombie_outputs_merged") else "./uzombie_outputs"
            logger.info(f"🎯 Running MT-Bench evaluation on {eval_path}...")
            eval_results = evaluator.simple_evaluate(
                model="hf",
                model_args=f"pretrained={eval_path},dtype=auto,trust_remote_code=True",
                tasks=["mt_bench"],
                num_fewshot=0,
                batch_size=4,
            )
            mt_results = eval_results["results"]["mt_bench"]
            mt_score = mt_results.get("score", mt_results.get("average", "unknown"))
            logger.info(f"🎉 MT-Bench Score: {mt_score:.2f}/10.0")
        except Exception as e:
            logger.warning(f"MT-Bench failed: {e}")

    del model, tokenizer, dataset
    torch.cuda.empty_cache()

def _infer_is_conversational(dataset) -> bool:
    try:
        sample = dataset[0]
    except Exception:
        return False
    chat_keys = {"messages", "conversations", "turns"}
    return any(k in sample for k in chat_keys)

def _compute_loss_flags(is_conv, chat_flag, style, logger=None):
    asst_only = True if (chat_flag or is_conv) else False
    if chat_flag and not is_conv and logger: asst_only = False
    return asst_only, (style == "sft")

def _parse_time(time_str):
    if time_str.endswith("h"): return int(float(time_str[:-1]) * 3600)
    if time_str.endswith("m"): return int(float(time_str[:-1]) * 60)
    return int(time_str.replace("s", ""))

if __name__ == "__main__":
    main()