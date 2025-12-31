# src/uzombie/utils/benchmarks.py
# REAL Speed Benchmark: Unsloth Baseline vs. Uzombie (arXiv:2403.03507 + Unsloth kernels, web:0, web:6, web:7)
# Measures time per iteration over 50 steps (Alpaca style: batch=2, accum=4, rank=16)
# Research: Timed SFTTrainer.train(max_steps=50); 2-4x speedup expected (Unsloth HF benchmarks)

import time
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import load_dataset
from transformers import TrainingArguments
from ..utils.logger import console

def run_speed_benchmark(model_name: str, dataset_name: str, mode: str = "unsloth", steps: int = 50):
    """
    Run timed benchmark: Unsloth baseline (mode='unsloth') or Uzombie-wrapped (mode='uzombie').
    Returns total time in seconds (lower = faster).
    Adapted from Unsloth HF benchmarks (web:6, web:7): 50 steps on Alpaca slice.
    """
    console.print(f"[bold yellow]Benchmarking {mode.upper()} on {model_name}... (50 steps)[/]")

    # Load small dataset slice (first 200 examples for speed)
    dataset = load_dataset(dataset_name, split="train[:200]")
    tokenizer = FastLanguageModel.from_pretrained(model_name, load_in_4bit=True)[1]  # Just tokenizer

    def formatting_prompts_func(example):
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output_text = example.get("output", "")
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
        return {"text": text[:1024]}  # Truncate for speed

    dataset = dataset.map(formatting_prompts_func, batched=False, num_proc=1)

    # Load model (shared for fairness)
    model, _ = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )

    if mode == "uzombie":
        # Apply Uzombie projector (your hybrid: GaLore + LoRA-FA + Universal stub)
        from ..core.hybrid_projector import UzombieProjector
        projector = UzombieProjector(rank=16, use_dora=False, prior_adapters=[])  # Fast goal
        model = projector.apply_to_model(model)

    # Minimal LoRA (rank=16 for benchmark consistency, web:6)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        use_dora=False,  # Off for baseline parity
        random_state=3407,
    )

    # Compact SFTConfig (batch=2, accum=4; matches Unsloth benchmarks )
    sft_config = SFTConfig(
        output_dir="/tmp/benchmark",
        max_steps=steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=steps,  # Silent
        save_strategy="no",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        warmup_steps=5,
        report_to=[],
        max_seq_length=1024,
        packing=False,
        dataset_text_field="text",
        dataloader_num_workers=0,  # Stability on Windows/laptop
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
        peft_config=None,  # Already applied
    )

    # Time the core training loop (real steps, no overhead)
    torch.cuda.synchronize()
    start_time = time.time()
    trainer.train()
    torch.cuda.synchronize()
    total_time = time.time() - start_time

    console.print(f"[bold green]{mode.upper()} Benchmark: {total_time:.2f}s for {steps} steps ({total_time/steps:.3f}s/step)[/]")

    # Cleanup
    del model, trainer, dataset
    torch.cuda.empty_cache()

    return total_time