import sys
import os
import yaml
import torch
from transformers import HfArgumentParser
from datasets import load_dataset
from trl import SFTConfig
from unsloth import FastLanguageModel

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from uzombie.trainer.uzombie_trainer import UzombieTrainer
from uzombie.core.hybrid_projector import UzombieProjector
from uzombie.modeling.patching import advanced_lora_init, apply_patches
from uzombie.utils.logger import console

def main():
    # Simple CLI to get config paths
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--data_config", type=str, required=False) # Optional for SFT if dataset in yaml
    parser.add_argument("--trainer_config", type=str, required=True)
    args = parser.parse_args()

    # Load YAMLs
    with open(args.model_config, 'r') as f: model_cfg = yaml.safe_load(f)
    with open(args.trainer_config, 'r') as f: trainer_cfg_dict = yaml.safe_load(f)

    # 1. Load Model (Unsloth)
    console.print(f"[bold green]Loading {model_cfg['model_name']}...[/]")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg['model_name'],
        max_seq_length=model_cfg['max_seq_length'],
        load_in_4bit=model_cfg['load_in_4bit'],
        dtype=model_cfg.get('dtype', None)
    )

    # 2. Apply Uzombie Projector (The Hybrid Engine)
    console.print("[bold magenta]Applying Uzombie Hybrid Projector...[/]")
    projector = UzombieProjector(
        rank=model_cfg['r'],
        use_dora=model_cfg.get('use_dora', False),
        # Pass other projector args if defined in yaml
    )
    model = projector.apply_to_model(model)

    # 3. Apply PEFT (LoRA/DoRA) via Unsloth
    # We filter model_cfg to only pass relevant args to get_peft_model
    peft_kwargs = {k:v for k,v in model_cfg.items() if k in ['r', 'target_modules', 'lora_alpha', 'use_dora', 'random_state']}
    
    # Advanced Init (arXiv 2510.03731) - Modify config before application if needed, 
    # but Unsloth applies immediately. We apply init logic logically here.
    if model_cfg.get("advanced_init", False):
        # We manually adjust alpha in the kwargs before passing to Unsloth
        peft_kwargs['lora_alpha'] = int(peft_kwargs['lora_alpha'] * 1.5)
        console.print(f"[cyan]Advanced Init: Scaled LoRA Alpha to {peft_kwargs['lora_alpha']}[/]")

    model = FastLanguageModel.get_peft_model(
        model,
        bias="none",
        use_gradient_checkpointing=model_cfg.get("use_gradient_checkpointing", "unsloth"),
        **peft_kwargs
    )

    # 4. Load Dataset
    # Assuming the trainer config or a data config specifies the dataset
    # For SFT, we usually look for 'dataset_name' in trainer config or args
    dataset_name = trainer_cfg_dict.get("dataset_name", "yahma/alpaca-cleaned") # Fallback
    console.print(f"[bold yellow]Loading dataset: {dataset_name}[/]")
    dataset = load_dataset(dataset_name, split="train")

    # 5. Trainer Setup
    # Convert dict to SFTConfig object
    sft_config = SFTConfig(**trainer_cfg_dict)
    
    # Apply Liger Patches
    apply_patches(sft_config)

    trainer = UzombieTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config
    )

    # 6. Train
    console.print("[bold green]Starting SFT Training...[/]")
    trainer.train()

if __name__ == "__main__":
    main()