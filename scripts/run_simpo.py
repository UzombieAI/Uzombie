import sys
import os
import yaml
from trl import CPOConfig
from unsloth import FastLanguageModel
from datasets import load_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from uzombie.trainer.simpo_trainer import UzombieSimPOTrainer
from uzombie.modeling.patching import apply_patches
from uzombie.utils.logger import console

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--data_config", type=str, required=False)
    parser.add_argument("--trainer_config", type=str, required=True)
    args = parser.parse_args()

    with open(args.model_config, 'r') as f: model_cfg = yaml.safe_load(f)
    with open(args.trainer_config, 'r') as f: trainer_cfg_dict = yaml.safe_load(f)

    # 1. Load Model
    console.print(f"[bold green]Loading {model_cfg['model_name']} for Alignment...[/]")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg['model_name'],
        max_seq_length=model_cfg['max_seq_length'],
        load_in_4bit=model_cfg['load_in_4bit'],
    )

    # 2. PEFT (Standard LoRA/DoRA for Alignment)
    # Note: Projector usually applied in SFT, but valid here too if desired.
    # We skip Projector for standard SimPO stability unless specified.
    
    peft_kwargs = {k:v for k,v in model_cfg.items() if k in ['r', 'target_modules', 'lora_alpha', 'use_dora', 'random_state']}
    model = FastLanguageModel.get_peft_model(model, bias="none", **peft_kwargs)

    # 3. Config & Dataset
    cpo_config = CPOConfig(**trainer_cfg_dict)
    
    # SimPO requires specific formatting usually (prompt, chosen, rejected)
    # We assume dataset is ready or use `configs/data` to find it
    if args.data_config:
        with open(args.data_config, 'r') as f: data_cfg = yaml.safe_load(f)
        dataset_name = data_cfg.get("repo_id", "UzombieAI/reasoning-preferences")
    else:
        dataset_name = "UzombieAI/reasoning-preferences"

    dataset = load_dataset(dataset_name, split="train")
    
    # 4. Train
    trainer = UzombieSimPOTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=cpo_config
    )
    
    console.print(f"[bold green]Starting {cpo_config.loss_type.upper()} Alignment...[/]")
    trainer.train()

if __name__ == "__main__":
    main()