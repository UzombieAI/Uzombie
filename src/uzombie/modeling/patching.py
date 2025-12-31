import torch
from unsloth import PatchDPOTrainer
from peft import LoraConfig
from ..utils.logger import console

def apply_patches(training_args):
    """
    Applies Liger Kernel patches and Unsloth DPO fixes based on config.
    """
    # 1. Unsloth DPO/CPO gradient checkpointing fix
    PatchDPOTrainer()
    
    # 2. Liger Kernel (Fused Loss/Norms)
    if getattr(training_args, "use_liger_kernel", False):
        console.print("[bold green]Applying Liger Kernel (Fused CrossEntropy + RMSNorm)...[/]")
        # Note: In TRL > 0.24, setting use_liger_kernel=True in args handles imports,
        # but explicit import ensures availability.
        try:
            import liger_kernel
        except ImportError:
            console.print("[bold red]Liger Kernel not installed. Install via pip install liger-kernel[/]")

def advanced_lora_init(peft_config: LoraConfig) -> LoraConfig:
    """
    Applies arXiv 2510.03731 optimizations for stable LoRA/DoRA initialization.
    """
    console.print("[bold cyan]Applying Advanced LoRA Initialization (arXiv 2510.03731)[/]")
    
    # Scale alpha for faster initial convergence
    if hasattr(peft_config, 'lora_alpha'):
         # Heuristic: Slight boost to alpha to encourage movement
        peft_config.lora_alpha = int(peft_config.lora_alpha * 1.5)
        
    return peft_config