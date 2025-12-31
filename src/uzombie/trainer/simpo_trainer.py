from trl import CPOTrainer, CPOConfig
from ..callbacks import PESORestartCallback, ResearchCallback, AutoDoRACallback
from ..utils.logger import console
from ..modeling.patching import apply_patches
import torch

class UzombieSimPOTrainer(CPOTrainer):
    def __init__(self, *args, **kwargs):
        # Apply Liger/Unsloth patches before init
        if 'args' in kwargs:
            apply_patches(kwargs['args'])
        
        super().__init__(*args, **kwargs)
        
        # Add Uzombie Callbacks
        self.add_callback(ResearchCallback())
        
        # Add PESO if model has projector
        if hasattr(self.model, "projector"):
            self.add_callback(PESORestartCallback(self))
            
        self.add_callback(AutoDoRACallback())

        # Check for RPO/MMPO extensions
        loss_type = getattr(self.args, "loss_type", "simpo")
        if loss_type in ['rpo', 'mmpo']:
            console.print(f"[bold magenta]Active Alignment Extension: {loss_type.upper()}[/]")
        
        console.print("[bold magenta]UZOMBIE SimPO Trainer Initialized[/]")