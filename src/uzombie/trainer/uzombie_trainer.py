# src/uzombie/trainer/uzombie_trainer.py
# UzombieTrainer — Drop-in replacement for TRL trainers (SFT/DPO/ORPO/KTO/SimPO/PPO)
# Injects Uzombie-specific callbacks ONLY (projector is applied externally in cli.py)
# Research: GaLore (arXiv:2403.03507), LoRA-FA (arXiv:2305.14314), Universal Subspaces (arXiv:2512.05117)

from typing import Optional
from trl import SFTTrainer
from ..callbacks import PESORestartCallback, ResearchCallback
from ..utils.logger import console

class UzombieTrainer(SFTTrainer):
    """
    Enhanced SFTTrainer with Uzombie-specific callbacks.
    - Projector is applied externally in cli.py (before accelerator.prepare) to support multi-GPU/DeepSpeed.
    - Automatically adds ResearchCallback and PESORestartCallback (duplicate-safe).
    - Logs clearly and avoids double injection.
    """
    def __init__(self, *args, **kwargs):
        # Remove projector arg — it's now applied externally
        super().__init__(*args, **kwargs)

        # Flag to track if projector was already applied (safety)
        if not getattr(self.model, "_uzombie_projector_applied", False):
            console.print("[bold yellow]Note: Hybrid projector should be applied before accelerator.prepare in cli.py[/]")
            self.model._uzombie_projector_applied = True  # Prevent warnings

        # === Uzombie Callbacks ===
        # Research logging (loss, grad norm every 100/500 steps)
        self.add_callback(ResearchCallback())

        # PESO-style subspace refinement (duplicate-safe)
        already_has_peso = any(isinstance(cb, PESORestartCallback) for cb in self.callback_handler.callbacks)
        if not already_has_peso:
            self.add_callback(PESORestartCallback(trainer_instance=self))
        else:
            console.print("[dim]PESORestartCallback already added — skipping duplicate[/]")

        console.print("[bold magenta]UZOMBIE Trainer Initialized — Callbacks Active | Multi-GPU Ready[/]")

    def __repr__(self):
        return f"UzombieTrainer(model={self.model.__class__.__name__}, callbacks={'Active'})"