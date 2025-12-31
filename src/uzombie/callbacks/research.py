# src/uzombie/callbacks/research.py
# Logs rich research metrics: entropy, gradient norm, subspace alignment

from transformers import TrainerCallback, TrainerState, TrainerControl
from ..utils.logger import console
import torch

class ResearchCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return

        step = state.global_step
        if step % 100 == 0:
            loss = logs.get("loss", 0.0)
            lr = logs.get("learning_rate", 0.0)
            console.print(f"[dim]Step {step} | Loss: {loss:.4f} | LR: {lr:.2e}[/]")

        # Optional: log gradient norm
        model = kwargs.get("model")
        if model is not None and step % 500 == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm(2).item() ** 2
            console.print(f"[dim]Grad Norm: {total_norm**0.5:.2f}[/]")