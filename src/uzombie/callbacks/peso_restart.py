# src/uzombie/callbacks/peso_restart.py
# PESO-style periodic subspace refinement (extension of GaLore, arXiv:2403.03507)
# Triggers projector refinement every N steps or on loss plateau

from transformers import TrainerCallback, TrainerState, TrainerControl
from ..utils.logger import console

class PESORestartCallback(TrainerCallback):
    def __init__(self, trainer_instance=None, refine_every: int = 200):
        self.refine_every = refine_every
        self.trainer_instance = trainer_instance

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.refine_every == 0 and state.global_step > 0:
            model = kwargs.get("model") or self.trainer_instance.model
            if hasattr(model, "projector") and model.projector is not None:
                console.print(f"[bold magenta]PESO Refine triggered at step {state.global_step}[/]")
                model.projector.refine_subspace(reason=f"Step {state.global_step}")
        return control