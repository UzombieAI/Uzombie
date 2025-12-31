# src/uzombie/callbacks/auto_dora.py
from transformers import TrainerCallback, TrainerState, TrainerControl
from ..utils.logger import console

class AutoDoRACallback(TrainerCallback):
    def __init__(self, patience: int = 80, min_improvement: float = 0.001):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_loss = float("inf")
        self.no_improve_count = 0
        self.dora_activated = False

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.dora_activated:
            return control

        # Get latest loss from log history
        if not state.log_history:
            return control
        latest_log = state.log_history[-1]
        current_loss = latest_log.get("loss")
        if current_loss is None:
            return control

        if current_loss < self.best_loss - self.min_improvement:
            self.best_loss = current_loss
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        if self.no_improve_count >= self.patience:
            model = kwargs.get("model")
            if model is not None and hasattr(model, "peft_config"):
                activated = False
                for config in model.peft_config.values():
                    if not config.use_dora:
                        config.use_dora = True
                        activated = True
                if activated:
                    console.print(f"[bold magenta]AUTO DoRA ACTIVATED at step {state.global_step} (loss plateau detected)[/]")
                    self.dora_activated = True
                    # Magnitude vectors will be initialized on next parameter access (PEFT handles it)

        return control