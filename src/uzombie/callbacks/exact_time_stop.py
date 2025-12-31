# src/uzombie/callbacks/exact_time_stop.py
# Exact-time training stop via callback (Transformers/Unsloth standard, 2025)
# Replaces invalid interrupt_callback
import time
from transformers import TrainerCallback, TrainerState, TrainerControl
from ..utils.logger import console

class ExactTimeStopCallback(TrainerCallback):
    def __init__(self, deadline: float):
        self.deadline = deadline

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if time.time() >= self.deadline:
            console.print(f"[bold red]⏱️ Exact time deadline reached at step {state.global_step} — forcing save & stop[/]")
            
            # FIXED: Explicitly trigger a save checkpoint BEFORE stopping
            control.should_save = True
            control.should_training_stop = True
            
        return control