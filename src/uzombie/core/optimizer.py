# src/uzombie/core/optimizer.py
# FINAL FIXED VERSION — No more NoneType * float errors
# Research: Goyal 2017 + Unsloth real-world configs (Dec 2025)

import time
import torch
from transformers import get_scheduler
from ..utils.logger import console

class ExactTimeScheduler:
    def __init__(self, total_seconds: int):
        self.deadline = time.time() + total_seconds
        self.steps_per_second = None
        self.calibrated = False

    def calibrate(self, trainer, calibration_steps: int = 25):
        console.print("[bold yellow]UZOMBIE CALIBRATION — Measuring real speed...[/]")
        trainer.model.train()
        dataloader = trainer.get_train_dataloader()
        batch = next(iter(dataloader))
        batch = {k: v.to(trainer.model.device) for k, v in batch.items()}

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(calibration_steps):
            with torch.cuda.amp.autocast():
                loss = trainer.model(**batch).loss
                loss = loss / trainer.args.gradient_accumulation_steps
            trainer.accelerator.backward(loss)
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
        torch.cuda.synchronize()

        elapsed = time.time() - start
        self.steps_per_second = calibration_steps / elapsed
        self.calibrated = True
        console.print(f"[bold green]CALIBRATED: {self.steps_per_second:.2f} steps/sec[/]")
        return self.steps_per_second

    def get_optimal_steps(self):
        if not self.calibrated:
            return 1000
        remaining = max(0, self.deadline - time.time())
        return max(100, int(remaining * self.steps_per_second))

    def should_stop(self):
        return time.time() >= self.deadline


def get_strategy_for_goal(
    goal: str,
    calibrated_speed: float = None,  # ← This is the SECOND argument now!
    base_lr: float = 2e-4,
    base_rank: int = 64
):
    """
    FIXED: calibrated_speed is now the first optional arg so base_lr stays safe
    """
    # Estimate steps from time budget (600s = 10min)
    estimated_steps = 1200 if calibrated_speed is None else int(600 * (calibrated_speed or 2.0))

    strategies = {
        "fast": {
            "rank": 32,
            "lr": base_lr * 2.0,
            "alpha": 16,
            "dropout": 0.0,
            "use_dora": False,
            "steps": int(estimated_steps * 1.8),
            "bsz": 8,
            "accum_steps": 2,
            "mode": "fast",
        },
        "balanced": {
            "rank": 64,
            "lr": base_lr,
            "alpha": 32,
            "dropout": 0.0,
            "use_dora": True,
            "steps": estimated_steps,
            "bsz": 2,
            "accum_steps": 8,
            "mode": "balanced",
        },
        "best": {
            "rank": 128,
            "lr": base_lr * 0.5,
            "alpha": 64,
            "dropout": 0.0,
            "use_dora": True,
            "steps": int(estimated_steps * 0.7),
            "bsz": 1,
            "accum_steps": 16,
            "mode": "best",
        }
    }
    return strategies.get(goal.lower(), strategies["balanced"])