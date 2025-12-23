# src/__init__.py
"""
Uzombie v1 — The fastest LLM fine-tuning engine on Earth
"""

__version__ = "1.0.0"
__author__ = "Kafoo"

# === Core public API (what people will use) ===
from .cli import main as tune
from .trainer.uzombie_trainer import UzombieTrainer
from .core.hardware import auto_optimize
from .utils.upload import push_to_hub_auto
from .core.hybrid_projector import UzombieProjector
from .utils.logger import get_logger
from .utils.benchmarks import run_speed_benchmark
from .core.optimizer import ExactTimeScheduler, get_strategy_for_goal
from .callbacks import (
    PESORestartCallback,
    ResearchCallback,
    ExactTimeStopCallback,
    AutoDoRACallback,  # ← Added: now publicly available
)

# Optional nice aliases
from .cli import main

__all__ = [
    "tune",
    "UzombieTrainer",
    "auto_optimize",
    "push_to_hub_auto",
    "UzombieProjector",
    "get_logger",
    "run_speed_benchmark",
    "ExactTimeScheduler",
    "get_strategy_for_goal",
    "PESORestartCallback",
    "ResearchCallback",
    "ExactTimeStopCallback",
    "AutoDoRACallback",  # ← Exposed in public API
    "main",
]