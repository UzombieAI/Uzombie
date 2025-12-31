# src/uzombie/__init__.py
"""
Uzombie v2 — The fastest LLM fine-tuning engine on Earth (SOTA Alignment + Hybrid Subspaces)
"""

__version__ = "2.0.0"
__author__ = "Kafoo"

# Core Hardware/Optimizer
from .core.hardware import auto_optimize
from .core.optimizer import ExactTimeScheduler, get_strategy_for_goal
from .core.hybrid_projector import UzombieProjector

# Trainers
from .trainer.uzombie_trainer import UzombieTrainer
from .trainer.simpo_trainer import UzombieSimPOTrainer

# Callbacks
from .callbacks import (
    PESORestartCallback,
    ResearchCallback,
    ExactTimeStopCallback,
    AutoDoRACallback,
)

# Utils
from .utils.logger import get_logger, console
from .utils.upload import push_to_hub_auto
from .utils.benchmarks import run_speed_benchmark

# New Data/Modeling
from .data import builders
from .modeling import patching, merging

__all__ = [
    "UzombieTrainer",
    "UzombieSimPOTrainer",
    "UzombieProjector",
    "auto_optimize",
    "ExactTimeScheduler",
    "PESORestartCallback",
    "ResearchCallback",
    "AutoDoRACallback",
    "get_logger",
    "console",
    "builders",
    "patching",
    "merging",
]