# src/uzombie/utils/logger.py
# Uzombie Rich Logger — Clean neon terminal

import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

# Custom neon theme
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "debug": "dim white",
})

console = Console(theme=custom_theme, markup=True)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_time=True, show_path=False)]
)

def get_logger(name: str):
    """Returns a beautiful Rich logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger

# Global logger
logger = get_logger("uzombie")