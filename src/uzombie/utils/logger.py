# src/uzombie/utils/logger.py
# Uzombie Rich Logger — Clean neon terminal with pixel zombie banner (Design 3)

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

# Print beautiful pixel zombie banner (Design 3) — only once
# No global needed — module-level variable is safe
_banner_printed = False

if not _banner_printed:
    banner = r"""
\033[38;2;0;255;65m
 ██╗   ██╗███████╗ ██████╗ ███╗   ███╗██████╗ ██╗███████╗
 ██║   ██║╚══███╔╝██╔═══██╗████╗ ████║██╔══██╗██║██╔════╝
 ██║   ██║  ███╔╝ ██║   ██║██╔████╔██║██████╔╝██║█████╗  
 ██║   ██║ ███╔╝  ██║   ██║██║╚██╔╝██║██╔══██╗██║██╔══╝  
 ╚██████╔╝███████╗╚██████╔╝██║ ╚═╝ ██║██████╔╝██║███████╗
  ╚═════╝ ╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚═════╝ ╚═╝╚══════╝
\033[0m
\033[38;2;180;200;180m              ████████          \033[0m
\033[38;2;160,190,160m            ██\033[38;2;255;255;255m████\033[38;2;160,190,160m██████      \033[0m  \033[38;2;255;80;80m╔═══════════════════════════╗\033[0m
\033[38;2;160,190,160m          ██\033[38;2;255;255;255m██\033[38;2;0;0;0m░\033[38;2;255;255;255m██\033[38;2;160,190,160m████\033[38;2;255;255;255m████\033[38;2;160,190,160m██    \033[0m  \033[38;2;255;100;100m║\033[0m \033[38;2;0;255;100m3.5-4.0× FASTER\033[0m than Unsloth \033[38;2;255;100;100m║\033[0m
\033[38;2;160,190,160m          ██\033[38;2;0;0;0m░░\033[38;2;160,190,160m████████████      \033[0m  \033[38;2;255;100;100m║\033[0m \033[38;2;100;255;150mHybrid Subspace Magic\033[0m      \033[38;2;255;100;100m║\033[0m
\033[38;2;160,190,160m            ████\033[38;2;255;200;200m▼▼▼▼\033[38;2;160,190,160m████        \033[0m  \033[38;2;255;100;100m║\033[0m \033[38;2;150;255;180mZero Config | Pure Speed\033[0m   \033[38;2;255;100;100m║\033[0m
\033[38;2;160,190,160m            ████████████████      \033[0m  \033[38;2;255;80;80m╚═══════════════════════════╝\033[0m
\033[38;2;140,170,140m          ████\033[38;2;80;60;40m▓▓▓▓▓▓▓▓\033[38;2;140,170,140m████      \033[0m
\033[38;2;140,170,140m        ██\033[38;2;180;180;180m████\033[38;2;80;60;40m▓▓▓▓▓▓\033[38;2;180;180;180m████\033[38;2;140,170,140m████    \033[0m  \033[38;2;100;200;255m→ github.com/kafoo/uzombie\033[0m
\033[38;2;140,170,140m        ████████████████████    \033[0m
\033[38;2;140,170,140m          ████████  ████████    \033[0m
"""
    console.print(banner[1:].rstrip())
    console.rule("[bold magenta]UZOMBIE v1 — The fastest fine-tuning engine on Earth[/]")
    _banner_printed = True  # ← No global needed