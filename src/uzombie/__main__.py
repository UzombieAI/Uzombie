# src/uzombie/__main__.py
"""
Run Uzombie CLI directly: python -m uzombie tune ...
Research: TRL SFTConfig best practices (web:40, web:43); Unsloth integration (web:0, web:1).
"""

from .cli import main

if __name__ == "__main__":
    main()