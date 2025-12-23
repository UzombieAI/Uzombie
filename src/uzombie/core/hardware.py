# src/uzombie/core/hardware.py
# Auto-optimize: Unsloth + dtype auto + QLoRA (web:0, web:1; 70% less VRAM, 2x speed)
# Research: BF16 on Ada (RTX 40-series) for stability (arXiv:2403.03507 GaLore compat)

import torch
import os
from unsloth import FastLanguageModel

def auto_optimize(model_name: str, max_seq_length: int = 32768, **kwargs):
    """
    Load Unsloth model with auto dtype (BF16 if supported) + 4-bit QLoRA.
    Forwards kwargs (e.g., load_in_4bit=True, attn_implementation='xformers').
    Unsloth best practices: dtype=None auto-detects (web:0, web:1).
    """
    # FIXED: Set defaults if not passed (QLoRA for low VRAM, web:0)
    if 'load_in_4bit' not in kwargs:
        kwargs['load_in_4bit'] = True
    if 'dtype' not in kwargs:
        kwargs['dtype'] = None  # Auto: BF16 on Ampere+ (your 4050 Ada), FP16 fallback (web:1)

    # FIXED: Handle attn_implementation via env (Unsloth auto-FlashAttn, web:0)
    if 'attn_implementation' in kwargs:
        attn_impl = kwargs.pop('attn_implementation')
        os.environ['FLASH_ATTENTION_FORCE_REENTRANT'] = '1' if attn_impl == 'flash_attn' else '0'
        os.environ['xformers_attention_backend'] = attn_impl if attn_impl == 'xformers' else 'None'

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        **kwargs  # Forward all (e.g., dtype=None, load_in_4bit=True)
    )

    return model, tokenizer