# src/uzombie/core/hardware.py
# Auto-optimize: Unsloth + dtype auto + QLoRA (web:0, web:1; 70% less VRAM, 2x speed)
# Research: BF16 on Ada (RTX 40-series) for stability (arXiv:2403.03507 GaLore compat)

import torch
import os
import math
from unsloth import FastLanguageModel
from ..utils.logger import console

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

def estimate_safe_batch_size(model_name: str, vram_gb: float, ctx_len: int, target_bsz: int = 4) -> int:
    """
    Physics-Based VRAM Estimator to push hardware to the edge.
    
    Formula:
    VRAM_Used = Model_Weights + KV_Cache + Activations + Overhead
    - Model (4-bit): ~0.75GB per billion params + Overhead
    - Activations: B * S * H * ... (Scales linearly with Batch & Context)
    """
    # Heuristic parameter counts (billions)
    name_lower = model_name.lower()
    if "70b" in name_lower: params = 70
    elif "8b" in name_lower: params = 8
    elif "7b" in name_lower: params = 7
    elif "3b" in name_lower: params = 3
    elif "1b" in name_lower: params = 1.1
    else: params = 7 # Conservative default if unknown
    
    # 1. Static Footprint (Weights + Cuda Context)
    # 4-bit weights (~0.5GB/B) + Metadata/Buffers (~0.25GB/B) + ~1.5GB CUDA Kernel Overhead
    static_mem = (params * 0.75) + 1.5 
    
    available_mem = vram_gb - static_mem
    
    if available_mem <= 0.5: # Leave small buffer
        return 1 # Barely fits model
        
    # 2. Dynamic Footprint per Batch Item (Gradient + Act)
    # Empirical slope for 4-bit LoRA training based on Unsloth benchmarks
    # 1 unit of (Batch * Context/2048) takes roughly X GB
    gb_per_batch_unit = 0.0
    if params > 10: gb_per_batch_unit = 2.5  # Large model
    elif params > 3: gb_per_batch_unit = 1.2 # 7B/8B range
    else: gb_per_batch_unit = 0.6            # TinyLlama / 1B-3B range
    
    # Scale by context length (linear approximation for LoRA gradients)
    ctx_scale = ctx_len / 2048.0
    mem_per_sample = gb_per_batch_unit * ctx_scale
    
    # Calculate Max Batch
    if mem_per_sample > 0:
        max_batch = int(available_mem / mem_per_sample)
    else:
        max_batch = target_bsz
    
    # Clamp to target (don't go crazy if memory is huge) and min 1
    safe_batch = max(1, min(max_batch, target_bsz))
    
    console.print(f"[dim]VRAM Calculus: Avail={available_mem:.2f}GB, Cost/Sample={mem_per_sample:.2f}GB -> MaxBS={max_batch} (Clamped to {safe_batch})[/]")
    
    return safe_batch