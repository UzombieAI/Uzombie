# src/uzombie/data/builders.py
import torch
import os
import gc
from datasets import Dataset
from ..utils.logger import console

# --- ENGINE IMPORTS ---
try:
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

def get_real_free_vram(gpu_id=0):
    """Returns the actual free VRAM in GB."""
    if not torch.cuda.is_available(): return 0.0, 0.0
    torch.cuda.empty_cache()
    gc.collect()
    free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
    return free_bytes / (1024**3), total_bytes / (1024**3)

def generate_magpie_reasoning_v2(
    teacher_model_id: str,
    num_samples: int,
    template: str = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
    active_iterations: int = 1,
    temperature_query: float = 1.0,
    temperature_response: float = 0.6,
    max_query_tokens: int = 512,
    max_response_tokens: int = 2048, 
    thinking_template: str = "<|start_header_id|>assistant<|end_header_id|>\n\n<thinking>\n",
):
    """
    Uzombie Hybrid Generator (v4.0)
    - Automatically selects the best engine based on VRAM constraints.
    - Prevents OOM crashes on 8GB cards while maximizing speed on 24GB+ cards.
    """
    gpu_count = torch.cuda.device_count()
    free_gb, total_gb = get_real_free_vram()
    
    console.print(f"[bold green]🧟 Starting Uzombie Hybrid Pipeline[/]")
    console.print(f" • Hardware: {gpu_count}x GPUs | VRAM: {free_gb:.2f}/{total_gb:.2f} GB Free")

    # --- DECISION ENGINE ---
    # 1. Estimate Model Weight Size (4-bit ~ 0.7GB per Billion Params)
    if "8b" in teacher_model_id.lower() or "7b" in teacher_model_id.lower(): 
        estimated_weights = 5.8 
    elif "70b" in teacher_model_id.lower(): 
        estimated_weights = 40.0
    elif "1b" in teacher_model_id.lower(): 
        estimated_weights = 1.2
    else: 
        estimated_weights = 6.0 # Safe default
        
    headroom = free_gb - estimated_weights
    console.print(f" • Headroom Calculation: {free_gb:.2f}GB (Free) - {estimated_weights:.2f}GB (Model) = {headroom:.2f}GB")

    use_unsloth = False
    
    # 2. The Switch Logic
    # vLLM generally needs ~2.0 GB overhead for KV Cache blocks + CUDA Graphs to run efficiently.
    # If we have less than that, we switch to Unsloth Native (which needs ~0.5 GB overhead).
    if headroom < 2.0:
        console.print(f"[yellow]⚠️ Tight VRAM Detected (Only {headroom:.2f}GB headroom).[/]")
        if UNSLOTH_AVAILABLE:
            console.print(f"[bold cyan]🔄 Switching to Unsloth Native Inference (Maximum Efficiency Mode)[/]")
            use_unsloth = True
        else:
            console.print("[bold red]⛔ Critical: VRAM too low for vLLM and Unsloth not found.[/]")
            return None
    elif not VLLM_AVAILABLE:
        console.print("[yellow]vLLM not found. Falling back to Unsloth.[/]")
        use_unsloth = True
    else:
        console.print(f"[bold green]🚀 Sufficient VRAM detected. Engaging vLLM Turbo Engine.[/]")

    all_data = []

    # =========================================================
    # PATH A: UNSLOTH NATIVE (The "Off-Road" Efficient Path)
    # =========================================================
    if use_unsloth:
        # Load Model in 4-bit (Minimal Memory Footprint)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=teacher_model_id,
            max_seq_length=max_query_tokens + max_response_tokens,
            dtype=None, 
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model) # Enables 2x faster inference optimization
        
        console.print(f"[cyan]▶ Generating {num_samples} samples using Unsloth...[/]")
        
        # Unsloth generation is sequential loop (slower but safe)
        for i in range(num_samples):
            # Step 1: Query Generation
            inputs = tokenizer([template], return_tensors="pt").to("cuda")
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_query_tokens,
                temperature=temperature_query,
                do_sample=True,
                stop_strings=[tokenizer.eos_token, "<|eot_id|>", "assistant", "\n\n"],
                tokenizer=tokenizer,
            )
            text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Simple parser to get just the query part
            # (Assumes template is at the start)
            query = text[len(template):].strip() if template in text else text.strip()
            
            if len(query) < 5: continue

            # Step 2: Response Generation
            prompt = f"{template}{query}\n{thinking_template}"
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_response_tokens,
                temperature=temperature_response,
                do_sample=True,
            )
            response_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Extract just the new response
            # (In production, use robust regex, but split works for simple templates)
            final_response = response_text.replace(template, "").replace(query, "").strip()
            
            if len(final_response) > 50:
                all_data.append({
                    "instruction": query,
                    "input": "",
                    "output": final_response,
                    "generator": teacher_model_id
                })
                # Progress indicator
                print(".", end="", flush=True)
                
            if len(all_data) >= num_samples: break
            
        console.print(f"\n[bold green]✅ Unsloth Generation Complete.[/]")
        del model, tokenizer
        torch.cuda.empty_cache()

    # =========================================================
    # PATH B: vLLM (The "Ferrari" High-Speed Path)
    # =========================================================
    else:
        # Calculate safe utilization for vLLM path
        # If we are here, we have >2GB headroom, so we can be generous.
        buffer_bytes = 0.5 * 1024**3
        safe_utilization = (free_gb * 1024**3 - buffer_bytes) / (total_gb * 1024**3)
        safe_utilization = min(safe_utilization, 0.95)
        
        try:
            llm = LLM(
                model=teacher_model_id,
                tensor_parallel_size=gpu_count, 
                max_model_len=4096, # Safe default for high-VRAM cards
                gpu_memory_utilization=safe_utilization,
                trust_remote_code=True,
                enforce_eager=True,
                dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
                quantization="bitsandbytes" if "4bit" in teacher_model_id.lower() else None
            )
            
            tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)
            stop_tokens = [tokenizer.eos_token]
            if "<|eot_id|>" in tokenizer.get_vocab(): stop_tokens.append("<|eot_id|>")
            if "<|start_header_id|>" in tokenizer.get_vocab(): stop_tokens.append("<|start_header_id|>")
            
            samples_per_iter = max(1, num_samples // active_iterations)
            
            for i in range(active_iterations):
                # Query Gen
                qp = SamplingParams(temperature=temperature_query, max_tokens=max_query_tokens, stop=stop_tokens)
                q_out = llm.generate([template] * samples_per_iter, qp)
                valid_q = [o.outputs[0].text.strip() for o in q_out if len(o.outputs[0].text) > 10]
                
                if not valid_q: continue

                # Response Gen
                rp = SamplingParams(temperature=temperature_response, max_tokens=max_response_tokens)
                prompts = [f"{template}{q}{tokenizer.eos_token}{thinking_template}" for q in valid_q]
                r_out = llm.generate(prompts, rp)
                
                for q, r in zip(valid_q, r_out):
                    resp = r.outputs[0].text.strip()
                    if len(resp) > 50:
                        all_data.append({
                            "instruction": q,
                            "input": "",
                            "output": f"<thinking>\n{resp}",
                            "generator": teacher_model_id
                        })
            
            destroy_model_parallel()
            del llm.llm_engine.model_executor
            del llm
            torch.cuda.empty_cache()
            
        except Exception as e:
            console.print(f"[bold red]vLLM Error: {e}[/]")
            return None

    return Dataset.from_list(all_data)