#!/bin/bash
# Needs sudo!
# 1. Enable Persistence Mode (Keeps driver loaded)
sudo nvidia-smi -pm 1
# 2. Lock Clocks (For RTX 3070 Ti, ~1700-1800MHz is sweet spot, usually max is better)
# Note: Laptop/Consumer GPUs might lock this capability.
sudo nvidia-smi --lock-gpu-clocks=1500,2100
# 3. Disable ECC (Not applicable to 3070 Ti usually, but good practice for A100s)
# sudo nvidia-smi -e 0
echo "🚀 GPU Clocks Locked for Stability"