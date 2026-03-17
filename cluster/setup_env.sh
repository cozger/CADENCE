#!/bin/bash
# Verify CADENCE environment on SSRDE cluster.
# CADENCE shares the MCCT conda env — no separate install needed.
# Usage: bash cluster/setup_env.sh

set -e

echo "Setting up CADENCE on $(hostname)..."

# Verify MCCT conda env exists
eval "$(conda shell.bash hook)"
conda activate MCCT

python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    print(f'  GPU {i}: {name} ({mem:.0f} GB)')
import scipy, numpy, sklearn, matplotlib, yaml
print('All dependencies available.')
"

# Symlink session_cache from MCCT (avoid 3.7GB duplicate)
ln -sfn ~/MCCT/session_cache ~/CADENCE/session_cache
echo "Symlinked session_cache -> ~/MCCT/session_cache"

# Create output directories
mkdir -p ~/CADENCE/logs ~/CADENCE/results

echo "Done. CADENCE environment ready."
