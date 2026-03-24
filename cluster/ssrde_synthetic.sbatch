#!/bin/bash
#SBATCH --job-name=cadence-synthetic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx6000ada:2
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/cadence_synthetic_%j.out
#SBATCH --error=logs/cadence_synthetic_%j.err

# CADENCE synthetic validation on dual GPUs.
# Splits 6 tests across 2 GPUs (3 tests each) for ~2x speedup.
#
# Usage:
#   sbatch cluster/ssrde_synthetic.sbatch              # default 1800s
#   sbatch cluster/ssrde_synthetic.sbatch 600           # custom duration

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "---"

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate MCCT
cd $HOME/CADENCE

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

DURATION="${1:-1800}"
OUTPUT="results/cadence_synthetic_v2"

echo "Duration: ${DURATION}s"
echo "Output:   ${OUTPUT}"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
START_S=$SECONDS

# Split 5 tests across 2 GPUs:
#   GPU 0: A (EEG), B (BL)              — wavelet CWT for A
#   GPU 1: C (Pose), D (EEG+BL), E (Null) — wavelet CWT for D
# One EEG wavelet test per GPU for balanced load.
# Each writes to a separate output dir to avoid file race conditions.

mkdir -p "${OUTPUT}/gpu0" "${OUTPUT}/gpu1"

CUDA_VISIBLE_DEVICES=0 python -u scripts/run_synthetic.py \
    --duration "$DURATION" \
    --device cuda:0 \
    --tests A,B \
    --output "${OUTPUT}/gpu0" \
    2>&1 | tee logs/synthetic_gpu0_${SLURM_JOB_ID}.log &
PID_GPU0=$!

CUDA_VISIBLE_DEVICES=1 python -u scripts/run_synthetic.py \
    --duration "$DURATION" \
    --device cuda:0 \
    --tests C,D,E \
    --output "${OUTPUT}/gpu1" \
    2>&1 | tee logs/synthetic_gpu1_${SLURM_JOB_ID}.log &
PID_GPU1=$!

echo "GPU 0 (tests A,B): PID $PID_GPU0"
echo "GPU 1 (tests C,D,E): PID $PID_GPU1"
echo "---"

# Wait for both to complete
EXIT_0=0
EXIT_1=0
wait $PID_GPU0 || EXIT_0=$?
wait $PID_GPU1 || EXIT_1=$?

echo ""
echo "=== Merge results ==="

# Merge results from both GPU runs into the top-level output directory
python -u -c "
import json, os

output = '${OUTPUT}'
results = []
for gpu_dir in ['gpu0', 'gpu1']:
    path = os.path.join(output, gpu_dir, 'synthetic_results.json')
    if os.path.exists(path):
        with open(path) as f:
            results.extend(json.load(f))
    else:
        print(f'WARNING: {path} not found')

results.sort(key=lambda r: r['test'])
n_pass = sum(1 for r in results if r['passed'])
n_total = len(results)

print(f'')
print('=' * 60)
print(f'COMBINED: {n_pass}/{n_total} tests passed')
print('=' * 60)
for r in results:
    icon = '[PASS]' if r['passed'] else '[FAIL]'
    print(f'  {icon} {r[\"test\"]}: {r[\"status\"]}')

merged = os.path.join(output, 'synthetic_results.json')
with open(merged, 'w') as f:
    json.dump(results, f, indent=2)
print(f'')
print(f'Merged results: {merged}')
"

ELAPSED=$(( SECONDS - START_S ))
echo "---"
echo "GPU 0 exit: $EXIT_0"
echo "GPU 1 exit: $EXIT_1"
echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total runtime: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"

# Fail the job if either GPU process failed
if [ $EXIT_0 -ne 0 ] || [ $EXIT_1 -ne 0 ]; then
    echo "ERROR: One or both GPU workers failed"
    exit 1
fi
