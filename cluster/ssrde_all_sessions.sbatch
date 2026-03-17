#!/bin/bash
#SBATCH --job-name=cadence-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:rtx6000ada:2
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/cadence_all_%j.out
#SBATCH --error=logs/cadence_all_%j.err

# Ensure logs directory exists
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "---"

# Initialize conda
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate MCCT
cd $HOME/CADENCE

# Unbuffered output
export PYTHONUNBUFFERED=1

echo "All sessions analysis"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
START_S=$SECONDS

python -u scripts/run_all_sessions.py \
    --config configs/cluster.yaml \
    --output results/cluster_all_sessions \
    --parallel-directions

ELAPSED=$(( SECONDS - START_S ))
echo "---"
echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total runtime: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
