#!/bin/bash
#SBATCH --job-name=cadence-session
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx6000ada:2
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/cadence_session_%j.out
#SBATCH --error=logs/cadence_session_%j.err

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

SESSION="${1:-y_06}"
echo "Session: $SESSION"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
START_S=$SECONDS

# 2-GPU parallel directions: p1->p2 on cuda:0, p2->p1 on cuda:1
python -u scripts/run_session.py \
    --session "$SESSION" \
    --config configs/cluster.yaml \
    --parallel-directions \
    --output results/cluster_session

ELAPSED=$(( SECONDS - START_S ))
echo "---"
echo "Session: $SESSION"
echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total runtime: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
