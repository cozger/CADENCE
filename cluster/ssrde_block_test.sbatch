#!/bin/bash
#SBATCH --job-name=cadence-block-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx6000ada:2
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/cadence_block_test_%j.out
#SBATCH --error=logs/cadence_block_test_%j.err

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

echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
START_S=$SECONDS

python -u scripts/test_block_size.py --output results/block_size_test

ELAPSED=$(( SECONDS - START_S ))
echo "---"
echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total runtime: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
