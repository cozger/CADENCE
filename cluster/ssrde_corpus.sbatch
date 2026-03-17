#!/bin/bash
#SBATCH --job-name=cadence-corpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --gres=gpu:rtx6000ada:2
#SBATCH --time=10:00:00
#SBATCH --output=logs/cadence_corpus_%j.out
#SBATCH --error=logs/cadence_corpus_%j.err

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

# Launch multi-GPU corpus pipeline
python -u cluster/launch.py \
    --script corpus \
    --config configs/cluster.yaml \
    --n-gpus 2 \
    --duration 3000 \
    --output results/corpus_3000s \
    "$@"
