#!/bin/bash
#SBATCH --job-name=cadence-pseudopair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/cadence_pseudopair_%j.out
#SBATCH --error=logs/cadence_pseudopair_%j.err

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "---"

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate MCCT
cd $HOME/CADENCE

export PYTHONUNBUFFERED=1

N_PAIRS="${1:-5}"
echo "Pseudo-pair null test: $N_PAIRS pairs"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
START_S=$SECONDS

python -u scripts/run_pseudopair.py \
    --config configs/cluster.yaml \
    --n-pairs "$N_PAIRS" \
    --output results/pseudopair_null

ELAPSED=$(( SECONDS - START_S ))
echo "---"
echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total runtime: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
