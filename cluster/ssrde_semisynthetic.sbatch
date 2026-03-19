#!/bin/bash
#SBATCH --job-name=cadence-semisyn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx6000ada:2
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/cadence_semisynthetic_%j.out
#SBATCH --error=logs/cadence_semisynthetic_%j.err

# CADENCE semi-synthetic validation on dual GPUs.
#
# Splits cross-dyad pairs across GPUs by index (same pairing seed).
# GPU 0: pairs 0,1  |  GPU 1: pairs 2,3
#
# Both GPUs build the SAME pair assignments (--pair-seed is shared),
# then each GPU only runs its assigned --pair-indices.  Base sessions
# (with interbrain CWT) are built once per pair; coupling injection
# is CPU-only numpy.
#
# Usage:
#   sbatch cluster/ssrde_semisynthetic.sbatch              # 4 pairs
#   sbatch cluster/ssrde_semisynthetic.sbatch 6             # 6 pairs

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

N_PAIRS="${1:-4}"
OUTPUT="results/semisynthetic"
SEED=42

echo "N_PAIRS: ${N_PAIRS}"
echo "Output:  ${OUTPUT}"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
START_S=$SECONDS

# Split pair indices across GPUs
# GPU 0: even pairs (0, 2, 4, ...)
# GPU 1: odd pairs  (1, 3, 5, ...)
GPU0_INDICES=""
GPU1_INDICES=""
for ((i=0; i<N_PAIRS; i++)); do
    if (( i % 2 == 0 )); then
        [ -n "$GPU0_INDICES" ] && GPU0_INDICES="${GPU0_INDICES},"
        GPU0_INDICES="${GPU0_INDICES}${i}"
    else
        [ -n "$GPU1_INDICES" ] && GPU1_INDICES="${GPU1_INDICES},"
        GPU1_INDICES="${GPU1_INDICES}${i}"
    fi
done

echo "GPU 0 pairs: ${GPU0_INDICES}"
echo "GPU 1 pairs: ${GPU1_INDICES}"

mkdir -p "${OUTPUT}/gpu0" "${OUTPUT}/gpu1"

# GPU 0
if [ -n "$GPU0_INDICES" ]; then
    CUDA_VISIBLE_DEVICES=0 python -u scripts/run_semisynthetic.py \
        --config configs/cluster.yaml \
        --n-pairs "$N_PAIRS" \
        --pair-indices "$GPU0_INDICES" \
        --pair-seed "$SEED" \
        --device cuda:0 \
        --output "${OUTPUT}/gpu0" \
        2>&1 | tee logs/semisyn_gpu0_${SLURM_JOB_ID}.log &
    PID_GPU0=$!
    echo "GPU 0 PID: $PID_GPU0"
fi

# GPU 1
if [ -n "$GPU1_INDICES" ]; then
    CUDA_VISIBLE_DEVICES=1 python -u scripts/run_semisynthetic.py \
        --config configs/cluster.yaml \
        --n-pairs "$N_PAIRS" \
        --pair-indices "$GPU1_INDICES" \
        --pair-seed "$SEED" \
        --device cuda:0 \
        --output "${OUTPUT}/gpu1" \
        2>&1 | tee logs/semisyn_gpu1_${SLURM_JOB_ID}.log &
    PID_GPU1=$!
    echo "GPU 1 PID: $PID_GPU1"
fi

echo "---"

# Wait for both
EXIT_0=0
EXIT_1=0
[ -n "${PID_GPU0:-}" ] && { wait $PID_GPU0 || EXIT_0=$?; }
[ -n "${PID_GPU1:-}" ] && { wait $PID_GPU1 || EXIT_1=$?; }

echo ""
echo "=== Merge results ==="

# Merge results from both GPU runs
python -u -c "
import json, os

output = '${OUTPUT}'
all_analyses = []
all_roc = {}
all_pairs = []

for gpu_dir in ['gpu0', 'gpu1']:
    path = os.path.join(output, gpu_dir, 'semisynthetic_results.json')
    if not os.path.exists(path):
        print(f'WARNING: {path} not found')
        continue
    with open(path) as f:
        data = json.load(f)
    all_analyses.extend(data.get('analyses', []))
    all_pairs.extend(data.get('pairs', []))

    for mod, roc in data.get('roc_data', {}).items():
        if mod not in all_roc:
            all_roc[mod] = {'labels': [], 'scores': []}
        all_roc[mod]['labels'].extend(roc.get('labels', []))
        all_roc[mod]['scores'].extend(roc.get('scores', []))

# Recompute AUC from merged data
auc_results = {}
for mod, roc in all_roc.items():
    labels = roc['labels']
    scores = roc['scores']
    if len(set(labels)) >= 2:
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels, [-s for s in scores])
        except ImportError:
            n_pos = sum(labels)
            n_neg = len(labels) - n_pos
            paired = sorted(zip([-s for s in scores], labels), reverse=True)
            tp = 0; auc = 0.0
            for sc, lb in paired:
                if lb == 1: tp += 1
                else: auc += tp
            auc = auc / (n_pos * n_neg) if (n_pos * n_neg) > 0 else float('nan')
    else:
        auc = float('nan')
    auc_results[mod] = auc
    roc['auc'] = auc

print()
print('=' * 60)
print('MERGED AUC RESULTS')
print('=' * 60)
mod_short = {'eeg_wavelet': 'EEGw', 'blendshapes_v2': 'BL', 'pose_features': 'Pose'}
for mod, auc in auc_results.items():
    short = mod_short.get(mod, mod)
    n_pos = sum(all_roc[mod]['labels'])
    n_neg = len(all_roc[mod]['labels']) - n_pos
    print(f'  {short:>6s}: AUC = {auc:.3f} (n_pos={n_pos}, n_neg={n_neg})')

# Null FPR
null_results = [r for r in all_analyses if r.get('kappa', -1) == 0.0
                and r.get('target_mod') != 'combined']
n_fp = sum(1 for r in null_results if r.get('detected', False))
n_null = len(null_results)
fpr = n_fp / max(n_null, 1)
print(f'  Null FPR: {n_fp}/{n_null} = {fpr:.1%}')

# Verification
all_pass = True
for mod, auc in auc_results.items():
    short = mod_short.get(mod, mod)
    if auc < 0.80:
        print(f'  [WARN] {short} AUC={auc:.3f} < 0.80')
        all_pass = False
    else:
        print(f'  [ OK ] {short} AUC={auc:.3f} >= 0.80')
if fpr > 0.10:
    print(f'  [WARN] Null FPR={fpr:.1%} > 10%')
    all_pass = False
else:
    print(f'  [ OK ] Null FPR={fpr:.1%} <= 10%')
print(f'')
print('ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED')

merged = {
    'n_pairs': len(all_pairs),
    'auc': auc_results,
    'analyses': all_analyses,
    'roc_data': {m: {'labels': d['labels'], 'scores': d['scores'], 'auc': d['auc']}
                 for m, d in all_roc.items()},
    'pairs': all_pairs,
}
merged_path = os.path.join(output, 'semisynthetic_results.json')
with open(merged_path, 'w') as f:
    json.dump(merged, f, indent=2)
print(f'Merged results: {merged_path}')
"

ELAPSED=$(( SECONDS - START_S ))
echo "---"
echo "GPU 0 exit: $EXIT_0"
echo "GPU 1 exit: $EXIT_1"
echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total runtime: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"

if [ $EXIT_0 -ne 0 ] || [ $EXIT_1 -ne 0 ]; then
    echo "ERROR: One or both GPU workers failed"
    exit 1
fi
