# SSRDE Cluster Guide

## Overview

The UCSD SSRDE cluster provides multi-GPU training for MCCT. This guide covers SSH setup, environment configuration, code/data sync, job submission, and DDP training.

**Cluster specs:**
- Login node: `ssrde.ucsd.edu` (1x RTX A5000, 24GB — for testing only)
- Compute nodes:
  - `ssrde-c-a01`, `ssrde-c-a02`: 256 CPUs (EPYC 9754), 232GB RAM, **2x RTX 6000 ADA** (48GB each)
  - `ssrde-c-b01`: 64 CPUs, 772GB RAM, **1x RTX 6000 ADA**
  - `ssrde-c-c01`: 80 CPUs, 772GB RAM, no GPU
- Scheduler: Slurm, single "general" partition, 7-day max wall time
- Home quota: 100GB (1TB grace for 35 days)
- Access: UCSD VPN required

## 1. SSH Setup (Windows)

### Generate key and configure SSH

```bash
# In Git Bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ssrde -N "" -C "mcct-cluster"
```

Create `~/.ssh/config`:
```
Host ssrde
    HostName ssrde.ucsd.edu
    User cozger
    StrictHostKeyChecking accept-new
    PreferredAuthentications keyboard-interactive,password
```

> **Note:** SSRDE requires password + Duo 2FA on every SSH session. Public key auth is not accepted by the server. Windows OpenSSH does not support ControlMaster, so each raw SSH/SCP connection requires a separate Duo approval. Use `pipeline.py` (Section 9) to batch operations into single SSH connections.

> **CRITICAL:** Never attempt automated/non-interactive SSH connections (from Claude Code, scripts, or subprocesses). The server detects non-interactive sessions and forces a disconnect. ALL SSH/SCP commands must be run manually by the user in their own terminal to complete the password + Duo 2FA flow.

### Connecting

```bash
ssh ssrde
# Enter password, then approve Duo push
```

## 2. Cluster Environment

### One-time setup (on the cluster)

```bash
# Install miniconda (if not already present)
curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc

# Accept conda TOS
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create MCCT environment
conda create -n MCCT python=3.11 -y
conda activate MCCT

# Install PyTorch (cu124 for Ada Lovelace GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy scikit-learn matplotlib pandas pyxdf einops pyyaml

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, NCCL: {torch.distributed.is_nccl_available()}')"
```

Or upload and run the setup script:
```bash
# From Git Bash on Windows
scp /c/Users/optilab/Desktop/MCCT/cluster/setup_env.sh ssrde:~/MCCT/cluster/
# On cluster
bash ~/MCCT/cluster/setup_env.sh
```

## 3. Syncing Code and Data

### Recommended: `pipeline.py` (1 Duo auth)

The pipeline uses tar-over-ssh to bundle all files into a single archive and pipe it through one SSH connection. Text files are automatically converted from Windows (`\r\n`) to Unix (`\n`) line endings.

```bash
cd C:\Users\optilab\Desktop\MCCT

# Sync code only (1 Duo auth)
python cluster/pipeline.py sync

# Sync code + extra files (e.g. pretrained checkpoint) (1 Duo auth)
python cluster/pipeline.py sync --include results/v5/pretrained/v5_self_attention.pt

# Sync code + session data (1 Duo auth, but large upload)
python cluster/pipeline.py sync --data

# Sync + submit job in one shot (1 Duo auth total)
python cluster/pipeline.py run --include results/v5/pretrained/v5_self_attention.pt
```

> **How it works:** Python's `tarfile` creates a `.tar.gz` archive locally, then pipes it through a single `ssh` connection: `ssh ssrde "cd ~/MCCT && tar xzf -" < archive.tar.gz`. For `run`, the remote command chains extraction with job submission (`tar xzf - && sbatch ...`), so the entire sync+submit happens in one authenticated session.

### Manual scp (multiple Duo auths)

If `pipeline.py` is unavailable, use raw scp. Each call requires a separate Duo approval:

```bash
# Code directories (one scp call)
scp -r /c/Users/optilab/Desktop/MCCT/common /c/Users/optilab/Desktop/MCCT/model /c/Users/optilab/Desktop/MCCT/training /c/Users/optilab/Desktop/MCCT/data ssrde:~/MCCT/

# More code directories (second scp call)
scp -r /c/Users/optilab/Desktop/MCCT/evaluation /c/Users/optilab/Desktop/MCCT/visualization /c/Users/optilab/Desktop/MCCT/scripts /c/Users/optilab/Desktop/MCCT/configs /c/Users/optilab/Desktop/MCCT/cluster ssrde:~/MCCT/
```

> **Important:** Use Git Bash, NOT PowerShell. PowerShell's `scp` is the Windows built-in copy command, not SSH scp, and will copy files locally instead of to the cluster.

> **Important:** Paste each scp command as a **single line**. If the terminal wraps it to multiple lines, the command will fail.

> **Important:** Shell scripts (`.sh`, `.sbatch`) must have Unix line endings (`\n`). If uploading manually from Windows, convert with `dos2unix` on the cluster or use `sed -i 's/\r$//'` before running. The pipeline handles this automatically.

### Upload preprocessed data

Upload the session cache (3.7GB) rather than raw XDF files (12GB) — the training pipeline reads from cache directly:

```bash
# Via pipeline (1 Duo auth):
python cluster/pipeline.py sync --data

# Or via raw scp:
scp -r /c/Users/optilab/Desktop/MCCT/session_cache ssrde:~/MCCT/
```

For full reproducibility, also upload raw XDF files:
```bash
scp -r /c/Users/optilab/Desktop/MCCT/sessions ssrde:~/MCCT/
```

### Verify on cluster

```bash
ssh ssrde
cd ~/MCCT && ls -d common model training data evaluation visualization scripts configs cluster
python -c "from training.trainer import Trainer; print('Imports OK')"
```

## 4. DDP Architecture

### How it works

PyTorch DistributedDataParallel (DDP) with torchrun:

1. `torchrun --nproc_per_node=2` launches 2 processes (one per GPU)
2. Each process gets env vars: `RANK` (0 or 1), `LOCAL_RANK` (GPU index), `WORLD_SIZE` (2)
3. `common/distributed.py:setup_ddp()` initializes NCCL backend
4. `DDPModelWrapper` wraps model + loss_fn so DDP syncs the loss's learnable `log_vars` parameters. All training forward passes go through the DDP wrapper (`Trainer._fwd_model`) so gradient sync hooks fire correctly.
5. Training data is sharded: rank 0 generates random permutation, broadcasts to all ranks, each takes its chunk (truncated to equal size across ranks)
6. After backward pass, DDP auto-reduces gradients via NCCL all-reduce
7. Validation loss is all-reduced across ranks for true average
8. Only rank 0 saves checkpoints, writes results, and runs CSGI evaluation

### Single-GPU backward compatibility

All DDP code paths are guarded by `world_size > 1`. When running locally (default `world_size=1`), there is zero overhead — no NCCL init, no data sharding, no all-reduce calls.

### Key files

| File | Purpose |
|------|---------|
| `common/distributed.py` | `setup_ddp()`, `cleanup_ddp()`, `is_main()`, `sync_scalar()` |
| `cluster/launch.py` | torchrun entry point — dispatches to training scripts |
| `cluster/ssrde.sbatch` | Slurm job template |
| `cluster/pipeline.py` | Automation (tar-over-ssh sync/submit/status/results) |
| `cluster/setup_env.sh` | One-time conda env setup |
| `configs/cluster.yaml` | Cluster-optimized hyperparameters (V4 base) |
| `configs/cluster_v7.yaml` | V7 config (bs=512, auto-merge, adaptive resampling) |
| `configs/cluster_v5_tuned.yaml` | V5 tuned config (differential LR, EEG scale fix) |
| `cluster/ssrde_v7.sbatch` | V7 Slurm job template |

## 5. Submitting Jobs

### Quick test (single GPU, login node)

```bash
conda activate MCCT
cd ~/MCCT
python -c "from training.trainer import Trainer; from training.config import default_config; from common.utils import build_model; c=default_config(); c['model_version']='v4'; c['device']='cuda'; m=build_model(c); t=Trainer(m,c); print('OK', m.count_parameters(), 'params')"
```

### DDP test (2 GPUs via Slurm)

```bash
cd ~/MCCT
sbatch --gres=gpu:rtx6000ada:2 --cpus-per-task=4 --mem=16G --time=00:05:00 --output=test_ddp_%j.out --wrap="torchrun --standalone --nproc_per_node=2 test_ddp.py"
```

### Full holdout training

Using the sbatch template:
```bash
cd ~/MCCT
mkdir -p logs
sbatch cluster/ssrde.sbatch
```

Or with custom arguments:
```bash
sbatch cluster/ssrde.sbatch --output results/custom_run
```

The sbatch template (`cluster/ssrde.sbatch`) requests:
- 2x RTX 6000 ADA (`--gres=gpu:rtx6000ada:2`)
- 32 CPUs, 128GB RAM
- 2-day wall time
- Runs: `torchrun --standalone --nproc_per_node=2 cluster/launch.py --config configs/cluster.yaml --script holdout`

### Available training scripts

The launcher (`cluster/launch.py`) supports:
```bash
--script holdout        # Hold-2-out evaluation (train on N-2, test on 2)
--script multi_session  # Train combined model on all sessions
--script session        # Single session training
--script pretrain       # V5 self-attention pre-training
```

## 6. Cluster Config

`configs/cluster.yaml` is optimized for 2x RTX 6000 ADA (48GB each):

| Parameter | Local (RTX 5080) | Cluster |
|-----------|------------------|---------|
| `batch_size` | 128 | 256 (per GPU) |
| `lr` | 1e-4 | 1.4e-4 (sqrt(2) scaling) |
| `vram_budget_gb` | 13.0 | 44.0 |
| Effective batch | 128 | 512 (256 x 2 GPUs) |

The learning rate is scaled by sqrt(2) because the effective batch size doubles with 2 GPUs (established practice from Hoffer et al. 2017).

## 7. Monitoring Jobs

```bash
# Check job status
squeue -u cozger

# See all jobs on the cluster
squeue -p general

# Check node availability
sinfo -N -l

# Tail latest log
ls -t ~/MCCT/logs/mcct_*.out | head -1 | xargs tail -30

# Cancel a job
scancel <JOBID>

# Cancel all your jobs
scancel -u cozger
```

### Job states
- `PD` — Pending (waiting for resources or priority)
- `R` — Running
- `CG` — Completing
- `(Resources)` — waiting for GPUs/CPUs to free up
- `(Priority)` — waiting behind higher-priority jobs

## 8. Retrieving Results

From **Git Bash on Windows**:

```bash
# Download results directory
scp -r ssrde:~/MCCT/results/cluster_run /c/Users/optilab/Desktop/MCCT/results/cluster/

# Download a specific file
scp ssrde:~/MCCT/results/cluster_run/comparison_summary.json /c/Users/optilab/Desktop/MCCT/results/cluster/
```

## 9. Pipeline Automation

`cluster/pipeline.py` batches operations over single SSH connections using tar-over-ssh, minimizing Duo 2FA prompts. Text files are automatically converted to Unix line endings.

### Commands

| Command | Duo auths | What it does |
|---------|-----------|--------------|
| `sync` | 1 | Upload code to cluster |
| `submit` | 1 | Submit Slurm job (returns job ID) |
| `status` | 1 | Check job queue + tail latest log |
| `results` | 1 | Download results from cluster |
| `run` | **1** | sync + submit (single SSH connection) |
| `all` | 1 + N | sync + submit + poll + retrieve |
| `setup` | 2 | One-time: create conda env on cluster |

### Usage

```bash
cd C:\Users\optilab\Desktop\MCCT

# Sync code + submit job (1 Duo auth total)
python cluster/pipeline.py run

# Include extra files (e.g. pretrained checkpoint)
python cluster/pipeline.py run --include results/v5/pretrained/v5_self_attention.pt

# Include session data in sync
python cluster/pipeline.py run --data

# Sync only (no submit)
python cluster/pipeline.py sync --include results/v5/pretrained/v5_self_attention.pt

# Submit only (code already synced)
python cluster/pipeline.py submit

# Check status
python cluster/pipeline.py status

# Download results
python cluster/pipeline.py results --output results/holdout_v5_tuned_fc30
```

### How tar-over-ssh works

Instead of multiple `scp` calls (each needing Duo), the pipeline:
1. Creates a `.tar.gz` archive locally using Python's `tarfile` (no local `tar` binary needed)
2. Converts text files from `\r\n` to `\n` during archiving (prevents `sbatch` DOS line-ending errors)
3. Pipes the archive through a single SSH connection: `ssh ssrde "cd ~/MCCT && tar xzf -"`
4. For `run`, chains extraction with submission: `tar xzf - && sbatch cluster/ssrde.sbatch`

> **Note:** The `all` command polls job status in a loop, requiring a Duo auth per poll (every 60s). For practical use, `run` + manual `status` checks is recommended.

> **IMPORTANT:** Never attempt automated SSH/SCP connections (e.g., from Claude Code or non-interactive subprocesses). The SSRDE server forces a disconnect for non-interactive sessions. All pipeline commands must be run manually by the user in their terminal to complete the password + Duo 2FA flow.

## 10. Troubleshooting

### SSH connection timeout
- Check UCSD VPN is connected. SSRDE is only reachable on the campus network (172.21.x.x).
- Reconnect VPN and retry.

### "Host key verification failed"
```bash
ssh-keygen -R ssrde.ucsd.edu
# Then reconnect — will be prompted to accept new host key
```

### PowerShell scp copies locally instead of to cluster
Use **Git Bash**, not PowerShell. PowerShell's `scp` is the Windows built-in copy command.

### Python command split across lines
The cluster's bash interprets multi-line pastes as separate commands. Always paste Python one-liners as a single line, or write to a `.py` file first.

### Out of disk quota
```bash
du -sh ~/* | sort -h
# Clean up old results, __pycache__, etc.
find ~/MCCT -name __pycache__ -exec rm -rf {} +
```

### NCCL timeout / DDP deadlock
If DDP hangs or crashes with NCCL timeout errors (e.g., "ALLGATHER SeqNum mismatch"):

1. **P2P access:** The SSRDE GPUs may not support peer-to-peer communication. `setup_ddp()` sets `NCCL_P2P_DISABLE=1` and `NCCL_SOCKET_IFNAME=lo` by default to force shared-memory fallback (fine for same-node multi-GPU).

2. **Verify P2P status:**
```bash
python -c "import torch; print(torch.cuda.can_device_access_peer(0, 1))"
```

3. **Enable NCCL diagnostics:** The sbatch template exports `NCCL_DEBUG=INFO` which logs NCCL init to stderr. Check `logs/mcct_<jobid>.err` for the handshake.

4. **Unequal batch counts:** If ranks have different numbers of batches per epoch, collectives will deadlock. The trainer truncates data to equal chunks across ranks (dropping at most 1 sample per epoch).

5. **NaN loss deadlock (fixed 2026-03-09):** If one rank gets NaN loss and skips `loss.backward()`, it skips the DDP gradient all-reduce. The other rank enters all-reduce while the NaN rank moves to the next batch's `sync_scalar()` (BROADCAST). Different NCCL collectives on different ranks at the same sequence number = deadlock. **Fix:** `_process_batch()` in `training/trainer.py` now calls `backward()` unconditionally. NaN gradients propagate harmlessly — `clip_grad_norm_` returns Inf, and `optimizer.step()` is gated on `torch.isfinite(loss) and torch.isfinite(grad_norm)`. The `return None` (to exclude NaN batches from loss statistics) happens after backward.

6. **DDP init in rank-0-only code (fixed 2026-03-09):** Evaluation and fine-tuning run only on rank 0 (inside `if is_main(rank):`), but if the config still carries `world_size=2`, any `Trainer()` created inside that block tries to initialize DDP — which requires both ranks to call `DDP()` simultaneously. **Fix:** `evaluate_held_out_session()` in `scripts/run_holdout.py` copies the config and forces `world_size=1, rank=0, local_rank=0` so all downstream code (fine-tuning, per-session baseline, checkpoint evaluation) runs in single-GPU mode.

## 11. DDP Fixes and Lessons Learned (2026-03-09)

This section documents the issues encountered during the first multi-GPU cluster runs and how they were resolved.

### Fix A: NaN loss causing NCCL collective mismatch

**Symptom:** Job freezes ~10 minutes after NCCL init. Watchdog kills both ranks. `.err` log shows rank 1 stuck on `BROADCAST` (SeqNum=82) while rank 0 is on `ALLREDUCE` (SeqNum=84) — a 2-sequence-number gap.

**Root cause:** Rank 1 hit NaN loss early on. The old code had:
```python
# OLD (broken for DDP)
if not torch.isfinite(loss):
    return None        # <-- skips backward() and DDP all-reduce
self.optimizer.zero_grad()
loss.backward()        # <-- DDP gradient sync happens here
```

When rank 1 returned early, it skipped `backward()` and the DDP gradient all-reduce. Rank 0 still called `backward()` and entered all-reduce (ALLREDUCE). On the next batch, rank 1 called `sync_scalar()` (BROADCAST) while rank 0 was still in the previous all-reduce. Different collective types at different sequence numbers = deadlock.

**Fix** (`training/trainer.py:_process_batch`):
```python
# NEW (DDP-safe)
self.optimizer.zero_grad()
loss.backward()        # <-- always runs, DDP sync always fires

grad_norm = clip_grad_norm_(...)

finite_loss = torch.isfinite(loss)
if finite_loss and torch.isfinite(grad_norm):
    self.optimizer.step()  # <-- gated on both loss and grad being finite

if not finite_loss:
    return None            # <-- only affects loss statistics, after backward
```

**Principle:** In DDP, `backward()` must run on every rank for every batch. It's the mechanism that triggers NCCL gradient all-reduce. Gate only `optimizer.step()`, never `backward()`.

### Fix B: DDP init inside rank-0-only evaluation block

**Symptom:** Combined training completes, but the job crashes immediately when starting fine-tuning with `RuntimeError: value cannot be converted to type int without overflow` at `DDP(model, ...)`.

**Root cause:** The evaluation loop (`evaluate_held_out_session`) is gated by `if is_main(rank):` — only rank 0 runs it. But the config still contains `world_size=2`. When `finetune_session()` creates a `Trainer(model, config)`, the constructor sees `world_size > 1` and calls `DDP(model, ...)`. DDP initialization is a collective operation requiring all ranks to participate. Rank 1 never enters this code path, so DDP init fails.

**Fix** (`scripts/run_holdout.py:evaluate_held_out_session`):
```python
def evaluate_held_out_session(session, session_name, role_direction, config, ...):
    # Force single-GPU mode — this function runs only on rank 0
    config = config.copy()
    config['world_size'] = 1
    config['rank'] = 0
    config['local_rank'] = 0
    ...
```

**Principle:** Any code that runs on only one rank must not create DDP wrappers. Override `world_size=1` in the config before creating Trainer/model instances in rank-0-only sections.

### Fix C: VRAM budget too conservative, causing session cycling

**Symptom:** Both GPUs show 5-6% utilization with brief spikes to 40-100%, then back to idle. Training is bottlenecked on CPU-GPU data transfers rather than compute.

**Root cause:** `vram_budget_gb: 42.0` in `configs/cluster.yaml`. The 5 training sessions total 42.9GB — just 0.9GB over budget — so the bin-packer splits them into 2 groups. Every epoch cycles ~20GB of session data between CPU and GPU twice (load group 1, train, unload, load group 2, train, unload). At PCIe Gen4 speeds, this adds seconds of idle time per epoch.

**Diagnosis:**
```bash
# On compute node — shows burst/idle pattern
nvidia-smi dmon -s u -d 1

# Check VRAM grouping in training log
grep "VRAM plan" ~/MCCT/logs/mcct_<jobid>.out
# Output: "VRAM plan: 2 group(s), 9.2GB, 8.0GB, 11.4GB, 8.3GB, 6.0GB = 42.9GB total"
```

**Fix** (`configs/cluster.yaml` and `configs/cluster_v5_tuned.yaml`):
```yaml
# OLD
vram_budget_gb: 42.0
# NEW
vram_budget_gb: 44.0
```

With 44GB budget, all 5 sessions (42.9GB) fit in 1 group — fully GPU-resident, no cycling. The remaining ~5GB accommodates the model (~50MB), optimizer states (~100MB), and activation memory during forward/backward.

**Expected log output after fix:**
```
VRAM plan: 1 group(s), ... = 42.9GB total
All sessions fit on GPU -> fully resident (no cycling)
```

**Principle:** Set `vram_budget_gb` to total GPU VRAM minus ~4GB for model/optimizer/activations. Being 1GB too conservative can cause a 2-3x slowdown due to cycling overhead.

### Fix D: Parallel session materialization

**Context:** Building `PreMaterializedHierarchicalDataset` for each session involves CPU-heavy work (NumPy array slicing, `searchsorted`, tensor stacking). With 5 sessions, this was sequential — each taking ~30-60s.

**Fix** (`scripts/run_holdout.py:train_combined`):
```python
from concurrent.futures import ThreadPoolExecutor

def _materialize_one(args):
    i, session = args
    base_ds = HierarchicalMCCTDataset(session, ...)
    pre_ds = PreMaterializedHierarchicalDataset(base_ds)
    ...
    return i, pre_ds, split, p_dir, p1r

n_parallel = min(len(train_sessions), 4)
with ThreadPoolExecutor(max_workers=n_parallel) as pool:
    results = list(pool.map(_materialize_one, enumerate(train_sessions)))
```

Threading works because NumPy and PyTorch release the GIL during heavy computation. `pool.map` preserves ordering for consistent indexing across DDP ranks.

### Monitoring GPU utilization on compute nodes

The login node and compute nodes are different machines. To monitor GPU usage on the node running your job:

```bash
# 1. Find which node your job is on
squeue -u $USER -o "%.8i %.9P %.20j %.2t %.10M %.6D %R"

# 2. SSH into that node
ssh ssrde-c-a02   # replace with actual node name

# 3. Monitor GPU utilization (streams every 1 second)
nvidia-smi dmon -s u -d 1
# Columns: sm = compute %, mem = memory bandwidth %
# Healthy DDP training: sustained 50-80%+ on both GPUs

# 4. Or use watch for the full nvidia-smi dashboard
watch -n 1 nvidia-smi

# 5. CPU utilization
htop
```

**Interpreting `nvidia-smi dmon` output:**
- **sm 5-6%, brief spikes to 40-100%**: Data-starved — GPUs waiting for CPU/transfers (check VRAM grouping)
- **sm 50-80% sustained on both GPUs**: Healthy DDP training
- **sm 90%+ on one GPU, 0% on other**: DDP not working — only one rank active
- **mem 40%+ sustained**: Memory bandwidth saturated — consider smaller batch size

### PyTorch version differences
- Local: PyTorch 2.10.0+cu128
- Cluster: PyTorch 2.6.0+cu124
- Known difference: `torch.cuda.get_device_properties(d).total_memory` (2.6) vs `.total_mem` (2.10)
- The training code is compatible with both versions.

## 12. Auto-Merge GPU-Resident Training (2026-03-09)

### Problem: Session cycling overhead

The original `train_prematerialized_cycling` groups sessions by VRAM and cycles groups on/off GPU each epoch. When all sessions fit in one group this is a no-op, but the per-group loop still adds unnecessary complexity. More importantly, the cycling path can't shuffle windows across session boundaries within an epoch, reducing sample diversity per gradient step.

### Solution: `merge_prematerialized_datasets()`

A new function in `training/dataset.py` concatenates multiple `PreMaterialized*Dataset` objects into a single unified dataset by `torch.cat`-ing each tensor store along dimension 0. The merged dataset has the same interface as individual datasets (`get_batch`, `to_device`, `set_noise_epoch`) and works directly with `Trainer.train_prematerialized()`.

```python
from training.dataset import merge_prematerialized_datasets

# After materializing per-session datasets
merged_ds, global_train_idx, global_val_idx = merge_prematerialized_datasets(
    session_datasets, session_train_indices, session_val_indices)
merged_ds.to_device('cuda')
trainer.train_prematerialized(merged_ds, global_train_idx, global_val_idx)
```

### Auto-detection logic

`train_combined()` in `scripts/run_holdout.py` automatically decides:

1. Estimates total dataset VRAM via `Trainer._estimate_dataset_vram_gb()`
2. Subtracts 4 GB headroom (model, optimizer, activations, CUDA overhead)
3. If data fits -> merge all sessions into one GPU-resident dataset
4. If data exceeds budget -> fall back to `train_prematerialized_cycling`

The `gpu_resident_all` config key can force merge mode (`true`) regardless of auto-detection.

### VRAM estimates for V7

Per-window data size: ~1.07 MB (source + target context + future targets + validity masks).

| Sessions | Estimated windows | Dataset VRAM | Fits in 48GB? | Mode |
|----------|------------------|-------------|--------------|------|
| 5 (current holdout) | ~7,000 | ~7.5 GB | Yes | Merged |
| 10 | ~14,000 | ~15 GB | Yes | Merged |
| 20 | ~28,000 | ~30 GB | Yes | Merged |
| 40+ | ~56,000+ | ~60+ GB | No | Cycling |

With the current dataset (5 sessions, ~7 GB), merging leaves ~36 GB headroom per GPU. Future data collection up to ~20 sessions will still auto-merge. Beyond ~40 sessions, the system degrades gracefully to session cycling with no config changes needed.

### Benefits of merged training

- **Cross-session shuffling**: Windows from all sessions are shuffled together each epoch, improving gradient diversity
- **Simpler code path**: Uses `train_prematerialized()` directly, avoiding the group cycling loop
- **Zero cycling overhead**: No CPU-GPU data transfers between groups within each epoch
- **DDP compatible**: The merged permutation is broadcast from rank 0, sharded across ranks identically to single-dataset training

## 13. Batch Size Scaling (2026-03-09)

### Analysis for RTX 6000 ADA (48 GB)

The V7 model (2.87M params, d_model=128, seq_len=120 tokens) has relatively small attention matrices compared to the available VRAM. Activation memory scales linearly with batch size:

| Batch Size (per GPU) | Effective (2 GPUs) | Activation VRAM | Total VRAM | Steps/epoch (~5900 train) |
|---------------------|-------------------|-----------------|-----------|--------------------------|
| 256 | 512 | ~0.8 GB | ~10 GB | ~12 |
| 512 | 1024 | ~1.6 GB | ~11 GB | ~6 |
| 1024 | 2048 | ~3.2 GB | ~12 GB | ~3 |
| 2048 | 4096 | ~6.3 GB | ~16 GB | ~2 |

### Recommended: bs=512 per GPU

Balance between throughput and convergence stability. At bs=1024, only ~3 gradient steps per epoch may not provide enough stochastic variance for good generalization.

### LR scaling rule

When increasing batch size, scale the learning rate using the **square root rule** (Hoffer et al. 2017), which is safer than linear scaling for small models with AdamW:

```
lr_new = lr_base * sqrt(bs_new / bs_base)
```

| Config | batch_size | lr | warmup_epochs |
|--------|-----------|------|--------------|
| Original cluster_v7 | 256 | 1.4e-4 | 5 |
| Updated cluster_v7 | 512 | 2.0e-4 | 8 |

Warmup is extended proportionally to prevent early training instability at higher learning rates.

### Config (`configs/cluster_v7.yaml`)

```yaml
batch_size: 512         # Per-GPU (48GB supports this with merged dataset)
lr: 2.0e-4              # sqrt(2) scaling from 1.4e-4 for 2x batch size
warmup_epochs: 8        # Longer warmup for larger batch size
vram_budget_gb: 44.0    # Per-GPU; auto-merge if data fits, else cycle
```

### Future: Gradient accumulation

If model size increases (e.g., d_model=256, ~13M params) and activation memory exceeds VRAM at the desired effective batch size, gradient accumulation can simulate larger batches without increasing peak memory:

```
effective_batch = physical_batch * accumulation_steps * n_gpus
```

This is not currently implemented but is straightforward to add to `Trainer._process_batch()` by separating the `backward()` call from `optimizer.step()` and only stepping every N mini-batches.
