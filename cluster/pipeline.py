"""SSRDE cluster pipeline automation for CADENCE.

Handles code/data sync, Slurm job submission, and result retrieval
from Windows. All operations batched over single SSH connections to
minimize Duo 2FA prompts (one auth per command).

Usage:
    python cluster/pipeline.py sync                # Upload code (1 Duo auth)
    python cluster/pipeline.py sync --data         # Include session_cache data
    python cluster/pipeline.py submit              # Submit Slurm job (1 Duo auth)
    python cluster/pipeline.py submit --job session --session y_06
    python cluster/pipeline.py submit --node ssrde-c-a01  # Force specific node
    python cluster/pipeline.py run                 # sync + submit (1 Duo auth total)
    python cluster/pipeline.py run --job session --session y_06
    python cluster/pipeline.py probe               # Show GPU node status (1 Duo auth)
    python cluster/pipeline.py status              # Check queue + tail log (1 Duo auth)
    python cluster/pipeline.py results             # Download results/ (1 Duo auth)
    python cluster/pipeline.py setup               # One-time env verification
"""

import io
import os
import sys
import subprocess
import argparse
import tarfile
import tempfile
import time
import glob as globmod


# Cluster connection
REMOTE_USER = "cozger"
REMOTE_HOST = "ssrde.ucsd.edu"
REMOTE = f"{REMOTE_USER}@{REMOTE_HOST}"
REMOTE_DIR = "/home/cozger/CADENCE"
SSH_HOST = "ssrde"  # Assumes ~/.ssh/config has Host ssrde entry

# Local paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories to sync (code only)
CODE_DIRS = ['cadence', 'scripts', 'configs', 'cluster']

# Data directories (large — opt-in)
DATA_DIRS = ['session_cache']  # 3.7GB, shared with MCCT
RAW_DATA_DIRS = ['raw sessions']  # 14GB XDF files — upload once

# Exclusion patterns
EXCLUDE_PARTS = {'.git', '__pycache__', '.pytest_cache', '.claude', 'results'}
EXCLUDE_EXTS = {'.pyc', '.egg-info'}

# Text file extensions that need \r\n -> \n conversion for Linux
_TEXT_EXTS = {
    '.py', '.yaml', '.yml', '.md', '.sh', '.txt', '.json', '.cfg',
    '.toml', '.ini', '.sbatch', '.bash', '.zsh',
}

# Sbatch file lookup
SBATCH_FILES = {
    'discovery': 'cluster/ssrde_discovery.sbatch',
    'session': 'cluster/ssrde_session.sbatch',
    'corpus': 'cluster/ssrde_corpus.sbatch',
    'all_sessions': 'cluster/ssrde_all_sessions.sbatch',
}
DEFAULT_JOB = 'discovery'

# Dual-GPU nodes eligible for CADENCE jobs
GPU_NODES = ['ssrde-c-a01', 'ssrde-c-a02']


def ssh(cmd, capture=False, stdin_file=None):
    """Run a command on the cluster via SSH.

    Args:
        cmd: remote command string
        capture: if True, return (stdout, stderr, returncode)
        stdin_file: open file object to pipe as stdin (for tar-over-ssh)
    """
    full_cmd = ['ssh', SSH_HOST, cmd]
    kwargs = {}
    if stdin_file is not None:
        kwargs['stdin'] = stdin_file
    if capture:
        kwargs['capture_output'] = True
        kwargs['text'] = True
        result = subprocess.run(full_cmd, **kwargs)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    else:
        return subprocess.run(full_cmd, **kwargs).returncode


def _should_exclude(path):
    """Check if a path component matches exclusion patterns."""
    parts = path.replace('\\', '/').split('/')
    for part in parts:
        if part in EXCLUDE_PARTS:
            return True
        for ext in EXCLUDE_EXTS:
            if part.endswith(ext):
                return True
    return False


def _is_text_file(path):
    """Check if a file is a text file that needs line-ending conversion."""
    _, ext = os.path.splitext(path)
    return ext.lower() in _TEXT_EXTS


def _add_with_unix_endings(tar, filepath, arcname):
    """Add a text file to tar archive with \\r\\n converted to \\n."""
    with open(filepath, 'rb') as f:
        data = f.read()
    data = data.replace(b'\r\n', b'\n')
    info = tar.gettarinfo(filepath, arcname=arcname)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _add_path(tar, local_path, arcname):
    """Add a file or directory, converting text file line endings to Unix."""
    if os.path.isfile(local_path):
        if _is_text_file(local_path):
            _add_with_unix_endings(tar, local_path, arcname)
        else:
            tar.add(local_path, arcname=arcname)
    elif os.path.isdir(local_path):
        for root, dirs, files in os.walk(local_path):
            # Filter excluded directories in-place
            dirs[:] = [d for d in dirs if not _should_exclude(d)]
            for fname in files:
                if _should_exclude(fname):
                    continue
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, os.path.dirname(local_path)).replace('\\', '/')
                if _is_text_file(fname):
                    _add_with_unix_endings(tar, full, rel)
                else:
                    tar.add(full, arcname=rel)


def create_sync_archive(include_data=False, include_raw=False, extra_files=None):
    """Create a tar archive of code (and optionally data) to sync.

    Uses Python's tarfile module — no local tar binary needed.
    Text files have \\r\\n converted to \\n for Linux compatibility.
    Uses fast gzip (level 1) for large payloads, normal gzip otherwise.
    Returns path to temporary archive file.
    """
    # Skip compression entirely for raw XDF uploads (binary, won't compress)
    if include_raw:
        archive_path = os.path.join(tempfile.gettempdir(), 'cadence_sync.tar')
        tar_mode = 'w:'
    else:
        archive_path = os.path.join(tempfile.gettempdir(), 'cadence_sync.tar.gz')
        tar_mode = 'w:gz'

    with tarfile.open(archive_path, tar_mode) as tar:
        # Code directories
        for d in CODE_DIRS:
            local = os.path.join(PROJECT_ROOT, d)
            if os.path.exists(local):
                _add_path(tar, local, d)

        # Top-level files
        for ext in ['*.py', '*.yaml', '*.yml', '*.md']:
            for f in globmod.glob(os.path.join(PROJECT_ROOT, ext)):
                fname = os.path.basename(f)
                if not fname.startswith('.'):
                    _add_path(tar, f, fname)

        # Data directories (large — opt-in only)
        if include_data:
            for d in DATA_DIRS:
                local = os.path.join(PROJECT_ROOT, d)
                if os.path.exists(local):
                    _add_path(tar, local, d)

        # Raw session XDF files (very large — one-time upload)
        if include_raw:
            for d in RAW_DATA_DIRS:
                local = os.path.join(PROJECT_ROOT, d)
                if os.path.exists(local):
                    _add_path(tar, local, d)

        # Extra files (explicitly included, bypass exclusions)
        if extra_files:
            for ef in extra_files:
                full_path = os.path.join(PROJECT_ROOT, ef) if not os.path.isabs(ef) else ef
                if os.path.exists(full_path):
                    arcname = os.path.relpath(full_path, PROJECT_ROOT).replace('\\', '/')
                    _add_path(tar, full_path, arcname)
                else:
                    print(f"  Warning: extra file not found: {ef}")

    return archive_path


# ── Commands ─────────────────────────────────────────────────────────


def cmd_setup(args):
    """One-time: verify/setup CADENCE environment on cluster (2 Duo auths)."""
    print("Setting up CADENCE environment on cluster...")

    # Upload setup script via tar-over-ssh
    setup_script = os.path.join(PROJECT_ROOT, 'cluster', 'setup_env.sh')
    archive_path = os.path.join(tempfile.gettempdir(), 'cadence_setup.tar.gz')
    with tarfile.open(archive_path, 'w:gz') as tar:
        _add_with_unix_endings(tar, setup_script, 'cluster/setup_env.sh')

    print("  Uploading setup script...")
    cmd = f'mkdir -p {REMOTE_DIR}/cluster && cd {REMOTE_DIR} && tar xzf -'
    with open(archive_path, 'rb') as f:
        rc = ssh(cmd, stdin_file=f)
    os.remove(archive_path)

    if rc != 0:
        print("Upload failed.")
        return rc

    # Run setup script (separate connection)
    print("  Running setup_env.sh on cluster...")
    rc = ssh(f'cd {REMOTE_DIR} && bash cluster/setup_env.sh')
    if rc == 0:
        print("Environment setup complete.")
    else:
        print(f"Setup failed with exit code {rc}")
    return rc


def cmd_sync(args):
    """Upload code to cluster via single SSH connection (1 Duo auth).

    Uses tar-over-ssh: creates archive locally with Python tarfile,
    pipes through single SSH connection to extract on remote.
    """
    include_data = getattr(args, 'data', False)
    include_raw = getattr(args, 'raw', False)
    extra_files = getattr(args, 'include', None)

    parts = ['code']
    if include_data:
        parts.append('data')
    if include_raw:
        parts.append('raw XDF')
    print(f"Creating {' + '.join(parts)} archive...")
    archive_path = create_sync_archive(
        include_data=include_data,
        include_raw=include_raw,
        extra_files=extra_files,
    )
    archive_size = os.path.getsize(archive_path) / (1024 * 1024)
    print(f"  Archive: {archive_size:.1f} MB")

    print("Syncing to cluster (single SSH connection)...")
    cmd = f'mkdir -p {REMOTE_DIR} && cd {REMOTE_DIR} && tar xzf -'
    with open(archive_path, 'rb') as f:
        rc = ssh(cmd, stdin_file=f)

    os.remove(archive_path)

    if rc == 0:
        print("Sync complete.")
    else:
        print("Sync failed.")
    return rc


def _resolve_sbatch(args):
    """Resolve sbatch file from --job or --sbatch arguments."""
    if hasattr(args, 'sbatch') and args.sbatch:
        return args.sbatch
    job = getattr(args, 'job', DEFAULT_JOB)
    return SBATCH_FILES.get(job, SBATCH_FILES[DEFAULT_JOB])


def _build_sbatch_args(args):
    """Build extra arguments to pass to the sbatch script."""
    parts = []
    session = getattr(args, 'session', None)
    if session:
        parts.append(session)
    extra = getattr(args, 'extra_args', None)
    if extra:
        parts.append(extra)
    return ' '.join(parts)


def _node_probe_cmd():
    """Bash snippet that probes GPU nodes, sets $NODEFLAG to --nodelist=<best>."""
    nodes_str = ' '.join(GPU_NODES)
    return (
        f'echo "=== Node Auto-Selection ==="; '
        f'BEST_NODE=""; BEST_JOBS=9999; '
        f'for N in {nodes_str}; do '
        f'  ST=$(sinfo -h -n $N -o "%T" 2>/dev/null | head -1); '
        f'  case "$ST" in idle*|mix*|alloc*) '
        f'    J=$(squeue -h -w $N 2>/dev/null | wc -l); '
        f'    echo "  $N: $ST, $J job(s)"; '
        f'    if [ "$J" -lt "$BEST_JOBS" ]; then BEST_NODE=$N; BEST_JOBS=$J; fi;; '
        f'  *) echo "  $N: $ST (skipped)";; '
        f'  esac; '
        f'done; '
        f'if [ -n "$BEST_NODE" ]; then '
        f'  echo "=> Selected: $BEST_NODE"; '
        f'  NODEFLAG="--nodelist=$BEST_NODE"; '
        f'else '
        f'  echo "=> WARNING: No GPU nodes available, using default scheduler"; '
        f'  NODEFLAG=""; '
        f'fi'
    )


def _node_select_cmd(node):
    """Return the bash snippet for node selection (manual override or auto-probe)."""
    if node:
        return f'NODEFLAG="--nodelist={node}"; echo "Node (manual): {node}"'
    return _node_probe_cmd()


def cmd_submit(args):
    """Submit Slurm job with auto node selection (1 Duo auth)."""
    sbatch = _resolve_sbatch(args)
    extra = _build_sbatch_args(args)
    node = getattr(args, 'node', None)
    node_cmd = _node_select_cmd(node)

    cmd = (f'mkdir -p {REMOTE_DIR}/logs && cd {REMOTE_DIR} && '
           f'{node_cmd} && sbatch $NODEFLAG {sbatch} {extra}')
    stdout, stderr, rc = ssh(cmd, capture=True)

    if rc == 0 and stdout:
        for line in stdout.strip().split('\n'):
            print(line)
        # Extract job ID from "Submitted batch job 12345"
        for line in stdout.strip().split('\n'):
            if 'Submitted batch job' in line:
                parts = line.split()
                if len(parts) >= 4:
                    job_id = parts[-1]
                    print(f"Job ID: {job_id}")
                    with open(os.path.join(PROJECT_ROOT, '.last_job_id'), 'w') as f:
                        f.write(job_id)
                break
    else:
        print(f"Submit failed: {stderr}")
    return rc


def cmd_run(args):
    """Sync + submit in a single SSH connection (1 Duo auth total).

    Pipes tar archive through SSH, extracts on remote, probes GPU nodes
    for availability, then submits to the best node — all through the
    same authenticated connection.
    """
    include_data = getattr(args, 'data', False)
    include_raw = getattr(args, 'raw', False)
    extra_files = getattr(args, 'include', None)
    sbatch = _resolve_sbatch(args)
    extra = _build_sbatch_args(args)
    node = getattr(args, 'node', None)
    node_cmd = _node_select_cmd(node)

    parts = ['code']
    if include_data:
        parts.append('data')
    if include_raw:
        parts.append('raw XDF')
    print(f"Creating {' + '.join(parts)} archive...")
    archive_path = create_sync_archive(
        include_data=include_data,
        include_raw=include_raw,
        extra_files=extra_files,
    )
    archive_size = os.path.getsize(archive_path) / (1024 * 1024)
    print(f"  Archive: {archive_size:.1f} MB")

    print("Syncing + submitting (single SSH connection)...")
    cmd = (f'mkdir -p {REMOTE_DIR}/logs && cd {REMOTE_DIR} && tar xzf - && '
           f'{node_cmd} && sbatch $NODEFLAG {sbatch} {extra}')

    with open(archive_path, 'rb') as f:
        stdout, stderr, rc = ssh(cmd, capture=True, stdin_file=f)

    os.remove(archive_path)

    if rc == 0 and stdout:
        for line in stdout.strip().split('\n'):
            line = line.strip()
            print(line)
            if 'Submitted batch job' in line:
                parts = line.split()
                if len(parts) >= 4:
                    job_id = parts[-1]
                    print(f"Job ID: {job_id}")
                    with open(os.path.join(PROJECT_ROOT, '.last_job_id'), 'w') as f:
                        f.write(job_id)
        print("Sync + submit complete.")
    else:
        print(f"Failed: {stderr}")
    return rc


def cmd_status(args):
    """Check job queue and tail latest log (1 Duo auth)."""
    cmd = (f'echo "=== Job Queue ===" && squeue -u {REMOTE_USER} && '
           f'echo "" && echo "=== Latest Log (last 30 lines) ===" && '
           f'ls -t {REMOTE_DIR}/logs/cadence_*.out 2>/dev/null | head -1 | xargs -r tail -30')
    return ssh(cmd)


def cmd_results(args):
    """Download results from cluster via tar-over-ssh (1 Duo auth).

    Streams a tar archive from the remote through SSH to extract locally.
    """
    output_dir = getattr(args, 'output', None) or 'results/cluster_discovery'
    local_dir = getattr(args, 'local', None)

    if local_dir:
        local_results = os.path.join(PROJECT_ROOT, local_dir)
    else:
        local_results = os.path.join(PROJECT_ROOT, output_dir)

    os.makedirs(local_results, exist_ok=True)

    print(f"Downloading {REMOTE_DIR}/{output_dir}/ -> {local_results}/")

    cmd = f'cd {REMOTE_DIR} && tar czf - {output_dir}'
    result = subprocess.run(
        ['ssh', SSH_HOST, cmd],
        capture_output=True,
    )

    if result.returncode == 0 and result.stdout:
        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:gz') as tar:
            for member in tar.getmembers():
                prefix = output_dir.rstrip('/') + '/'
                if member.name.startswith(prefix):
                    member.name = member.name[len(prefix):]
                elif member.name == output_dir.rstrip('/'):
                    continue
                tar.extract(member, local_results)
        print(f"Results downloaded to {local_results}/")
    else:
        print(f"Download failed: {result.stderr.decode()}")
        return 1
    return 0


def cmd_all(args):
    """Sync + submit + poll + retrieve.

    Sync+submit uses 1 Duo auth. Each poll check and final download
    each use 1 additional auth.
    """
    rc = cmd_run(args)
    if rc != 0:
        return rc

    # Poll until job completes
    job_id_file = os.path.join(PROJECT_ROOT, '.last_job_id')
    if not os.path.exists(job_id_file):
        print("No job ID found, cannot poll.")
        return 1

    with open(job_id_file) as f:
        job_id = f.read().strip()

    print(f"\nPolling job {job_id}...")
    while True:
        stdout, _, _ = ssh(f'squeue -j {job_id} -h', capture=True)
        if not stdout.strip():
            print("Job completed.")
            break
        parts = stdout.split()
        state = parts[4] if len(parts) > 4 else '?'
        print(f"  Status: {state}", end='\r')
        time.sleep(60)

    # Show final log
    print("\n=== Final Log (last 50 lines) ===")
    ssh(f'tail -50 {REMOTE_DIR}/logs/cadence_*_{job_id}.out')

    # Download results
    return cmd_results(args)


def cmd_probe(args):
    """Show GPU node status without submitting (1 Duo auth)."""
    nodes = ' '.join(GPU_NODES)
    cmd = (
        f'echo "=== GPU Node Status ===" && '
        f'for N in {nodes}; do '
        f'  ST=$(sinfo -h -n $N -o "%T %c %m %G" 2>/dev/null); '
        f'  J=$(squeue -h -w $N 2>/dev/null | wc -l); '
        f'  echo "  $N: $ST | $J running job(s)"; '
        f'done && echo "" && '
        f'echo "=== Your Jobs ===" && squeue -u {REMOTE_USER}'
    )
    return ssh(cmd)


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description='CADENCE cluster pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Pipeline command')

    # setup
    subparsers.add_parser('setup', help='Verify/setup env on cluster')

    # sync
    p_sync = subparsers.add_parser('sync', help='Upload code to cluster (1 Duo auth)')
    p_sync.add_argument('--data', action='store_true',
                        help='Also sync session_cache data (3.7GB)')
    p_sync.add_argument('--raw', action='store_true',
                        help='Also sync raw XDF session files (14GB)')
    p_sync.add_argument('--include', nargs='+', default=None,
                        help='Extra files to include')

    # submit
    p_submit = subparsers.add_parser('submit', help='Submit Slurm job (1 Duo auth)')
    p_submit.add_argument('--job', choices=['discovery', 'session', 'corpus', 'all_sessions'],
                          default=DEFAULT_JOB, help='Job type')
    p_submit.add_argument('--session', default=None,
                          help='Session name (for --job session)')
    p_submit.add_argument('--sbatch', default=None,
                          help='Custom sbatch file override')
    p_submit.add_argument('--node', default=None,
                          help='Force specific node (skip auto-selection)')
    p_submit.add_argument('extra_args', nargs='?', default='',
                          help='Extra args to pass to sbatch')

    # probe
    subparsers.add_parser('probe', help='Show GPU node availability (1 Duo auth)')

    # status
    subparsers.add_parser('status', help='Check job status + tail log (1 Duo auth)')

    # results / pull
    p_results = subparsers.add_parser('results', help='Download results (1 Duo auth)')
    p_results.add_argument('--output', default='results/cluster_discovery',
                           help='Remote output directory')
    p_results.add_argument('--local', default=None,
                           help='Local destination (default: same as --output)')

    p_pull = subparsers.add_parser('pull', help='Download results (alias)')
    p_pull.add_argument('--output', default='results/cluster_discovery',
                        help='Remote output directory')
    p_pull.add_argument('--local', default=None,
                        help='Local destination')

    # run (sync + submit)
    p_run = subparsers.add_parser('run', help='Sync + submit (1 Duo auth total)')
    p_run.add_argument('--job', choices=['discovery', 'session', 'corpus', 'all_sessions'],
                       default=DEFAULT_JOB, help='Job type')
    p_run.add_argument('--session', default=None,
                       help='Session name (for --job session)')
    p_run.add_argument('--data', action='store_true',
                       help='Also sync session_cache data')
    p_run.add_argument('--raw', action='store_true',
                       help='Also sync raw XDF session files (14GB)')
    p_run.add_argument('--include', nargs='+', default=None,
                       help='Extra files to include')
    p_run.add_argument('--sbatch', default=None,
                       help='Custom sbatch file override')
    p_run.add_argument('--node', default=None,
                       help='Force specific node (skip auto-selection)')
    p_run.add_argument('extra_args', nargs='?', default='',
                       help='Extra args to pass to sbatch')

    # all (sync + submit + poll + retrieve)
    p_all = subparsers.add_parser('all', help='Sync + submit + poll + retrieve')
    p_all.add_argument('--job', choices=['discovery', 'session', 'corpus', 'all_sessions'],
                       default=DEFAULT_JOB, help='Job type')
    p_all.add_argument('--session', default=None)
    p_all.add_argument('--output', default='results/cluster_discovery')
    p_all.add_argument('--data', action='store_true')
    p_all.add_argument('--raw', action='store_true')
    p_all.add_argument('--include', nargs='+', default=None)
    p_all.add_argument('--sbatch', default=None)
    p_all.add_argument('--node', default=None,
                       help='Force specific node (skip auto-selection)')
    p_all.add_argument('extra_args', nargs='?', default='')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        'setup': cmd_setup,
        'sync': cmd_sync,
        'submit': cmd_submit,
        'probe': cmd_probe,
        'status': cmd_status,
        'results': cmd_results,
        'pull': cmd_results,
        'run': cmd_run,
        'all': cmd_all,
    }

    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main() or 0)
