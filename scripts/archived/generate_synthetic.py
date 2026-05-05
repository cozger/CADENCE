"""Generate a synthetic corpus for CADENCE training/validation.

Usage:
    python scripts/generate_synthetic.py
    python scripts/generate_synthetic.py --n-coupled 50 --n-null 20
"""

import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadence.synthetic import build_synthetic_session_permod, plan_corpus


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic corpus')
    parser.add_argument('--output', default='synthetic_corpus')
    parser.add_argument('--duration', type=int, default=2700)
    parser.add_argument('--n-coupled', type=int, default=50)
    parser.add_argument('--n-null', type=int, default=150)
    parser.add_argument('--duty-cycle', type=float, default=None)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    specs = plan_corpus(args.n_coupled, args.n_null)
    total = len(specs)
    print(f"Generating {total} sessions ({args.n_coupled} coupled, "
          f"{args.n_null} null), {args.duration}s each")

    manifest = []
    t0 = time.time()

    for i, (name, kappa_dict, seed) in enumerate(specs):
        coupled = [m for m, k in kappa_dict.items() if k > 0]
        coupled_str = ', '.join(f'{m.split("_")[0]}={kappa_dict[m]:.2f}'
                                for m in coupled) if coupled else 'null'
        print(f"  [{i+1}/{total}] {name} [{coupled_str}]...", end=' ', flush=True)

        t1 = time.time()
        session = build_synthetic_session_permod(
            args.duration, kappa_dict, seed=seed,
            duty_cycle_override=args.duty_cycle)

        path = os.path.join(args.output, f'{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(session, f, protocol=pickle.HIGHEST_PROTOCOL)

        dt = time.time() - t1
        print(f"{dt:.1f}s")

        manifest.append({
            'name': name, 'file': f'{name}.pkl',
            'kappa_dict': kappa_dict, 'duration': args.duration,
        })

    manifest_path = os.path.join(args.output, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    total_time = time.time() - t0
    print(f"\nDone: {total} sessions in {total_time:.0f}s")
    print(f"Manifest: {manifest_path}")


if __name__ == '__main__':
    main()
