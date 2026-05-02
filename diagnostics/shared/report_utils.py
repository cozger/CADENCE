from __future__ import annotations
import os
from datetime import datetime
from typing import List, Tuple


def make_output_dir(base_dir: str, module_name: str) -> str:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(base_dir, f'{ts}_{module_name}')
    os.makedirs(out, exist_ok=True)
    return out


def write_md_report(output_dir: str, filename: str, sections: List[Tuple[str, str]]) -> None:
    path = os.path.join(output_dir, filename)
    lines = []
    for title, body in sections:
        lines.append(f'## {title}\n')
        lines.append(body.strip())
        lines.append('\n')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def screening_report_path(outputs_dir: str) -> str:
    return os.path.join(outputs_dir, 'screening_report.md')
