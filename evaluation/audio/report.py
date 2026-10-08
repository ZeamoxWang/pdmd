"""Summarize mentor-harness shards using its four-decimal per-clip means."""
import argparse
import json
from pathlib import Path
import numpy as np
import h3audio_core as core


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--parts', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    ds = core.load_dataset(a.dataset)
    aes = core.read_parts(a.parts / 'aesthetics')
    av = {r['col']: r for r in core.read_parts(a.parts / 'avbench')}
    columns = {}
    for col in ds['columns']:
        rows = {str(r['stem']): r for r in aes if r['col'] == col and 'error' not in r}
        scores = {k: round(float(np.mean([r[k] for r in rows.values()])), 4)
                  for k in ('PQ', 'CE', 'CU')} if rows else {}
        scores.update({k: av[col][k] for k in ('IS', 'IB', 'DeSync')} if col in av else {})
        columns[col] = {'aesthetics_n': len(rows), 'av_n': av.get(col, {}).get('n', 0), 'scores': scores}
    result = {'dataset': ds['name'], 'columns': columns,
              'errors': [r for r in aes if 'error' in r]}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
