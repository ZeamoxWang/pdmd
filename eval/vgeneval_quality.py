#!/usr/bin/env python3
"""Run one official VBench quality dimension on an explicit video manifest."""
import argparse
import importlib
import json
from pathlib import Path

from vgeneval_eval_info import QUALITY_DIMS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--info", required=True, help="vbench_eval_info.json")
    parser.add_argument("--dimension", required=True, choices=QUALITY_DIMS)
    parser.add_argument("--out", required=True, help="quality results directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local", action="store_true", help="use VBench's local model assets")
    args = parser.parse_args()

    import torch
    from vbench.utils import init_submodules

    assets = init_submodules([args.dimension], local=args.local, read_frame=False)
    module = importlib.import_module(f"vbench.{args.dimension}")
    compute = getattr(module, f"compute_{args.dimension}")
    result = compute(str(Path(args.info).resolve()), torch.device(args.device), assets[args.dimension])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.dimension}_eval_results.json").write_text(
        json.dumps({args.dimension: result}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
