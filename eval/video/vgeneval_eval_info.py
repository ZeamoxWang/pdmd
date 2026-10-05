#!/usr/bin/env python3
"""Prepare the VBench quality input manifest: <id>.mp4 -> <id>-0.mp4 hard
links + vbench_eval_info.json carrying only the QUALITY dims.

The nine semantic dims are not listed here on purpose: VBench's detectors need its
own per-prompt auxiliary info, which VideoGen-Eval prompts do not have. Those dims
are scored by vgeneval_semantic_judge.py against the same clips.

The quality runner passes this manifest directly to VBench dimension functions.
Entries use absolute video paths; no prompt-named files or VBench patches are needed."""
import argparse, json, os

QUALITY_DIMS = ["subject_consistency", "background_consistency", "temporal_flickering",
                "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality"]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--videos", required=True, help="directory containing <id>.mp4 videos")
    p.add_argument("--prompts", required=True, help="jsonl {id, prompt} (original VGenEval text)")
    p.add_argument("--out", required=True, help="eval dir (hard links + vbench_eval_info.json)")
    p.add_argument("--expect", type=int, default=387, help="0 = whatever is present (probe)")
    p.add_argument("--min-bytes", type=int, default=10240)
    a = p.parse_args()

    prompts = {}
    for line in open(a.prompts):
        r = json.loads(line)
        prompts[str(r["id"])] = r["prompt"]
    os.makedirs(a.out, exist_ok=True)
    info, missing, small = [], [], []
    for pid in sorted(prompts, key=int):
        src = os.path.join(a.videos, f"{pid}.mp4")
        if not os.path.exists(src):
            missing.append(pid); continue
        if os.path.getsize(src) < a.min_bytes:
            small.append(pid); continue
        dst = os.path.join(a.out, f"{pid}-0.mp4")
        if not os.path.exists(dst):
            os.link(src, dst)
        info.append({"prompt_en": prompts[pid], "dimension": list(QUALITY_DIMS),
                     "video_list": [os.path.abspath(dst)], "vgeneval_id": int(pid)})
    with open(os.path.join(a.out, "vbench_eval_info.json"), "w") as f:
        json.dump(info, f, indent=1, ensure_ascii=False)
    print(f"EVAL_INFO entries={len(info)} missing={len(missing)} small={len(small)} expect={a.expect}")
    if small:
        raise SystemExit(f"EVAL_INFO_SMALL {small[:8]}")
    if a.expect and len(info) != a.expect:
        raise SystemExit(f"EVAL_INFO_SHORT {len(info)}/{a.expect} missing={missing[:8]}")


if __name__ == "__main__":
    main()
