#!/usr/bin/env python3
"""Fold one column's VBench quality results + Qwen semantic judgements into VBench's
Quality / Semantic / Total, and write summary.json + final_scores.txt.

Quality (7 dims): read from vbench's *_eval_results.json, normalised with the same
(min, max) table cal_scores.py uses (VBench leaderboard constants), weighted like the
leaderboard (dynamic_degree 0.5, others 1).
Semantic (9 dims): each dim aggregated by its own vbench/<dim>.py rule over the judge
records, then normalised on (0, 1) -- the judge's hit rates / P(Yes) already live there
(VBench's CLIP-similarity ranges do not apply to a VLM judge).
Total = (4 * Quality + Semantic) / 5, as VBench."""
import argparse, glob, json, os, statistics

QUALITY = {"subject_consistency": (0.1462, 1.0, 1), "background_consistency": (0.2615, 1.0, 1),
           "temporal_flickering": (0.6293, 1.0, 1), "motion_smoothness": (0.706, 0.9975, 1),
           "dynamic_degree": (0.0, 1.0, 0.5), "aesthetic_quality": (0.0, 1.0, 1), "imaging_quality": (0.0, 1.0, 1)}
SEMANTIC = ["object_class", "multiple_objects", "human_action", "color", "spatial_relationship", "scene",
            "appearance_style", "temporal_style", "overall_consistency"]


def quality_scores(eval_dir):
    raw = {}
    for f in sorted(glob.glob(os.path.join(eval_dir, "*_eval_results.json"))):
        for k, v in json.load(open(f)).items():
            if k in QUALITY and isinstance(v, list) and v:
                raw[k] = float(v[0])
    return raw


def semantic_scores(judge_dir):
    per = {d: [] for d in SEMANTIC}
    n_videos = 0
    for f in sorted(glob.glob(os.path.join(judge_dir, "*.jsonl"))):
        for line in open(f):
            r = json.loads(line); n_videos += 1
            for rec in r["records"]:
                per[rec["dim"]].append(rec)
    out, n = {}, {}
    for d, recs in per.items():
        n[d] = len(recs)
        if not recs:
            continue
        if d in ("object_class", "multiple_objects", "scene"):          # pooled success frames / frames
            out[d] = sum(r["frame_hits"] for r in recs) / sum(r["frame_n"] for r in recs)
        elif d == "color":                                                # videos where the object was seen
            v = [r["video_results"] for r in recs if r["video_results"] is not None]
            n[d] = len(v)
            if v:
                out[d] = statistics.mean(v)
        elif d == "appearance_style":                                     # per-frame mean over all frames
            out[d] = sum(r["frame_p_sum"] for r in recs) / sum(r["frame_n"] for r in recs)
        else:                                                             # mean over videos
            out[d] = statistics.mean(r["video_results"] for r in recs)
    return out, n, n_videos


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-dir", required=True, help="vbench *_eval_results.json dir")
    ap.add_argument("--judge-dir", required=True, help="judge shard jsonl dir")
    ap.add_argument("--col", required=True)
    ap.add_argument("--out", required=True, help="dir for summary.json + final_scores.txt")
    ap.add_argument("--meta", default="{}", help="json string recorded verbatim (judge model, prompt field, ...)")
    a = ap.parse_args()
    q = quality_scores(a.eval_dir)
    s, sn, nvid = semantic_scores(a.judge_dir)
    qn = {k: (v - QUALITY[k][0]) / (QUALITY[k][1] - QUALITY[k][0]) for k, v in q.items()}
    qw = {k: QUALITY[k][2] for k in q}
    Q = sum(qn[k] * qw[k] for k in q) / sum(qw.values()) if q else None
    S = sum(s.values()) / len(s) if s else None
    T = (4 * Q + S) / 5 if Q is not None and S is not None else None
    summary = {"column": a.col, "n_videos_judged": nvid, "quality_raw": q, "quality_norm": qn, "quality_dims": len(q),
               "semantic": s, "semantic_n": sn, "semantic_dims": len(s),
               "quality_score": Q, "semantic_score": S, "total_score": T, "meta": json.loads(a.meta),
               "protocol": "VBench 7 quality dims (official code, leaderboard normalisation) + 9 semantic dims by "
                           "Qwen3.8-27B under VBench's per-dim rules (hit rates / P(Yes), normalised on (0,1)); "
                           "Total=(4Q+S)/5"}
    os.makedirs(a.out, exist_ok=True)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    lines = [f"column: {a.col}  videos judged: {nvid}", "", f"{'dimension':24s} {'raw':>8s} {'norm':>8s} {'n':>5s}"]
    for k in QUALITY:
        if k in q:
            lines.append(f"{k:24s} {q[k]:8.4f} {qn[k]:8.4f}")
        else:
            lines.append(f"{k:24s} {'MISSING':>8s}")
    for k in SEMANTIC:
        lines.append(f"{k:24s} {s[k]:8.4f} {s[k]:8.4f} {sn.get(k,0):5d}" if k in s else f"{k:24s} {'n/a':>8s} {'':>8s} {sn.get(k,0):5d}")
    lines += ["", f"Quality Score : {Q if Q is None else round(Q,4)}  ({len(q)}/7 dims)",
              f"Semantic Score: {S if S is None else round(S,4)}  ({len(s)}/9 dims)",
              f"Total Score   : {T if T is None else round(T,4)}"]
    open(os.path.join(a.out, "final_scores.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    missing_q = [k for k in QUALITY if k not in q]
    if missing_q:
        raise SystemExit(f"QUALITY_DIMS_MISSING {missing_q}")


if __name__ == "__main__":
    main()
