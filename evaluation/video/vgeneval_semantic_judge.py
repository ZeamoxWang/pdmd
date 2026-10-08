#!/usr/bin/env python3
"""VBench's nine semantic dimensions on VideoGen-Eval prompts, judged by Qwen3.8-27B.

VBench scores a semantic dim only on the prompts of that category, and needs per-prompt
auxiliary info (object / colour / relation / scene / style) that exists for VBench's own
prompts only. The detectors (GRiT, Tag2Text, UMT, ViCLIP, CLIP) are the other half of
that contract. This file keeps VBench's RULES -- which frames, what counts as a hit, how
hits aggregate (see vbench/<dim>.py, quoted per dim below) -- and swaps both halves for
one VLM:

  --mode aux    Qwen reads each prompt once (text only) and fills the auxiliary fields,
                null where the prompt has no such content = prompt not in that VBench
                category. Written once per prompt set, shared by every column.
  --mode judge  per-frame yes/no for the frame-counted dims (P(Yes) over {Yes,No} from
                the first answer token; hit = P(Yes) >= 0.5, the binary detector
                replaced by a binary judge), one video-level question for the
                video-level dims (continuous P(Yes) where VBench used a CLIP-family
                similarity). Frames: 16 uniform per clip, as vbench/utils.load_video
                (num_frames=16) for the detector dims; also 16 for the video-level dims
                (UMT uses 16, ViCLIP 8).

Per-video records mirror VBench's video_results fields so vgeneval_vbench_aggregate.py
can apply each dim's own aggregation. One shard = one GPU; --shard i/N."""
import argparse, json, os, re, sys, time

import numpy as np
import torch

FRAME_DIMS = ["object_class", "multiple_objects", "color", "spatial_relationship", "scene",
              "appearance_style"]
VIDEO_DIMS = ["human_action", "temporal_style", "overall_consistency"]
ALL_DIMS = FRAME_DIMS + VIDEO_DIMS
N_FRAMES = 16
RELATIONS = ["on the left of", "on the right of", "on the top of", "on the bottom of"]

AUX_SYSTEM = (
    "You annotate text-to-video prompts for the VBench benchmark. For the prompt you are "
    "given, fill the JSON fields below EXACTLY as VBench's prompt categories define them. "
    "Use null for a field when the prompt does not explicitly contain that kind of content -- "
    "do not guess or infer. Answer with one JSON object and nothing else.\n"
    "Fields:\n"
    '  "object": the single most important concrete, countable, visible object, animal or person '
    'named in the prompt (a short noun, e.g. "dog", "car", "woman"; no articles); null if none.\n'
    '  "multiple_objects": [a, b] -- two DIFFERENT classes of concrete, countable objects/animals/'
    "people that the prompt says appear together (e.g. [\"cat\", \"dog\"]). Not places, rooms, "
    "landscapes, weather or materials; not the same class twice (\"man\" and \"man\" is null); "
    "null if fewer than two such classes.\n"
    '  "human_action": a short verb phrase for a physical action a human performs in the prompt '
    '(e.g. "playing the guitar", "riding a bicycle", "shaking hands"); null if no human performs a '
    "nameable action.\n"
    '  "color": {"object": noun, "color": colour} -- an object whose colour the prompt states '
    "explicitly with one of these colour words: white, red, pink, blue, silver, purple, orange, "
    'green, gray, yellow, black (e.g. "a yellow umbrella" -> {"object": "umbrella", "color": '
    '"yellow"}); null otherwise ("colorful", "bright", "dark" are not colours).\n'
    '  "spatial_relationship": {"object_a": noun, "object_b": noun, "relationship": one of '
    '"on the left of", "on the right of", "on the top of", "on the bottom of"} -- only when the '
    "prompt states such a left/right/above/below relation between two DIFFERENT object classes; "
    "null otherwise.\n"
    '  "scene": a short phrase for the explicitly stated place or environment (e.g. "beach", '
    '"kitchen", "snowy mountain"); null if the prompt does not say where.\n'
    '  "appearance_style": ONLY a non-photographic artistic rendering style that the prompt '
    'states, as VBench uses: painting styles ("oil painting", "watercolor", "Van Gogh style", '
    '"ukiyo-e"), animation/illustration ("anime", "pixel art", "3D cartoon", "claymation"), '
    'or a named visual aesthetic ("cyberpunk", "black and white", "vintage film", "surrealism"). '
    'Photographic/realistic footage of any genre is null: "cinematic", "documentary", "realistic", '
    '"naturalistic", "dramatic", "candid", "minimalist", "high quality", "4K" are NOT styles.\n'
    '  "temporal_style": ONLY an explicitly stated camera MOTION or temporal effect, as VBench '
    'uses: "zoom in", "zoom out", "pan left/right", "tilt up/down", "camera rotates/orbits", '
    '"tracking shot", "dolly in", "slow motion", "time-lapse", "hyperlapse", "fast forward". '
    'Shot size or framing ("close-up", "wide shot", "medium shot") and "static camera" are NOT '
    "temporal styles -> null.\n"
)

AUX_VOCAB_SYSTEM = (
    "You annotate text-to-video prompts for a VBench-style benchmark. Three fields, each chosen "
    "from a closed list (copy the list entry EXACTLY) or null. Answer with one JSON object only.\n"
    '  "object": the single most important concrete, visible object / animal / person in the prompt, '
    "chosen from OBJECT_LIST; null only if none of the list applies.\n"
    '  "scene": the explicitly stated place or environment, chosen from SCENE_LIST; null if the prompt '
    "does not say where it happens or no entry fits.\n"
    '  "human_action": the physical action a HUMAN performs in the prompt, chosen from ACTION_LIST; '
    "null if no human performs a listed action (animal actions do not count).\n"
    "OBJECT_LIST: {objects}\nSCENE_LIST: {scenes}\nACTION_LIST: {actions}\n"
)

# VBench temporal styles are single phrases ("zoom in"); the judge question for each vocabulary
# value is spelled out so a video-level yes/no is unambiguous.
TEMPORAL_Q = {
    "static camera": "Does the camera stay completely static throughout this video (no panning, tilting, zooming or travelling)?",
    "handheld shaky": "Is this video shot handheld, with visible camera shake?",
    "zoom in": "Does the camera zoom in during this video (the subject grows progressively larger in the frame)?",
    "zoom out": "Does the camera zoom out during this video (the view progressively widens)?",
    "push in (dolly in)": "Does the camera travel closer to the subject during this video (a push-in / dolly-in)?",
    "pull out (dolly out)": "Does the camera travel away from the subject during this video (a pull-out / dolly-out), revealing more of the surroundings?",
    "pan left": "Does the camera pan or move to the left during this video?",
    "pan right": "Does the camera pan or move to the right during this video?",
    "pan (direction unspecified)": "Does the camera pan horizontally across the scene during this video?",
    "tilt up": "Does the camera tilt or move upward during this video?",
    "tilt down": "Does the camera tilt or move downward during this video?",
    "tilt (direction unspecified)": "Does the camera tilt vertically during this video?",
    "tracking shot (camera follows the subject)": "Does the camera follow the moving subject, keeping it in frame as it travels?",
    "orbit / arc around the subject": "Does the camera orbit or arc around the subject during this video?",
    "crane up (camera rises)": "Does the camera rise upward (a crane shot) during this video?",
    "slow motion": "Is this video in slow motion?",
    "time-lapse": "Is this video a time-lapse (a slow real-world process visibly compressed so it progresses within the clip)?",
    "high-speed (fast motion)": "Does this video show very fast, high-speed motion?",
    "racking focus": "Does the focus shift from one subject to another (rack focus) during this video?",
    "dolly zoom": "Does this video show a dolly-zoom effect (the background perspective changes while the subject keeps its size)?",
}

# ---- prompts. Frame questions see ONE frame; video questions see all 16 in order.
def q_frame(dim, aux, sub=None):
    if dim == "object_class":
        o = aux["object"]
        if o.split()[0] in ("two", "three", "four", "five", "six", "several", "people") or " and " in o:
            return f'Are there {o} clearly visible in this image? Answer strictly Yes or No.'
        art = "" if o.split()[0] in ("a", "an", "the") else "a "
        return f'Is there {art}{o} clearly visible in this image? Answer strictly Yes or No.'
    if dim == "multiple_objects":
        a, b = aux["multiple_objects"]
        return f'Are both a {a} and a {b} visible in this image? Answer strictly Yes or No.'
    if dim == "color":
        o, c = aux["color"]["object"], aux["color"]["color"]
        if sub == "object":
            return f'Is there a {o} visible in this image? Answer strictly Yes or No.'
        return f'Is the {o} in this image {c} in colour? Answer strictly Yes or No.'
    if dim == "spatial_relationship":
        s = aux["spatial_relationship"]
        return (f'In this image, is the {s["object_a"]} {s["relationship"]} the {s["object_b"]}, '
                f'judged by their positions in the frame? Answer strictly Yes or No.')
    if dim == "scene":
        return f'Does this image show the following scene or place: {aux["scene"]}? Answer strictly Yes or No.'
    if dim == "appearance_style":
        return f'Is this image rendered in the following visual style: {aux["appearance_style"]}? Answer strictly Yes or No.'
    raise KeyError(dim)


def q_video(dim, aux, prompt, n, sub=None):
    head = f"These {n} images are frames sampled uniformly, in order, from one short video. "
    if dim == "human_action":
        return head + f'Is a person performing the action "{aux["human_action"]}" in this video? Answer strictly Yes or No.'
    if dim == "temporal_style":
        v = aux["temporal_style"] if sub is None else sub
        q = TEMPORAL_Q.get(v) or (f'Do the camera work and temporal dynamics of this video match the following description: "{v}"?')
        return head + q + " Answer strictly Yes or No."
    if dim == "overall_consistency":
        return head + f'Does this video match the following description? Description: "{prompt}" Answer strictly Yes or No.'
    raise KeyError(dim)


def applicable(dim, aux):
    """Whether the prompt is in this VBench category (its aux field is filled). Always a bool."""
    v = aux.get(dim if dim != "object_class" else "object")
    if dim == "overall_consistency":
        return True
    if dim == "multiple_objects":
        return bool(isinstance(v, list) and len(v) == 2 and all(isinstance(x, str) and x for x in v) and v[0] != v[1])
    if dim == "color":
        return bool(isinstance(v, dict) and v.get("object") and v.get("color"))
    if dim == "spatial_relationship":
        return bool(isinstance(v, dict) and v.get("object_a") and v.get("object_b") and v.get("relationship") in RELATIONS)
    if dim == "temporal_style" and isinstance(v, list):
        return bool(v)
    return bool(isinstance(v, str) and v.strip())


# ---- model
def load_model(path, device):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(path)
    proc.tokenizer.padding_side = "left"          # last-position logits across a padded batch
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map={"": device})
    model.eval()
    return model, proc


def chat_text(proc, content_items, question):
    messages = [{"role": "user", "content": content_items + [{"type": "text", "text": question}]}]
    return proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


class YesNo:
    def __init__(self, tok):
        def ids(words):
            out = set()
            for w in words:
                t = tok.encode(w, add_special_tokens=False)
                if len(t) == 1:
                    out.add(t[0])
            return sorted(out)
        self.yes = ids(["Yes", " Yes", "yes", " yes"])
        self.no = ids(["No", " No", "no", " no"])
        assert self.yes and self.no, "Yes/No are not single tokens for this tokenizer"

    def p_yes(self, logits):                       # logits: [B, V] at the answer position
        pr = logits.float().softmax(-1)
        py = pr[:, self.yes].max(-1).values
        pn = pr[:, self.no].max(-1).values
        return (py / (py + pn + 1e-9)).tolist()


@torch.no_grad()
def frame_pyes(model, proc, yn, frames, question, device, batch=16):
    """One question asked of every frame; returns P(Yes) per frame."""
    from PIL import Image
    text = chat_text(proc, [{"type": "image"}], question)
    out = []
    for i in range(0, len(frames), batch):
        ims = [Image.fromarray(f) for f in frames[i:i + batch]]
        inputs = proc(text=[text] * len(ims), images=ims, padding=True, return_tensors="pt").to(device)
        logits = model(**inputs).logits[:, -1]
        out += yn.p_yes(logits)
    return out


@torch.no_grad()
def video_pyes(model, proc, yn, frames, question, device):
    from PIL import Image
    text = chat_text(proc, [{"type": "image"} for _ in frames], question)
    inputs = proc(text=[text], images=[Image.fromarray(f) for f in frames], return_tensors="pt").to(device)
    return yn.p_yes(model(**inputs).logits[:, -1])[0]


def load_frames(path, n=N_FRAMES, max_side=None):
    """16 uniform frames, segment-middle sampling like vbench.utils.get_frame_indices('middle')."""
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(path, ctx=cpu(0))
        total = len(vr)
        idx = [int((k + 0.5) * total / n) for k in range(n)]
        idx = [min(max(i, 0), total - 1) for i in idx]
        arr = vr.get_batch(idx).asnumpy()
    except Exception:
        import cv2
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        want = {min(max(int((k + 0.5) * total / n), 0), total - 1) for k in range(n)}
        fr, i = [], 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i in want:
                fr.append(f[:, :, ::-1].copy())
            i += 1
        arr = np.stack(fr)
    if max_side and max(arr.shape[1:3]) > max_side:
        from PIL import Image
        s = max_side / max(arr.shape[1:3])
        arr = np.stack([np.asarray(Image.fromarray(f).resize((int(f.shape[1] * s), int(f.shape[0] * s)), Image.BICUBIC)) for f in arr])
    return arr


# ---- aux
def parse_json_obj(s):
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < 0:
        raise ValueError("no json object")
    return json.loads(s[i:j + 1])


VBENCH_COLOURS = ["white", "red", "pink", "blue", "silver", "purple", "orange", "green", "gray", "grey", "yellow", "black"]
NOT_STYLES = ["documentary", "cinematic", "realistic", "photorealistic", "naturalistic", "natural", "dramatic",
              "candid", "minimalist", "action", "culinary", "lifestyle", "commercial", "vlog", "news", "sport",
              "high quality", "4k", "hd", "professional", "editorial", "fashion", "travel", "food", "nature",
              "wildlife", "live", "broadcast", "handheld", "footage", "photography", "photo", "film", "modern",
              "elegant", "cozy", "warm", "vibrant", "moody", "clean", "simple", "casual"]
TEMPORAL_WORDS = ["zoom", "pan ", "pans", "panning", "tilt", "rotat", "orbit", "dolly", "tracking", "track ",
                  "follow", "slow motion", "slow-motion", "time-lapse", "timelapse", "time lapse", "hyperlapse",
                  "fast forward", "fast-forward", "speed", "crane", "push in", "pull out", "fly", "aerial",
                  "circl", "sweep", "revolv", "rack focus", "racking"]


def _sing(w):
    w = str(w).strip().lower()
    for a, b in (("men", "man"), ("women", "woman"), ("people", "person"), ("children", "child")):
        if w == a or w.endswith(" " + a):
            return w[: -len(a)] + b
    return w[:-1] if w.endswith("s") and not w.endswith("ss") else w


def normalise_aux(d):
    """Coerce the model's JSON into VBench's field shapes, and null out what VBench would
    not count: non-VBench colour words, photographic 'styles', shot sizes as temporal
    styles, same-class pairs. Deterministic, so the rule is in the code, not the prompt."""
    out = {k: None for k in ["object", "multiple_objects", "human_action", "color",
                             "spatial_relationship", "scene", "appearance_style", "temporal_style"]}
    if not isinstance(d, dict):
        return out
    for k in out:
        v = d.get(k)
        if v in ("", [], {}, "null", "None"):
            v = None
        out[k] = v
    # colour: keep only a VBench colour word; "dark blue" -> "blue", "grey" -> "gray"
    c = out["color"]
    if isinstance(c, dict) and c.get("object") and c.get("color"):
        cw = [w for w in VBENCH_COLOURS if w in str(c["color"]).lower()]
        out["color"] = {"object": str(c["object"]).strip(), "color": cw[0].replace("grey", "gray")} if cw else None
    else:
        out["color"] = None
    # appearance style: photographic genres are not styles
    st = out["appearance_style"]
    if isinstance(st, str):
        s_ = st.lower().strip()
        if not s_ or any(w in s_ for w in NOT_STYLES):
            out["appearance_style"] = None
    else:
        out["appearance_style"] = None
    # temporal style: needs a camera motion / temporal effect word
    ts = out["temporal_style"]
    if isinstance(ts, str):
        t_ = ts.lower().strip()
        if not t_ or "static" in t_ or not any(w in t_ for w in TEMPORAL_WORDS):
            out["temporal_style"] = None
    else:
        out["temporal_style"] = None
    # multiple objects / spatial: two different classes
    mo = out["multiple_objects"]
    if isinstance(mo, list) and len(mo) == 2 and all(isinstance(x, str) and x.strip() for x in mo):
        if _sing(mo[0]) == _sing(mo[1]):
            out["multiple_objects"] = None
        else:
            out["multiple_objects"] = [str(mo[0]).strip(), str(mo[1]).strip()]
    else:
        out["multiple_objects"] = None
    sp = out["spatial_relationship"]
    if isinstance(sp, dict):
        r = str(sp.get("relationship", "")).lower().strip()
        alias = {"left of": "on the left of", "to the left of": "on the left of", "right of": "on the right of",
                 "to the right of": "on the right of", "above": "on the top of", "on top of": "on the top of",
                 "top of": "on the top of", "below": "on the bottom of", "under": "on the bottom of",
                 "beneath": "on the bottom of", "bottom of": "on the bottom of"}
        r = alias.get(r, r)
        a, b = sp.get("object_a"), sp.get("object_b")
        if r in RELATIONS and a and b and _sing(a) != _sing(b):
            out["spatial_relationship"] = {"object_a": str(a).strip(), "object_b": str(b).strip(), "relationship": r}
        else:
            out["spatial_relationship"] = None
    else:
        out["spatial_relationship"] = None
    for k in ("object", "human_action", "scene"):
        out[k] = str(out[k]).strip() if isinstance(out[k], str) and out[k].strip() else None
    return out


@torch.no_grad()
def run_aux(model, proc, prompts, device, out_path, model_name, prompt_field, shard="0/1", vocab=None):
    """One JSON per prompt; each shard appends to <out>.shard{i}.jsonl as it goes (resumable),
    and writes the final document itself only when it is the single shard -- otherwise
    --mode aux-merge folds the shard files from the per-GPU outputs."""
    si, sn = map(int, shard.split("/"))
    prompts = [pp for k, pp in enumerate(prompts) if k % sn == si]
    part = out_path + f".shard{si}.jsonl"
    items, n_fail = {}, 0
    if os.path.exists(part):
        for line in open(part):
            r = json.loads(line); items[r["id"]] = r["aux"]; n_fail += int(r.get("fail", 0))
        print(f"AUX_RESUME {len(items)} from {part}", flush=True)
    fpart = open(part, "a")
    t0 = time.time()
    for k, (pid, prompt) in enumerate(prompts):
        if str(pid) in items:
            continue
        system = AUX_SYSTEM if vocab is None else AUX_VOCAB_SYSTEM.format(
            objects=", ".join(vocab["object_class_vocab"]), scenes=", ".join(vocab["scene_vocab"]),
            actions=", ".join(vocab["human_action_vocab"]))
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": f"Prompt: {prompt}\nJSON:"}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = proc(text=[text], return_tensors="pt").to(device)
        d = None
        for attempt in range(2):
            kw = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20) if attempt else dict(do_sample=False)
            gen = model.generate(**inputs, max_new_tokens=320, **kw)
            ans = proc.tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            try:
                d = parse_json_obj(ans); break
            except Exception:
                d = None
        fail = int(d is None); n_fail += fail
        items[str(pid)] = (d if isinstance(d, dict) else {}) if vocab else normalise_aux(d)
        fpart.write(json.dumps({"id": str(pid), "aux": items[str(pid)], "fail": fail}) + "\n"); fpart.flush()
        if k % 25 == 0:
            print(f"AUX {k}/{len(prompts)} fail={n_fail} {time.time()-t0:.0f}s", flush=True)
    fpart.close()
    print(f"AUX_SHARD_DONE shard={shard} n={len(items)} fail={n_fail}", flush=True)
    if sn == 1:
        return aux_merge(out_path, model_name, prompt_field, vocab)


def apply_vocab(pid, a, vocab):
    """v3: the five hand-assigned dims come from the vocabulary file verbatim; the three open
    dims must be entries of their closed lists (else null)."""
    out = {k: None for k in a}
    for k in ("object", "scene", "human_action"):
        lst = {"object": vocab["object_class_vocab"], "scene": vocab["scene_vocab"], "human_action": vocab["human_action_vocab"]}[k]
        v = a.get(k)
        if isinstance(v, str):
            m = [w for w in lst if w.lower() == v.strip().lower()]
            out[k] = m[0] if m else None
    styles = [st for st, ids in vocab["appearance_style"].items() if int(pid) in ids]
    out["appearance_style"] = styles[0] if styles else None
    temps = [tv for tv, ids in vocab["temporal_style"].items() if int(pid) in ids]
    out["temporal_style"] = temps or None
    c = vocab["color"].get(str(pid)); out["color"] = dict(c) if c else None
    sp = vocab["spatial_relationship"].get(str(pid)); out["spatial_relationship"] = dict(sp) if sp else None
    mo = vocab["multiple_objects"].get(str(pid)); out["multiple_objects"] = list(mo) if mo else None
    return out


def aux_merge(out_path, model_name, prompt_field, vocab=None):
    import glob
    items, n_fail = {}, 0
    for f in sorted(glob.glob(out_path + ".shard*.jsonl")):
        for line in open(f):
            r = json.loads(line); n_fail += int(r.get("fail", 0))
            items[r["id"]] = apply_vocab(r["id"], r["aux"], vocab) if vocab else normalise_aux(r["aux"])
    cover = {dim: int(sum(applicable(dim, a) for a in items.values())) for dim in ALL_DIMS}
    doc = {"model": model_name, "prompt_field": prompt_field, "n": len(items), "parse_failures": n_fail,
           "vocab": (vocab or {}).get("version"), "coverage": cover,
           "items": dict(sorted(items.items(), key=lambda kv: int(kv[0])))}
    json.dump(doc, open(out_path, "w"), indent=1, ensure_ascii=False)
    print("AUX_DONE", json.dumps({"n": len(items), "fail": n_fail, "coverage": cover}), flush=True)
    return doc


# ---- judge
def judge_video(model, proc, yn, frames, aux, prompt, device):
    recs = []
    for dim in FRAME_DIMS:
        if not applicable(dim, aux):
            continue
        if dim == "color":
            # vbench/color.py: per frame, object detected? and its caption carries the colour?
            # video = colour_frames / object_frames, only videos with object_frames > 0 count.
            po = frame_pyes(model, proc, yn, frames, q_frame(dim, aux, "object"), device)
            pc = frame_pyes(model, proc, yn, frames, q_frame(dim, aux, "color"), device)
            obj = [p >= 0.5 for p in po]
            col = [a and (p >= 0.5) for a, p in zip(obj, pc)]
            recs.append({"dim": dim, "object_frames": sum(obj), "color_frames": sum(col), "frame_n": len(frames),
                         "video_results": (sum(col) / sum(obj)) if sum(obj) else None})
        elif dim == "appearance_style":
            # vbench/appearance_style.py: CLIP(frame, style) per frame, averaged over ALL frames.
            p = frame_pyes(model, proc, yn, frames, q_frame(dim, aux), device)
            recs.append({"dim": dim, "frame_p_sum": float(sum(p)), "frame_n": len(p), "video_results": float(np.mean(p))})
        else:
            # object_class / multiple_objects / scene: success frames over total frames (pooled);
            # spatial_relationship: per-frame score, video = mean over frames, dim = mean over videos.
            p = frame_pyes(model, proc, yn, frames, q_frame(dim, aux), device)
            hits = sum(x >= 0.5 for x in p)
            recs.append({"dim": dim, "frame_hits": hits, "frame_n": len(p), "video_results": hits / len(p)})
    for dim in VIDEO_DIMS:
        if not applicable(dim, aux):
            continue
        if dim == "temporal_style" and isinstance(aux["temporal_style"], list):
            ps = [video_pyes(model, proc, yn, frames, q_video(dim, aux, prompt, len(frames), sub=v), device) for v in aux["temporal_style"]]
            recs.append({"dim": dim, "values": aux["temporal_style"], "p_yes_each": ps, "p_yes": float(np.mean(ps)),
                         "video_results": float(np.mean(ps))})
            continue
        p = video_pyes(model, proc, yn, frames, q_video(dim, aux, prompt, len(frames)), device)
        # human_action: UMT top-5 hit -> binary; temporal_style / overall_consistency: ViCLIP
        # similarity -> continuous. Same shape here: binary hit vs P(Yes).
        recs.append({"dim": dim, "p_yes": p, "video_results": float(p >= 0.5) if dim == "human_action" else p})
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["aux", "aux-merge", "aux-from-vocab", "judge"], required=True)
    ap.add_argument("--model", required=True, help="local Qwen3.8-27B dir")
    ap.add_argument("--prompts", required=True, help="jsonl {id, prompt}")
    ap.add_argument("--prompt-field", default="original", help="label recorded in outputs")
    ap.add_argument("--aux", required=True, help="aux json (written by --mode aux, read by --mode judge)")
    ap.add_argument("--videos", help="dir of <id>.mp4 (judge)")
    ap.add_argument("--out", help="output jsonl for this shard (judge)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=0, help="downscale frames to this long side (0 = native)")
    ap.add_argument("--dims", default=",".join(ALL_DIMS))
    ap.add_argument("--vocab", help="annotation JSON; use evaluation/data/vgeneval/vocab_v4.json with aux-from-vocab")
    a = ap.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    prompts = [(str(int(r["id"])), r["prompt"]) for r in map(json.loads, open(a.prompts))]
    prompts.sort(key=lambda x: int(x[0]))
    if a.limit:
        prompts = prompts[:a.limit]
    model_name = os.path.basename(a.model.rstrip("/"))
    vocab = json.load(open(a.vocab)) if a.vocab else None
    if a.mode == "aux-from-vocab":
        # v4: every field of every prompt is fixed in the annotation file -- no model in the loop.
        items = {k: vocab["items"][k] for k, _ in prompts if k in vocab["items"]}
        assert len(items) == len(prompts), f"AUX_FROM_VOCAB_SHORT {len(items)}/{len(prompts)}"
        cover = {dim: int(sum(applicable(dim, v) for v in items.values())) for dim in ALL_DIMS}
        doc = {"model": None, "prompt_field": a.prompt_field, "n": len(items), "parse_failures": 0,
               "vocab": vocab.get("version"), "coverage": cover, "items": items}
        json.dump(doc, open(a.aux, "w"), indent=1, ensure_ascii=False)
        print("AUX_DONE", json.dumps({"n": len(items), "fail": 0, "coverage": cover}), flush=True)
        return
    if a.mode == "aux-merge":
        doc = aux_merge(a.aux, model_name, a.prompt_field, vocab)
        assert doc["n"] == len(prompts), f"AUX_MERGE_SHORT {doc['n']}/{len(prompts)}"
        return
    t0 = time.time()
    model, proc = load_model(a.model, device)
    print(f"MODEL_LOADED {a.model} {time.time()-t0:.0f}s mem={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)

    if a.mode == "aux":
        run_aux(model, proc, prompts, device, a.aux, model_name, a.prompt_field, shard=a.shard, vocab=vocab)
        return

    auxdoc = json.load(open(a.aux))
    items = auxdoc["items"]
    i, n = map(int, a.shard.split("/"))
    todo = [(pid, pr) for k, (pid, pr) in enumerate(prompts) if k % n == i and os.path.exists(os.path.join(a.videos, f"{pid}.mp4"))]
    done = set()
    if a.out and os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    todo = [t for t in todo if t[0] not in done]
    yn = YesNo(proc.tokenizer)
    dims = set(a.dims.split(","))
    print(f"JUDGE shard={a.shard} todo={len(todo)} done={len(done)}", flush=True)
    fout = open(a.out, "a")
    for k, (pid, prompt) in enumerate(todo):
        aux = items.get(pid) or normalise_aux(None)
        frames = load_frames(os.path.join(a.videos, f"{pid}.mp4"), max_side=a.max_side or None)
        recs = [r for r in judge_video(model, proc, yn, frames, aux, prompt, device) if r["dim"] in dims]
        fout.write(json.dumps({"id": pid, "stem": pid, "prompt_field": a.prompt_field, "frames": int(frames.shape[0]),
                               "hw": [int(frames.shape[1]), int(frames.shape[2])], "records": recs}) + "\n")
        fout.flush()
        if k % 10 == 0:
            print(f"JUDGE {k}/{len(todo)} {time.time()-t0:.0f}s mem={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
    fout.close()
    print(f"JUDGE_DONE shard={a.shard} n={len(todo)} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
