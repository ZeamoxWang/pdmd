"""Shared dataset, PyAV audio decoder, and shard I/O from the supplied harness."""

import json
import os

import numpy as np

_DATASET = None


def load_dataset(path):
    global _DATASET
    d = json.load(open(path))
    d.setdefault("expect_per_column", len(d["stems"]))
    _DATASET = d
    return d


def dataset():
    assert _DATASET is not None, "load_dataset() first"
    return _DATASET


def work_units(root, columns=None, limit=0):
    """[(column, stem, path)] in a fixed order, so rank r takes units[r::world]."""
    ds = dataset()
    columns = columns or ds["columns"]
    stems = ds["stems"][:limit] if limit else ds["stems"]
    return [(c, s, os.path.join(root, c, f"{s}.mp4")) for c in columns for s in stems]


def decode_audio(path):
    """mp4 -> (float32 [C, T] in [-1, 1], sample_rate). Raises on a file with no
    audio stream: a silent substitute would score as a valid clip."""
    import av

    with av.open(path) as c:
        if not c.streams.audio:
            raise RuntimeError(f"no audio stream in {path}")
        a = c.streams.audio[0]
        sr = int(a.rate)
        frames = []
        for f in c.decode(a):
            x = f.to_ndarray()
            if x.ndim == 1:
                x = x[None]
            # packed (non-planar) formats interleave channels along the last axis
            if not f.format.is_planar and x.shape[0] == 1 and f.layout.nb_channels > 1:
                x = x.reshape(-1, f.layout.nb_channels).T
            frames.append(x)
    x = np.concatenate(frames, axis=1)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    else:
        x = x.astype(np.float32)
    return x, sr


class ShardWriter:
    """JSONL for one rank (a rerun overwrites that rank); flushes per record so a killed rank leaves
    a readable prefix, and drops a .done marker only at the very end."""

    def __init__(self, out_dir, rank):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"rank{rank:02d}.jsonl")
        self.done = os.path.join(out_dir, f"rank{rank:02d}.done")
        self.f = open(self.path, "w")
        self.n = 0

    def __call__(self, rec):
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()
        self.n += 1

    def close(self):
        self.f.close()
        open(self.done, "w").write(str(self.n))


def read_parts(parts_dir):
    recs = []
    for fn in sorted(os.listdir(parts_dir)):
        if fn.endswith(".jsonl"):
            for line in open(os.path.join(parts_dir, fn)):
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs
