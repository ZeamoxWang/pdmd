"""Precomputed Qwen3-VL prompt features with deterministic, prefetched sampling."""
import hashlib
import json
import queue
import threading
from pathlib import Path
import torch
from safetensors import safe_open


def stream_seed(*parts):
    """A stable 63-bit seed from named parts, independent of Python's hash salt."""
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(digest[:8], 'little') & ((1 << 63) - 1)


class PromptCache:
    """Rank manifests plus safetensors shards, as written by tools/data/cache_prompts.py.

    Rows are sorted by source row, so an index into this cache is stable across
    machines and worker counts.
    """
    def __init__(self, root):
        self.root = Path(root)
        rows = []
        for manifest in sorted(self.root.glob('rank-*.jsonl')):
            rows.extend(json.loads(line) for line in manifest.read_text().splitlines() if line)
        rows.sort(key=lambda r: r['source_row'])
        if not rows:
            raise ValueError(f'No cached prompts under {self.root}')
        if len({r['source_row'] for r in rows}) != len(rows):
            raise ValueError('Duplicate cache records')
        self.rows = rows
        self._handles = {}
        self._lock = threading.Lock()

    def __len__(self):
        return len(self.rows)

    def signature(self):
        return hashlib.sha256(json.dumps(self.rows, sort_keys=True).encode()).hexdigest()

    def load(self, index, pin=True):
        row = self.rows[index]
        with self._lock:
            handle = self._handles.get(row['shard'])
            if handle is None:
                if len(self._handles) > 64:
                    self._handles.clear()
                handle = self._handles[row['shard']] = safe_open(self.root / row['shard'], framework='pt', device='cpu')
        emb, tags = handle.get_tensor(row['key']), handle.get_tensor(row['key'] + '_tags')
        if pin and torch.cuda.is_available():
            emb, tags = emb.pin_memory(), tags.pin_memory()
        return emb, tags


class SampleStream:
    """Which prompt and bucket each global batch slot sees, without replacement.

    Student updates reuse the previous critic update's rollouts and consume no
    prompts. Critic update number c fills global slots c*B .. c*B+B-1 of a
    seeded permutation of the corpus, so the stream does not depend on the
    number of GPUs or on resuming.
    """
    def __init__(self, size, global_batch, critic_steps, buckets, seed):
        self.size, self.global_batch = size, global_batch
        self.period, self.buckets, self.seed = critic_steps + 1, buckets, seed
        self._perm_epoch, self._perm = None, None

    def critic_updates_before(self, iteration):
        return iteration - iteration // self.period

    def position(self, iteration, slot):
        return self.critic_updates_before(iteration) * self.global_batch + slot

    def index(self, iteration, slot):
        position = self.position(iteration, slot)
        epoch, offset = divmod(position, self.size)
        if epoch != self._perm_epoch:
            g = torch.Generator().manual_seed(stream_seed(self.seed, 'permutation', epoch))
            self._perm, self._perm_epoch = torch.randperm(self.size, generator=g), epoch
        return int(self._perm[offset])

    def bucket(self, iteration, slot):
        g = torch.Generator().manual_seed(stream_seed(self.seed, 'bucket', self.position(iteration, slot)))
        return self.buckets[int(torch.randint(len(self.buckets), (1,), generator=g))]


class Prefetcher:
    """Reads the prompt features of upcoming critic updates on a background thread.

    Keys are produced in exactly the order the training loop consumes them, into
    a bounded queue, so the loop only blocks when the disk is slower than a whole
    update. The time it does block is reported as data_seconds.
    """
    def __init__(self, cache, stream, slots, start, stop, is_critic, depth=2):
        self.cache, self.stream = cache, stream
        self.order = [(it, s) for it in range(start, stop) if is_critic(it) for s in slots]
        self.scheduled = set(self.order)
        self.queue = queue.Queue(maxsize=max(1, depth * len(slots)))
        self.closed = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        for key in self.order:
            if self.closed:
                return
            try:
                value = self.cache.load(self.stream.index(*key))
            except Exception as error:  # handed to the consumer, never swallowed
                value = error
            self.queue.put((key, value))

    def get(self, iteration, slot):
        key = (iteration, slot)
        if key not in self.scheduled:
            # Not part of the forward schedule, e.g. the critic batch of the
            # iteration before a resume point: load it inline.
            return self.cache.load(self.stream.index(iteration, slot))
        while True:
            got, value = self.queue.get()
            if isinstance(value, Exception):
                raise value
            if got == key:
                return value
            if got > key:
                raise RuntimeError(f'Prefetch order broken: wanted {key}, got {got}')

    def close(self):
        self.closed = True
        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass
