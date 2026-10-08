"""Asynchronous checkpoints: a rolling resumable state and LoRA-only milestones.

The GPU->host copy happens on the training thread into reused pinned buffers
(about one second for the full state); serialization and fsync run on a
background thread while training continues. A new snapshot first waits for the
previous write, so buffers are never overwritten while being written.
"""
import json
import os
import threading
import time
from pathlib import Path
import torch
from safetensors.torch import save_file


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    _fsync_dir(path.parent)


def atomic_safetensors(tensors, path, metadata=None):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    save_file(tensors, str(tmp), metadata=metadata)
    with tmp.open('rb') as f:
        os.fsync(f.fileno())
    tmp.replace(path)
    _fsync_dir(path.parent)


def atomic_json(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


class AsyncCheckpointer:
    def __init__(self, output, enabled=True):
        self.output = Path(output)
        self.enabled = enabled
        self.buffers = {}
        self.thread = None
        self.error = None
        self.last_write_seconds = None

    def _host(self, key, tensor):
        if not torch.is_tensor(tensor):
            return tensor
        buf = self.buffers.get(key)
        if buf is None or buf.shape != tensor.shape or buf.dtype != tensor.dtype:
            buf = torch.empty(tensor.shape, dtype=tensor.dtype,
                              pin_memory=tensor.is_cuda and torch.cuda.is_available())
            self.buffers[key] = buf
        buf.copy_(tensor.detach(), non_blocking=True)
        return buf

    def _host_tree(self, prefix, obj):
        if torch.is_tensor(obj):
            return self._host(prefix, obj)
        if isinstance(obj, dict):
            return {k: self._host_tree(f'{prefix}/{k}', v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._host_tree(f'{prefix}/{i}', v) for i, v in enumerate(obj))
        return obj

    def wait(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if self.error is not None:
            error, self.error = self.error, None
            raise RuntimeError('Asynchronous checkpoint write failed') from error

    def _launch(self, fn):
        def run():
            started = time.monotonic()
            try:
                fn()
            except BaseException as error:  # re-raised on the training thread
                self.error = error
            self.last_write_seconds = time.monotonic() - started
        self.thread = threading.Thread(target=run, daemon=False)
        self.thread.start()

    def save_rolling(self, state, iteration, blocking=False):
        """Overwrite checkpoints/latest.pt with a full resumable state."""
        if not self.enabled:
            return
        self.wait()
        host = self._host_tree('rolling', state)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        directory = self.output / 'checkpoints'
        directory.mkdir(parents=True, exist_ok=True)

        def write():
            atomic_torch_save(host, directory / 'latest.pt')
            atomic_json({'next_iteration': iteration, 'time': time.time()}, directory / 'latest.json')
        self._launch(write)
        if blocking:
            self.wait()

    def save_lora(self, adapters, iteration, metadata, blocking=False):
        """Write milestones/iter_NNNNNN/{student,critic}_lora.safetensors (fp32 masters)."""
        if not self.enabled:
            return
        self.wait()
        host = {k: self._host(f'lora/{k}', v) for k, v in adapters.items()}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        directory = self.output / 'milestones' / f'iter_{iteration:06d}'
        directory.mkdir(parents=True, exist_ok=True)
        meta = {k: str(v) for k, v in metadata.items()}

        def write():
            for role in ('student', 'critic'):
                part = {k: v.contiguous() for k, v in host.items() if k.endswith('.' + role)}
                atomic_safetensors(part, directory / f'{role}_lora.safetensors', dict(meta, role=role))
            atomic_json(dict(metadata, iteration=iteration), directory / 'meta.json')
            (directory / 'COMPLETE').write_text(f'{iteration}\n')
        self._launch(write)
        if blocking:
            self.wait()
