"""Keep metrics aligned with the checkpoint selected for recovery."""
import hashlib
import json
import os
from pathlib import Path


def recover_metrics(path, completed_steps):
    path = Path(path)
    original = path.read_bytes()
    lines = original.splitlines(keepends=True)
    if len(lines) < completed_steps:
        raise ValueError('Metrics are missing checkpointed steps')
    for index, line in enumerate(lines[:completed_steps], 1):
        if json.loads(line)['iteration'] != index:
            raise ValueError('Metrics before checkpoint are inconsistent')
    retained = b''.join(lines[:completed_steps])
    if retained and not retained.endswith(b'\n'):
        retained += b'\n'
    if retained == original:
        return
    # Preserve even partial trailing JSON as evidence of the interrupted run.
    digest = hashlib.sha256(original).hexdigest()
    backup = path.with_name(path.name + f'.before-resume-{completed_steps}-{digest}')
    if backup.exists():
        if backup.read_bytes() != original:
            raise ValueError('Existing recovery backup differs')
    else:
        with backup.open('xb') as f:
            f.write(original)
            f.flush()
            os.fsync(f.fileno())
    temporary = path.with_suffix(path.suffix + '.recovery.tmp')
    with temporary.open('wb') as f:
        f.write(retained)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
