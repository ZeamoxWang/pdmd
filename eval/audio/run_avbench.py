"""Portable adaptation of the supplied av-benchmark orchestration.

Preserves PyAV decoding, native-channel PCM16 conversion, duration, official
feature extraction and same-clip evaluation. Reports IS, IB and DeSync only.
"""
import argparse
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--dataset', required=True)
ap.add_argument('--root', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--models', required=True, help='supplied model bundle containing avbench/{av-benchmark,home,cwd}')
ap.add_argument('--work', required=True)
ap.add_argument('--rank', type=int, default=0)
ap.add_argument('--world', type=int, default=1)
ap.add_argument('--columns', default='')
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--batch', type=int, default=32)
ap.add_argument('--workers', type=int, default=4)
a = ap.parse_args()
for k in ('dataset', 'root', 'out', 'models', 'work'):
    setattr(a, k, os.path.abspath(getattr(a, k)))
a.work = os.path.join(a.work, f'r{a.rank}')
AVB = Path(a.models) / 'avbench'
os.environ['HF_HOME'] = str(AVB / 'home/.cache/huggingface')
os.environ['TORCH_HOME'] = str(AVB / 'home/.cache/torch')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.chdir(AVB / 'cwd')
sys.path.insert(0, str(AVB / 'av-benchmark'))
import numpy as np
import torch
import torchaudio
import h3audio_core as core
import av_bench.data.video_dataset as vds
import av_bench.data.audio_dataset as ads
import av_bench.evaluate as ev
from av_bench.extract import extract

DEV = 'cuda'
torch.set_num_threads(2)
ev.device = DEV
vds._VIDEO_BACKEND = 'pyav'


def _sf_load(path, *args, **kwargs):
    import soundfile as sf
    data, sr = sf.read(str(path), dtype='float32', always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _sf_load
ads.torchaudio.load = _sf_load


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] r{a.rank} {msg}', flush=True)


def write_wavs(col, stems):
    """mp4 audio -> PCM16 wav at native rate, same stem names. Returns #written."""
    import wave

    wdir = os.path.join(a.work, col, "wav")
    os.makedirs(wdir, exist_ok=True)
    n = 0
    for s in stems:
        dst = os.path.join(wdir, f"{s}.wav")
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            n += 1
            continue
        x, sr = core.decode_audio(os.path.join(a.root, col, f"{s}.mp4"))
        pcm = (np.clip(x.T, -1, 1) * 32767).astype("<i2")
        with wave.open(dst + ".part", "wb") as w:
            w.setnchannels(x.shape[0])
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
        os.replace(dst + ".part", dst)
        n += 1
    return wdir, n


def main():
    ds = core.load_dataset(a.dataset)
    cols = [c for c in a.columns.split(',') if c] or ds['columns']
    stems = ds['stems'][:a.limit] if a.limit else ds['stems']
    dur = float(ds.get('duration', 5.175))
    if a.limit:
        a.work = os.path.join(a.work, f'lim{a.limit}')
    writer = core.ShardWriter(a.out, a.rank)
    for col in cols[a.rank::a.world]:
        t0 = time.time()
        wdir, n = write_wavs(col, stems)
        cache = Path(a.work) / col / 'cache'
        if not (cache / 'passt_logits.pth').exists():
            extract(audio_path=Path(wdir), output_path=cache, audio_length=dur,
                    device=DEV, batch_size=a.batch, num_workers=a.workers,
                    skip_video_related=False, skip_clap=True)
        if not (cache / 'synchformer_video.pth').exists():
            # Restrict extraction to the same selected stems as the audio.
            # Hardlinks preserve the source files and require the same filesystem.
            selected = Path(a.work) / col / 'video'
            selected.mkdir(parents=True, exist_ok=True)
            for stem in stems:
                target = selected / f'{stem}.mp4'
                if not target.exists():
                    os.link(Path(a.root) / col / f'{stem}.mp4', target)
            import extract_video as ex
            ex.device = DEV
            ex.extract(argparse.Namespace(video_path=selected, gt_audio=None,
                       gt_cache=cache, audio_length=dur, num_workers=a.workers,
                       gt_batch_size=max(1, a.batch // 4)))
        scores = ev.evaluate(cache, cache, is_paired=True, num_samples=1,
                             skip_video_related=False, skip_clap=True)
        rec = {'col': col, 'n': n, 'IS': round(float(scores['ISC-PASST-mean']), 6),
               'IB': round(float(scores['IB-Score']), 6),
               'DeSync': round(float(scores['DeSync']), 6)}
        writer(rec)
        log(f'{col}: {rec}; seconds={time.time()-t0:.1f}')
    writer.close()


if __name__ == '__main__':
    main()
