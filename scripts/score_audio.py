"""The paper's six audio metrics on a directory of generated mp4 files.

Each is computed over the directory's clips, higher is better except DeSync:

  PQ      Production Quality    clarity, no distortion or artifacts, balanced levels  (Audiobox Aesthetics)
  CE      Content Enjoyment     how pleasant or engaging the audio is                  (Audiobox Aesthetics)
  CU      Content Usefulness    how usable the audio is for content creation           (Audiobox Aesthetics)
  IS      inception score       confidence and diversity of PaSST's AudioSet labels    (av-benchmark)
  IB      ImageBind score       agreement of the audio with its own video              (av-benchmark)
  DeSync  Synchformer offset    audio-video misalignment in seconds, lower is better   (av-benchmark)

PQ, CE and CU: Audiobox Aesthetics on the full audio track at its native sample rate and channel
count, batch 16, each clip rounded to four decimals before averaging.
IS, IB and DeSync: MMAudio's av-benchmark, run as the paper ran it. The audio track is written as
native-rate PCM16 wav, av-benchmark extracts the audio features of those files and the ImageBind
and Synchformer features of the videos themselves (PyAV decoding), and its evaluate() compares the
directory with itself, so no reference videos are needed. --duration is the audio length it reads.

  python scripts/score_audio.py --videos renders/pdmd_4nfe_2500/videos --out scores/pdmd_4nfe_2500/audio \\
      --av-benchmark /path/to/av-benchmark --expect 387

--av-benchmark is a source checkout of https://github.com/hkchengrex/av-benchmark with its weights/
directory (see requirements-audio.txt). The other weights download on first use.
"""
import argparse
import json
import math
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch

AXES = ('PQ', 'CE', 'CU')


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def decode_audio(path):
    """mp4 -> float32 [channels, samples] in [-1, 1] and the sample rate, via PyAV."""
    import av
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise RuntimeError(f'{path} has no audio stream')
        stream = container.streams.audio[0]
        rate = int(stream.rate)
        chunks = []
        for frame in container.decode(stream):
            x = frame.to_ndarray()
            if x.ndim == 1:
                x = x[None]
            channels = frame.layout.nb_channels
            if not frame.format.is_planar and x.shape[0] == 1 and channels > 1:
                x = x.reshape(-1, channels).T  # packed formats interleave the channels
            chunks.append(x)
    x = np.concatenate(chunks, axis=1)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    return x.astype(np.float32), rate


def write_wav(path, x, rate):
    """float [channels, samples] -> PCM16 wav at the native rate and channel count."""
    pcm = (np.clip(x.T, -1, 1) * 32767).astype('<i2')
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(x.shape[0])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def aesthetics(clips, ckpt, batch):
    """Per-clip PQ, CE and CU."""
    from audiobox_aesthetics.infer import AesPredictor
    predictor = AesPredictor(checkpoint_pth=ckpt, batch_size=batch)
    out = {}
    for i in range(0, len(clips), batch):
        chunk = clips[i:i + batch]
        items = []
        for stem, path in chunk:
            x, rate = decode_audio(path)
            items.append({'path': torch.from_numpy(x), 'sample_rate': rate})
        for (stem, _), s in zip(chunk, predictor.forward(items)):
            out[stem] = {k: round(float(s[k]), 4) for k in AXES}
        log(f'aesthetics {min(i + batch, len(clips))}/{len(clips)}')
    return out


def av_benchmark(clips, videos, work, repo, duration, batch, video_batch, workers):
    """IS, IB and DeSync from av-benchmark's own extraction and evaluation code."""
    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))  # av_bench and extract_video from the checkout, so weights/ resolves
    import soundfile as sf
    import torchaudio
    import av_bench.data.audio_dataset as audio_dataset
    import av_bench.data.video_dataset as video_dataset
    import av_bench.evaluate as evaluate
    import extract_video
    from av_bench.extract import extract

    def load_pcm(path, *args, **kwargs):
        # Read our PCM16 wavs the same way under every torchaudio backend.
        data, rate = sf.read(str(path), dtype='float32', always_2d=True)
        return torch.from_numpy(data.T.copy()), rate

    torchaudio.load = audio_dataset.torchaudio.load = load_pcm
    video_dataset._VIDEO_BACKEND = 'pyav'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluate.device = extract_video.device = device

    wav_dir, cache = Path(work) / 'wav', Path(work) / 'cache'
    wav_dir.mkdir(parents=True, exist_ok=True)
    for stem, path in clips:
        x, rate = decode_audio(path)
        write_wav(wav_dir / f'{stem}.wav', x, rate)
    log(f'{len(clips)} wav files in {wav_dir}')

    cwd = os.getcwd()
    os.chdir(repo)  # ImageBind keeps its checkpoint in ./.checkpoints
    try:
        extract(audio_path=wav_dir, output_path=cache, audio_length=duration, device=device,
                batch_size=batch, num_workers=workers, skip_video_related=False, skip_clap=True)
        extract_video.extract(argparse.Namespace(video_path=Path(videos).resolve(), gt_audio=None, gt_cache=cache,
                                                 audio_length=duration, num_workers=workers,
                                                 gt_batch_size=video_batch))
        m = evaluate.evaluate(cache, cache, is_paired=True, num_samples=1, skip_clap=True)
    finally:
        os.chdir(cwd)
    return {'IS': float(m['ISC-PASST-mean']), 'IS_std': float(m['ISC-PASST-std']),
            'IB': float(m['IB-Score']), 'DeSync': float(m['DeSync'])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--videos', required=True, help='directory of <id>.mp4 files with sound')
    ap.add_argument('--out', required=True, help='output directory for per_clip.jsonl, summary.json and work files')
    ap.add_argument('--av-benchmark', required=True, help='source checkout of hkchengrex/av-benchmark with weights/')
    ap.add_argument('--expect', type=int, default=0, help='refuse to score unless exactly this many clips')
    ap.add_argument('--duration', type=float, default=5.175, help='seconds of audio av-benchmark reads (paper: 5.175)')
    ap.add_argument('--audiobox-ckpt', default=None, help='local Audiobox checkpoint.pt; default downloads it')
    ap.add_argument('--batch', type=int, default=16, help='Audiobox batch (paper: 16)')
    ap.add_argument('--av-batch', type=int, default=32, help='av-benchmark audio batch (paper: 32)')
    ap.add_argument('--video-batch', type=int, default=8, help='av-benchmark video batch (paper: 8)')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()

    clips = sorted(((p.stem, p) for p in Path(a.videos).glob('*.mp4')), key=lambda c: c[0])
    if not clips or (a.expect and len(clips) != a.expect):
        raise SystemExit(f'{a.videos}: {len(clips)} mp4 files, expected {a.expect or "at least one"}')
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log(f'{len(clips)} clips from {a.videos}')

    per_clip = aesthetics(clips, a.audiobox_ckpt, a.batch)
    av = av_benchmark(clips, a.videos, out / 'work', a.av_benchmark, a.duration, a.av_batch, a.video_batch,
                      a.workers)

    with open(out / 'per_clip.jsonl', 'w') as f:
        for stem, _ in clips:
            f.write(json.dumps({'id': stem, **per_clip[stem]}) + '\n')
    summary = {'videos': str(a.videos), 'n': len(clips),
               **{k: round(sum(per_clip[s][k] for s, _ in clips) / len(clips), 4) for k in AXES},
               **{k: round(v, 4) for k, v in av.items()}}
    assert all(math.isfinite(v) for k, v in summary.items() if k not in ('videos', 'n'))
    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
