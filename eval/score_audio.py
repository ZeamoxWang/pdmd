#!/usr/bin/env python3
"""PDMD audio evaluation: PQ, CE, CU, PaSST IS, ImageBind IB, DeSync."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

AV_REV = 'f351b9a6fc6abde746d5f8e1d4c47c883319cb41'
AES_REV = '2618e9d451b456e9328b39495b5e6234678aa550'


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videos', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--av-benchmark', type=Path, required=True)
    p.add_argument('--audiobox-checkpoint', type=Path)
    p.add_argument('--duration', type=float, default=124 / 24)
    p.add_argument('--is-splits', type=int, default=10)
    p.add_argument('--seed', type=int, default=2020)
    p.add_argument('--stage', choices=['all', 'prepare', 'aesthetics', 'isst', 'av', 'summary'], default='all')
    args = p.parse_args()
    if args.duration < 4.8:
        p.error("--duration must be at least 4.8 seconds for DeSync")
    args.out = args.out.resolve()
    args.videos = args.videos.resolve()
    args.av_benchmark = args.av_benchmark.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.av_benchmark))
    videos = sorted(args.videos.glob('*.mp4'), key=lambda x: x.stem)
    if not videos:
        raise ValueError('No MP4 videos found')
    if not 1 <= args.is_splits <= len(videos):
        p.error('--is-splits must be between 1 and the number of videos')
    wav_dir = args.out / 'wav'
    wav_dir.mkdir(exist_ok=True)
    wavs = [wav_dir / (v.stem + '.wav') for v in videos]
    if args.stage == 'all':
        for stage in ['prepare', 'aesthetics', 'isst', 'av', 'summary']:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--stage', stage], check=True)
        return
    if args.stage == 'prepare':
        for video, wav in zip(videos, wavs):
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(video),
                            '-map', '0:a:0', '-t', str(args.duration), '-ac', '1', '-c:a', 'pcm_f32le', str(wav)], check=True)
        save(args.out / 'protocol.json', {
            'videos': [str(v) for v in videos], 'duration_seconds': args.duration,
            'av_benchmark_revision': AV_REV, 'audiobox_revision': AES_REV,
            'is_splits': args.is_splits, 'is_shuffle_seed': args.seed,
            'audio': 'first audio stream; mono float PCM; native sample rate; crop from t=0',
            'audiobox': '16 kHz; official 10 s windows and duration-weighted aggregation',
            'passt': '32 kHz; mean removal; crop/pad to 10 s; softmax logits; shuffled splits',
            'imagebind': 'upstream 0.5 fps, 224 px, 3 spatial crops; three 2 s audio clips; unscaled cosine',
            'sync_effective_duration_seconds': (16 + ((int(args.duration * 25) - 16) // 8) * 8) / 25,
            'desync': 'upstream 25 fps, 224 px; 16 kHz; mean absolute argmax offset of first/last 14 segments',
            'versions': {n: importlib.metadata.version(n) for n in ['torch', 'torchaudio', 'torchvision', 'av_bench', 'audiobox_aesthetics']}})
        return
    if args.stage == 'summary':
        scores = {}
        for name in ['aesthetics', 'isst', 'av']:
            scores.update(json.loads((args.out / f'{name}.json').read_text())['scores'])
        save(args.out / 'summary.json', {'num_videos': len(videos), 'scores': scores})
        print(json.dumps(scores, indent=2))
        return
    import torch
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    with torch.inference_mode():
        if args.stage == 'aesthetics':
            from audiobox_aesthetics.infer import initialize_predictor
            model = initialize_predictor(str(args.audiobox_checkpoint) if args.audiobox_checkpoint else None)
            rows = []
            for i, wav in enumerate(wavs):
                prediction = model.forward([{'path': str(wav)}])[0]
                rows.append({'id': wav.stem, **{k: prediction[k] for k in ['PQ', 'CE', 'CU']}})
                print(f'AESTHETICS {i+1}/{len(wavs)}', flush=True)
            save(args.out / 'aesthetics.json', {'scores': {k: sum(r[k] for r in rows)/len(rows) for k in ['PQ', 'CE', 'CU']}, 'per_video': rows})
        elif args.stage == 'isst':
            from av_bench.data.audio_dataset import AudioDataset
            from av_bench.metrics.isc import compute_isc
            from hear21passt.base import get_basic_model
            model = get_basic_model(mode='all').cuda().eval()
            dataset = AudioDataset(wavs, audio_length=args.duration, sr=32000)
            logits = []
            for i, wav in enumerate(wavs):
                audio = dataset.read_from_file(wav).cuda()
                audio = torch.nn.functional.pad(audio[..., :320000], (0, max(0, 320000-audio.shape[-1])))
                logits.append(model(audio)[:, :527].cpu())
                print(f'PASST {i+1}/{len(wavs)}', flush=True)
            logits = torch.cat(logits)
            torch.save({'ids': [w.stem for w in wavs], 'logits': logits}, args.out / 'passt_logits.pt')
            result = compute_isc(logits, feat_layer_name=None, rng_seed=args.seed, samples_shuffle=True, splits=args.is_splits)
            save(args.out / 'isst.json', {'scores': {'IS': result['inception_score_mean']}, 'split_std': result['inception_score_std']})
        elif args.stage == 'av':
            import torchaudio
            from av_bench.data.audio_dataset import ImageBindAudioDataset, SynchformerAudioDataset
            from av_bench.data.video_dataset import VideoDataset
            from av_bench.extract import encode_audio_with_sync
            from av_bench.synchformer.synchformer import Synchformer, make_class_grid
            from extract_video import encode_video_with_sync, encode_video_with_imagebind
            from imagebind.models import imagebind_model
            from imagebind.models.imagebind_model import ModalityType
            sync = Synchformer().cuda().eval()
            sync.load_state_dict(torch.load(args.av_benchmark / 'weights/synchformer_state_dict.pth', map_location='cpu', weights_only=True))
            ib = imagebind_model.imagebind_huge(pretrained=False)
            ib.load_state_dict(torch.hub.load_state_dict_from_url(
                'https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth',
                map_location='cpu', weights_only=True))
            ib = ib.cuda().eval()
            mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, win_length=400, hop_length=160, n_fft=1024, n_mels=128).cuda()
            # Synchformer consumes 16-frame windows at stride 8. The encoder
            # discards trailing incomplete strides anyway. Request only that
            # used prefix to avoid torio's terminal-frame rounding at 24->25 fps.
            sync_frames = 16 + ((int(args.duration * 25) - 16) // 8) * 8
            sync_duration = sync_frames / 25
            vd = VideoDataset(videos, duration_sec=sync_duration)
            ad = ImageBindAudioDataset(wavs)
            sd = SynchformerAudioDataset(wavs, duration=sync_duration)
            grid = make_class_grid(-2, 2, 21)
            rows = []
            for i, video in enumerate(videos):
                data = vd.sample(i)
                vf = encode_video_with_imagebind(ib, data['ib_video'].unsqueeze(0).cuda())
                audio, _ = ad[i]
                af = ib({ModalityType.AUDIO: audio.cuda()})[ModalityType.AUDIO]
                cosine = torch.cosine_similarity(vf, af, dim=-1).item()
                sv = encode_video_with_sync(sync, data['sync_video'].unsqueeze(0).cuda())
                audio, _ = sd.sample(i)
                sa = encode_audio_with_sync(sync, audio.unsqueeze(0).cuda(), mel)
                offsets = [abs(grid[sync.compare_v_a(v, a).argmax(-1).item()].item()) for v, a in [(sv[:, :14], sa[:, :14]), (sv[:, -14:], sa[:, -14:])]]
                rows.append({'id': video.stem, 'IB': cosine, 'DeSync': sum(offsets)/2})
                print(f'AV {i+1}/{len(videos)} {rows[-1]}', flush=True)
            save(args.out / 'av.json', {'scores': {k: sum(r[k] for r in rows)/len(rows) for k in ['IB', 'DeSync']}, 'per_video': rows})


if __name__ == '__main__':
    main()
