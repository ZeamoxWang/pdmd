# Audio evaluation

Compute PQ, CE, CU, PaSST IS, ImageBind IB and Synchformer DeSync for generated videos.

## One video directory

For a training reproduction or a single directory of MP4 files, use
[`score_directory.py`](score_directory.py). Install
[`requirements.txt`](requirements.txt) on top of the training environment and set up
av-benchmark as described in the root [Evaluation guide](../../README.md#evaluation):

```bash
python evaluation/audio/score_directory.py --videos renders/pdmd_4nfe_2500/videos \
  --out scores/pdmd_4nfe_2500/audio --av-benchmark av-benchmark --expect 387
```

This writes `per_clip.jsonl` and `summary.json` under `--out`. The workflow below
is the paper's offline harness for comparing multiple video directories.

## Offline harness setup

Use Linux with CUDA-enabled PyTorch 2.8 and matching torchaudio. Install
[Audiobox Aesthetics](https://github.com/facebookresearch/audiobox-aesthetics)
(revision `2618e9d451b456e9328b39495b5e6234678aa550`) and
[av-benchmark](https://github.com/hkchengrex/av-benchmark)
(revision `f351b9a6fc6abde746d5f8e1d4c47c883319cb41`) with their dependencies, then:

```bash
pip install -r evaluation/audio/requirements-offline.txt
pip install transformers==4.37.2
```

Prepare the model weights and caches following the upstream setup instructions.
Arrange the model directory as follows; the AV runner uses offline caches.

```text
model-bundle/
  audiobox/checkpoint.pt
  avbench/
    av-benchmark/          # source checkout, including weights/
    home/.cache/
      huggingface/
      torch/
    cwd/                  # working directory for upstream model assets
```

## Run

Place videos at `video-root/PDMD/730.mp4` through `1116.mp4`.
Keep videos and output on the same filesystem because the runner uses hard links.
From the repository root:

```bash
AUDIO_PYTHON=/path/to/env/bin/python \
  bash evaluation/audio/score.sh /path/to/video-root /path/to/model-bundle /path/to/output
```

To score another video directory, edit `columns` in a copy of `evaluation/data/vgeneval/audio_dataset.json`
and pass that file as the fourth argument. The launcher uses one GPU.

The default duration is 5.175 seconds. Audiobox uses the full PyAV-decoded audio
at its native sample rate and channel count, with batch size 16. AV evaluation
uses native-rate, native-channel PCM16 audio, PyAV video decoding, audio batch
size 32, video batch size 8 and four workers. PQ, CE and CU are rounded per clip
to four decimal places before averaging. IS, IB and DeSync use the upstream
column-level evaluation; no reference videos are required.

## Outputs

- `parts/aesthetics/rank00.jsonl`: per-video PQ, CE and CU.
- `parts/avbench/rank00.jsonl`: IS, IB and DeSync for each video directory.
- `summary/summary.json`: aggregated scores.

Use a fresh output directory when changing inputs, models or scoring settings;
the AV stage reuses cached features. Run individual scripts with `--help` for
stage-specific and sharding options.
