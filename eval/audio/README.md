# PDMD audio evaluation

This directory adapts the audio harness supplied in `zimo-internship-stuff.zip`.
It reports PQ (Production Quality), CE (Content Enjoyment), CU (Content
Usefulness), PaSST IS, ImageBind IB and Synchformer DeSync.

## Preserved protocol

- `h3audio_core.py`: decode the entire first audio stream with PyAV, preserving
  the native sample rate and channels.
- `run_aesthetics.py`: pass the decoded tensor directly to `AesPredictor`, using
  an explicit Audiobox `checkpoint.pt`. The default batch is 16. Per-clip values
  are rounded to four decimals before aggregation, as in the supplied runner.
- `run_avbench.py`: convert the full decoded audio to native-rate, native-channel
  PCM16 (`clip(x, -1, 1) * 32767`), read it using SoundFile, and call the supplied
  av-benchmark extraction and evaluation entry points. Use PyAV for video.
  The dataset duration is **5.175 seconds**, copied from the supplied configuration.
  Defaults are batch 32, video batch 8, and four data-loader workers.
- IS is the upstream `ISC-PASST-mean`; IB and DeSync compare each column's audio
  against that same column's video. No teacher videos are needed for these outputs.
- `report.py` produces the per-column means for PQ/CE/CU and passes through
  the column-level IS/IB/DeSync, with the source harness's rounding.

We removed company storage/download commands and deployment-specific directories.
The published dataset uses a generic `PDMD` column and the original 387 IDs.
Signal diagnostics and outputs outside the paper's six metrics are omitted.
The original model calls, decoder choices and PCM conversion are retained.
The local launcher uses one GPU, limits CPU threads, and runs groups sequentially.
The source AV runner is column-sharded; the Audiobox runner is clip-sharded.

## Inputs and model bundle

Arrange videos as `/path/to/video-root/PDMD/730.mp4` through `1116.mp4`.
To evaluate another column, edit `columns` in a copy of `dataset.json` and pass
its path as the fourth launcher argument. Keep work/output on the same filesystem
as the videos because the AV stage makes hardlinks to its selected inputs.

The original harness requires this bundle:

```text
model-bundle/
  audiobox/checkpoint.pt
  avbench/
    av-benchmark/          # exact source snapshot, including weights/
    home/.cache/
      huggingface/        # model cache used by the supplied harness
      torch/
    cwd/                  # working directory, including .checkpoints/ if required
```

**The ZIP contains the scoring scripts, but not this model bundle, its upstream
source snapshot, or its wheel/version snapshot.** Supply those assets to retain
the original environment. Installing a current av-benchmark checkout is a
separate environment choice; this draft does not claim it matches that snapshot.
The AV stage retains the source harness's offline cache configuration, without
changing the user's HOME.

Use the matching CUDA PyTorch/torchaudio environment, install the supplied
av-benchmark and Audiobox implementations, and install the small adapter
requirements in `../requirements-audio.txt`. The existing public environment
needed Transformers 4.37.2 for Synchformer's AST API; that is a known compatibility
setting, not a recovered version from this archive.

```bash
AUDIO_PYTHON=/path/to/env/bin/python \
  bash eval/audio/score.sh /path/to/video-root /path/to/model-bundle /path/to/output
```

Outputs are `parts/aesthetics/rank00.jsonl`, `parts/avbench/rank00.jsonl`,
and `summary/summary.json`. A rerun overwrites the rank shard and may reuse the
AV work cache; use a fresh output directory after changing videos, models,
dataset duration or other settings. For separate stages and sharding, invoke
the Python runners with `--help`.

## Draft status

This replacement has been checked for syntax and report aggregation only.
No GPU job has been started for it. Previous scores were produced by the removed
runner and are not results of this supplied-harness adaptation.
