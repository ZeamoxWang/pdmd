# Video evaluation

Score existing videos with the PDMD visual evaluation protocol: seven official
VBench quality dimensions and nine semantic dimensions judged by Qwen3.8-27B on
387 VideoGen-Eval T2V prompts (IDs 730–1116). This is a VBench-style evaluation on
VideoGen-Eval prompts with a custom semantic judge, not the official VideoGen-Eval
agent evaluator or the unmodified VBench semantic implementation.

## Data

| File | Purpose |
| --- | --- |
| `data/prompts_vgeneval_t2v.jsonl` | Original benchmark text used for semantic evaluation. |
| `data/prompts_vgeneval_h3pe_v2.jsonl` | Fixed H3 prompt rewrites used for video generation. |
| `data/vocab_v4.json` | Fixed per-prompt semantic annotations and category assignments. |

The original text is the ID 730–1116 subset of the official
[VGenEval_T2V.xlsx](https://github.com/AILab-CVC/VideoGen-Eval/blob/main/docs/specifc_model/prompts/VGenEval_T2V.xlsx).
Both JSONL files contain `id` and `prompt`. Generation uses the H3 rewrites;
evaluation uses the original text. The v4 annotations are fixed: scoring does not
ask the judge to regenerate them. The older annotation-extraction modes remain in
the semantic script for reference; the supplied runner uses `aux-from-vocab` only.
The upstream VideoGen-Eval license is included in `data/LICENSE-VideoGen-Eval.txt`.

## Setup

Use Linux with CUDA. The Qwen judge is loaded in bf16 on one GPU per shard, without
quantization or model sharding. Its weights alone require approximately 56 GB;
allow additional memory for image inputs and activations. Increasing `NUM_GPUS`
distributes videos across independent model replicas, not one model across GPUs.

Install [VBench](https://github.com/Vchitect/VBench#installation) and its dependencies
in an environment with a CUDA-enabled PyTorch build. The public runner calls
VBench's dimension functions directly, so no internal scripts or `video_list`
patches are needed. VBench downloads its metric weights on first use; its cache
location can be set with `VBENCH_CACHE_DIR`.

For the judge, install a CUDA-enabled PyTorch build and torchvision, then:

```bash
pip install -r eval/requirements-judge.txt
```

Use separate environments for VBench and the judge if their dependencies conflict.
The runner accepts `QUALITY_PYTHON` and `JUDGE_PYTHON` as executable paths.
Supply a local Hugging Face-format Qwen3.8-27B model directory containing its
weights, configuration, tokenizer and processor. The judge dependency version
matches the supplied scoring scripts; the public VBench installation is not the
archived environment used for the paper, so exact numerical parity has not been
verified.

## Run

Place the generated videos in one directory, named `730.mp4` through `1116.mp4`.
For the paper's four-step distilled evaluation, generation uses the supplied H3
rewrites, seed 42, 124 frames at 24 fps, 544p, video shift 12 and audio shift 3.
In ascending prompt-ID order, cycle through aspect ratios 21:9, 9:21, 16:9, 9:16,
1:1, 4:3 and 3:4. Video generation and audio scoring are separate from this runner.

From the repository root:

```bash
bash eval/score.sh /path/to/videos /path/to/Qwen3.8-27B outputs/eval/run1
```

The quality dimensions run sequentially on GPU 0, followed by the semantic judge.
The quality adapter limits PyTorch and OpenCV to two CPU threads to avoid
thread oversubscription in CPU-limited GPU containers. When calling
`vgeneval_quality.py` directly, override this with `--cpu-threads`.
For four judge replicas, with separate environments:

```bash
QUALITY_PYTHON=/path/to/vbench-env/bin/python \
JUDGE_PYTHON=/path/to/judge-env/bin/python \
bash eval/score.sh /path/to/videos /path/to/Qwen3.8-27B outputs/eval/run1 4
```

`QUALITY_GPU` selects the physical GPU for quality scoring. `JUDGE_GPUS`, such as
`2,3,4,5`, selects physical GPUs for the judge; otherwise the runner uses 0 through
`NUM_GPUS - 1`. These settings replace `CUDA_VISIBLE_DEVICES` for child processes.
Keep the output directory specific to one video set and configuration. Semantic
shards append records and resume completed IDs; use a new directory if the inputs,
judge or shard count change. The input preparation uses hard links, so videos and
the output directory must be on the same filesystem.

## Outputs and scoring

* `quality/*_eval_results.json`: official VBench quality results.
* `judge/shard*.jsonl`: per-video semantic records; adjacent logs show progress.
* `aux_v4.json`: the fixed annotations used for this run.
* `summary/summary.json` and `summary/final_scores.txt`: dimension and total scores.

Quality uses VBench leaderboard normalization, with weight 0.5 for dynamic degree
and 1 for each other quality dimension. Semantic scoring samples 16 frames per
video with the supplied script's sampling formula. Frame-level dimensions use
individual images; video-level dimensions use the ordered images together.
The judge derives a Yes/No score from the first-answer-token probabilities;
binary dimensions threshold it at 0.5, while continuous dimensions retain it.
Only prompts with the relevant annotation participate in each dimension.

Semantic is the equal-weight mean of available semantic dimension scores, and
`Total = (4 * Quality + Semantic) / 5`. Results are on the 0–1 scale; multiply by
100 for percentage-style reporting. The supplied scoring and aggregation rules
are preserved. This directory does not include an audio evaluator or prompt
rewriting model pipeline.

## Files

* `vgeneval_eval_info.py`: prepare the explicit video manifest and hard links.
* `vgeneval_quality.py`: run one official VBench quality dimension.
* `vgeneval_semantic_judge.py`: prepare annotations and compute semantic judgments.
* `vgeneval_vbench_aggregate.py`: combine quality and semantic scores.
* `score.sh`: local orchestration without remote storage or cluster dependencies.

Please cite [VideoGen-Eval](https://arxiv.org/abs/2503.23452),
[VBench](https://github.com/Vchitect/VBench), and the
[PDMD paper](https://arxiv.org/abs/2609.35768) when using this evaluation.

## Audio evaluation

`score_audio.py` reports the paper's six audio metrics: Audiobox Aesthetics
**PQ, CE, CU**, PaSST **IS**, ImageBind **IB**, and Synchformer **DeSync**.
It uses Audiobox's official predictor and the MMAudio
[av-benchmark](https://github.com/hkchengrex/av-benchmark) preprocessing and models.
A single GPU runs the stages sequentially. Input MP4 files must contain audio.
The requirements pin Transformers 4.37.2 for Synchformer's bundled AST implementation.

In a separate Python environment, install matching CUDA builds of
`torch==2.8.0`, `torchvision==0.23.0`, and `torchaudio==2.8.0`, plus system
`ffmpeg` version 4–6 and Git, then:

```bash
git clone https://github.com/hkchengrex/av-benchmark.git /path/to/av-benchmark
git -C /path/to/av-benchmark checkout f351b9a6fc6abde746d5f8e1d4c47c883319cb41
pip install -e /path/to/av-benchmark
pip install -r eval/requirements-audio.txt
mkdir -p /path/to/av-benchmark/weights
curl -L https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth \
  -o /path/to/av-benchmark/weights/synchformer_state_dict.pth
python eval/score_audio.py --videos /path/to/videos \
  --av-benchmark /path/to/av-benchmark --out outputs/eval/audio
```

Model weights are downloaded on first use; set `HF_HOME` and `TORCH_HOME` and
run from a writable working directory for model caches. Run setup and scoring
on the machine where the data and GPU reside.

The default evaluated duration is `124/24` seconds, matching the paper's H3
clips; override `--duration` for other clip lengths (at least 4.8 seconds).
FFmpeg extracts the first audio stream as mono float PCM at its native sample
rate, starting at time zero and cropped to this duration. The official model
preprocessors perform their own resampling. Audiobox uses 16 kHz audio,
10-second windows and its duration-weighted window average. PaSST uses 32 kHz
audio, removes its mean, and pads/truncates to 10 seconds as in av-benchmark.
IS uses softmax logits with 10 shuffled splits and seed 2020. IB uses the
upstream 0.5 fps video sampler and three 2-second audio clips; the cosine is
reported without multiplying by 100. DeSync averages the absolute predicted
offsets of the first and last 4.8-second windows. The AV decoder requests
only complete Synchformer 16-frame windows at stride 8 (5.12 seconds for
124/24-second clips), avoiding terminal-frame rounding during 24-to-25 fps
resampling; the encoder would discard those trailing incomplete strides anyway. PQ, CE, CU, IB and DeSync are
then averaged equally over videos. These are explicit official-default settings
for this public runner; the original paper environment has not been compared.

`summary.json` contains exactly the six scores. `aesthetics.json` and `av.json`
include per-video scores; `passt_logits.pt` stores logits used for IS.
`protocol.json` records input paths, duration, parameters and installed versions.
Stages run in separate processes to release model memory. To rerun an individual
stage, use `--stage prepare`, `aesthetics`, `isst`, `av`, or `summary` with the
same arguments and output directory. Reuse outputs only for the same inputs and
settings. This audio runner is separate from the visual `score.sh` entry point.
