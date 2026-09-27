# MiniMax-H3 Turbo LoRA：单张 24GB 显卡推理

在 **单张 24GB 显存的 GPU**（NRP Nautilus 上的 NVIDIA A10）上跑
[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) +
[Turbo LoRA](https://github.com/ModelTC/Minimax-H3-Turbo)（FL2VA Turbo 4-step v0.1）的配置和脚本。

> ⚠️ **这个版本是专门针对 24GB 显存优化的。**
>
> 如果在 **80GB 显存的卡**（A100 80G / H100 等）上，**不应该用这个脚本**。更合适的是
> Turbo 仓库的官方脚本 [`inference_minimax_h3.py`](https://github.com/ModelTC/Minimax-H3-Turbo/blob/main/inference_minimax_h3.py)
> 加上 `--fuse-lora`：它默认用 `ComponentsManager` 的自动 offload，按组件整体搬上/搬下 GPU，
> 保持 bf16 精度。本仓库为了挤进 24GB 做的 int8 量化和细粒度 offload，在 80GB 卡上只会
> 带来额外的加载时间、更慢的 VAE 解码和轻微的画质损失（见下文"80GB 卡上的效率"）。

## 为什么需要特殊处理

H3 的 transformer 在 bf16 下是 61.7GB，Qwen3-VL-32B 文本编码器是 62.1GB，直接跑的显存峰值在
50GB 以上。这里采用 Diffusers 文档里 24–32GB 显卡的方案，并做了几处补充：

| 组件 | 处理方式 |
|---|---|
| Transformer | 先把 Turbo LoRA 融合进 bf16 权重（int8 权重无法直接 fuse），加载时量化为 int8（torchao weight-only），按 block 用 CUDA stream 从 CPU 流式搬到 GPU |
| 文本编码器 | int8 量化，leaf 级 offload |
| 视频 VAE | leaf 级 offload，**不常驻 GPU**，否则去噪时 FFN 激活值会 OOM |
| 音频 VAE | 常驻 GPU（很小） |

权重主要常驻在 CPU 内存里，pod 需要约 160Gi 内存。

## 文件

| 文件 | 作用 |
|---|---|
| `deployment_a10_h3.yaml` | Nautilus deployment：1 张 A10、160Gi 内存、挂载 `zmw-vol-haosu` PVC 到 `/pv` |
| `fuse_lora.py` | 一次性把 Turbo LoRA 融合进 bf16 transformer 并存盘（CPU 上完成，约 20 分钟） |
| `run_a10.py` | 常驻推理 worker：模型只加载一次，之后轮询队列目录执行任务 |
| `jobs/giant_cat_harbor.json` | 示例任务（14 秒、三个镜头的 T2VA prompt） |

## 使用方法

所有路径都在 PVC 上的 `/pv/h3` 下。

### 1. 启动 pod

```bash
kubectl apply -f deployment_a10_h3.yaml
```

启动命令会安装依赖。**注意 torch 需要 ≥ 2.11**，最新版 torchao 在镜像自带的 torch 2.8 上无法 import；
yaml 里已经先升级 torch。

### 2. 准备代码和权重

```bash
# 在 pod 里
export HF_HOME=/pv/h3/hf_cache
git clone --depth 1 https://github.com/ModelTC/Minimax-H3-Turbo.git /pv/h3/Minimax-H3-Turbo
hf download lightx2v/Minimax-h3-Turbo minimax_h3_fl2v_turbo_4step_v0.1.safetensors --local-dir /pv/h3/loras
hf download MiniMaxAI/MiniMax-H3 --exclude "transformer_ref/*"
```

把本仓库的 `fuse_lora.py`、`run_a10.py` 拷到 `/pv/h3/`（例如用 `kubectl cp`）。

> 基础模型仓库里还带有原始格式的 `FL2VA/`、`Ref2VA/` 目录（约 124GB），Diffusers 用不到，
> 可以在下载时一并 `--exclude`。

### 3. 融合 LoRA（只需一次）

```bash
cd /pv/h3
python fuse_lora.py \
  --lora-path /pv/h3/loras/minimax_h3_fl2v_turbo_4step_v0.1.safetensors \
  --lora-alpha 8 \
  --output /pv/h3/transformer_turbo4step_v0.1_fused
```

v0.1 4-step LoRA 的 rank 是 128、alpha 是 8（与官方脚本默认值一致）。

### 4. 启动推理 worker

```bash
cd /pv/h3
export HF_HOME=/pv/h3/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python run_a10.py \
  --transformer-path /pv/h3/transformer_turbo4step_v0.1_fused \
  --inference-steps 4 > run.log 2>&1 &
```

加载完成后日志会出现 `watching /pv/h3/queue`。

### 5. 提交任务

把 jobs JSON（格式同 Turbo 仓库的 `examples/prompts_t2va_test.json`）放进队列目录：

```bash
cp jobs/giant_cat_harbor.json /pv/h3/queue/
```

- 成功：视频写到 `/pv/h3/outputs/`，JSON 移到 `queue/done/`
- 失败（例如 OOM）：JSON 移到 `queue/failed/`，worker 继续等待下一个任务，**不需要重新加载模型**

## A10 实测（960×544，345 帧 ≈ 14.4 秒，4 NFE）

| 阶段 | 耗时 |
|---|---|
| 加载 + int8 量化 | 约 23.5 分钟 |
| offload 准备（pinned memory） | 约 4–8 分钟 |
| 4 步去噪 | 7 分 28 秒（约 112 秒/步），显存约 18.4GB |
| VAE 解码 | 很慢（实测超过 17 分钟），瓶颈是 leaf 级 offload 的逐层同步搬运 |

加载只在 worker 启动时付一次；之后每条视频的耗时是去噪 + VAE 解码。

## 已知问题与补丁

- **torchao int8 + group offload stream**：Diffusers 的 group offload 开启 `use_stream` 后会调用
  `tensor.to(device, non_blocking=True)`，torchao 的 int8 tensor 只接受 `dtype/layout/device`，
  会触发 `AssertionError`。`run_a10.py` 开头对 torchao tensor 改用同步拷贝（对速度影响可忽略）。
- **`low_cpu_mem_usage=False`**：Diffusers 文档示例里带了这个参数，但当前 Diffusers 版本在量化加载时
  会直接报错，这里已经去掉。
- 文本编码时会打印大量 `visual.*` 层 "not executed" 的警告：纯文本 prompt 不经过 Qwen3-VL 的视觉编码器，可以忽略。

## 80GB 卡上的效率

本脚本在 80GB 卡上**可以正常运行**（没有依赖显存大小的逻辑），但效率明显低于应有水平：

| 阶段 | 本脚本 | 80GB 卡的合理做法 |
|---|---|---|
| 加载 | bf16 读取 → int8 量化 → pinned memory，约 28 分钟 | 直接 bf16 读取，约 5–10 分钟 |
| 文本编码 | leaf 级同步 offload | 整体上 GPU，几秒 |
| 去噪 | int8 weight-only 需要先反量化，长序列下通常慢 1.2–1.5 倍，画质有轻微损失 | bf16 |
| VAE 解码 | leaf 级同步 offload，瓶颈在 PCIe 和 Python hook，换更快的 GPU 也不会快多少 | VAE 常驻 GPU，1–2 分钟 |

所以在 80GB 卡上请使用官方的 `inference_minimax_h3.py --fuse-lora`。

## 许可

MiniMax-H3 使用 MiniMax H3 Community License，美国、欧盟、英国、韩国不在默认授权范围内，
需要先通过 [MiniMax 的授权申请](https://platform.minimax.io/h3-license)。
