"""Thin read-only DiffSynth H3 adapter: no pipeline/VAE/encoder in training."""
from types import SimpleNamespace
import torch
from pdmd.adapters import set_role


def enable_cached_attention_bounds():
    """Avoid one GPU->CPU sync per attention call in DiffSynth's varlen SDPA loop.

    The upstream helper reads the packed-sequence boundaries with
    cu_seqlens.tolist() inside every attention call. That blocks the CPU until
    the GPU has drained, so it cannot queue the next block's kernels or the
    FSDP all-gather ahead of time. The boundaries of one tensor never change,
    so they are read once and remembered on that tensor object; every later
    block that receives the same tensor reuses them. A new tensor (e.g. the
    next sample) is read afresh, so a stale value is impossible.
    """
    import diffsynth.models.minimax_h3_dit as h3
    if getattr(h3, '_pdmd_cached_bounds', False):
        return
    original = h3._sdpa_varlen_attention

    def cached(q, k, v, cu_seqlens, softmax_scale):
        if getattr(cu_seqlens, '_pdmd_bounds', None) is None:
            try:
                cu_seqlens._pdmd_bounds = cu_seqlens.tolist()
            except AttributeError:
                return original(q, k, v, cu_seqlens, softmax_scale)
        bounds = cu_seqlens._pdmd_bounds
        out = torch.empty_like(q)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if stop == start:
                continue
            seg = [t[start:stop].transpose(0, 1).unsqueeze(0) for t in (q, k, v)]
            out[start:stop] = h3.attention_forward(*seg, scale=softmax_scale).squeeze(0).transpose(0, 1)
        return out

    h3._sdpa_varlen_attention = cached
    h3._pdmd_cached_bounds = True


class H3Backend:
    def __init__(self, dit, device, dtype=torch.bfloat16, checkpoint=True):
        from diffsynth.pipelines.minimax_h3_audio_video import (
            MiniMaxH3Unit_PackedSequenceBuilder, model_fn_minimax_h3)
        self.dit, self.device, self.dtype = dit, device, dtype
        self.pack = MiniMaxH3Unit_PackedSequenceBuilder()
        self.model_fn, self.checkpoint = model_fn_minimax_h3, checkpoint

    @staticmethod
    def latent_shapes(frames, height, width):
        if (frames - 5) % 17 or frames < 5 or height % 32 or width % 32:
            raise ValueError('H3 requires frames=17k+5 and spatial multiples of 32')
        vf = ((frames - 5) // 17) * 5 + 2
        af = round(frames / 24 * 40)
        return (1, 24, vf, height // 16, width // 16), (2, 32, af)

    def noise(self, frames, height, width, generator):
        """fp32 Gaussian noise on the training device, drawn from `generator`."""
        shapes = self.latent_shapes(frames, height, width)
        return tuple(torch.randn(s, device=self.device, dtype=torch.float32, generator=generator)
                     for s in shapes)

    def condition(self, embedding, tags, noise):
        v, a = noise
        embedding = embedding.to(self.device, self.dtype, non_blocking=True)
        tags = tags.to(self.device, non_blocking=True)
        packed = self.pack.process(SimpleNamespace(device=self.device), embedding,
                                   v.to(self.dtype), a.to(self.dtype),
                                   text_token_tags=tags)['packed']
        return {'prompt_embeds': embedding, 'packed': packed}

    def predict(self, role, states, sigmas, condition):
        """Velocity v = z - x0 for each modality, in fp32."""
        set_role(self.dit, role)
        out = self.model_fn(self.dit, video_latents=states[0].to(self.dtype),
                            audio_latents=states[1].to(self.dtype),
                            timestep_video=sigmas[0] * 1000,
                            timestep_audio=sigmas[1] * 1000,
                            use_gradient_checkpointing=self.checkpoint and torch.is_grad_enabled(),
                            **condition)
        return tuple(v.float() for v in out)
