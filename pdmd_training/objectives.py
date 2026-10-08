"""Endpoint-space PDMD, Eq. 4 of arXiv:2609.35768.

All inputs have an explicit sample dimension. In H3, stereo is NOT a sample
dimension: [2, 32, T] audio must be wrapped as [1, 2, 32, T]. Video and audio
are projected and normalized independently. Teacher and critic predictions are
always stop-gradient.
"""
import torch
import torch.nn.functional as F


def project_update(direction, residual, eps=1e-8):
    """Remove the component of `direction` parallel to `residual`, per sample.

    d' = d - <d, r> / (<r, r> + eps) * r, inner products over all non-batch
    dimensions in fp32. eps only matters when the residual is numerically zero,
    in which case the update is left unchanged.
    """
    if direction.shape != residual.shape or direction.ndim < 2:
        raise ValueError("Expected equal shapes [batch, ...]")
    d, r = direction.float(), residual.detach().float()
    dims = tuple(range(1, d.ndim))
    rr = r.square().sum(dims, keepdim=True)
    dr = (d * r).sum(dims, keepdim=True)
    return d - dr / (rr + eps) * r


def endpoint(latents, velocity, sigma):
    """x0 estimate: x_sigma=(1-sigma)x0+sigma*z, v=z-x0  =>  x0 = x - sigma*v."""
    return latents.float() - sigma * velocity.float()


def noise_estimate(latents, velocity, sigma):
    """z estimate from the same prediction: z = x + (1-sigma)*v."""
    return latents.float() + (1 - sigma) * velocity.float()


@torch.no_grad()
def endpoint_update(student, teacher_endpoint, critic_endpoint, projected=True,
                    normalizer_floor=1e-5, eps=1e-8):
    """DMD (projected=False) or PDMD (projected=True) endpoint-space update.

    p_real = x - x_teacher, p_fake = x - x_critic, d = p_real - p_fake.
    PDMD drops the part of d parallel to the student-critic residual p_fake.
    Both variants divide by the SAME normalizer mean|p_real|, computed per
    sample before projection, so the projected and unprojected controls differ
    only in the projection.
    """
    if not (student.shape == teacher_endpoint.shape == critic_endpoint.shape):
        raise ValueError("Endpoint shapes differ")
    student = student.float()
    p_real = student - teacher_endpoint.float()
    p_fake = student - critic_endpoint.float()
    direction = p_real - p_fake
    if projected:
        direction = project_update(direction, p_fake, eps)
    dims = tuple(range(1, student.ndim))
    normalizer = p_real.abs().mean(dims, keepdim=True).clamp_min(normalizer_floor)
    update = direction / normalizer
    # A non-finite update would otherwise poison the student through the
    # surrogate target. Zero it, as the original training run did, and let the
    # caller count it.
    bad = ~torch.isfinite(update)
    return torch.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0), int(bad.sum())


def surrogate_loss(student, update, clamp_max=5.0):
    """MSE to the detached target x - update; its gradient is 2 * update / numel.

    The loss value is mean(update^2). Clamping it at `clamp_max` turns the
    gradient off for a sample whose normalized update is pathologically large,
    as in the original training run.
    """
    target = (student.float() - update.float()).detach()
    loss = F.mse_loss(student.float(), target)
    return loss.clamp(0.0, clamp_max) if clamp_max else loss


def renoise(clean, noise, sigma):
    """Forward process at sigma with a given noise sample."""
    if clean.shape != noise.shape:
        raise ValueError("Endpoint and noise shapes differ")
    return (1 - sigma) * clean.float() + sigma * noise.float()


def critic_loss(velocity, clean, noise, clamp_max=5.0):
    """Flow-matching loss of the critic on fresh-noise re-noised student samples."""
    target = noise.detach().float() - clean.detach().float()
    loss = F.mse_loss(velocity.float(), target)
    return loss.clamp(0.0, clamp_max) if clamp_max else loss
