"""Time grids and update ordering for H3 PDMD training.

Noise-time convention throughout: sigma=1 is pure noise, sigma=0 is clean,
x_sigma = (1-sigma) * x0 + sigma * z and the network velocity is v = z - x0.
"""
import torch


def shift_time(t, shift):
    """SD3 time shift, sigma' = s*t / (1 + (s-1)*t). shift=1 is the identity."""
    if shift <= 0:
        raise ValueError("shift must be positive")
    return shift * t / (1 + (shift - 1) * t)


def student_grid(nfe, shift, device='cpu'):
    """Trailing uniform grid [1, 1-1/n, ..., 1/n] plus the terminal 0, then shifted.

    This is DiffSynth's MiniMax-H3 scheduler grid (linspace(1, 0, n+1)[:-1],
    shifted) with the terminal 0 appended.
    """
    if nfe < 1:
        raise ValueError("nfe must be positive")
    t = torch.linspace(1, 0, nfe + 1, device=device, dtype=torch.float64)
    return shift_time(t, shift).float()


def schedules(nfe=4, video_shift=1.0, audio_shift=1.0, device='cpu'):
    """Per-modality student grids, each of length nfe+1 ending at 0."""
    return student_grid(nfe, video_shift, device), student_grid(nfe, audio_shift, device)


def update_kind(iteration, critic_steps=5):
    """Five critic updates FOLLOWED by one student update; each update is one iteration.

    Iterations 0..4 update the critic and iteration 5 the student, so the first
    student update already sees a critic that has moved off the teacher. The
    student update reuses the rollouts of the critic update right before it.
    """
    if iteration < 0 or critic_steps < 1:
        raise ValueError("Invalid TTUR arguments")
    return 'student' if iteration % (critic_steps + 1) == critic_steps else 'critic'


def critic_times(generator, device, video_threshold=0.95, audio_floor=0.85):
    """Critic re-noise times for one sample, in the noise-time convention.

    Video sigma ~ U(0, 1). Audio sigma ~ U(0, 1) independently, except that when
    the video sample is almost pure noise (sigma > video_threshold) the audio
    sigma is redrawn from U(audio_floor, 1). A near-noise video is then never
    paired with near-clean audio.
    """
    u = torch.rand((3,), generator=generator, device=device, dtype=torch.float64)
    video = u[0]
    audio = torch.where(video > video_threshold, audio_floor + (1 - audio_floor) * u[2], u[1])
    return video.float(), audio.float()


def score_times(video_grid, audio_grid, index, ratio):
    """DMD query time inside the rollout interval below the student's step.

    sigma'' = sigma[k+1] + r * (sigma[k] - sigma[k+1]), with one r ~ U(0,1) per
    sample shared by both modalities and mapped onto each modality's own grid.
    """
    return tuple(g[index + 1] + ratio * (g[index] - g[index + 1]) for g in (video_grid, audio_grid))
