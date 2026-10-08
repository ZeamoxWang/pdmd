"""Student rollout and PDMD losses, independent of H3's model implementation.

One training sample is a pair of latents (video [1,24,T,H,W], audio [2,32,T]).
States are kept in fp32; the backend casts them to the model dtype at its boundary.
"""
from dataclasses import dataclass, field
import torch
from pdmd.adapters import set_role
from pdmd.objectives import (endpoint, noise_estimate, endpoint_update, surrogate_loss,
                                      renoise, critic_loss)
from pdmd.schedule import score_times


@dataclass
class Trajectory:
    noise: tuple
    states: list
    condition: dict
    video_sigmas: torch.Tensor
    audio_sigmas: torch.Tensor
    meta: dict = field(default_factory=dict)


@torch.no_grad()
def rollout(backend, noise, condition, video_grid, audio_grid):
    """Few-step Euler sampling with the current student, no gradient.

    states[0] is the initial noise and states[-1] the generated endpoint.
    """
    states = [tuple(x.float() for x in noise)]
    for i in range(len(video_grid) - 1):
        velocities = backend.predict('student', states[-1], (video_grid[i], audio_grid[i]), condition)
        states.append(tuple(x + (g[i + 1] - g[i]) * v
                            for x, v, g in zip(states[-1], velocities, (video_grid, audio_grid))))
    return Trajectory(states[0], states, condition, video_grid, audio_grid)


def critic_step_loss(backend, trajectory, sigmas, noise, weights=(0.8, 0.8), clamp_max=5.0):
    """Fit the critic to the student's endpoint distribution.

    The endpoint of the full rollout is re-noised with FRESH noise at the critic
    time and the critic regresses the flow-matching velocity noise - endpoint.
    `noise` is that fresh sample, passed in so the caller owns the RNG stream.
    """
    clean = trajectory.states[-1]
    query = tuple(renoise(x, z, s) for x, z, s in zip(clean, noise, sigmas))
    velocities = backend.predict('critic', query, sigmas, trajectory.condition)
    parts = [critic_loss(v, x, z, clamp_max) for v, x, z in zip(velocities, clean, noise)]
    loss = weights[0] * parts[0] + weights[1] * parts[1]
    return loss, {'critic_video': parts[0].detach(), 'critic_audio': parts[1].detach()}


def student_step_loss(backend, trajectory, index, ratio, projected=True, weights=(0.8, 0.8),
                      clamp_max=5.0, normalizer_floor=1e-5, eps=1e-8):
    """One differentiable student call at rollout state `index`, then the (P)DMD surrogate.

    The teacher and critic are queried at a point reached from the
    student's own prediction at this state, (1-s'')*x0_hat + s''*z_hat with
    z_hat = x_k + (1-s_k)*v, i.e. a deterministic Euler step along the
    student's trajectory, instead of re-noising x0_hat with fresh noise.
    s'' lies inside the rollout interval just below the student's step.
    """
    start = trajectory.states[index]
    start_sigmas = (trajectory.video_sigmas[index], trajectory.audio_sigmas[index])
    velocity = backend.predict('student', start, start_sigmas, trajectory.condition)
    clean = tuple(endpoint(x, v, s) for x, v, s in zip(start, velocity, start_sigmas))
    query_sigmas = score_times(trajectory.video_sigmas, trajectory.audio_sigmas, index, ratio)
    with torch.no_grad():
        noise_hat = tuple(noise_estimate(x, v, s) for x, v, s in zip(start, velocity, start_sigmas))
        query = tuple(renoise(x.detach(), z, s) for x, z, s in zip(clean, noise_hat, query_sigmas))
        vc = backend.predict('critic', query, query_sigmas, trajectory.condition)
        vt = backend.predict('teacher', query, query_sigmas, trajectory.condition)
        ce = tuple(endpoint(x, v, s) for x, v, s in zip(query, vc, query_sigmas))
        te = tuple(endpoint(x, v, s) for x, v, s in zip(query, vt, query_sigmas))
        updates, nonfinite = [], 0
        for x, t, c in zip(clean, te, ce):
            # Explicit sample dimension: H3 audio's first axis is stereo, not batch.
            u, bad = endpoint_update(x.detach().unsqueeze(0), t.unsqueeze(0), c.unsqueeze(0),
                                     projected=projected, normalizer_floor=normalizer_floor, eps=eps)
            updates.append(u.squeeze(0))
            nonfinite += bad
    # Activation-checkpoint recomputation in backward MUST see the student's adapters.
    set_role(backend.dit, 'student')
    parts = [surrogate_loss(x, u, clamp_max) for x, u in zip(clean, updates)]
    loss = weights[0] * parts[0] + weights[1] * parts[1]
    stats = {'dmd_video': parts[0].detach(), 'dmd_audio': parts[1].detach(),
             'update_rms_video': updates[0].square().mean().sqrt(),
             'update_rms_audio': updates[1].square().mean().sqrt(),
             'nonfinite_update': nonfinite}
    return loss, stats
