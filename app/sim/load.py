"""Synthetic unbalanced three-phase load profile generator."""

from __future__ import annotations

import numpy as np

from .params import LoadParams


def generate_three_phase_load(
    params: LoadParams,
    duration_s: float,
    dt_s: float,
) -> np.ndarray:
    """Generate a (3, N) array of per-phase load in kW.

    Base load per phase (unbalanced) + small noise + Poisson-distributed peaks
    (1–4 kW, 30 s–5 min) landing on one phase at a time.
    """
    rng = np.random.default_rng(params.seed)
    n = int(np.ceil(duration_s / dt_s))
    base = np.array(params.base_w_per_phase, dtype=float) / 1000.0  # kW
    out = np.tile(base[:, None], (1, n))                            # (3, N)

    # Noise: low-frequency wander per phase (random walk smoothed)
    noise_kw = params.noise_w / 1000.0
    for ph in range(3):
        walk = rng.standard_normal(n) * noise_kw * np.sqrt(dt_s)
        out[ph] += np.cumsum(walk) * 0.3        # damp the drift
        out[ph] += rng.standard_normal(n) * noise_kw * 0.3  # high-freq jitter

    # Peaks: Poisson arrivals
    expected_peaks = params.peaks_per_hour * (duration_s / 3600.0)
    n_peaks = rng.poisson(expected_peaks)
    for _ in range(int(n_peaks)):
        start_s = rng.uniform(0.0, max(duration_s - params.peak_min_s, 1.0))
        dur_s = rng.uniform(params.peak_min_s, params.peak_max_s)
        amp_kw = rng.uniform(params.peak_min_kw, params.peak_max_kw)
        phase = rng.integers(0, 3)
        i0 = int(start_s / dt_s)
        i1 = min(int((start_s + dur_s) / dt_s), n)
        # Soft on/off (linear ramp over 5 s) to avoid step changes that the
        # controller would respond to in one tick
        ramp_n = max(1, int(5.0 / dt_s))
        for k in range(i0, i1):
            t_in = k - i0
            t_out = i1 - 1 - k
            scale = min(1.0, t_in / ramp_n, t_out / ramp_n) if (i1 - i0) > 2 * ramp_n else 1.0
            out[phase, k] += amp_kw * max(0.0, scale)

    return np.clip(out, 0.0, None)
