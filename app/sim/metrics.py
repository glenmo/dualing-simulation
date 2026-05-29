"""Oscillation / control-quality metrics for a finished simulation.

These quantify how badly the two uncoordinated controllers "duel" around the POC
target: how far the POC swings, how fast it cycles, how much the inverters hunt,
and how much grid export the fight causes. Computed from the 1 Hz logged series
so they match what the UI plots.
"""

from __future__ import annotations

import numpy as np


def _dominant_period_s(sig: np.ndarray, dt_s: float) -> float:
    """Period (s) of the strongest non-DC spectral component, 0 if none/flat."""
    n = len(sig)
    if n < 8 or dt_s <= 0:
        return 0.0
    # Linear-detrend first: a slow drift across the window otherwise dominates the
    # spectrum's lowest bin and masks the actual controller oscillation.
    idx = np.arange(n)
    slope, intercept = np.polyfit(idx, sig, 1)
    x = sig - (slope * idx + intercept)
    if not np.any(np.abs(x) > 1e-9):
        return 0.0
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, d=dt_s)
    spec[0] = 0.0                      # ignore the DC bin
    k = int(np.argmax(spec))
    f = float(freqs[k])
    return 1.0 / f if f > 0 else 0.0


def compute_metrics(
    t_s: np.ndarray,
    poc_total_w: np.ndarray,
    estore_ac_kw: np.ndarray,
    solax_ac_total_kw: np.ndarray,
    target_w: float,
    settle_frac: float = 1.0 / 3.0,
) -> dict:
    """Return a dict of oscillation metrics.

    Oscillation/effort metrics use the *settled* window (everything after the
    first ``settle_frac`` of the run) so the startup transient doesn't dominate.
    Export is integrated over the whole run, since any export is a real cost.
    """
    n = len(t_s)
    if n < 2:
        return {
            "poc_rms_error_w": 0.0,
            "poc_peak_to_peak_w": 0.0,
            "poc_dominant_period_s": 0.0,
            "target_crossings_per_min": 0.0,
            "control_effort_kw": 0.0,
            "export_energy_kwh": 0.0,
            "export_fraction_pct": 0.0,
        }

    dt_s = float(t_s[1] - t_s[0])
    dt_h = dt_s / 3600.0
    i0 = int(n * settle_frac)
    poc_s = poc_total_w[i0:]
    err = poc_s - target_w

    # POC swing about the target.
    rms = float(np.sqrt(np.mean(err**2))) if err.size else 0.0
    ptp = float(poc_s.max() - poc_s.min()) if poc_s.size else 0.0

    # How fast it cycles.
    period = _dominant_period_s(poc_s, dt_s)
    crossings = int(np.sum(err[1:] * err[:-1] < 0)) if err.size > 1 else 0
    settled_min = (poc_s.size * dt_s) / 60.0
    crossings_per_min = crossings / settled_min if settled_min > 0 else 0.0

    # Control effort: cumulative AC-output movement of both inverters (hunting).
    effort = 0.0
    for series in (estore_ac_kw[i0:], solax_ac_total_kw[i0:]):
        if series.size > 1:
            effort += float(np.abs(np.diff(series)).sum())

    # Export caused by the duel (POC < 0), over the whole run.
    export_kw = np.clip(-poc_total_w / 1000.0, 0.0, None)
    export_kwh = float(export_kw.sum() * dt_h)
    export_frac = float(np.mean(poc_total_w < 0.0) * 100.0)

    return {
        "poc_rms_error_w": rms,
        "poc_peak_to_peak_w": ptp,
        "poc_dominant_period_s": period,
        "target_crossings_per_min": crossings_per_min,
        "control_effort_kw": effort,
        "export_energy_kwh": export_kwh,
        "export_fraction_pct": export_frac,
    }
