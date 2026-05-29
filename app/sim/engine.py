"""Simulation engine — ticks fine time, runs each controller at its own rate."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_t
from datetime import datetime

import numpy as np

from . import controller as ctl
from .load import generate_three_phase_load
from .metrics import compute_metrics
from .params import SimParams
from .solar import array_ac_power_series


@dataclass
class SimResult:
    """Down-sampled time series for plotting (1 Hz)."""
    t_s: np.ndarray
    pv_estore_kw: np.ndarray
    pv_solax_kw: np.ndarray
    estore_ac_kw: np.ndarray                # total across phases (single-phase)
    solax_ac_per_phase_kw: np.ndarray       # shape (3, N)
    solax_ac_total_kw: np.ndarray
    estore_soc_pct: np.ndarray
    solax_soc_pct: np.ndarray
    estore_curtail_kw: np.ndarray           # PV curtailed (battery full)
    solax_curtail_kw: np.ndarray
    poc_total_w: np.ndarray
    poc_per_phase_w: np.ndarray             # shape (3, N)
    load_total_kw: np.ndarray
    load_per_phase_kw: np.ndarray           # shape (3, N)
    target_w: float
    metrics: dict = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        return {
            "t_s": self.t_s.tolist(),
            "pv_estore_kw": self.pv_estore_kw.tolist(),
            "pv_solax_kw": self.pv_solax_kw.tolist(),
            "estore_ac_kw": self.estore_ac_kw.tolist(),
            "solax_ac_per_phase_kw": self.solax_ac_per_phase_kw.tolist(),
            "solax_ac_total_kw": self.solax_ac_total_kw.tolist(),
            "estore_soc_pct": self.estore_soc_pct.tolist(),
            "solax_soc_pct": self.solax_soc_pct.tolist(),
            "estore_curtail_kw": self.estore_curtail_kw.tolist(),
            "solax_curtail_kw": self.solax_curtail_kw.tolist(),
            "poc_total_w": self.poc_total_w.tolist(),
            "poc_per_phase_w": self.poc_per_phase_w.tolist(),
            "load_total_kw": self.load_total_kw.tolist(),
            "load_per_phase_kw": self.load_per_phase_kw.tolist(),
            "target_w": self.target_w,
            "metrics": self.metrics,
        }


def _resolve_date(date_str: str) -> date_t:
    if date_str == "today":
        return datetime.now().date()
    return datetime.fromisoformat(date_str).date()


def run(params: SimParams) -> SimResult:
    sim_date = _resolve_date(params.date)
    n = int(np.ceil(params.duration_s / params.dt_s))

    pv_estore = array_ac_power_series(
        sim_date, params.duration_s, params.dt_s,
        params.estore.pv_capacity_kw, params.estore.pv_tilt_deg,
        params.estore.pv_azimuth_deg, params.cloud_factor,
        start_hour=params.start_hour,
    )
    pv_solax = array_ac_power_series(
        sim_date, params.duration_s, params.dt_s,
        params.solax.pv_capacity_kw, params.solax.pv_tilt_deg,
        params.solax.pv_azimuth_deg, params.cloud_factor,
        start_hour=params.start_hour,
    )
    load_phase = generate_three_phase_load(params.load, params.duration_s, params.dt_s)

    # Down-sample to 1 Hz for plotting (keep all engine fidelity in the loop,
    # but the UI doesn't need 100 ms detail across 24 h).
    log_every = max(1, int(round(1.0 / params.dt_s)))
    n_log = int(np.ceil(n / log_every))

    out = SimResult(
        t_s=np.zeros(n_log),
        pv_estore_kw=np.zeros(n_log),
        pv_solax_kw=np.zeros(n_log),
        estore_ac_kw=np.zeros(n_log),
        solax_ac_per_phase_kw=np.zeros((3, n_log)),
        solax_ac_total_kw=np.zeros(n_log),
        estore_soc_pct=np.zeros(n_log),
        solax_soc_pct=np.zeros(n_log),
        estore_curtail_kw=np.zeros(n_log),
        solax_curtail_kw=np.zeros(n_log),
        poc_total_w=np.zeros(n_log),
        poc_per_phase_w=np.zeros((3, n_log)),
        load_total_kw=np.zeros(n_log),
        load_per_phase_kw=np.zeros((3, n_log)),
        target_w=params.estore.target_poc_import_w,
    )

    estore_state = ctl.make_state(params.estore)
    solax_state = ctl.make_state(params.solax)

    log_idx = 0
    last_poc_w = 0.0
    last_poc_per_phase_w = np.zeros(3)

    for k in range(n):
        t = k * params.dt_s

        # Controllers see the POC measurement from the previous tick — they are
        # not omniscient; this is what physical CT-clamp controllers do.
        ctl.update_single_phase(estore_state, params.estore, last_poc_w, t)
        # SolaX needs per-phase load to allocate; use last-tick estimate
        last_load_kw = load_phase[:, max(k - 1, 0)]
        ctl.update_three_phase(
            solax_state, params.solax, last_poc_w, last_load_kw, t
        )

        # Ramp actuals toward commands
        ctl.ramp_toward_command(estore_state, params.estore, params.dt_s)
        ctl.ramp_toward_command(solax_state, params.solax, params.dt_s)

        # Battery SOC update (uses each system's own PV)
        ctl.integrate_soc(estore_state, params.estore, pv_estore[k], params.dt_s)
        ctl.integrate_soc(solax_state, params.solax, pv_solax[k], params.dt_s)

        # POC accounting. actual_per_phase_kw is the hybrid inverter's *total* AC
        # output (PV pass-through + battery discharge — see integrate_soc, where
        # battery_kw = actual - pv). So PV is already inside `actual`; subtracting
        # it again here would double-count generation and show phantom export.
        load_now = load_phase[:, k]
        poc_per_phase_kw = (
            load_now
            - estore_state.actual_per_phase_kw
            - solax_state.actual_per_phase_kw
        )
        last_poc_per_phase_w = poc_per_phase_kw * 1000.0
        last_poc_w = float(last_poc_per_phase_w.sum())

        if k % log_every == 0:
            out.t_s[log_idx] = t
            out.pv_estore_kw[log_idx] = pv_estore[k]
            out.pv_solax_kw[log_idx] = pv_solax[k]
            out.estore_ac_kw[log_idx] = estore_state.actual_total_kw
            out.solax_ac_per_phase_kw[:, log_idx] = solax_state.actual_per_phase_kw
            out.solax_ac_total_kw[log_idx] = solax_state.actual_total_kw
            out.estore_soc_pct[log_idx] = estore_state.soc_pct
            out.solax_soc_pct[log_idx] = solax_state.soc_pct
            out.estore_curtail_kw[log_idx] = estore_state.curtail_kw
            out.solax_curtail_kw[log_idx] = solax_state.curtail_kw
            out.poc_total_w[log_idx] = last_poc_w
            out.poc_per_phase_w[:, log_idx] = last_poc_per_phase_w
            out.load_total_kw[log_idx] = float(load_now.sum())
            out.load_per_phase_kw[:, log_idx] = load_now
            log_idx += 1

    # Trim trailing zeros if log buffer wasn't fully filled
    if log_idx < n_log:
        out.t_s = out.t_s[:log_idx]
        out.pv_estore_kw = out.pv_estore_kw[:log_idx]
        out.pv_solax_kw = out.pv_solax_kw[:log_idx]
        out.estore_ac_kw = out.estore_ac_kw[:log_idx]
        out.solax_ac_per_phase_kw = out.solax_ac_per_phase_kw[:, :log_idx]
        out.solax_ac_total_kw = out.solax_ac_total_kw[:log_idx]
        out.estore_soc_pct = out.estore_soc_pct[:log_idx]
        out.solax_soc_pct = out.solax_soc_pct[:log_idx]
        out.estore_curtail_kw = out.estore_curtail_kw[:log_idx]
        out.solax_curtail_kw = out.solax_curtail_kw[:log_idx]
        out.poc_total_w = out.poc_total_w[:log_idx]
        out.poc_per_phase_w = out.poc_per_phase_w[:, :log_idx]
        out.load_total_kw = out.load_total_kw[:log_idx]
        out.load_per_phase_kw = out.load_per_phase_kw[:, :log_idx]

    out.metrics = compute_metrics(
        out.t_s, out.poc_total_w, out.estore_ac_kw,
        out.solax_ac_total_kw, out.target_w,
    )
    return out
