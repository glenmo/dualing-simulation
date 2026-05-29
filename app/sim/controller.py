"""Inverter controllers.

Two implementations: a single-phase proportional controller (eStore) and a
three-phase controller with per-phase priority allocation (SolaX). Both target
total POC import = +target_W; both act independently — no coordination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .params import SystemParams


@dataclass
class InverterState:
    """Mutable runtime state for an inverter under control.

    Holds both the commanded set-point (updated each control sample) and the
    actual output (ramped each engine tick). Per-phase fields are arrays of
    length 3 even for single-phase systems — the unused phases stay at zero.
    """
    commanded_total_kw: float = 0.0
    actual_total_kw: float = 0.0
    commanded_per_phase_kw: np.ndarray = None       # type: ignore[assignment]
    actual_per_phase_kw: np.ndarray = None          # type: ignore[assignment]
    soc_pct: float = 0.0
    last_sample_t: float = -1e9

    def __post_init__(self) -> None:
        if self.commanded_per_phase_kw is None:
            self.commanded_per_phase_kw = np.zeros(3)
        if self.actual_per_phase_kw is None:
            self.actual_per_phase_kw = np.zeros(3)


def make_state(sp: SystemParams) -> InverterState:
    return InverterState(soc_pct=sp.soc_initial_pct)


def _apply_proportional(
    current_cmd_kw: float,
    poc_total_w: float,
    sp: SystemParams,
) -> float:
    """Proportional control toward the +target_W aggregate-POC setpoint.

    error > deadband: importing more than target → discharge more
    error < -deadband: importing less / exporting → back off
    """
    err_w = poc_total_w - sp.target_poc_import_w
    if abs(err_w) <= sp.deadband_w:
        return current_cmd_kw
    delta_kw = (err_w * sp.proportional_gain) / 1000.0
    return current_cmd_kw + delta_kw


def update_single_phase(
    state: InverterState,
    sp: SystemParams,
    poc_total_w: float,
    t_now: float,
) -> None:
    """eStore-style single-phase controller. Mutates state on a sample tick."""
    if (t_now - state.last_sample_t) + 1e-9 < sp.sample_interval_s:
        return
    state.last_sample_t = t_now

    cmd = _apply_proportional(state.commanded_total_kw, poc_total_w, sp)
    cmd = float(np.clip(cmd, 0.0, sp.inverter_max_kw))
    state.commanded_total_kw = cmd

    assert sp.phase is not None, "single-phase system must specify a phase"
    state.commanded_per_phase_kw[:] = 0.0
    state.commanded_per_phase_kw[_phase_to_idx(sp.phase)] = cmd


def update_three_phase(
    state: InverterState,
    sp: SystemParams,
    poc_total_w: float,
    load_per_phase_kw: np.ndarray,
    t_now: float,
) -> None:
    """SolaX-style three-phase controller with per-phase priority allocation.

    Commands a total output via the same proportional law as the single-phase
    controller, then distributes the total across phases greedily, starting with
    the most-loaded phase. Each phase is clipped at max_kw_per_phase and the
    total at inverter_max_kw.
    """
    if (t_now - state.last_sample_t) + 1e-9 < sp.sample_interval_s:
        return
    state.last_sample_t = t_now

    desired_total = _apply_proportional(state.commanded_total_kw, poc_total_w, sp)
    desired_total = float(np.clip(desired_total, 0.0, sp.inverter_max_kw))
    state.commanded_total_kw = desired_total

    # Greedy fill from the most-loaded phase down, capping each at max_kw_per_phase.
    per_phase = np.zeros(3)
    remaining = desired_total
    for idx in np.argsort(-load_per_phase_kw):
        take = min(remaining, sp.max_kw_per_phase)
        per_phase[idx] = take
        remaining -= take
        if remaining <= 1e-6:
            break
    state.commanded_per_phase_kw = per_phase


def ramp_toward_command(state: InverterState, sp: SystemParams, dt_s: float) -> None:
    """Ramp actual_per_phase_kw toward commanded_per_phase_kw at ramp_rate_w_per_s."""
    max_step_kw = sp.ramp_rate_w_per_s * dt_s / 1000.0
    delta = state.commanded_per_phase_kw - state.actual_per_phase_kw
    np.clip(delta, -max_step_kw, max_step_kw, out=delta)
    state.actual_per_phase_kw += delta
    state.actual_total_kw = float(state.actual_per_phase_kw.sum())


def integrate_soc(
    state: InverterState,
    sp: SystemParams,
    pv_kw: float,
    dt_s: float,
) -> None:
    """Update battery SOC given the inverter's actual AC output and PV available.

    Net battery flow = actual_ac_out - pv_available (positive = discharging).
    SOC drops by discharge_kWh / capacity_kWh; charging gains efficiency × |power|.
    """
    cap_kwh = sp.battery_capacity_kwh
    hrs = dt_s / 3600.0
    eta_one_way = float(np.sqrt(sp.round_trip_efficiency))
    battery_kw = state.actual_total_kw - pv_kw

    if battery_kw > 0:
        # Discharging: pull more from battery than is delivered to AC (efficiency loss)
        dsoc = -(battery_kw / eta_one_way) * hrs / cap_kwh * 100.0
    else:
        # Charging from PV surplus: store less than is generated
        dsoc = -(battery_kw * eta_one_way) * hrs / cap_kwh * 100.0

    new_soc = state.soc_pct + dsoc
    if new_soc < sp.soc_min_pct:
        # Out of usable charge — cannot discharge further; clamp output to PV-only
        state.soc_pct = sp.soc_min_pct
        # Inform the engine on next tick by capping commanded to PV (cheap hack:
        # zero the commanded delta — controller will re-react next sample).
        if battery_kw > 0:
            scale = pv_kw / max(state.actual_total_kw, 1e-6)
            state.actual_per_phase_kw *= max(0.0, scale)
            state.actual_total_kw = float(state.actual_per_phase_kw.sum())
    elif new_soc > sp.soc_max_pct:
        state.soc_pct = sp.soc_max_pct
    else:
        state.soc_pct = new_soc


def _phase_to_idx(p: str) -> int:
    return {"A": 0, "B": 1, "C": 2}[p]


def split_pv_per_phase(sp: SystemParams, pv_total_kw: float) -> np.ndarray:
    """How a system's PV appears on the three phases.

    Single-phase: all PV on the configured phase.
    Three-phase: split evenly.
    """
    out = np.zeros(3)
    if sp.three_phase:
        out[:] = pv_total_kw / 3.0
    else:
        assert sp.phase is not None
        out[_phase_to_idx(sp.phase)] = pv_total_kw
    return out
