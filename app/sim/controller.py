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
    curtail_kw: float = 0.0          # PV curtailed this tick (battery full)
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
    state.commanded_per_phase_kw = _greedy_per_phase(
        desired_total, sp.max_kw_per_phase, load_per_phase_kw
    )


def _greedy_per_phase(
    total_kw: float, max_kw_per_phase: float, load_per_phase_kw: np.ndarray
) -> np.ndarray:
    """Distribute a total across phases, filling the most-loaded phase first and
    capping each at ``max_kw_per_phase``."""
    per_phase = np.zeros(3)
    remaining = total_kw
    for idx in np.argsort(-load_per_phase_kw):
        take = min(remaining, max_kw_per_phase)
        per_phase[idx] = take
        remaining -= take
        if remaining <= 1e-6:
            break
    return per_phase


def update_coordinated(
    estore_state: InverterState,
    estore_sp: SystemParams,
    solax_state: InverterState,
    solax_sp: SystemParams,
    poc_total_w: float,
    load_per_phase_kw: np.ndarray,
    t_now: float,
) -> None:
    """Coordinated site controller — the cooperative contrast to the duel.

    Instead of each inverter independently chasing the full POC error (which
    double-counts the response and oscillates), a single coordinator measures the
    error once, increments one *combined* command, and splits it across the two
    inverters by inverter capacity. Because the capacity weights sum to one and
    the combined command is clipped to the total capacity, each share lands within
    its own inverter limit without further juggling.

    Runs at the faster of the two sample intervals; uses the average gain,
    deadband and target of the two systems as the site setpoint.
    """
    interval = min(estore_sp.sample_interval_s, solax_sp.sample_interval_s)
    if (t_now - estore_state.last_sample_t) + 1e-9 < interval:
        return
    estore_state.last_sample_t = t_now
    solax_state.last_sample_t = t_now

    gain = 0.5 * (estore_sp.proportional_gain + solax_sp.proportional_gain)
    deadband = 0.5 * (estore_sp.deadband_w + solax_sp.deadband_w)
    target = 0.5 * (estore_sp.target_poc_import_w + solax_sp.target_poc_import_w)
    total_cap = estore_sp.inverter_max_kw + solax_sp.inverter_max_kw

    err_w = poc_total_w - target
    combined = estore_state.commanded_total_kw + solax_state.commanded_total_kw
    if abs(err_w) > deadband:
        combined += gain * err_w / 1000.0
    combined = float(np.clip(combined, 0.0, total_cap))

    # Capacity-weighted split (both shares are inherently within their caps).
    e_cmd = combined * estore_sp.inverter_max_kw / total_cap
    s_cmd = combined * solax_sp.inverter_max_kw / total_cap

    assert estore_sp.phase is not None, "single-phase system must specify a phase"
    estore_state.commanded_total_kw = e_cmd
    estore_state.commanded_per_phase_kw[:] = 0.0
    estore_state.commanded_per_phase_kw[_phase_to_idx(estore_sp.phase)] = e_cmd

    solax_state.commanded_total_kw = s_cmd
    solax_state.commanded_per_phase_kw = _greedy_per_phase(
        s_cmd, solax_sp.max_kw_per_phase, load_per_phase_kw
    )


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
    """Update battery SOC from the inverter's AC output and available PV, holding
    the SOC limits with physically honest fallbacks.

    ``actual_total_kw`` is the hybrid inverter's *total* AC output (PV
    pass-through + battery discharge), so the intended battery flow is::

        battery_kw = actual_total_kw - pv_kw     # >0 discharge, <0 charge

    - **At the SOC floor** the battery can't sustain the discharge, so the AC
      output is reduced toward PV-only (a partial discharge that lands exactly on
      ``soc_min_pct`` for the tick).
    - **At the SOC ceiling** the battery can't absorb the PV surplus; a
      zero-export system curtails the excess PV rather than exporting it. The
      curtailed power is recorded in ``state.curtail_kw`` so the engine can
      report wasted solar — it is no longer silently discarded.
    """
    state.curtail_kw = 0.0
    cap_kwh = sp.battery_capacity_kwh
    hrs = dt_s / 3600.0
    eta = float(np.sqrt(sp.round_trip_efficiency))
    battery_kw = state.actual_total_kw - pv_kw

    if battery_kw >= 0:
        # Discharging: DC drawn exceeds AC delivered by the one-way efficiency.
        dc_kw = battery_kw / eta
        usable_kwh = (state.soc_pct - sp.soc_min_pct) / 100.0 * cap_kwh
        if dc_kw * hrs <= usable_kwh:
            state.soc_pct -= dc_kw * hrs / cap_kwh * 100.0
        else:
            # Battery hits the floor mid-tick: it can only supply `usable_kwh`.
            max_batt_ac = usable_kwh / hrs * eta
            _scale_actual(state, pv_kw + max_batt_ac)
            state.soc_pct = sp.soc_min_pct
    else:
        # Charging from PV surplus: stored DC energy gains the efficiency factor.
        dc_kw = -battery_kw * eta
        headroom_kwh = (sp.soc_max_pct - state.soc_pct) / 100.0 * cap_kwh
        if dc_kw * hrs <= headroom_kwh:
            state.soc_pct += dc_kw * hrs / cap_kwh * 100.0
        else:
            # Battery fills mid-tick: store what fits, curtail the surplus PV.
            absorbable_ac = (headroom_kwh / hrs) / eta   # PV kW the battery takes
            state.curtail_kw = max(0.0, -battery_kw - absorbable_ac)
            state.soc_pct = sp.soc_max_pct


def _scale_actual(state: InverterState, target_total_kw: float) -> None:
    """Scale actual per-phase output so the total becomes ``target_total_kw``."""
    cur = state.actual_total_kw
    if cur > 1e-9:
        state.actual_per_phase_kw *= max(0.0, target_total_kw / cur)
    state.actual_total_kw = float(state.actual_per_phase_kw.sum())


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
