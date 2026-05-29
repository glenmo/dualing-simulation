"""Smoke + property tests for the simulation engine."""

from __future__ import annotations

import numpy as np
import pytest

from app.sim import controller as ctl
from app.sim.engine import run
from app.sim.load import generate_three_phase_load
from app.sim.metrics import compute_metrics
from app.sim.params import LoadParams, SimParams, SystemParams
from app.sim.solar import array_ac_power_series


def _short_params(duration_s: float = 60.0, dt_s: float = 0.1) -> SimParams:
    p = SimParams.default()
    p.date = "2026-01-15"   # midsummer in VIC — sun is up
    p.duration_s = duration_s
    p.dt_s = dt_s
    return p


def test_load_shape_and_bounds() -> None:
    load = generate_three_phase_load(LoadParams(), duration_s=300.0, dt_s=0.1)
    assert load.shape == (3, 3000)
    assert np.all(load >= 0.0)
    # Base load is 180–250 W/phase ⇒ total 0.6–0.7 kW typical, peaks up to ~4 kW
    assert load.sum(axis=0).max() < 10.0
    assert load.sum(axis=0).mean() > 0.3


def test_load_reproducible_with_seed() -> None:
    a = generate_three_phase_load(LoadParams(seed=7), 60.0, 0.1)
    b = generate_three_phase_load(LoadParams(seed=7), 60.0, 0.1)
    np.testing.assert_array_equal(a, b)


def test_solar_nonnegative_and_capped() -> None:
    from datetime import date
    pv = array_ac_power_series(
        date(2026, 1, 15), 3600.0, 1.0, capacity_kw=5.0,
        tilt_deg=50.0, azimuth_deg=0.0, cloud_factor=1.0,
    )
    assert pv.shape == (3600,)
    assert np.all(pv >= 0.0)
    assert np.all(pv <= 5.0 + 1e-6)


def test_solar_cloud_factor_zero_yields_no_power() -> None:
    from datetime import date
    pv = array_ac_power_series(
        date(2026, 1, 15), 600.0, 1.0, capacity_kw=5.0,
        tilt_deg=50.0, azimuth_deg=0.0, cloud_factor=0.0,
    )
    assert pv.max() == 0.0


def test_start_hour_shifts_window_into_daylight() -> None:
    from datetime import date
    # Same midsummer day: a 1 h window at midnight is dark; at noon it has sun.
    night = array_ac_power_series(
        date(2026, 1, 15), 3600.0, 60.0, capacity_kw=5.0,
        tilt_deg=50.0, azimuth_deg=0.0, cloud_factor=1.0, start_hour=0.0,
    )
    noon = array_ac_power_series(
        date(2026, 1, 15), 3600.0, 60.0, capacity_kw=5.0,
        tilt_deg=50.0, azimuth_deg=0.0, cloud_factor=1.0, start_hour=12.0,
    )
    assert night.max() == 0.0
    assert noon.max() > 1.0


def test_ramp_respected() -> None:
    sp = SystemParams.estore_default()
    sp.ramp_rate_w_per_s = 100.0   # 0.1 kW/s
    state = ctl.make_state(sp)
    state.commanded_per_phase_kw[0] = 5.0    # huge command on phase A
    ctl.ramp_toward_command(state, sp, dt_s=1.0)
    assert state.actual_per_phase_kw[0] == pytest.approx(0.1, abs=1e-9)


def test_soc_ceiling_curtails_surplus_pv() -> None:
    sp = SystemParams.estore_default()
    state = ctl.make_state(sp)
    state.soc_pct = sp.soc_max_pct          # already full
    state.actual_per_phase_kw[:] = 0.0      # inverter exporting nothing
    state.actual_total_kw = 0.0
    ctl.integrate_soc(state, sp, pv_kw=3.0, dt_s=1.0)
    # Nowhere to put 3 kW of PV: it must be curtailed, not stored or exported.
    assert state.soc_pct == pytest.approx(sp.soc_max_pct)
    assert state.curtail_kw == pytest.approx(3.0)


def test_soc_floor_caps_discharge_to_pv() -> None:
    sp = SystemParams.estore_default()
    state = ctl.make_state(sp)
    state.soc_pct = sp.soc_min_pct          # empty
    state.actual_per_phase_kw[:] = 0.0
    state.actual_per_phase_kw[0] = 2.0      # trying to discharge 2 kW
    state.actual_total_kw = 2.0
    ctl.integrate_soc(state, sp, pv_kw=0.0, dt_s=1.0)
    # No charge left and no PV: output must fall to zero, SOC pinned at the floor.
    assert state.soc_pct == pytest.approx(sp.soc_min_pct)
    assert state.actual_total_kw == pytest.approx(0.0)


def test_full_batteries_curtail_without_exporting() -> None:
    """Sunny noon with batteries already full: surplus PV is curtailed (recorded),
    and the POC still holds near target rather than dumping power to the grid."""
    p = _short_params(duration_s=600.0, dt_s=0.5)
    p.start_hour = 12.0
    p.cloud_factor = 1.0
    p.load.peaks_per_hour = 0.0
    p.estore.soc_initial_pct = p.estore.soc_max_pct
    p.solax.soc_initial_pct = p.solax.soc_max_pct
    r = run(p)
    assert float(r.estore_curtail_kw.sum() + r.solax_curtail_kw.sum()) > 0.0
    settled = r.poc_total_w[len(r.poc_total_w) // 3:]
    assert abs(settled.mean() - 100.0) < 600.0, settled.mean()
    # SOC can't climb past max.
    assert r.estore_soc_pct.max() <= p.estore.soc_max_pct + 1e-6
    assert r.solax_soc_pct.max() <= p.solax.soc_max_pct + 1e-6


def test_three_phase_priority_allocation() -> None:
    sp = SystemParams.solax_default()
    sp.deadband_w = 0.0
    state = ctl.make_state(sp)
    state.last_sample_t = -1e6
    # Phase B heavily loaded, A and C light
    load = np.array([0.2, 4.0, 0.2])
    # POC error big enough to demand max output (15 kW)
    ctl.update_three_phase(state, sp, poc_total_w=20000.0, load_per_phase_kw=load, t_now=10.0)
    cmd = state.commanded_per_phase_kw
    # Phase B should be filled first to its 5 kW cap, then C/A in some order to 5 each
    assert cmd[1] == pytest.approx(5.0)
    assert cmd.sum() == pytest.approx(15.0)
    assert np.all(cmd <= 5.0 + 1e-9)


def test_three_phase_modest_total_goes_to_loaded_phase() -> None:
    sp = SystemParams.solax_default()
    sp.deadband_w = 0.0
    state = ctl.make_state(sp)
    state.last_sample_t = -1e6
    load = np.array([0.1, 2.0, 0.1])
    # POC 1000 W vs target 100 W ⇒ err 900 W ⇒ command +0.9 kW
    ctl.update_three_phase(state, sp, poc_total_w=1000.0, load_per_phase_kw=load, t_now=10.0)
    cmd = state.commanded_per_phase_kw
    assert cmd[1] == pytest.approx(0.9)
    assert cmd[0] == 0.0 and cmd[2] == 0.0


def test_metrics_on_clean_oscillation() -> None:
    # Sample at 0.25 s so the discrete series actually reaches the sine's peaks.
    t = np.arange(0.0, 120.0, 0.25)
    target = 100.0
    poc = target + 100.0 * np.sin(2.0 * np.pi * t / 10.0)   # 10 s period, ±100 W
    zeros = np.zeros_like(t)
    m = compute_metrics(t, poc, zeros, zeros, target_w=target, settle_frac=0.0)
    assert m["poc_dominant_period_s"] == pytest.approx(10.0, abs=0.5)
    assert m["poc_peak_to_peak_w"] == pytest.approx(200.0, abs=2.0)
    assert m["poc_rms_error_w"] == pytest.approx(70.7, abs=1.0)
    # ~12 cycles over 120 s ⇒ ~24 target crossings ⇒ ~12 per minute.
    assert m["target_crossings_per_min"] == pytest.approx(12.0, abs=1.0)


def test_metrics_flat_at_target_are_quiet() -> None:
    t = np.arange(0.0, 60.0, 1.0)
    poc = np.full_like(t, 100.0)
    zeros = np.zeros_like(t)
    m = compute_metrics(t, poc, zeros, zeros, target_w=100.0, settle_frac=0.0)
    assert m["poc_rms_error_w"] == pytest.approx(0.0)
    assert m["poc_peak_to_peak_w"] == pytest.approx(0.0)
    assert m["target_crossings_per_min"] == pytest.approx(0.0)
    assert m["poc_dominant_period_s"] == pytest.approx(0.0)
    assert m["export_energy_kwh"] == pytest.approx(0.0)


def test_metrics_export_accounting() -> None:
    # 10 samples at -3600 W (export) over 1 s each = -1 Wh? check kWh integral.
    t = np.arange(0.0, 10.0, 1.0)
    poc = np.full_like(t, -3600.0)            # 3.6 kW export, all samples
    zeros = np.zeros_like(t)
    m = compute_metrics(t, poc, zeros, zeros, target_w=100.0, settle_frac=0.0)
    assert m["export_fraction_pct"] == pytest.approx(100.0)
    # 3.6 kW × 10 s = 0.01 kWh
    assert m["export_energy_kwh"] == pytest.approx(0.01, abs=1e-4)


def test_run_attaches_metrics() -> None:
    r = run(_short_params(duration_s=120.0, dt_s=0.5))
    assert set(r.metrics) >= {
        "poc_rms_error_w", "poc_peak_to_peak_w", "poc_dominant_period_s",
        "target_crossings_per_min", "control_effort_kw",
        "export_energy_kwh", "export_fraction_pct",
    }
    assert r.to_json_dict()["metrics"] is r.metrics


def _duel_scenario(mode: str, gain: float = 3.0) -> SimParams:
    p = _short_params(duration_s=600.0, dt_s=0.2)
    p.cloud_factor = 0.0           # remove PV variability — isolate the control duel
    p.load.peaks_per_hour = 0.0    # smooth load
    p.mode = mode
    # The duel only erupts once the loop gain is high enough that the
    # uncoordinated pair's doubled effective gain destabilises it.
    p.estore.proportional_gain = gain
    p.solax.proportional_gain = gain
    return p


def test_coordinated_mode_tames_the_duel() -> None:
    un = run(_duel_scenario("uncoordinated"))
    co = run(_duel_scenario("coordinated"))
    # At a destabilising gain the duel roughly doubles the swing; coordination
    # (one controller's worth of correction, split) keeps it well under control.
    assert co.metrics["poc_rms_error_w"] < 0.8 * un.metrics["poc_rms_error_w"]
    assert co.metrics["poc_peak_to_peak_w"] < 0.8 * un.metrics["poc_peak_to_peak_w"]
    # ...while still steering the POC to the target on average.
    settled = co.poc_total_w[len(co.poc_total_w) // 3:]
    assert abs(settled.mean() - 100.0) < 500.0, settled.mean()


def test_coordinated_respects_inverter_caps() -> None:
    r = run(_duel_scenario("coordinated"))
    assert np.all(r.estore_ac_kw <= 5.0 + 1e-6)        # eStore inverter_max
    assert np.all(r.solax_ac_per_phase_kw <= 5.0 + 1e-6)  # SolaX max_kw_per_phase


def test_full_run_produces_consistent_shapes() -> None:
    p = _short_params(duration_s=120.0, dt_s=0.5)
    r = run(p)
    n = len(r.t_s)
    assert n > 0
    assert r.pv_estore_kw.shape == (n,)
    assert r.solax_ac_per_phase_kw.shape == (3, n)
    assert r.poc_per_phase_w.shape == (3, n)
    # SOC remains in configured window
    assert np.all((r.estore_soc_pct >= p.estore.soc_min_pct - 1e-3)
                  & (r.estore_soc_pct <= p.estore.soc_max_pct + 1e-3))
    assert np.all((r.solax_soc_pct >= p.solax.soc_min_pct - 1e-3)
                  & (r.solax_soc_pct <= p.solax.soc_max_pct + 1e-3))


def test_controllers_steer_poc_toward_target() -> None:
    """With both controllers active and modest load, POC should average near +100 W."""
    p = _short_params(duration_s=600.0, dt_s=0.2)
    p.cloud_factor = 0.0          # remove PV variability
    p.load.peaks_per_hour = 0.0    # smooth load
    r = run(p)
    # After a startup window, the mean POC should be near the +100 W target,
    # though the duelling-controller oscillation means it's not exact.
    settled = r.poc_total_w[len(r.poc_total_w) // 3:]
    assert abs(settled.mean() - 100.0) < 500.0, settled.mean()


def test_sunny_midday_does_not_export_massively() -> None:
    """Regression: PV is part of each inverter's AC output, not a separate export.

    With clear-sky PV at noon, the controllers should absorb surplus into their
    batteries (charge) and hold POC near the +100 W target — not show kilowatts
    of phantom export from double-counting PV.
    """
    p = _short_params(duration_s=600.0, dt_s=0.2)
    p.start_hour = 12.0
    p.cloud_factor = 1.0
    p.load.peaks_per_hour = 0.0
    r = run(p)
    settled = r.poc_total_w[len(r.poc_total_w) // 3:]
    assert abs(settled.mean() - 100.0) < 600.0, settled.mean()
    # And the batteries should be charging (SOC rising) off the PV surplus.
    assert r.estore_soc_pct[-1] >= r.estore_soc_pct[0] - 1e-6
    assert r.solax_soc_pct[-1] >= r.solax_soc_pct[0] - 1e-6
