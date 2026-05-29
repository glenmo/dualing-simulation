"""Parameter dataclasses for the simulation.

Decoupled from any web framework — these are the single source of truth for what
the engine accepts. The API layer maps incoming JSON to these classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Phase = Literal["A", "B", "C"]
Mode = Literal["uncoordinated", "coordinated"]


@dataclass
class SystemParams:
    name: str
    inverter_max_kw: float
    max_kw_per_phase: float          # ignored for single-phase systems
    three_phase: bool
    phase: Optional[Phase]           # None for three-phase systems
    pv_capacity_kw: float
    pv_tilt_deg: float
    pv_azimuth_deg: float            # 0 = true north
    battery_capacity_kwh: float
    soc_min_pct: float
    soc_max_pct: float
    soc_initial_pct: float
    round_trip_efficiency: float     # 0..1, applied as sqrt on charge and discharge

    # Control loop
    sample_interval_s: float
    deadband_w: float
    ramp_rate_w_per_s: float
    target_poc_import_w: float
    proportional_gain: float         # commanded delta = gain * error_W

    @staticmethod
    def estore_default() -> "SystemParams":
        return SystemParams(
            name="eStore",
            inverter_max_kw=5.0,
            max_kw_per_phase=5.0,
            three_phase=False,
            phase="A",
            pv_capacity_kw=3.5,
            pv_tilt_deg=50.0,
            pv_azimuth_deg=0.0,
            battery_capacity_kwh=10.0,
            soc_min_pct=10.0,
            soc_max_pct=95.0,
            soc_initial_pct=50.0,
            round_trip_efficiency=0.90,
            sample_interval_s=2.0,
            deadband_w=50.0,
            ramp_rate_w_per_s=500.0,
            target_poc_import_w=100.0,
            proportional_gain=1.0,
        )

    @staticmethod
    def solax_default() -> "SystemParams":
        return SystemParams(
            name="SolaX",
            inverter_max_kw=15.0,
            max_kw_per_phase=5.0,
            three_phase=True,
            phase=None,
            pv_capacity_kw=12.0,
            pv_tilt_deg=6.0,
            pv_azimuth_deg=0.0,
            battery_capacity_kwh=27.0,
            soc_min_pct=10.0,
            soc_max_pct=95.0,
            soc_initial_pct=50.0,
            round_trip_efficiency=0.92,
            sample_interval_s=3.0,
            deadband_w=50.0,
            ramp_rate_w_per_s=800.0,
            target_poc_import_w=100.0,
            proportional_gain=1.0,
        )


@dataclass
class LoadParams:
    seed: int = 42
    base_w_per_phase: tuple[float, float, float] = (180.0, 240.0, 200.0)  # unbalanced
    noise_w: float = 30.0
    peaks_per_hour: float = 8.0
    peak_min_kw: float = 1.0
    peak_max_kw: float = 4.0
    peak_min_s: float = 30.0
    peak_max_s: float = 300.0


@dataclass
class SimParams:
    date: str                   # ISO YYYY-MM-DD; "today" resolved by API
    start_hour: float           # local hour-of-day the window begins (0..24)
    duration_s: float
    dt_s: float
    cloud_factor: float         # 0..1; 1.0 = clear sky, 0.0 = fully overcast
    # "uncoordinated": each controller chases the full POC error (they duel).
    # "coordinated": one site coordinator splits the response between them.
    mode: Mode = "uncoordinated"
    estore: SystemParams = field(default_factory=SystemParams.estore_default)
    solax: SystemParams = field(default_factory=SystemParams.solax_default)
    load: LoadParams = field(default_factory=LoadParams)

    @staticmethod
    def default() -> "SimParams":
        return SimParams(
            date="today",
            start_hour=10.0,        # mid-morning: sun up so the PV duel is visible
            duration_s=3600.0,
            dt_s=0.1,
            cloud_factor=1.0,
            mode="uncoordinated",
        )
