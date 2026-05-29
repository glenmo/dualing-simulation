"""Solar AC power for a fixed-tilt array using pvlib clear-sky."""

from __future__ import annotations

from datetime import date as date_t
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pvlib

from .site import ALT_M, LAT, LON, TIMEZONE

# AC-side derate (inverter efficiency × DC-AC losses × soiling) lumped into one
# linear factor. The simulation's purpose is duelling-controller behaviour, not
# exact yield prediction, so a single scalar is appropriate.
SYSTEM_DERATE = 0.85


def array_ac_power_series(
    sim_date: date_t,
    duration_s: float,
    dt_s: float,
    capacity_kw: float,
    tilt_deg: float,
    azimuth_deg: float,
    cloud_factor: float,
    start_hour: float = 0.0,
) -> np.ndarray:
    """Compute AC power for one array across the simulation window.

    The window begins at ``start_hour`` (local hour-of-day, may be fractional) on
    ``sim_date`` and runs for ``duration_s`` seconds. Returns an array of length
    ceil(duration_s / dt_s) in kW. Uses pvlib clear-sky GHI/DNI/DHI, projects to
    plane-of-array, and applies a linear capacity × (POA / 1000 W/m²) × derate ×
    cloud model.
    """
    n_samples = int(np.ceil(duration_s / dt_s))

    # We sample irradiance at 1-minute resolution and step-and-hold to dt — the
    # sun moves slowly enough that this is invisible at the controller timescale.
    start_local = datetime.combine(sim_date, datetime.min.time()) + timedelta(
        hours=start_hour
    )
    end_local = start_local + timedelta(seconds=duration_s)
    times_minute = pd.date_range(
        start=start_local, end=end_local, freq="1min", tz=TIMEZONE
    )

    location = pvlib.location.Location(LAT, LON, tz=TIMEZONE, altitude=ALT_M)
    solpos = location.get_solarposition(times_minute)
    clearsky = location.get_clearsky(times_minute, model="ineichen")

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_deg,
        surface_azimuth=azimuth_deg,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
    )
    poa_total = poa["poa_global"].fillna(0.0).clip(lower=0.0).to_numpy()

    # Linear capacity model: P_kW = cap * (POA / 1000) * derate * cloud
    p_kw_minute = capacity_kw * (poa_total / 1000.0) * SYSTEM_DERATE * cloud_factor
    p_kw_minute = np.clip(p_kw_minute, 0.0, capacity_kw)

    # Step-and-hold to dt resolution
    minutes_elapsed = np.arange(n_samples) * dt_s / 60.0
    idx = np.clip(minutes_elapsed.astype(int), 0, len(p_kw_minute) - 1)
    return p_kw_minute[idx]
