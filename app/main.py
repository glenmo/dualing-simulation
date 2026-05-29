from pathlib import Path

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.sim import params as sp_mod
from app.sim.engine import run as sim_run

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Dualing Simulation", docs_url="/dualing-simulation/api/docs")

app.mount(
    "/dualing-simulation/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


@app.get("/dualing-simulation/api/health")
def health() -> dict:
    return {"ok": True, "phase": 3, "stage": "ui"}


@app.get("/dualing-simulation/")
@app.get("/dualing-simulation")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# ---- Pydantic schemas that mirror the dataclasses in sim.params ----


class SystemParamsIn(BaseModel):
    name: str
    inverter_max_kw: float = Field(gt=0)
    max_kw_per_phase: float = Field(gt=0)
    three_phase: bool
    phase: str | None = None
    pv_capacity_kw: float = Field(ge=0)
    pv_tilt_deg: float = Field(ge=0, le=90)
    pv_azimuth_deg: float = Field(ge=-360, le=360)
    battery_capacity_kwh: float = Field(gt=0)
    soc_min_pct: float = Field(ge=0, le=100)
    soc_max_pct: float = Field(ge=0, le=100)
    soc_initial_pct: float = Field(ge=0, le=100)
    round_trip_efficiency: float = Field(gt=0, le=1)
    sample_interval_s: float = Field(gt=0)
    deadband_w: float = Field(ge=0)
    ramp_rate_w_per_s: float = Field(gt=0)
    target_poc_import_w: float
    proportional_gain: float = Field(gt=0)

    def to_dc(self) -> sp_mod.SystemParams:
        d = self.model_dump()
        if d.get("phase") not in ("A", "B", "C", None):
            d["phase"] = None
        return sp_mod.SystemParams(**d)


class LoadParamsIn(BaseModel):
    seed: int = 42
    base_w_per_phase: tuple[float, float, float] = (180.0, 240.0, 200.0)
    noise_w: float = 30.0
    peaks_per_hour: float = 8.0
    peak_min_kw: float = 1.0
    peak_max_kw: float = 4.0
    peak_min_s: float = 30.0
    peak_max_s: float = 300.0

    def to_dc(self) -> sp_mod.LoadParams:
        return sp_mod.LoadParams(**self.model_dump())


class SimParamsIn(BaseModel):
    date: str = "today"
    start_hour: float = Field(default=10.0, ge=0, le=24)
    duration_s: float = Field(gt=0, le=86400 * 2)
    dt_s: float = Field(gt=0, le=1.0)
    cloud_factor: float = Field(ge=0, le=1)
    mode: Literal["uncoordinated", "coordinated"] = "uncoordinated"
    estore: SystemParamsIn
    solax: SystemParamsIn
    load: LoadParamsIn = LoadParamsIn()

    def to_dc(self) -> sp_mod.SimParams:
        return sp_mod.SimParams(
            date=self.date,
            start_hour=self.start_hour,
            duration_s=self.duration_s,
            dt_s=self.dt_s,
            cloud_factor=self.cloud_factor,
            mode=self.mode,
            estore=self.estore.to_dc(),
            solax=self.solax.to_dc(),
            load=self.load.to_dc(),
        )


@app.get("/dualing-simulation/api/defaults")
def defaults() -> dict:
    """Return default parameters so the UI can populate the panel."""
    p = sp_mod.SimParams.default()
    return {
        "date": p.date,
        "start_hour": p.start_hour,
        "duration_s": p.duration_s,
        "dt_s": p.dt_s,
        "cloud_factor": p.cloud_factor,
        "mode": p.mode,
        "estore": p.estore.__dict__,
        "solax": p.solax.__dict__,
        "load": {
            **p.load.__dict__,
            "base_w_per_phase": list(p.load.base_w_per_phase),
        },
    }


@app.post("/dualing-simulation/api/run")
def run_simulation(req: SimParamsIn) -> dict:
    try:
        result = sim_run(req.to_dc())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.to_json_dict()
