from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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
    return {"ok": True, "phase": 1, "stage": "skeleton"}


@app.get("/dualing-simulation/")
@app.get("/dualing-simulation")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
