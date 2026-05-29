# Dualing Simulation

A simulation of **two uncoordinated zero-export inverters sharing one point of
connection (POC)**. Each inverter independently tries to hold grid import at a
small positive target by discharging its battery — but neither knows the other
exists, so they "duel": one backs off as the other ramps up, producing
oscillation around the setpoint. The tool lets you tune both systems and the
site load, run a clear-sky day, and watch the interaction.

Live: **https://smartenergylab.online/dualing-simulation/**

The two modelled systems are based on a real site (Mount Toolebewong, VIC):

| System  | Connection    | Inverter | PV       | Battery  |
|---------|---------------|----------|----------|----------|
| eStore  | single-phase  | 5 kW     | 3.5 kW   | 10 kWh   |
| SolaX   | three-phase   | 15 kW    | 12 kW    | 27 kWh   |

## The model

- **Solar** (`app/sim/solar.py`) — pvlib clear-sky GHI/DNI/DHI for the site,
  projected to each array's plane-of-array, times a single lumped derate and a
  `cloud_factor`. The window can start at any `start_hour` of day.
- **Load** (`app/sim/load.py`) — synthetic unbalanced three-phase base load
  plus low-frequency wander and Poisson-arrival peaks (seeded, reproducible).
- **Controllers** (`app/sim/controller.py`) — both use the same proportional
  law toward a `+target_poc_import_w` setpoint, with a deadband, a control
  sample interval, and a slew-rate ramp. The single-phase controller drives one
  phase; the three-phase controller allocates its commanded total across phases
  greedily, most-loaded first. Each sees only the *previous tick's* POC
  measurement, like a real CT-clamp controller.
- **Control mode** (`mode`) — `uncoordinated` (default) runs each controller
  independently: both chase the *full* POC error, so their combined response is
  ~2× the error. `coordinated` runs one site controller that measures the error
  once, increments a single combined command, and splits it across the two
  inverters by capacity. Because the uncoordinated pair has double the effective
  loop gain, it goes unstable — the "duel" — at roughly half the gain the
  coordinated controller tolerates: at the conservative default gain both are
  well-damped and similar, but raise `proportional_gain` past ~2 and the
  uncoordinated POC oscillation roughly doubles while coordinated stays in hand.
  Toggle the mode (and watch the oscillation metrics) to see it.
- **Engine** (`app/sim/engine.py`) — ticks at `dt_s`, runs each controller at
  its own sample rate, ramps actual output toward command, integrates battery
  SOC, and accounts the POC. An inverter's `actual` AC output is the **total**
  hybrid output (PV pass-through + battery discharge), so PV is *not* subtracted
  again at the POC — surplus PV charges the battery instead of exporting. At the
  SOC ceiling the battery can't absorb more, so the surplus is **curtailed**
  (reported per system) rather than exported; at the floor the AC output falls
  back to PV-only. Results are down-sampled to 1 Hz for plotting.

## Architecture

```
static/            Browser UI (no build step): form + Chart.js plots
  index.html
  app.js           Fetches /api/defaults, POSTs /api/run, renders charts
  style.css
app/
  main.py          FastAPI app: /api/health, /api/defaults, /api/run, static, index
  sim/
    params.py      Dataclasses — single source of truth for engine inputs
    site.py        Site lat/lon/tz constants
    solar.py       pvlib clear-sky → AC power series
    load.py        Synthetic three-phase load generator
    controller.py  Single-phase + three-phase inverter controllers
    engine.py      Time-stepping loop, SOC integration, POC accounting
tests/             pytest smoke + property tests
deploy/            Apache vhost, systemd unit, install.sh
```

All routes are namespaced under `/dualing-simulation/` so the app can live
behind a reverse proxy alongside others on the same domain.

## Running locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload --port 8765
```

Open <http://127.0.0.1:8765/dualing-simulation/>.

## API

| Method | Path                            | Purpose                                   |
|--------|---------------------------------|-------------------------------------------|
| GET    | `/dualing-simulation/api/health`   | Liveness + current phase/stage         |
| GET    | `/dualing-simulation/api/defaults` | Default parameters to populate the UI  |
| POST   | `/dualing-simulation/api/run`      | Run a simulation, return time series   |
| POST   | `/dualing-simulation/api/gain-sweep` | Sweep gain in both modes → metrics per gain |
| GET    | `/dualing-simulation/api/docs`     | Swagger UI                             |

`POST /api/run` takes the full parameter object (same shape as `/api/defaults`)
and returns 1 Hz series for PV, inverter AC output (per-phase for SolaX),
battery SOC, POC power, and load.

## Tests

```bash
.venv/bin/python -m pytest
```

Covers load bounds/reproducibility, solar non-negativity and cloud/`start_hour`
behaviour, ramp limiting, three-phase priority allocation, full-run shape/SOC
invariants, and that the controllers steer the POC toward target without
phantom export on a sunny day.

## Deployment

`deploy/install.sh` (run with `sudo`) provisions the venv, installs an Apache
reverse-proxy vhost and a systemd unit (`dualing-simulation.service`, uvicorn on
`127.0.0.1:8765`), and obtains a Let's Encrypt certificate.

TLS uses `certbot certonly --apache` (authenticator only) plus the explicit
`:80`/`:443` vhost in `deploy/smartenergylab.conf` — the `certbot --apache`
*installer* is avoided because it trips on "vhost ambiguity" for the `www.` SAN.

> After changing anything under `app/`, restart the service so it reloads the
> code: `sudo systemctl restart dualing-simulation.service`. Static files are
> served from disk and update without a restart.
