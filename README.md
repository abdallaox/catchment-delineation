# Catchment Delineation Tool

A browser-based watershed delineation tool. Draw a study area, download terrain data, extract the river network, and delineate catchments from any outlet point — all in one page.

![workflow](https://img.shields.io/badge/workflow-draw%20→%20DEM%20→%20rivers%20→%20delineate-blue)
![stack](https://img.shields.io/badge/stack-Flask%20%2B%20Leaflet%20%2B%20pysheds-informational)

---

## Workflow

1. **Draw** a polygon or rectangle over your study area on the map
2. **Download DEM** — fetches 30 m SRTM terrain from OpenTopography (falls back to AWS Terrain Tiles if rate-limited)
3. **Adjust threshold** — control how dense the river network is
4. **Click a stream** — snaps to the nearest stream cell and immediately delineates the upstream catchment
5. **Export** — download results as GeoPackage or Shapefile ZIP

---

## Features

- D8 flow routing via **pysheds** (pit fill → depression fill → flat resolution → flow direction → accumulation)
- **Segment-level Strahler ordering** — strict, consistent stream ordering with no cell-level artefacts
- River polylines smoothed with **Chaikin corner-cutting**
- Toggle DEM hillshade, flow accumulation, flow direction, and Strahler-order river layers independently
- Upload external **Shapefiles** (ZIP) or **GeoJSON** files as overlay layers, with per-layer visibility toggle
- Export includes catchments, outlets, rivers (vector) + DEM, flow accumulation, flow direction (raster GeoTIFFs) — all in **WGS84 (EPSG:4326)**

---

## Running Locally

```bash
pip install -r requirements.txt
python web_app.py
# Open http://localhost:5000
```

Requires an OpenTopography API key. Set it as an environment variable or edit the default in `web_app.py`:

```bash
export OPENTOPO_API_KEY=your_key_here
```

---

## Deployment

The app is configured for **Railway** (or any Docker host):

```bash
docker build -t catchment-delineation .
docker run -p 8080:8080 -e OPENTOPO_API_KEY=your_key catchment-delineation
```

Railway auto-deploys on push to `master` via the included `Dockerfile`.

---

## Stack

| Layer | Library |
|-------|---------|
| Backend | Flask, pysheds, rasterio, GeoPandas, NumPy |
| Frontend | Leaflet.js, Leaflet.draw |
| DEM source | OpenTopography SRTM GL1 (30 m) / AWS Terrain Tiles fallback |
| Deployment | Docker, gunicorn, Railway |

---

## Export Contents

| File | Description |
|------|-------------|
| `catchment_results.gpkg` | Catchment polygons, outlet points, river network |
| `dem.tif` | Clipped elevation raster (Float32) |
| `flow_accumulation.tif` | Upstream contributing cells (Float32) |
| `flow_direction.tif` | D8 direction codes 1–128 (Int16) |

All layers: **WGS84 geographic coordinates (EPSG:4326)**.
