"""
Catchment Delineation Web App  —  SRTM GL1 Edition
===================================================
Draw a polygon  →  download 30 m SRTM DEM  →  compute flow routing
→  view fdir / acc / river layers  →  click outlet(s)  →  delineate catchments.

Usage:
    python web_app.py
    Open http://localhost:5000 in your browser.

Environment variables:
    OPENTOPO_API_KEY  — OpenTopography API key (fallback to hard-coded key)
"""

import os, sys, io, json, math, re, zipfile, tempfile, warnings, traceback, shutil, datetime, base64, html as _html_mod, uuid, threading
from contextlib import contextmanager
import numpy as np
import rasterio
import rasterio.features
import rasterio.transform as rio_transform
import rasterio.mask as rio_mask
from rasterio.merge import merge as rio_merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
from matplotlib.backends.backend_pdf import PdfPages
from flask import Flask, request, jsonify, send_file, render_template
import geopandas as gpd
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from pysheds.grid import Grid
from collections import deque
import requests

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB upload limit
OPENTOPO_API_KEY = os.environ.get("OPENTOPO_API_KEY", "4e570ef4c138e88ff14c204a28aaf17e")
OPENTOPO_URL     = "https://portal.opentopography.org/API/globaldem"
TILE_DIR  = os.path.join(tempfile.gettempdir(), "catchment_dem_tiles")
os.makedirs(TILE_DIR, exist_ok=True)

# Persistent basemap tile cache — reused across server restarts
_BASEMAP_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "basemap_tiles")
os.makedirs(_BASEMAP_CACHE, exist_ok=True)
try:
    import contextily as _ctx_init
    _ctx_init.set_cache_dir(_BASEMAP_CACHE)
except Exception:
    pass
NODATA = -9999.0

# ── Stripe ─────────────────────────────────────────────────────────────────────
try:
    import stripe as _stripe
    _stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
except ImportError:
    _stripe = None

STRIPE_PRICE_CENTS = int(os.environ.get("STRIPE_PRICE_CENTS", "500"))   # $5.00
STRIPE_CURRENCY    = os.environ.get("STRIPE_CURRENCY", "usd")
STRIPE_PRODUCT_NAME = os.environ.get("STRIPE_PRODUCT_NAME", "Catchment Results Export")

# Temporary in-memory store: export_token → export params
# Tokens are single-use (cleared after download)
_pending_exports: dict = {}

# ── D8 lookup ──────────────────────────────────────────────────────────────────
D8_DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)
D8_DELTAS = {64:(-1,0), 128:(-1,1), 1:(0,1), 2:(1,1),
             4:(1,0),   8:(1,-1),   16:(0,-1), 32:(-1,-1)}
DELTA_TO_DIR = {v: k for k, v in D8_DELTAS.items()}

# ── Report design constants ────────────────────────────────────────────────────
_A4P = (8.27, 11.69)   # portrait A4 inches
_RN  = "#0d2440"       # navy
_RB  = "#1d5c9e"       # blue
_RLB = "#d4eaf8"       # light blue
_RG  = "#f5f7fa"       # light grey
_RMG = "#7a8a9a"       # muted grey
_RT  = "#1a1a2e"       # text dark
_RMT = "#445566"       # text mid

# ── Global state ───────────────────────────────────────────────────────────────
_cache        = {}            # Conditioned DEM + fdir/acc arrays
_all_results  = []            # All delineation results, one per outlet placed
                                               # River network is cached inside _cache["rivers_cache"]

# ── Agent API v1 — per-session state ──────────────────────────────────────────
_sessions:          dict = {}   # session_id → {cache, results, dem_progress}
_session_lock            = threading.Lock()
_v1_export_tokens:  dict = {}   # token → (zip_bytes, safe_title)


@contextmanager
def _use_session(session_id):
    """Temporarily swap module-level globals to this session's state (thread-safe).

    All existing helper functions (load_and_condition, delineate, extract_rivers_global,
    _emit*, etc.) read/write _cache, _all_results, _dem_progress as globals.  This context
    manager redirects those globals to the session's own dicts for the duration of the
    request, then restores the originals — leaving the web-UI routes completely unaffected.
    """
    global _cache, _all_results, _dem_progress
    if session_id not in _sessions:
        raise KeyError(f"Session '{session_id}' not found. "
                       "Create one via POST /api/v1/session")
    sess = _sessions[session_id]
    with _session_lock:
        _saved        = (_cache, _all_results, _dem_progress)
        _cache        = sess["cache"]
        _all_results  = sess["results"]
        _dem_progress = sess["dem_progress"]
        try:
            yield sess
        finally:
            _cache, _all_results, _dem_progress = _saved


def _v1_sid(data=None):
    """Extract session_id from JSON body, ?session_id= query param, or X-Session-Id header."""
    if data:
        sid = data.get("session_id")
        if sid:
            return sid
    sid = request.args.get("session_id")
    if sid:
        return sid
    return request.headers.get("X-Session-Id")

# ── Naturalearth countries cache (downloaded once per process) ─────────────────
_NE_COUNTRIES_CACHE = os.path.join(os.path.dirname(__file__), "data", "ne_110m_countries.geojson")
_NE_COUNTRIES_URL   = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector"
                        "/master/geojson/ne_110m_admin_0_countries.geojson")

def _ensure_ne_countries():
    """Return path to cached NE 110m countries GeoJSON, downloading if needed."""
    if os.path.exists(_NE_COUNTRIES_CACHE):
        return _NE_COUNTRIES_CACHE
    try:
        os.makedirs(os.path.dirname(_NE_COUNTRIES_CACHE), exist_ok=True)
        r = requests.get(_NE_COUNTRIES_URL, timeout=20)
        if r.status_code == 200:
            with open(_NE_COUNTRIES_CACHE, "wb") as f:
                f.write(r.content)
            print("  [NE] countries cached to", _NE_COUNTRIES_CACHE)
            return _NE_COUNTRIES_CACHE
    except Exception as e:
        print(f"  [NE] download failed: {e}")
    return None

# ── Real-time progress (polled by /api/dem_progress) ───────────────────────────
_dem_progress = {"stage": "", "detail": "", "pct": 0, "source": "", "fallback_reason": "", "error": None, "done": False}

def _emit(stage, detail="", pct=0):
    """Update the progress store — readable by the frontend via polling."""
    _dem_progress.update({"stage": stage, "detail": detail,
                          "pct": int(pct), "error": None, "done": False})
    print(f"  [{int(pct):3d}%] {stage}" + (f" — {detail}" if detail else ""))

def _emit_source(label):
    """Record which DEM source is being used — shown as a badge in the overlay."""
    _dem_progress["source"] = label
    print(f"  [source] {label}")

def _emit_error(msg):
    _dem_progress.update({"stage": "Error", "detail": msg,
                          "error": msg, "done": True})
    print(f"  [ERR] {msg}")

def _emit_done(summary=""):
    _dem_progress.update({"stage": "Complete", "detail": summary,
                          "pct": 100, "error": None, "done": True})
    print(f"  [100%] Done — {summary}")


# ═══════════════════════════════════════════════════════════════════════════════
#  DEM DOWNLOAD  —  SRTM GL1 (30 m) via OpenTopography
# ═══════════════════════════════════════════════════════════════════════════════

def _download_srtm_chunk(south, north, west, east):
    """Returns (path, error_reason). path is None on failure."""
    key  = f"srtm_{south:.5f}_{north:.5f}_{west:.5f}_{east:.5f}"
    path = os.path.join(TILE_DIR, f"{key}.tif")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path, None
    params = {
        "demtype": "SRTMGL1", "south": round(south,6), "north": round(north,6),
        "west": round(west,6), "east": round(east,6),
        "outputFormat": "GTiff", "API_Key": OPENTOPO_API_KEY,
    }
    try:
        r = requests.get(OPENTOPO_URL, params=params, timeout=120)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            return path, None
        reason = f"HTTP {r.status_code}"
        body = r.text.strip()[:200]
        if body:
            reason += f" — {body}"
        print(f"  OpenTopography error: {reason}")
        return None, reason
    except Exception as e:
        reason = str(e)
        print(f"  SRTM chunk download error: {reason}")
        return None, reason


def fetch_dem_bbox(south, north, west, east):
    """Download + mosaic SRTM GL1 tiles. Falls back to AWS Terrain Tiles if OpenTopography fails."""
    # ── Try OpenTopography first ──────────────────────────────────────────────
    CHUNK = 1.0
    lat_edges = list(np.arange(south, north, CHUNK)) + [north]
    lon_edges = list(np.arange(west,  east,  CHUNK)) + [east]
    chunks = [(lat_edges[i], lat_edges[i+1], lon_edges[j], lon_edges[j+1])
              for i in range(len(lat_edges)-1) for j in range(len(lon_edges)-1)]
    n_chunks = len(chunks)
    _emit("Contacting OpenTopography…",
          f"SRTM GL1 (30 m) — {n_chunks} tile(s) to download", pct=2)
    _emit_source("OpenTopography SRTM GL1 (30 m)")
    paths = []
    srtm_errors = []
    for i, (s, n, w, e) in enumerate(chunks):
        pct_tile = 2 + int(16 * i / n_chunks)
        _emit(f"Downloading SRTM tile {i+1}/{n_chunks}…",
              f"OpenTopography — chunk {i+1} of {n_chunks}", pct=pct_tile)
        p, err = _download_srtm_chunk(s, n, w, e)
        if p:
            paths.append(p)
        elif err:
            srtm_errors.append(err)
    # Final tile done → 18 %
    if paths:
        # OpenTopography succeeded
        if len(paths) == 1:
            return paths[0]
        _emit("Mosaicing tiles…", f"Merging {len(paths)} SRTM chunks", pct=18)
        datasets = [rasterio.open(p) for p in paths]
        mosaic, mosaic_tf = rio_merge(datasets, nodata=NODATA)
        meta = datasets[0].meta.copy()
        for ds in datasets:
            ds.close()
        arr = mosaic[0].astype(float)
        arr[arr <= -32767] = NODATA
        arr[arr > 9000]    = NODATA
        mosaic[0] = arr
        meta.update(height=mosaic.shape[1], width=mosaic.shape[2],
                    transform=mosaic_tf, nodata=NODATA, dtype="float32")
        out = os.path.join(TILE_DIR, f"srtm_mosaic_{south:.4f}_{north:.4f}_{west:.4f}_{east:.4f}.tif")
        with rasterio.open(out, "w", **meta) as dst:
            dst.write(mosaic[0:1].astype("float32"))
        return out

    # ── Fallback: AWS Terrain Tiles (GeoTIFF, no API key required) ───────────
    fallback_reason = srtm_errors[0] if srtm_errors else "No tiles returned"
    _dem_progress["fallback_reason"] = fallback_reason
    _emit("OpenTopography unavailable — switching to AWS",
          f"Reason: {fallback_reason[:120]}", pct=3)
    _emit_source("AWS Terrain Tiles (~76 m) — OpenTopography unavailable")
    return _fetch_dem_aws(south, north, west, east)


# AWS Terrain Tiles (public GeoTIFFs, no API key, ~76 m/pixel at zoom 11)
AWS_TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif"


def _wgs84_to_tile(lat, lon, zoom):
    n  = 2 ** zoom
    x  = int((lon + 180.0) / 360.0 * n)
    lr = math.radians(lat)
    y  = int((1.0 - math.asinh(math.tan(lr)) / math.pi) / 2.0 * n)
    return x, y


def _download_aws_tile(z, x, y):
    path = os.path.join(TILE_DIR, f"aws_z{z}_x{x}_y{y}.tif")
    if os.path.exists(path) and os.path.getsize(path) > 100:
        return path
    url = AWS_TILE_URL.format(z=z, x=x, y=y)
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            return path
        print(f"  AWS tile error {r.status_code} ({z}/{x}/{y})")
    except Exception as e:
        print(f"  AWS tile download error ({z}/{x}/{y}): {e}")
    return None


def _fetch_dem_aws(south, north, west, east):
    """Fetch DEM from AWS Terrain Tiles for a bounding box."""
    # Select zoom: aim for ~30 tiles max, ~30-80 m resolution
    lat_mid  = (south + north) / 2.0
    width_km = (east - west) * 111.0 * abs(math.cos(math.radians(lat_mid)))
    # Tile width at zoom z (km) ≈ 40075*cos(lat)/2^z
    circ_km  = 40075.0 * abs(math.cos(math.radians(lat_mid)))
    zoom     = 11  # default ~76 m/px
    for z in range(8, 14):
        tile_km = circ_km / (2 ** z)
        n_tiles = math.ceil(width_km / tile_km)
        if n_tiles <= 8:
            zoom = z
            break

    x0, y0 = _wgs84_to_tile(north, west, zoom)   # y0 is northernmost row
    x1, y1 = _wgs84_to_tile(south, east, zoom)   # y1 is southernmost row
    # Clamp to prevent runaway downloads
    x1 = min(x1, x0 + 9); y1 = min(y1, y0 + 9)

    n_tiles = (x1 - x0 + 1) * (y1 - y0 + 1)
    _emit("Downloading AWS terrain tiles…",
          f"Zoom {zoom} — {x1-x0+1}×{y1-y0+1} grid ({n_tiles} tile(s))", pct=5)

    tile_paths = []
    done = 0
    for xi in range(x0, x1 + 1):
        for yi in range(y0, y1 + 1):
            pct_tile = 5 + int(13 * done / n_tiles)
            _emit(f"Downloading AWS tile {done+1}/{n_tiles}…",
                  f"Zoom {zoom} — tile ({xi},{yi})", pct=pct_tile)
            p = _download_aws_tile(zoom, xi, yi)
            if p:
                tile_paths.append(p)
            done += 1

    if not tile_paths:
        raise RuntimeError("Could not download elevation data from OpenTopography or AWS. "
                           "Check your internet connection and try again.")

    # Merge tiles
    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic, mosaic_tf = rio_merge(datasets, nodata=NODATA)
    src_crs = datasets[0].crs
    meta    = datasets[0].meta.copy()
    for ds in datasets:
        ds.close()

    arr = mosaic[0].astype(float)
    arr[arr <= -32767] = NODATA
    arr[arr > 9000]    = NODATA
    mosaic[0] = arr
    meta.update(height=mosaic.shape[1], width=mosaic.shape[2],
                transform=mosaic_tf, nodata=NODATA, dtype="float32")

    key = f"aws_{south:.4f}_{north:.4f}_{west:.4f}_{east:.4f}"
    tmp_merged = os.path.join(TILE_DIR, f"aws_merged_{key}.tif")
    with rasterio.open(tmp_merged, "w", **meta) as dst:
        dst.write(mosaic[0:1].astype("float32"))

    # Reproject to WGS84 if needed (AWS tiles are already WGS84 at zoom >= 9)
    if src_crs and src_crs.to_epsg() == 4326:
        return tmp_merged

    print("  Reprojecting AWS mosaic to WGS84…")
    dst_crs  = CRS.from_epsg(4326)
    tmp_wgs  = os.path.join(TILE_DIR, f"aws_wgs84_{key}.tif")
    with rasterio.open(tmp_merged) as src:
        tf_dst, w_dst, h_dst = calculate_default_transform(
            src_crs or CRS.from_epsg(3857), dst_crs,
            src.width, src.height, *src.bounds
        )
        meta2 = src.meta.copy()
        meta2.update(crs=dst_crs, transform=tf_dst,
                     width=w_dst, height=h_dst, nodata=NODATA, dtype="float32")
        with rasterio.open(tmp_wgs, "w", **meta2) as dst:
            reproject(source=rasterio.band(src, 1),
                      destination=rasterio.band(dst, 1),
                      src_crs=src_crs or CRS.from_epsg(3857),
                      dst_crs=dst_crs,
                      src_nodata=NODATA, dst_nodata=NODATA,
                      resampling=Resampling.bilinear)
    return tmp_wgs


def clip_dem_to_polygon(dem_path, polygon_geojson):
    """Clip DEM to polygon with crop=True — outside cells become NODATA."""
    import hashlib
    poly_hash = hashlib.md5(json.dumps(polygon_geojson, sort_keys=True).encode()).hexdigest()[:12]
    out_path = os.path.join(TILE_DIR, f"masked_{poly_hash}.tif")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return out_path
    geom = shape(polygon_geojson)
    with rasterio.open(dem_path) as src:
        masked, transform = rio_mask.mask(src, [geom], crop=True, nodata=NODATA, all_touched=True)
        meta = src.meta.copy()
    meta.update(nodata=NODATA, dtype="float32",
                height=masked.shape[1], width=masked.shape[2], transform=transform)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(masked[0:1].astype("float32"))
    n_nodata = int((masked[0] == NODATA).sum())
    print(f"  Clipped DEM: shape {masked.shape[1]}×{masked.shape[2]}, {n_nodata} nodata cells outside polygon")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
#  DEM CONDITIONING
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_condition(dem_path):
    """Read, condition, and fully cache the DEM with a walled pysheds run."""
    global _cache
    _emit("Reading DEM…", os.path.basename(dem_path), pct=22)

    with rasterio.open(dem_path) as src:
        raw    = src.read(1).astype(float)
        nodata = src.nodata
        affine = src.transform
        crs    = src.crs
        nrows, ncols = src.height, src.width

    if nodata is not None:
        raw[raw == nodata] = np.nan
    raw[raw <= -9990]  = np.nan
    raw[raw <= -32767] = np.nan
    raw[raw >   9000]  = np.nan

    left, top = affine.c, affine.f
    bounds_wgs = (left, top + affine.e * nrows, left + affine.a * ncols, top)

    if crs is None:
        tmp = dem_path.replace(".tif", "_crs.tif")
        with rasterio.open(dem_path) as s:
            profile = s.profile.copy()
            data = s.read()
        profile["crs"] = CRS.from_epsg(4326)
        with rasterio.open(tmp, "w", **profile) as d:
            d.write(data)
        dem_path = tmp
        crs = CRS.from_epsg(4326)

    # valid_mask: True only inside the drawn polygon (non-NaN cells)
    valid_mask = np.isfinite(raw)
    n_valid = int(valid_mask.sum())
    elev_info = (f"{np.nanmin(raw):.0f} – {np.nanmax(raw):.0f} m"
                 if n_valid > 0 else "no valid data")
    print(f"  Valid cells: {n_valid}/{raw.size}  elev {elev_info}")

    if n_valid == 0:
        raise RuntimeError("No valid elevation data found in the selected area. "
                           "Try a different location or a larger polygon.")

    # Wall DEM: outside polygon cells → very high elevation so pysheds
    # can't route flow through them, preventing catchment leakage.
    _emit("Conditioning DEM…",
          f"{nrows}×{ncols} cells · elevation {elev_info}", pct=28)
    wall_elev = float(np.nanmax(raw)) + 5000.0
    dem_walled = np.where(valid_mask, raw, wall_elev)

    walled_path = dem_path.replace(".tif", "_walled.tif")
    with rasterio.open(dem_path) as src:
        meta_w = src.meta.copy()
    meta_w.update(nodata=None, dtype="float32")
    with rasterio.open(walled_path, "w", **meta_w) as dst:
        dst.write(dem_walled.astype("float32")[np.newaxis])

    _emit("Filling pits…", "Removing single-cell sinks (pysheds)", pct=34)
    grid    = Grid.from_raster(walled_path)
    dem_ps  = grid.read_raster(walled_path)
    pit     = grid.fill_pits(dem_ps)

    _emit("Filling depressions…", "Removing enclosed basins — may take a moment for large areas", pct=44)
    dep     = grid.fill_depressions(pit)

    _emit("Resolving flat areas…", "Adding gradient to flat cells so flow can route through them", pct=64)
    flat    = grid.resolve_flats(dep)

    _emit("Computing flow direction…", "D8 routing — assigning each cell to 1 of 8 neighbours", pct=74)
    fdir_ps = grid.flowdir(flat, dirmap=D8_DIRMAP)

    _emit("Computing flow accumulation…", "Counting upstream contributing cells for every pixel", pct=83)
    acc_ps  = grid.accumulation(fdir_ps, dirmap=D8_DIRMAP)

    fdir_arr = np.array(fdir_ps).astype(np.int32)
    acc_arr  = np.array(acc_ps).astype(float)

    # Hard-zero outside polygon — belt-and-suspenders
    fdir_arr[~valid_mask] = 0
    acc_arr[~valid_mask]  = 0

    # Explicitly release old large objects before overwriting the cache.
    # pysheds Grid keeps file handles open on Windows; dropping it here
    # ensures the old _walled.tif is not locked when reused on next download.
    import gc
    _cache.pop("ps_grid",  None)
    _cache.pop("ps_fdir",  None)
    _cache.pop("dem_arr",  None)
    _cache.pop("fdir_arr", None)
    _cache.pop("acc_arr",  None)
    _cache.pop("valid_mask", None)
    _cache.pop("strahler_cache", None)
    _cache.pop("rivers_cache",   None)
    gc.collect()

    _cache.clear()
    _cache.update(
        dem_arr=raw, affine=affine, crs=crs,
        fdir_arr=fdir_arr, acc_arr=acc_arr,
        valid_mask=valid_mask,
        bounds_wgs=bounds_wgs, shape=(nrows, ncols),
        ps_grid=grid, ps_fdir=fdir_ps,
        strahler_cache={}, rivers_cache={},
    )
    print(f"  Ready. shape={nrows}×{ncols}  bounds={[round(v,4) for v in bounds_wgs]}")
    return _cache


def _dem_covers(lat, lon):
    if not _cache:
        return False
    w = _cache["bounds_wgs"]
    return w[0] <= lon <= w[2] and w[1] <= lat <= w[3]


# ═══════════════════════════════════════════════════════════════════════════════
#  HYDROLOGICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def latlon_to_rowcol(lat, lon):
    affine = _cache["affine"]
    nrows, ncols = _cache["shape"]
    row, col = rio_transform.rowcol(affine, lon, lat)
    return int(np.clip(row, 0, nrows-1)), int(np.clip(col, 0, ncols-1))


def snap_to_stream(row, col, threshold, radius=15):
    """Snap to the NEAREST stream cell within radius — preserves click position."""
    acc = _cache["acc_arr"]
    nrows, ncols = _cache["shape"]
    best_r, best_c, best_dist = row, col, float("inf")
    for dr in range(-radius, radius+1):
        for dc in range(-radius, radius+1):
            nr, nc = row+dr, col+dc
            if 0 <= nr < nrows and 0 <= nc < ncols:
                if acc[nr, nc] >= threshold:
                    dist = dr*dr + dc*dc
                    if dist < best_dist:
                        best_dist = dist
                        best_r, best_c = nr, nc
    return best_r, best_c


def bfs_upstream(start_r, start_c):
    """Pysheds catchment, clipped to valid_mask (polygon boundary)."""
    try:
        grid    = _cache["ps_grid"]
        fdir_ps = _cache["ps_fdir"]
        catch   = grid.catchment(x=start_c, y=start_r, fdir=fdir_ps,
                                 dirmap=D8_DIRMAP, xytype="index")
        result  = np.asarray(catch).astype(bool)
    except Exception as exc:
        print(f"  [warn] pysheds catchment failed ({exc}), falling back to BFS")
        result = _bfs_python(start_r, start_c)
    valid_mask = _cache.get("valid_mask")
    if valid_mask is not None:
        result &= valid_mask
    return result


def _bfs_python(start_r, start_c):
    fdir_arr = _cache["fdir_arr"]
    nrows, ncols = _cache["shape"]
    visited = np.zeros((nrows, ncols), dtype=bool)
    visited[start_r, start_c] = True
    q = deque([(start_r, start_c)])
    while q:
        r, c = q.popleft()
        for (dr, dc) in D8_DELTAS.values():
            nr, nc = r+dr, c+dc
            if 0 <= nr < nrows and 0 <= nc < ncols and not visited[nr, nc]:
                needed = DELTA_TO_DIR.get((-dr, -dc))
                if needed and fdir_arr[nr, nc] == needed:
                    visited[nr, nc] = True
                    q.append((nr, nc))
    return visited


def _d8_slices(dr, dc, nrows, ncols):
    rh = slice(max(0,-dr), nrows-dr if dr > 0 else None)
    rn = slice(max(0, dr), nrows+dr if dr < 0 else None)
    ch = slice(max(0,-dc), ncols-dc if dc > 0 else None)
    cn = slice(max(0, dc), ncols+dc if dc < 0 else None)
    return rh, ch, rn, cn


def compute_strahler(stream_mask):
    fdir_arr = _cache["fdir_arr"]
    acc_arr  = _cache["acc_arr"]
    nrows, ncols = _cache["shape"]
    up_checks = [(dr, dc, DELTA_TO_DIR[(-dr,-dc)])
                 for dr, dc in D8_DELTAS.values() if (-dr,-dc) in DELTA_TO_DIR]
    order = np.zeros((nrows, ncols), dtype=np.int32)
    rs, cs = np.where(stream_mask)
    idx = np.argsort(acc_arr[rs, cs])
    rs, cs = rs[idx], cs[idx]
    for i in range(len(rs)):
        r, c = int(rs[i]), int(cs[i])
        up = [order[r+dr, c+dc]
              for dr, dc, needed in up_checks
              if 0 <= r+dr < nrows and 0 <= c+dc < ncols
              and stream_mask[r+dr, c+dc] and fdir_arr[r+dr, c+dc] == needed]
        if not up:
            order[r, c] = 1
        else:
            mx = max(up)
            order[r, c] = mx+1 if up.count(mx) >= 2 else mx
    return order


def _get_strahler(acc_threshold):
    sc = _cache.setdefault("strahler_cache", {})
    if acc_threshold not in sc:
        print(f"  Computing Strahler (threshold={acc_threshold})…")
        stream_mask = _cache["acc_arr"] >= acc_threshold
        valid_mask  = _cache.get("valid_mask")
        if valid_mask is not None:
            stream_mask &= valid_mask
        sc[acc_threshold] = compute_strahler(stream_mask)
    return sc[acc_threshold]


MAX_RIVER_SEGMENTS = 3000   # cap to keep browser responsive

def _auto_threshold(initial):
    """Pick a threshold that keeps segment count under MAX_RIVER_SEGMENTS.
    Uses binary search on stream-cell count to avoid building the full network
    repeatedly — only builds once at the estimated threshold."""
    acc  = _cache["acc_arr"]
    valid = _cache.get("valid_mask")

    def stream_cells(t):
        mask = acc >= t
        if valid is not None:
            mask &= valid
        return int(mask.sum())

    # Target stream cell count that empirically gives ~MAX_RIVER_SEGMENTS segments
    # (avg segment length ≈ 12-15 cells)
    target_cells = MAX_RIVER_SEGMENTS * 12

    # Binary search for threshold that gives ≈ target_cells stream cells
    lo, hi = initial, max(initial, int(acc.max()))
    # Only search if the initial threshold already has too many cells
    if stream_cells(initial) > target_cells:
        for _ in range(20):
            if hi - lo <= 1:
                break
            mid = (lo + hi) // 2
            if stream_cells(mid) > target_cells:
                lo = mid
            else:
                hi = mid
        initial = hi

    rivers = extract_rivers_global(initial)
    # Safety: if still over budget double the threshold (cached, so cheap)
    threshold = initial
    for _ in range(6):
        if len(rivers["features"]) <= MAX_RIVER_SEGMENTS:
            break
        threshold = int(threshold * 2)
        rivers = extract_rivers_global(threshold)
    return threshold, rivers

def extract_rivers_global(acc_threshold):
    if not _cache:
        return {"type": "FeatureCollection", "features": []}
    rc = _cache.setdefault("rivers_cache", {})
    if acc_threshold not in rc:
        stream_mask = _cache["acc_arr"] >= acc_threshold
        valid_mask  = _cache.get("valid_mask")
        if valid_mask is not None:
            stream_mask &= valid_mask
        if not stream_mask.any():
            rc[acc_threshold] = {"type": "FeatureCollection", "features": []}
        else:
            print(f"  Building river network (threshold={acc_threshold})…")
            rc[acc_threshold] = build_river_network(stream_mask)
            print(f"  {len(rc[acc_threshold]['features'])} river segments")
    return rc[acc_threshold]


def _no_holes(geom):
    from shapely.geometry import Polygon, MultiPolygon
    if geom is None: return None
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    return geom


def mask_to_polygon(mask):
    affine = _cache["affine"]
    geoms = [shape(s) for s, v in
             rasterio.features.shapes(mask.astype(np.uint8),
                                      mask=mask.astype(np.uint8),
                                      transform=affine) if v == 1]
    if not geoms:
        return None
    return _no_holes(unary_union(geoms))


def _chaikin(coords, iterations=3):
    for _ in range(iterations):
        out = [coords[0]]
        for i in range(len(coords)-1):
            x0,y0 = coords[i]; x1,y1 = coords[i+1]
            out.append([0.75*x0+0.25*x1, 0.75*y0+0.25*y1])
            out.append([0.25*x0+0.75*x1, 0.25*y0+0.75*y1])
        out.append(coords[-1])
        coords = out
    return coords


def build_river_network(stream_mask, _strahler=None):
    """
    Build river network with segment-level Strahler ordering.
    Strahler rules are applied between segments (reaches), not individual cells,
    so order is always consistent: two tributaries of order N → one of order N+1.
    """
    from collections import defaultdict
    fdir_arr = _cache["fdir_arr"]
    acc_arr  = _cache["acc_arr"]
    affine   = _cache["affine"]
    nrows, ncols = _cache["shape"]

    # --- Count upstream / downstream stream neighbours ---
    n_up = np.zeros((nrows, ncols), dtype=np.int8)
    n_dn = np.zeros((nrows, ncols), dtype=np.int8)
    for (dr, dc), dir_val in DELTA_TO_DIR.items():
        needed = DELTA_TO_DIR.get((-dr, -dc))
        if needed is None:
            continue
        rh, ch, rn, cn = _d8_slices(dr, dc, nrows, ncols)
        contrib = stream_mask[rn, cn] & (fdir_arr[rn, cn] == needed)
        n_up[rh, ch] += contrib.view(np.uint8)
        flows = stream_mask[rh, ch] & (fdir_arr[rh, ch] == dir_val) & stream_mask[rn, cn]
        n_dn[rh, ch] |= flows.view(np.uint8)

    nodes = {(r, c) for r, c in zip(*np.where(stream_mask))
             if n_up[r, c] == 0 or n_up[r, c] >= 2 or n_dn[r, c] == 0}

    # --- Trace all segments from each node ---
    seg_list = []
    for (sr, sc) in nodes:
        coords, cells = [], []
        r, c = sr, sc
        while True:
            x, y = rio_transform.xy(affine, r, c)
            coords.append([x, y])
            cells.append((r, c))
            d = fdir_arr[r, c]
            if d not in D8_DELTAS:
                break
            dr, dc = D8_DELTAS[d]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < nrows and 0 <= nc < ncols) or not stream_mask[nr, nc]:
                break
            r, c = nr, nc
            if (r, c) in nodes:
                x2, y2 = rio_transform.xy(affine, r, c)
                coords.append([x2, y2])
                cells.append((r, c))
                break
        if len(coords) < 2:
            continue
        seg_list.append({'start': (sr, sc), 'end': (r, c),
                         'coords': coords, 'cells': cells})

    # --- Segment-level Strahler ---
    # For each node, which segments flow INTO it (end there)
    segs_into = defaultdict(list)
    for i, seg in enumerate(seg_list):
        segs_into[seg['end']].append(i)

    seg_order = [0] * len(seg_list)
    # Sort by start-node accumulation (ascending) so headwaters are processed first
    for i in sorted(range(len(seg_list)),
                    key=lambda k: int(acc_arr[seg_list[k]['start']])):
        sr, sc = seg_list[i]['start']
        if n_up[sr, sc] == 0:
            seg_order[i] = 1          # true headwater — always order 1
        else:
            incoming = [seg_order[j] for j in segs_into[(sr, sc)]
                        if seg_order[j] > 0]
            if not incoming:
                seg_order[i] = 1
            else:
                mx = max(incoming)
                # Classic Strahler: promote only when ≥2 tributaries share the max order
                seg_order[i] = mx + 1 if incoming.count(mx) >= 2 else mx

    # --- GeoJSON features ---
    features = []
    for i, seg in enumerate(seg_list):
        sr, sc = seg['start']
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": _chaikin(seg['coords'])},
            "properties": {"order": seg_order[i],
                           "acc": int(acc_arr[sr, sc])}
        })
    return {"type": "FeatureCollection", "features": features}


def delineate(lat, lon, acc_threshold):
    """
    Catchment-only delineation for a single outlet.
    Returns catchment polygon + outlet point + stats.
    All results are strictly clipped to the polygon boundary (valid_mask).
    """
    import time
    t0 = time.time()

    if not _cache:
        return {"error": "No DEM loaded. Download a DEM first."}
    if not _dem_covers(lat, lon):
        return {"error": "Click is outside the loaded DEM boundary. Re-draw polygon and re-download."}

    affine  = _cache["affine"]
    acc_arr = _cache["acc_arr"]

    # 1. Snap to stream
    row, col     = latlon_to_rowcol(lat, lon)
    s_row, s_col = snap_to_stream(row, col, acc_threshold)
    print(f"  [snap] ({row},{col}) → ({s_row},{s_col})  acc={int(acc_arr[s_row,s_col])}")

    # 2. Catchment BFS — already clipped to valid_mask in bfs_upstream
    catch_mask = bfs_upstream(s_row, s_col)
    n_catch    = int(catch_mask.sum())
    print(f"  [t={time.time()-t0:.2f}s] catchment: {n_catch} cells")
    if n_catch < 10:
        return {"error": "Catchment too small — lower the threshold or click closer to a stream."}

    # 3. Quick stats
    stream_mask  = acc_arr >= acc_threshold
    stream_mask &= catch_mask
    strahler     = _get_strahler(acc_threshold)
    max_ord      = int(strahler[stream_mask].max()) if stream_mask.any() else 0
    cell_m       = abs(affine.a) * 111320
    area_km2     = round(n_catch * cell_m**2 / 1e6, 2)

    # 4. Vectorise
    catch_poly = mask_to_polygon(catch_mask)
    if catch_poly is None:
        return {"error": "Could not vectorise catchment boundary."}

    catch_fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(catch_poly),
         "properties": {"cells": n_catch, "area_km2": area_km2}}
    ]}

    ox, oy = rio_transform.xy(affine, s_row, s_col)
    out_fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature",
         "geometry": {"type": "Point", "coordinates": [ox, oy]},
         "properties": {"acc": int(acc_arr[s_row, s_col])}}
    ]}

    print(f"  [t={time.time()-t0:.2f}s] done  area={area_km2} km²  max_order={max_ord}")

    return {
        "catchment": catch_fc,
        "outlet":    out_fc,
        "stats": {
            "catchment_cells": n_catch,
            "area_km2":        area_km2,
            "max_strahler":    max_ord,
            "outlet_lat":      round(oy, 6),
            "outlet_lon":      round(ox, 6),
            "snapped":         (s_row != row or s_col != col),
            "dem_res_m":       round(cell_m, 1),
            "acc_threshold":   acc_threshold,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED MAP HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_hillshade_rgba(catchment_union):
    """Return cached (rgba_array, [west,east,south,north]) for the DEM hillshade.

    Keyed by the number of delineation results so it auto-invalidates when
    new outlets are added.  Both api_print_map and _rpt_html_map call this
    instead of re-computing the identical LightSource.shade every time.
    Returns (None, None) if no DEM or the computation fails.
    """
    if not _cache or "dem_arr" not in _cache or catchment_union is None:
        return None, None

    key = len(_all_results)
    hs  = _cache.get("_hillshade")
    if hs is not None and hs.get("key") == key:
        return hs["rgba"], hs["extent"]

    try:
        dem   = _cache["dem_arr"].astype(float)
        aff   = _cache["affine"]
        nr, nc = dem.shape
        west  = aff.c
        east  = aff.c + aff.a * nc
        north = aff.f
        south = aff.f + aff.e * nr  # aff.e is negative

        cmask = rasterio.features.rasterize(
            [(catchment_union, 1)],
            out_shape=(nr, nc),
            transform=aff,
            fill=0, dtype="uint8",
        ).astype(bool)

        dem_c = np.where(cmask & np.isfinite(dem), dem, np.nan)
        valid = np.isfinite(dem_c)
        if not valid.any():
            return None, None

        vmin = float(np.percentile(dem_c[valid], 2))
        vmax = float(np.percentile(dem_c[valid], 98))
        if vmax <= vmin:
            vmax = vmin + 1.0

        dem_hs = np.where(valid, dem_c, vmin)
        ls     = LightSource(azdeg=315, altdeg=40)
        rgb    = ls.shade(dem_hs, cmap=plt.cm.gist_earth,
                          vmin=vmin, vmax=vmax,
                          blend_mode="soft", vert_exag=1.5)
        alpha  = np.where(cmask, 0.65, 0.0)
        rgba   = np.dstack([rgb[..., :3], alpha])
        extent = [west, east, south, north]

        _cache["_hillshade"] = {"key": key, "rgba": rgba, "extent": extent}
        return rgba, extent
    except Exception:
        import traceback; traceback.print_exc()
        return None, None


# ═══════════════════════════════════════════════════════════════════════════════
#  PRINT MAP HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _pm_north_arrow(ax):
    """Clean north arrow drawn in its own inset axes (immune to aspect-ratio distortion)."""
    # Inset in top-right, 8 % wide × 12 % tall of main axes
    ax_na = ax.inset_axes([0.886, 0.855, 0.096, 0.130])
    ax_na.set_xlim(-1, 1)
    ax_na.set_ylim(-1.2, 1.2)
    ax_na.set_aspect("equal")
    ax_na.axis("off")
    ax_na.patch.set_facecolor("white")
    ax_na.patch.set_edgecolor("#aaa")
    ax_na.patch.set_linewidth(0.7)

    # Upper half of arrow = black filled triangle (north)
    ax_na.fill([0, -0.55, 0.55], [1.0, 0.0, 0.0], color="#1a2030", zorder=2)
    # Lower half = white filled triangle (south)
    ax_na.fill([0, -0.55, 0.55], [-1.0, 0.0, 0.0],
               color="white", zorder=2)
    ax_na.plot([0, -0.55, 0.55, 0, -0.55], [-1.0, 0.0, 0.0, 1.0, 0.0],
               color="#1a2030", lw=0.8, zorder=3)
    # Centre dot
    ax_na.plot(0, 0, "o", ms=3, color="#1a2030", zorder=4)
    # N label
    ax_na.text(0, 1.25, "N", ha="center", va="bottom",
               fontsize=8, fontweight="bold", color="#1a2030")


def _pm_locator_map(ax, catches_gdf, main_xmin, main_xmax, main_ymin, main_ymax):
    """Country-level context inset with CartoDB basemap tiles."""
    from matplotlib.patches import Rectangle as MplRect
    from shapely.geometry import Point
    import contextily as ctx

    ax_ins = ax.inset_axes([0.005, 0.68, 0.29, 0.31])

    centroid = catches_gdf.unary_union.centroid
    cx, cy   = centroid.x, centroid.y

    # ── Regional extent centred on the catchment ─────────────────────
    # Base: 1.5× the country span, always centred on the catchment.
    # Hard floor of 20°×16° so small countries still get regional context.
    hw, hh = 12.0, 9.0   # fallback if country lookup fails
    countries_path = _ensure_ne_countries()
    if countries_path:
        try:
            countries = gpd.read_file(countries_path)
            pt = Point(cx, cy)
            containing = countries[countries.geometry.contains(pt)]
            if containing.empty:
                countries = countries.copy()
                countries["_d"] = countries.geometry.centroid.distance(pt)
                containing = countries.nsmallest(1, "_d")
            if not containing.empty:
                b = containing.iloc[0].geometry.bounds  # (minx,miny,maxx,maxy)
                # 1.5× country span, min 20°×16°, max 50°×40°
                # Always centred on the CATCHMENT, not the country centroid
                hw = max(min((b[2] - b[0]) * 0.75, 25.0), 10.0)
                hh = max(min((b[3] - b[1]) * 0.75, 20.0),  8.0)
        except Exception as e:
            print(f"  [locator] country lookup: {e}")

    ix0 = max(cx - hw, -180);  ix1 = min(cx + hw, 180)
    iy0 = max(cy - hh,  -90);  iy1 = min(cy + hh,  90)

    ax_ins.set_facecolor("#b8d4e8")
    ax_ins.set_xlim(ix0, ix1)
    ax_ins.set_ylim(iy0, iy1)

    # CartoDB tiles — same style as main map
    try:
        ctx.add_basemap(ax_ins, crs="EPSG:4326",
                        source=ctx.providers.CartoDB.Positron,
                        attribution=False, zoom=6)
        ax_ins.set_xlim(ix0, ix1)   # restore after contextily
        ax_ins.set_ylim(iy0, iy1)
    except Exception:
        try:
            import geodatasets
            land = gpd.read_file(geodatasets.get_path("naturalearth.land"))
            land.clip([ix0, iy0, ix1, iy1]).plot(
                ax=ax_ins, color="#e8e4d8", edgecolor="#bbb", lw=0.4, zorder=1)
        except Exception:
            pass

    # Catchment — solid red so it reads at country scale
    catches_gdf.plot(ax=ax_ins, color="#e02020", alpha=0.9,
                     edgecolor="#900000", linewidth=0.8, zorder=4)

    ax_ins.set_xticks([]); ax_ins.set_yticks([])
    for sp in ax_ins.spines.values():
        sp.set_edgecolor("#888"); sp.set_linewidth(0.8)
    ax_ins.set_title("Country context", fontsize=6, color="#333",
                     pad=2.5, fontweight="bold")


def _pm_scale_bar(ax, xmin, xmax, ymin, ymax):
    """Draw a two-tone GIS scale bar at the bottom-left of *ax*."""
    from matplotlib.patches import FancyBboxPatch, Rectangle as MplRect

    center_lat  = (ymin + ymax) / 2
    km_per_deg  = 111.32 * math.cos(math.radians(center_lat))
    map_km      = (xmax - xmin) * km_per_deg

    # Aim for ~20 % of map width, rounded to a nice number
    target = map_km * 0.20
    mag    = 10 ** math.floor(math.log10(max(target, 0.001)))
    nice   = min([mag, 2*mag, 5*mag, 10*mag], key=lambda v: abs(v - target))
    nice   = max(nice, 0.01)

    bar_frac = min((nice / km_per_deg) / (xmax - xmin), 0.35)
    bx0, bx1 = 0.05, 0.05 + bar_frac
    by,  bh  = 0.058, 0.012

    # Background
    ax.add_patch(FancyBboxPatch(
        (bx0 - 0.01, by - 0.024), bar_frac + 0.02, 0.060,
        boxstyle="square,pad=0.005", transform=ax.transAxes,
        fc="white", ec="#aaa", lw=0.6, zorder=11
    ))
    mid = (bx0 + bx1) / 2
    # Dark first half
    ax.add_patch(MplRect((bx0, by), (bx1-bx0)/2, bh,
                         transform=ax.transAxes, fc="#333", ec="none", zorder=12))
    # White second half
    ax.add_patch(MplRect((mid, by), (bx1-bx0)/2, bh,
                         transform=ax.transAxes, fc="white", ec="none", zorder=12))
    # Outline
    ax.add_patch(MplRect((bx0, by), bx1-bx0, bh,
                         transform=ax.transAxes, fc="none", ec="#333", lw=0.8, zorder=13))
    # Labels
    lbl = f"{nice:.0f} km" if nice >= 1 else f"{nice*1000:.0f} m"
    for xf, txt in [(bx0, "0"), (mid, f"{nice/2:.0f}"), (bx1, lbl)]:
        ax.text(xf, by + bh + 0.006, txt, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=6.5, color="#333", zorder=14)


def _pm_legend(ax, outlet_info, rivers_gdf):
    """Build and attach a styled legend to *ax*."""
    from matplotlib.patches import Patch
    from matplotlib.lines  import Line2D

    handles, labels = [], []

    RIVER_COLORS = ["#aacce8","#6aaed6","#3182bd","#1a5ea0","#0d3a78","#062050"]
    RIVER_LABELS = ["Stream order 1","Stream order 2","Stream order 3",
                    "Stream order 4","Stream order 5","Stream order 6+"]

    for o in outlet_info:
        name  = o.get("name") or f'Catchment {o["id"]}'
        color = o.get("color", "#3399ff")
        handles.append(Patch(facecolor=color, edgecolor=color,
                             alpha=0.55, linewidth=1.5, linestyle="--"))
        labels.append(name)

    handles.append(Line2D([0], [0], marker="o", color="none", markersize=7,
                          markerfacecolor="#888", markeredgecolor="white",
                          markeredgewidth=1.5, linewidth=0))
    labels.append("Outlet")

    if not rivers_gdf.empty and "order" in rivers_gdf.columns:
        for order in sorted(rivers_gdf["order"].unique()):
            idx = min(int(order) - 1, 5)
            handles.append(Line2D([0], [0], color=RIVER_COLORS[idx],
                                  linewidth=max(0.8, int(order)*0.6 + 0.4)))
            labels.append(RIVER_LABELS[idx])

    legend = ax.legend(
        handles, labels,
        loc="lower right", fontsize=7.5,
        title="Legend", title_fontsize=8.5,
        framealpha=0.55, edgecolor="#aaa", fancybox=False,
        handlelength=2.2, handleheight=1.3,
        borderpad=0.9, labelspacing=0.55,
    )
    legend.get_frame().set_linewidth(0.8)
    legend.get_title().set_fontweight("bold")


# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGE RENDERING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _blank_png():
    import base64
    px = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9Q"
        "DwADhgGAWjR9awAAAABJRU5ErkJggg==")
    return send_file(io.BytesIO(px), mimetype="image/png")


def _render_rgba_png(rgba):
    MAX_PX = 1024
    h, w = rgba.shape[:2]
    scale = min(1.0, MAX_PX / max(h, w))
    if scale < 1.0:
        from PIL import Image as _PIL
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        img  = _PIL.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8))
        rgba = np.array(img.resize((new_w, new_h), _PIL.NEAREST)) / 255.0
        h, w = rgba.shape[:2]
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    ax.imshow(rgba, origin="upper", interpolation="nearest")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dem_progress")
def api_dem_progress():
    """Polled by the frontend every ~600 ms to get real-time DEM processing status."""
    return jsonify(_dem_progress)


@app.route("/api/fetch_dem", methods=["POST"])
def api_fetch_dem():
    """Download + clip + condition DEM. Body: {polygon} GeoJSON geometry."""
    data = request.json or {}
    _dem_progress.update({"stage": "Starting…", "detail": "", "pct": 0, "source": "", "fallback_reason": "", "error": None, "done": False})
    try:
        if "polygon" in data:
            polygon = data["polygon"]
            coords  = polygon["coordinates"][0]
            lats = [c[1] for c in coords]; lons = [c[0] for c in coords]
            south, north = min(lats), max(lats)
            west,  east  = min(lons), max(lons)
            area_deg2 = (north - south) * (east - west)
            print(f"[DEM] Polygon bbox: S={south:.4f} N={north:.4f} W={west:.4f} E={east:.4f}")
            _emit("Contacting OpenTopography…",
                  f"Bbox: {south:.3f}°–{north:.3f}°N, {west:.3f}°–{east:.3f}°E", pct=1)
            dem_path = fetch_dem_bbox(south, north, west, east)
            _emit("Clipping to study area…", "Masking cells outside drawn polygon", pct=20)
            dem_path = clip_dem_to_polygon(dem_path, polygon)
        else:
            return jsonify({"error": "Only polygon-based DEM download is supported."}), 400

        # Reset previous results when a new DEM is loaded
        _all_results.clear()
        load_and_condition(dem_path)

        w   = _cache["bounds_wgs"]
        acc = _cache["acc_arr"]
        p95 = float(np.percentile(acc[acc > 0], 95)) if (acc > 0).any() else 500
        default_acc = int(round(p95 / 50) * 50)
        _emit("Extracting river network…",
              f"Auto-selecting stream threshold (starting at {default_acc} cells)", pct=93)
        default_acc, rivers = _auto_threshold(default_acc)
        n_segs = len(rivers.get("features", []))
        _emit_done(f"DEM ready · {_cache['shape'][0]}×{_cache['shape'][1]} cells · {n_segs} river segments")
        return jsonify({
            "bounds":      {"south": w[1], "west": w[0], "north": w[3], "east": w[2]},
            "res_m":       round(abs(_cache["affine"].a) * 111320, 1),
            "shape":       list(_cache["shape"]),
            "acc_p95":     p95,
            "default_acc": default_acc,
            "rivers":      rivers,
            "dem_source":       _dem_progress.get("source", "OpenTopography SRTM GL1 (30 m)"),
            "fallback_reason":  _dem_progress.get("fallback_reason", ""),
        })
    except Exception as e:
        msg = str(e) or repr(e) or "Unknown server error"
        _emit_error(msg)
        return jsonify({"error": msg, "trace": traceback.format_exc()}), 500




@app.route("/api/upload_dem", methods=["POST"])
def api_upload_dem():
    """Accept a user-supplied GeoTIFF, reproject to WGS84 if needed, run same pipeline."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected"}), 400

    polygon_str = request.form.get("polygon")
    polygon = json.loads(polygon_str) if polygon_str else None

    _dem_progress.update({"stage": "Reading uploaded file…", "detail": f.filename,
                          "pct": 0, "source": "", "fallback_reason": "", "error": None, "done": False})
    try:
        with tempfile.TemporaryDirectory() as td:
            raw_path = os.path.join(td, "upload.tif")
            f.save(raw_path)

            with rasterio.open(raw_path) as src:
                src_crs    = src.crs
                src_nodata = src.nodata
                src_bounds = src.bounds
                _emit("Inspecting uploaded DEM…",
                      f"{src.width}×{src.height} px · dtype={src.dtypes[0]} · CRS={src_crs}", pct=5)
                print(f"  Uploaded DEM: {src.width}×{src.height} px  dtype={src.dtypes[0]}  "
                      f"nodata={src_nodata}  crs={src_crs}  bounds={src_bounds}")

            dst_crs = CRS.from_epsg(4326)

            # If no CRS, infer from bounds — geographic coords → assume WGS84
            if src_crs is None:
                if (-180 <= src_bounds.left <= 180 and -90 <= src_bounds.bottom <= 90):
                    print("  No CRS — bounds look geographic, assuming WGS84")
                    src_crs = dst_crs
                else:
                    return jsonify({"error":
                        "Cannot determine CRS. Reproject to WGS84 (EPSG:4326) before uploading."}), 400

            # Always normalise to float32 / single-band / WGS84 / nodata=-9999.
            # Using reproject even for same-CRS inputs avoids all WKT comparison
            # pitfalls, multi-band files, Int16/nodata=0 edge-cases, etc.
            # Warn early for very large DEMs (reprojection will still run but
            # the user gets a heads-up that processing will be slow)
            with rasterio.open(raw_path) as src:
                approx_cells = src.width * src.height
            if approx_cells > 50_000_000:
                return jsonify({"error":
                    f"DEM is too large ({approx_cells//1_000_000} M cells). "
                    "Clip to your study area in QGIS first, then re-upload."}), 400

            norm_path = os.path.join(td, "upload_norm.tif")
            _emit("Normalising DEM…", "Reprojecting to WGS84 float32", pct=18)
            with rasterio.open(raw_path) as src:
                tf_dst, w_dst, h_dst = calculate_default_transform(
                    src_crs, dst_crs, src.width, src.height, *src.bounds)
                meta = {"driver": "GTiff", "dtype": "float32", "nodata": NODATA,
                        "width": w_dst, "height": h_dst, "count": 1, "crs": dst_crs,
                        "transform": tf_dst}
                with rasterio.open(norm_path, "w", **meta) as dst_f:
                    reproject(source=rasterio.band(src, 1),
                              destination=rasterio.band(dst_f, 1),
                              src_crs=src_crs, dst_crs=dst_crs,
                              src_nodata=src_nodata, dst_nodata=NODATA,
                              resampling=Resampling.bilinear)

            persistent = os.path.join(TILE_DIR, "user_uploaded_dem.tif")
            shutil.copy2(norm_path, persistent)

        _all_results.clear()
        load_and_condition(persistent)

        if not _cache.get("valid_mask", np.array([])).any():
            return jsonify({"error":
                "DEM loaded but contains no valid elevation cells. "
                "Check nodata values or try a different file."}), 400

        w   = _cache["bounds_wgs"]
        acc = _cache["acc_arr"]
        p95 = float(np.percentile(acc[acc > 0], 95)) if (acc > 0).any() else 500
        default_acc = int(round(p95 / 50) * 50)
        _emit("Extracting river network…",
              f"Auto-selecting stream threshold (starting at {default_acc} cells)", pct=93)
        default_acc, rivers = _auto_threshold(default_acc)
        n_segs = len(rivers.get("features", []))
        _emit_done(f"DEM ready · {_cache['shape'][0]}×{_cache['shape'][1]} cells · {n_segs} river segments")
        return jsonify({
            "bounds":      {"south": w[1], "west": w[0], "north": w[3], "east": w[2]},
            "res_m":       round(abs(_cache["affine"].a) * 111320, 1),
            "shape":       list(_cache["shape"]),
            "acc_p95":     p95,
            "default_acc": default_acc,
            "rivers":      rivers,
            "source":      "upload",
        })
    except Exception as e:
        msg = str(e)
        _emit_error(msg)
        return jsonify({"error": msg, "trace": traceback.format_exc()}), 500


@app.route("/api/snap", methods=["POST"])
def api_snap():
    """Snap a lat/lon to the nearest stream cell."""
    data = request.json or {}
    if not _cache:
        return jsonify({"error": "No DEM loaded"}), 400
    try:
        lat           = float(data["lat"])
        lon           = float(data["lon"])
        acc_threshold = int(data.get("acc_threshold", 500))
        row, col      = latlon_to_rowcol(lat, lon)
        s_row, s_col  = snap_to_stream(row, col, acc_threshold)
        ox, oy        = rio_transform.xy(_cache["affine"], s_row, s_col)
        return jsonify({
            "lat":     float(oy),
            "lon":     float(ox),
            "snapped": (s_row != row or s_col != col),
            "acc":     int(_cache["acc_arr"][s_row, s_col]),
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/rivers", methods=["POST"])
def api_rivers():
    data = request.json or {}
    if not _cache:
        return jsonify({"error": "No DEM loaded"}), 400
    try:
        acc_threshold = int(data.get("acc_threshold", 500))
        rivers = extract_rivers_global(acc_threshold)
        return jsonify({"rivers": rivers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dem_preview.png")
def api_dem_preview():
    if not _cache:
        return _blank_png()
    dem   = _cache["dem_arr"].copy()
    valid = np.isfinite(dem)
    if not valid.any():
        return _blank_png()
    valid_vals = dem[valid]
    vmin = float(np.percentile(valid_vals, 2))
    vmax = float(np.percentile(valid_vals, 98))
    if vmax - vmin < 1.0:
        vmax = vmin + 1.0
    dem_f = np.where(valid, np.clip(dem, vmin, vmax), vmin)
    MAX_PX = 1024
    h_orig, w_orig = dem_f.shape
    scale = min(1.0, MAX_PX / max(h_orig, w_orig))
    if scale < 1.0:
        from PIL import Image as _PIL
        new_h = max(1, int(h_orig * scale))
        new_w = max(1, int(w_orig * scale))
        dem_f = np.array(_PIL.fromarray(dem_f.astype(np.float32)).resize((new_w,new_h), _PIL.BILINEAR))
        valid = np.array(_PIL.fromarray(valid.astype(np.uint8)).resize((new_w,new_h), _PIL.NEAREST)).astype(bool)
    relief = vmax - vmin
    vert_exag = float(np.clip(500.0 / max(relief, 1.0), 1.0, 15.0))
    ls = LightSource(azdeg=315, altdeg=40)
    hs = ls.hillshade(dem_f, vert_exag=vert_exag)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = plt.cm.terrain(norm(dem_f))
    rgba[..., :3] = rgba[..., :3] * (0.55 + 0.45 * hs[..., None])
    rgba[..., 3]  = np.where(valid, 1.0, 0.0)
    return _render_rgba_png(rgba)


@app.route("/api/acc_image")
def api_acc_image():
    """Log-scaled flow accumulation as a Blues colormap PNG overlay."""
    if not _cache:
        return _blank_png()
    acc   = _cache["acc_arr"].copy().astype(float)
    valid = _cache.get("valid_mask", np.ones(acc.shape, dtype=bool))
    log_acc = np.log1p(acc)
    log_acc[~valid] = 0
    vmax = log_acc[valid].max() if valid.any() else 1.0
    if vmax < 1e-6:
        return _blank_png()
    norm = log_acc / vmax
    rgba = plt.cm.Blues(norm)
    rgba[~valid, 3] = 0.0              # transparent outside polygon
    rgba[valid & (acc <= 0), 3] = 0.0  # transparent zero-acc cells
    rgba[valid & (acc > 0), 3] = 0.80  # semi-transparent inside
    return _render_rgba_png(rgba)


@app.route("/api/fdir_image")
def api_fdir_image():
    """D8 flow direction as a coloured-by-direction PNG overlay."""
    if not _cache:
        return _blank_png()
    fdir  = _cache["fdir_arr"].copy()
    valid = _cache.get("valid_mask", np.ones(fdir.shape, dtype=bool))
    # 8 D8 directions → 8 hues equally spaced on colour wheel
    # N=64,NE=128,E=1,SE=2,S=4,SW=8,W=16,NW=32
    DIR_HUE = {64: 0.60, 128: 0.50, 1: 0.33, 2: 0.17,
               4: 0.08,   8: 0.00, 16: 0.83, 32: 0.72}
    rgba = np.zeros((*fdir.shape, 4), dtype=float)
    for dval, hue in DIR_HUE.items():
        mask = (fdir == dval) & valid
        r, g, b = matplotlib.colors.hsv_to_rgb([hue, 0.75, 0.90])
        rgba[mask] = [r, g, b, 0.75]
    rgba[~valid] = [0, 0, 0, 0]
    return _render_rgba_png(rgba)


@app.route("/api/delineate", methods=["POST"])
def api_delineate():
    data = request.json or {}
    try:
        result = delineate(
            lat=float(data["lat"]),
            lon=float(data["lon"]),
            acc_threshold=int(data.get("acc_threshold", 500)),
        )
        if "error" not in result:
            _all_results.append(result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/clear_results", methods=["POST"])
def api_clear_results():
    _all_results.clear()
    return jsonify({"ok": True})


@app.route("/api/remove_result", methods=["POST"])
def api_remove_result():
    idx = (request.json or {}).get("idx", -1)
    if 0 <= idx < len(_all_results):
        _all_results.pop(idx)
    return jsonify({"ok": True})


def _write_raster_tif(arr, dtype, nodata, path):
    """Write a 2-D numpy array to a GeoTIFF using cached affine/CRS."""
    affine = _cache["affine"]
    crs    = _cache["crs"]
    nrows, ncols = arr.shape
    with rasterio.open(
        path, "w", driver="GTiff",
        height=nrows, width=ncols,
        count=1, dtype=dtype,
        crs=crs, transform=affine,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(arr.astype(dtype), 1)


@app.route("/api/export/<fmt>")
def api_export(fmt):
    if not _all_results:
        return jsonify({"error": "No delineation results to export."}), 400

    # Merge all catchments and outlets with outlet_id
    all_catches, all_outlets = [], []
    for i, r in enumerate(_all_results):
        for feat in r["catchment"]["features"]:
            feat = dict(feat)
            feat["properties"] = {**feat.get("properties", {}), "outlet_id": i+1}
            all_catches.append(feat)
        for feat in r["outlet"]["features"]:
            feat = dict(feat)
            feat["properties"] = {**feat.get("properties", {}), "outlet_id": i+1}
            all_outlets.append(feat)

    # Rivers at last-used threshold
    acc_threshold = int(_all_results[-1]["stats"].get("acc_threshold", 500)) \
        if _all_results else 500
    rivers_fc = extract_rivers_global(acc_threshold) if _cache \
        else {"type": "FeatureCollection", "features": []}

    vec_layers = [
        ("catchments", gpd.GeoDataFrame.from_features(all_catches, crs=4326)),
        ("outlets",    gpd.GeoDataFrame.from_features(all_outlets, crs=4326)),
        ("rivers",     gpd.GeoDataFrame.from_features(rivers_fc["features"], crs=4326)),
    ]

    # Build raster GeoTIFFs in a temp directory
    raster_files = []   # list of (arcname, file_path)
    if _cache:
        with tempfile.TemporaryDirectory() as rtd:
            dem_path  = os.path.join(rtd, "dem.tif")
            acc_path  = os.path.join(rtd, "flow_accumulation.tif")
            fdir_path = os.path.join(rtd, "flow_direction.tif")

            dem_arr  = _cache["dem_arr"].copy()
            dem_arr[~np.isfinite(dem_arr)] = -9999.0
            _write_raster_tif(dem_arr,              "float32", -9999.0, dem_path)

            acc_arr  = _cache["acc_arr"].copy()
            _write_raster_tif(acc_arr,              "float32", -1.0,    acc_path)

            fdir_arr = _cache["fdir_arr"].copy().astype(np.int16)
            _write_raster_tif(fdir_arr,             "int16",   0,       fdir_path)

            # Read bytes now (before TemporaryDirectory is cleaned up)
            raster_files = []
            for arcname, fpath in [("dem.tif", dem_path),
                                    ("flow_accumulation.tif", acc_path),
                                    ("flow_direction.tif", fdir_path)]:
                with open(fpath, "rb") as f:
                    raster_files.append((arcname, f.read()))

    if fmt == "gpkg":
        # Bundle GPKG + raster GeoTIFFs in a ZIP
        tmp_gpkg = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        tmp_gpkg.close()
        for name, gdf in vec_layers:
            if not gdf.empty:
                gdf.to_file(tmp_gpkg.name, layer=name, driver="GPKG")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(tmp_gpkg.name, "catchment_results.gpkg")
            for arcname, data in raster_files:
                zf.writestr(arcname, data)
        os.unlink(tmp_gpkg.name)
        buf.seek(0)
        return send_file(buf, mimetype="application/zip",
                         as_attachment=True, download_name="catchment_results.zip")

    elif fmt == "shp":
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, gdf in vec_layers:
                if gdf.empty: continue
                with tempfile.TemporaryDirectory() as td:
                    p = os.path.join(td, f"{name}.shp")
                    gdf.to_file(p)
                    for ext in (".shp",".shx",".dbf",".prj",".cpg"):
                        fp = p.replace(".shp", ext)
                        if os.path.exists(fp):
                            zf.write(fp, f"{name}{ext}")
            for arcname, data in raster_files:
                zf.writestr(arcname, data)
        buf.seek(0)
        return send_file(buf, mimetype="application/zip",
                         as_attachment=True, download_name="catchment_results.zip")

    return jsonify({"error": f"Unknown format: {fmt}"}), 400


@app.route("/api/print_map", methods=["POST"])
def api_print_map():
    """Render a print-quality GIS map as PNG and return it."""
    data = request.json or {}
    if not _all_results:
        return jsonify({"error": "No delineation results to print."}), 400

    fig = None
    try:
        title       = data.get("title", "Catchment Map")
        outlet_info = data.get("outlets", [])      # [{id, name, color}]
        paper       = data.get("paper", "A4")
        orient      = data.get("orientation", "landscape")
        # Slider value sent by JS is the source of truth; fall back to stored
        # threshold from the last delineation if the request omits it
        _req_thresh = data.get("acc_threshold")
        acc_thresh  = int(_req_thresh) if _req_thresh is not None else (
            _all_results[-1]["stats"].get("acc_threshold") or 500)

        # Paper size (landscape by default; swap axes for portrait)
        sizes = {"A4": (13.69, 8.27), "A3": (18.54, 11.69), "Letter": (13.0, 8.5)}
        fw, fh = sizes.get(paper, sizes["A4"])
        if orient == "portrait":
            fw, fh = fh, fw

        # Build lookup by outlet_id (1-indexed)
        lookup = {int(o["id"]): o for o in outlet_info}

        # Assemble GeoDataFrames
        all_catches, all_outlet_feats = [], []
        for i, r in enumerate(_all_results):
            oid  = i + 1
            info = lookup.get(oid, {})
            name  = (info.get("name") or "").strip() or f"Catchment {oid}"
            color = info.get("color", "#3399ff")
            for feat in r["catchment"]["features"]:
                f2 = dict(feat)
                f2["properties"] = {**f2.get("properties", {}),
                                     "outlet_id": oid, "name": name, "color": color}
                all_catches.append(f2)
            for feat in r["outlet"]["features"]:
                f2 = dict(feat)
                f2["properties"] = {**f2.get("properties", {}),
                                     "outlet_id": oid, "name": name, "color": color}
                all_outlet_feats.append(f2)

        catches_gdf = gpd.GeoDataFrame.from_features(all_catches, crs=4326)
        outlets_gdf = gpd.GeoDataFrame.from_features(all_outlet_feats, crs=4326)

        # Union of all catchment polygons — used for clipping rivers & DEM
        catchment_union = catches_gdf.unary_union

        # Rivers clipped to catchment area only
        rivers_fc  = extract_rivers_global(acc_thresh)
        rivers_gdf_full = (gpd.GeoDataFrame.from_features(rivers_fc["features"], crs=4326)
                           if rivers_fc.get("features") else gpd.GeoDataFrame())
        rivers_gdf = (rivers_gdf_full.clip(catchment_union)
                      if not rivers_gdf_full.empty else gpd.GeoDataFrame())

        # Map extent: bounding box of all catchments + 12 % padding
        tb  = catches_gdf.total_bounds          # [minx, miny, maxx, maxy]
        pw  = (tb[2] - tb[0]) * 0.12
        ph  = (tb[3] - tb[1]) * 0.12
        xmin, xmax = tb[0] - pw, tb[2] + pw
        ymin, ymax = tb[1] - ph, tb[3] + ph

        # Force equal aspect: expand the shorter axis
        data_w = xmax - xmin
        data_h = ymax - ymin
        map_aspect = fw * 0.82 / (fh * 0.80)
        if data_w / data_h < map_aspect:
            delta = data_h * map_aspect - data_w
            xmin -= delta / 2; xmax += delta / 2
        else:
            delta = data_w / map_aspect - data_h
            ymin -= delta / 2; ymax += delta / 2

        # ── Figure ─────────────────────────────────────────────────────────
        RIVER_COLORS = ["#aacce8","#6aaed6","#3182bd","#1a5ea0","#0d3a78","#062050"]

        fig = plt.figure(figsize=(fw, fh), facecolor="white")
        ax  = fig.add_axes([0.09, 0.11, 0.82, 0.80])

        ax.set_facecolor("#dde8f0")   # fallback colour if tiles fail
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # ── Basemap tiles (CartoDB Positron — clean print style) ───────
        # Cap zoom so we never fetch hundreds of tiles for large extents.
        # Target ≤ ~8 tiles across; zoom = log2(8 * 360 / span), capped 7–11.
        # Zoom 11 → ~9 tiles for a 0.5° catchment; zoom 13 → ~90 tiles (too slow).
        _deg_span  = max(xmax - xmin, ymax - ymin)
        _map_zoom  = max(7, min(11, int(math.log2(max(8 * 360 / max(_deg_span, 0.001), 1)))))
        try:
            import contextily as ctx
            ctx.add_basemap(ax, crs="EPSG:4326",
                            source=ctx.providers.CartoDB.Positron,
                            attribution=False, zoom=_map_zoom)
        except Exception:
            pass   # offline / tile error — plain background remains
        # contextily may shift limits — restore our pre-computed extent
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Ticks only, no grid
        ax.tick_params(labelsize=7, direction="out", color="#777")
        ax.set_xlabel("Longitude (°E)", fontsize=8, color="#444", labelpad=4)
        ax.set_ylabel("Latitude (°N)",  fontsize=8, color="#444", labelpad=4)
        for spine in ax.spines.values():
            spine.set_edgecolor("#888")
            spine.set_linewidth(0.8)

        # ── DEM hillshade clipped to catchment polygon ─────────────────
        if _cache and "dem_arr" in _cache:
            try:
                _rgba, _hs_ext = _get_hillshade_rgba(catchment_union)
                if _rgba is not None:
                    ax.imshow(_rgba, extent=_hs_ext,
                              origin="upper", zorder=1, interpolation="nearest",
                              aspect="auto")
            except Exception:
                traceback.print_exc()   # log but don't abort the map

        # Catchment fills + dashed borders (on top of hillshade)
        for _, row in catches_gdf.iterrows():
            color = row.get("color", "#3399ff")
            gdf_r = gpd.GeoDataFrame([row], crs=4326)
            gdf_r.plot(ax=ax, color=color, alpha=0.12, linewidth=0, zorder=2)
            gdf_r.boundary.plot(ax=ax, color=color, linewidth=1.8,
                                linestyle="--", zorder=3)

        # River network (clipped to catchment)
        if not rivers_gdf.empty and "order" in rivers_gdf.columns:
            for order in sorted(rivers_gdf["order"].unique()):
                sub = rivers_gdf[rivers_gdf["order"] == order]
                lw  = max(0.5, int(order) * 0.6 + 0.4)
                sub.plot(ax=ax, color=RIVER_COLORS[min(int(order)-1, 5)],
                         linewidth=lw, zorder=4)

        # Outlet markers + name labels
        for _, row in outlets_gdf.iterrows():
            color = row.get("color", "#ff4444")
            name  = row.get("name", f"Outlet {row.get('outlet_id','')}")
            x, y  = row.geometry.x, row.geometry.y
            ax.plot(x, y, "o", color=color, markersize=7,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=6)
            ax.annotate(
                name, (x, y),
                textcoords="offset points", xytext=(8, 5),
                fontsize=7.5, fontweight="bold", color="#1a2030", zorder=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
            )

        # GIS map furniture
        _pm_north_arrow(ax)
        _pm_scale_bar(ax, xmin, xmax, ymin, ymax)
        _pm_legend(ax, outlet_info, rivers_gdf)
        _pm_locator_map(ax, catches_gdf, xmin, xmax, ymin, ymax)

        # Title + credits
        fig.text(0.5, 0.945, title, ha="center", va="center",
                 fontsize=15, fontweight="bold", color="#1a2030")
        fig.text(
            0.09, 0.030,
            "Projection: WGS84 (EPSG:4326)  ·  Stream network: Strahler ordering"
            "  ·  DEM: SRTM GL1 30 m  ·  Generated with Catchment Delineation Tool",
            ha="left", va="bottom", fontsize=6.0, color="#999", style="italic",
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110,
                    pil_kwargs={"compress_level": 0})
        buf.seek(0)
        return send_file(buf, mimetype="image/png", as_attachment=True,
                         download_name="catchment_map.png")

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
    finally:
        if fig is not None:
            plt.close(fig)


def _build_export_zip(data: dict) -> io.BytesIO:
    """Render map PNG + GeoJSON layers into a ZIP buffer. Raises on error."""
    if not _all_results:
        raise ValueError("No delineation results to export.")

    fig = None
    try:
        # ── Render PNG (same logic as api_print_map) ────────────────────────
        title       = data.get("title", "Catchment Map")
        outlet_info = data.get("outlets", [])
        paper       = data.get("paper", "A4")
        orient      = data.get("orientation", "landscape")
        _req_thresh = data.get("acc_threshold")
        acc_thresh  = int(_req_thresh) if _req_thresh is not None else (
            _all_results[-1]["stats"].get("acc_threshold") or 500)

        sizes = {"A4": (13.69, 8.27), "A3": (18.54, 11.69), "Letter": (13.0, 8.5)}
        fw, fh = sizes.get(paper, sizes["A4"])
        if orient == "portrait":
            fw, fh = fh, fw

        lookup = {int(o["id"]): o for o in outlet_info}
        all_catches, all_outlet_feats = [], []
        for i, r in enumerate(_all_results):
            oid  = i + 1
            info = lookup.get(oid, {})
            name  = (info.get("name") or "").strip() or f"Catchment {oid}"
            color = info.get("color", "#3399ff")
            for feat in r["catchment"]["features"]:
                f2 = dict(feat)
                f2["properties"] = {**f2.get("properties", {}),
                                     "outlet_id": oid, "name": name, "color": color}
                all_catches.append(f2)
            for feat in r["outlet"]["features"]:
                f2 = dict(feat)
                f2["properties"] = {**f2.get("properties", {}),
                                     "outlet_id": oid, "name": name, "color": color}
                all_outlet_feats.append(f2)

        catches_gdf = gpd.GeoDataFrame.from_features(all_catches, crs=4326)
        outlets_gdf = gpd.GeoDataFrame.from_features(all_outlet_feats, crs=4326)
        catchment_union = catches_gdf.unary_union

        rivers_fc   = extract_rivers_global(acc_thresh)
        rivers_gdf_full = (gpd.GeoDataFrame.from_features(rivers_fc["features"], crs=4326)
                           if rivers_fc.get("features") else gpd.GeoDataFrame())
        rivers_gdf = (rivers_gdf_full.clip(catchment_union)
                      if not rivers_gdf_full.empty else gpd.GeoDataFrame())

        tb = catches_gdf.total_bounds
        pw = (tb[2] - tb[0]) * 0.12; ph = (tb[3] - tb[1]) * 0.12
        xmin, xmax = tb[0] - pw, tb[2] + pw
        ymin, ymax = tb[1] - ph, tb[3] + ph
        data_w = xmax - xmin; data_h = ymax - ymin
        map_aspect = fw * 0.82 / (fh * 0.80)
        if data_w / data_h < map_aspect:
            delta = data_h * map_aspect - data_w; xmin -= delta/2; xmax += delta/2
        else:
            delta = data_w / map_aspect - data_h; ymin -= delta/2; ymax += delta/2

        RIVER_COLORS = ["#aacce8","#6aaed6","#3182bd","#1a5ea0","#0d3a78","#062050"]
        fig = plt.figure(figsize=(fw, fh), facecolor="white")
        ax  = fig.add_axes([0.09, 0.11, 0.82, 0.80])
        ax.set_facecolor("#dde8f0"); ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

        _deg_span = max(xmax - xmin, ymax - ymin)
        _map_zoom = max(7, min(11, int(math.log2(max(8 * 360 / max(_deg_span, 0.001), 1)))))
        try:
            import contextily as ctx
            ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron,
                            attribution=False, zoom=_map_zoom)
        except Exception:
            pass
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.tick_params(labelsize=7, direction="out", color="#777")
        ax.set_xlabel("Longitude (°E)", fontsize=8, color="#444", labelpad=4)
        ax.set_ylabel("Latitude (°N)",  fontsize=8, color="#444", labelpad=4)
        for spine in ax.spines.values():
            spine.set_edgecolor("#888"); spine.set_linewidth(0.8)

        if _cache and "dem_arr" in _cache:
            try:
                _rgba, _hs_ext = _get_hillshade_rgba(catchment_union)
                if _rgba is not None:
                    ax.imshow(_rgba, extent=_hs_ext, origin="upper", zorder=1,
                              interpolation="nearest", aspect="auto")
            except Exception:
                traceback.print_exc()

        for _, row in catches_gdf.iterrows():
            color = row.get("color", "#3399ff")
            gdf_r = gpd.GeoDataFrame([row], crs=4326)
            gdf_r.plot(ax=ax, color=color, alpha=0.12, linewidth=0, zorder=2)
            gdf_r.boundary.plot(ax=ax, color=color, linewidth=1.8, linestyle="--", zorder=3)

        if not rivers_gdf.empty and "order" in rivers_gdf.columns:
            for order in sorted(rivers_gdf["order"].unique()):
                sub = rivers_gdf[rivers_gdf["order"] == order]
                lw  = max(0.5, int(order) * 0.6 + 0.4)
                sub.plot(ax=ax, color=RIVER_COLORS[min(int(order)-1, 5)], linewidth=lw, zorder=4)

        for _, row in outlets_gdf.iterrows():
            color = row.get("color", "#ff4444")
            name  = row.get("name", f"Outlet {row.get('outlet_id','')}")
            x, y  = row.geometry.x, row.geometry.y
            ax.plot(x, y, "o", color=color, markersize=7,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=6)
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(8, 5),
                        fontsize=7.5, fontweight="bold", color="#1a2030", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

        _pm_north_arrow(ax); _pm_scale_bar(ax, xmin, xmax, ymin, ymax)
        _pm_legend(ax, outlet_info, rivers_gdf); _pm_locator_map(ax, catches_gdf, xmin, xmax, ymin, ymax)
        fig.text(0.5, 0.945, title, ha="center", va="center",
                 fontsize=15, fontweight="bold", color="#1a2030")
        fig.text(0.09, 0.030,
                 "Projection: WGS84 (EPSG:4326)  ·  Stream network: Strahler ordering"
                 "  ·  DEM: SRTM GL1 30 m  ·  Generated with Catchment Delineation Tool",
                 ha="left", va="bottom", fontsize=6.0, color="#999", style="italic")

        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=110, pil_kwargs={"compress_level": 0})
        png_buf.seek(0)
        png_bytes = png_buf.read()

        # ── Build GeoJSON layers ────────────────────────────────────────────
        rivers_fc_full = extract_rivers_global(acc_thresh)
        rivers_all_gdf = (gpd.GeoDataFrame.from_features(rivers_fc_full["features"], crs=4326)
                          if rivers_fc_full.get("features") else gpd.GeoDataFrame())

        def _gdf_to_geojson_bytes(gdf):
            return gdf.to_json(indent=2).encode("utf-8") if not gdf.empty else b'{"type":"FeatureCollection","features":[]}'

        # ── Assemble ZIP ────────────────────────────────────────────────────
        zip_buf = io.BytesIO()
        safe_title = re.sub(r'[^a-zA-Z0-9 _-]', '', title).strip() or "catchment_results"
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("catchment_map.png",   png_bytes)
            zf.writestr("catchments.geojson",  _gdf_to_geojson_bytes(catches_gdf))
            zf.writestr("outlets.geojson",     _gdf_to_geojson_bytes(outlets_gdf))
            zf.writestr("rivers.geojson",      _gdf_to_geojson_bytes(rivers_all_gdf))
        zip_buf.seek(0)
        return zip_buf, safe_title

    except Exception:
        raise
    finally:
        if fig is not None:
            plt.close(fig)


@app.route("/api/export_results", methods=["POST"])
def api_export_results():
    """Direct export (no payment) — kept for internal use / testing."""
    if not _all_results:
        return jsonify({"error": "No delineation results to export."}), 400
    try:
        zip_buf, safe_title = _build_export_zip(request.json or {})
        return send_file(zip_buf, mimetype="application/zip",
                         as_attachment=True, download_name=f"{safe_title}.zip")
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ── Stripe checkout ────────────────────────────────────────────────────────────

@app.route("/api/create_checkout_session", methods=["POST"])
def api_create_checkout_session():
    if not _stripe or not _stripe.api_key:
        return jsonify({"error": "Stripe is not configured on the server."}), 503
    if not _all_results:
        return jsonify({"error": "No delineation results — run the analysis first."}), 400

    params = request.get_json(force=True) or {}
    base   = request.host_url.rstrip("/")

    # Store params under a random token so the success redirect can retrieve them
    token = uuid.uuid4().hex
    _pending_exports[token] = params

    try:
        session = _stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency":    STRIPE_CURRENCY,
                    "unit_amount": STRIPE_PRICE_CENTS,
                    "product_data": {"name": STRIPE_PRODUCT_NAME},
                },
                "quantity": 1,
            }],
            mode="payment",
            success_url=f"{base}/payment/success?token={token}&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{base}/",
            metadata={"export_token": token},
        )
        return jsonify({"checkout_url": session.url})
    except Exception as e:
        _pending_exports.pop(token, None)
        return jsonify({"error": str(e)}), 500


@app.route("/payment/success")
def payment_success():
    token      = request.args.get("token", "")
    session_id = request.args.get("session_id", "")

    # Verify payment with Stripe
    error_msg = None
    if not _stripe or not _stripe.api_key:
        error_msg = "Stripe not configured."
    elif not session_id:
        error_msg = "Missing session ID."
    else:
        try:
            sess = _stripe.checkout.Session.retrieve(session_id)
            if sess.payment_status != "paid":
                error_msg = "Payment has not been completed."
        except Exception as e:
            error_msg = f"Could not verify payment: {e}"

    if error_msg:
        return f"""<!DOCTYPE html><html><head><title>Payment Error</title></head>
<body style="font-family:sans-serif;text-align:center;padding:60px">
  <h2 style="color:#c00">Payment verification failed</h2>
  <p>{error_msg}</p>
  <a href="/" style="color:#1a7acc">← Return to app</a>
</body></html>""", 402

    title = ((_pending_exports.get(token) or {}).get("title") or "catchment_results")
    safe_title = re.sub(r'[^a-zA-Z0-9 _-]', '', title).strip() or "catchment_results"
    download_url = f"/api/download_export/{token}"

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Payment successful — Catchment Results</title>
  <style>
    body {{ margin:0; font-family:'Segoe UI',sans-serif; background:#0b151f; color:#cdd9e5;
           display:flex; align-items:center; justify-content:center; min-height:100vh; }}
    .card {{ background:#0f1f2e; border:1px solid #1e3a55; border-radius:12px;
             padding:48px 56px; text-align:center; max-width:460px; }}
    h2 {{ color:#4fc97e; font-size:1.6rem; margin:0 0 12px; }}
    p  {{ color:#8dafc4; font-size:14px; margin:0 0 28px; line-height:1.6; }}
    .dl-btn {{ display:inline-block; background:#155f38; color:#fff; text-decoration:none;
               padding:13px 28px; border-radius:8px; font-weight:600; font-size:14px;
               transition:background .2s; }}
    .dl-btn:hover {{ background:#1f7046; }}
    .back {{ display:block; margin-top:18px; font-size:12px; color:#5a7f99; text-decoration:none; }}
    .back:hover {{ color:#8dafc4; }}
  </style>
</head>
<body>
  <div class="card">
    <div style="font-size:48px;margin-bottom:16px">✅</div>
    <h2>Payment successful!</h2>
    <p>Your catchment results are ready.<br>
       Click below to download the ZIP file containing the map image and GeoJSON layers.</p>
    <a class="dl-btn" href="{download_url}" download="{safe_title}.zip">
      ⬇ Download {safe_title}.zip
    </a>
    <a class="back" href="/">← Return to the app</a>
  </div>
  <script>
    // Auto-trigger download after a short delay
    setTimeout(() => window.location.href = '{download_url}', 1200);
  </script>
</body>
</html>"""


@app.route("/api/download_export/<token>")
def api_download_export(token):
    params = _pending_exports.get(token)
    if not params:
        return jsonify({"error": "Export not found or already downloaded. "
                                 "Return to the app and generate a new export."}), 404
    try:
        zip_buf, safe_title = _build_export_zip(params)
        _pending_exports.pop(token, None)   # single-use token
        return send_file(zip_buf, mimetype="application/zip",
                         as_attachment=True, download_name=f"{safe_title}.zip")
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ═══════════════════════════════════════════════════════════════════════════════
#  HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _rpt_compute_stats(catches_gdf, outlets_gdf):
    """Return per-catchment morphometric stats as a list of dicts."""
    cea = "+proj=cea +datum=WGS84"
    stats = []
    try:
        ids = catches_gdf["outlet_id"].unique() if "outlet_id" in catches_gdf.columns else []
    except Exception:
        ids = []
    for oid in sorted(ids):
        sub = catches_gdf[catches_gdf["outlet_id"] == oid]
        name  = str(sub.iloc[0].get("name",  f"Catchment {oid}")) if len(sub) else f"Catchment {oid}"
        color = str(sub.iloc[0].get("color", "#3399ff"))           if len(sub) else "#3399ff"
        try:
            merged_geom = sub.geometry.unary_union
            sub_union   = gpd.GeoDataFrame(geometry=[merged_geom], crs=4326)
            sub_cea     = sub_union.to_crs(cea)
            area_m2     = float(sub_cea.area.iloc[0])
            perim_m     = float(sub_cea.length.iloc[0])
            area_km2    = area_m2 / 1e6
            perim_km    = perim_m / 1e3
            circ        = (4 * math.pi * area_m2 / (perim_m ** 2)) if perim_m > 0 else 0.0
            b           = sub_union.total_bounds   # [minx, miny, maxx, maxy]
            diag_deg    = math.sqrt((b[2]-b[0])**2 + (b[3]-b[1])**2)
            diag_km     = diag_deg * 111.32
            elong       = (2 * math.sqrt(area_km2 / math.pi) / diag_km) if diag_km > 0 else 0.0
        except Exception:
            area_km2 = perim_km = circ = elong = 0.0

        # Outlet lon/lat
        lon_o, lat_o = None, None
        try:
            out_sub = outlets_gdf[outlets_gdf["outlet_id"] == oid]
            if not out_sub.empty:
                lon_o = round(float(out_sub.geometry.x.iloc[0]), 5)
                lat_o = round(float(out_sub.geometry.y.iloc[0]), 5)
        except Exception:
            pass

        # Elevation stats from DEM masked by catchment
        elev = {}
        if _cache and "dem_arr" in _cache:
            try:
                dem_arr  = _cache["dem_arr"]
                aff      = _cache["affine"]
                nr, nc   = dem_arr.shape
                merged_geom2 = sub.geometry.unary_union
                cmask = rasterio.features.rasterize(
                    [(merged_geom2, 1)],
                    out_shape=(nr, nc),
                    transform=aff,
                    fill=0, dtype="uint8",
                ).astype(bool)
                vals = dem_arr[cmask & np.isfinite(dem_arr)]
                if len(vals) > 0:
                    elev = {
                        "min":  round(float(np.percentile(vals, 2)),  1),
                        "max":  round(float(np.percentile(vals, 98)), 1),
                        "mean": round(float(np.mean(vals)),            1),
                        "std":  round(float(np.std(vals)),             1),
                    }
            except Exception:
                pass

        stats.append({
            "id":      int(oid),
            "name":    name,
            "color":   color,
            "area":    round(area_km2, 2),
            "perim":   round(perim_km, 2),
            "circ":    round(circ,     3),
            "elong":   round(elong,    3),
            "lon":     lon_o,
            "lat":     lat_o,
            "elev":    elev,
        })
    return stats


def _rpt_compute_river_stats(rivers_gdf):
    """Return per-Strahler-order stats as list of dicts."""
    if rivers_gdf.empty or "order" not in rivers_gdf.columns:
        return []
    cea = "+proj=cea +datum=WGS84"
    rows = []
    try:
        riv_cea = rivers_gdf.to_crs(cea)
    except Exception:
        riv_cea = rivers_gdf
    orders = sorted(rivers_gdf["order"].unique())
    counts = {}
    for ord_val in orders:
        sub = riv_cea[rivers_gdf["order"] == ord_val]
        total_m  = float(sub.geometry.length.sum())
        counts[int(ord_val)] = len(sub)
        rows.append({
            "order":    int(ord_val),
            "count":    len(sub),
            "total_km": round(total_m / 1e3, 2),
            "mean_km":  round(total_m / max(len(sub), 1) / 1e3, 3),
        })
    return rows


# ── figure → base64 ─────────────────────────────────────────────────────────
def _fig_to_b64(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi,
                facecolor=fig.get_facecolor(),
                pil_kwargs={"compress_level": 0})
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return data


# ── map image ─────────────────────────────────────────────────────────────────
def _rpt_html_map(outlet_info, acc_thresh, rivers_gdf_prebuilt=None):
    """Render the study-area map identical in style to the PNG export."""
    import contextily as ctx

    if not _cache or not _all_results:
        return ""

    RIVER_COLORS = ["#aacce8","#6aaed6","#3182bd","#1a5ea0","#0d3a78","#062050"]

    dem_arr = _cache.get("dem_arr")
    affine  = _cache["affine"]
    nr = nc = 0
    if dem_arr is not None:
        nr, nc = dem_arr.shape

    # ── Build catchment GeoDataFrames ──────────────────────────────────────
    all_catches, all_outlet_feats = [], []
    lookup = {int(o["id"]): o for o in outlet_info}
    for i, r in enumerate(_all_results):
        oid   = i + 1
        info  = lookup.get(oid, {})
        name  = (info.get("name") or "").strip() or f"Catchment {oid}"
        color = info.get("color", "#3399ff")
        for feat in r["catchment"]["features"]:
            f2 = dict(feat)
            f2["properties"] = {**f2.get("properties", {}),
                                 "outlet_id": oid, "name": name, "color": color}
            all_catches.append(f2)
        for feat in r["outlet"]["features"]:
            f2 = dict(feat)
            f2["properties"] = {**f2.get("properties", {}),
                                 "outlet_id": oid, "name": name, "color": color}
            all_outlet_feats.append(f2)

    catches_gdf = gpd.GeoDataFrame.from_features(all_catches,      crs=4326)
    outlets_gdf = gpd.GeoDataFrame.from_features(all_outlet_feats, crs=4326)
    catchment_union = catches_gdf.unary_union

    # ── Extent: catchment bbox + 12 % padding, equal aspect ───────────────
    tb  = catches_gdf.total_bounds          # [minx, miny, maxx, maxy]
    pw  = (tb[2] - tb[0]) * 0.12
    ph  = (tb[3] - tb[1]) * 0.12
    xmin, xmax = tb[0] - pw, tb[2] + pw
    ymin, ymax = tb[1] - ph, tb[3] + ph
    # Force equal aspect (approx — figure is 14×10)
    data_w, data_h = xmax - xmin, ymax - ymin
    map_aspect = 14 * 0.82 / (10 * 0.80)
    if data_w / max(data_h, 1e-9) < map_aspect:
        delta = data_h * map_aspect - data_w
        xmin -= delta / 2; xmax += delta / 2
    else:
        delta = data_w / map_aspect - data_h
        ymin -= delta / 2; ymax += delta / 2

    # ── Rivers clipped to catchment union ─────────────────────────────────
    # rivers_gdf_prebuilt is already clipped (done in api_report before calling us)
    if rivers_gdf_prebuilt is not None:
        rivers_gdf = rivers_gdf_prebuilt  # already clipped — use directly
    else:
        rivers_fc = extract_rivers_global(acc_thresh)
        rivers_gdf_full = (gpd.GeoDataFrame.from_features(rivers_fc["features"], crs=4326)
                           if rivers_fc.get("features") else gpd.GeoDataFrame())
        rivers_gdf = (rivers_gdf_full.clip(catchment_union)
                      if not rivers_gdf_full.empty else gpd.GeoDataFrame())

    # ── Figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10), facecolor="white")
    ax  = fig.add_axes([0.09, 0.11, 0.82, 0.80])

    ax.set_facecolor("#dde8f0")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Basemap — same zoom cap as print map (≤ 11, target 8 tiles across)
    _rdeg   = max(xmax - xmin, ymax - ymin)
    _rzoom  = max(7, min(11, int(math.log2(max(8 * 360 / max(_rdeg, 0.001), 1)))))
    try:
        ctx.add_basemap(ax, crs="EPSG:4326",
                        source=ctx.providers.CartoDB.Positron,
                        attribution=False, zoom=_rzoom)
    except Exception:
        pass
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

    ax.tick_params(labelsize=7, direction="out", color="#777")
    ax.set_xlabel("Longitude (°E)", fontsize=8, color="#444", labelpad=4)
    ax.set_ylabel("Latitude (°N)",  fontsize=8, color="#444", labelpad=4)
    for spine in ax.spines.values():
        spine.set_edgecolor("#888"); spine.set_linewidth(0.8)

    # ── DEM hillshade clipped to catchment (identical to PNG export) ──────
    if dem_arr is not None and nr > 0:
        try:
            _rgba, _hs_ext = _get_hillshade_rgba(catchment_union)
            if _rgba is not None:
                ax.imshow(_rgba, extent=_hs_ext,
                          origin="upper", zorder=1, interpolation="nearest",
                          aspect="auto")
        except Exception:
            import traceback as _tb; _tb.print_exc()

    # ── Catchment fills + dashed borders ──────────────────────────────────
    for _, row in catches_gdf.iterrows():
        color = row.get("color", "#3399ff")
        gdf_r = gpd.GeoDataFrame([row], crs=4326)
        gdf_r.plot(ax=ax, color=color, alpha=0.12, linewidth=0, zorder=2)
        gdf_r.boundary.plot(ax=ax, color=color, linewidth=1.8,
                            linestyle="--", zorder=3)

    # ── Rivers clipped, colored by Strahler order ──────────────────────────
    if not rivers_gdf.empty and "order" in rivers_gdf.columns:
        for order in sorted(rivers_gdf["order"].unique()):
            sub = rivers_gdf[rivers_gdf["order"] == order]
            lw  = max(0.5, int(order) * 0.6 + 0.4)
            sub.plot(ax=ax, color=RIVER_COLORS[min(int(order)-1, 5)],
                     linewidth=lw, zorder=4)

    # ── Outlet markers + labels ────────────────────────────────────────────
    for _, row in outlets_gdf.iterrows():
        color = row.get("color", "#e53935")
        name  = row.get("name", f"Outlet {row.get('outlet_id','')}")
        x, y  = row.geometry.x, row.geometry.y
        ax.plot(x, y, "o", color=color, markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, zorder=6)
        ax.annotate(
            name, (x, y), textcoords="offset points", xytext=(8, 5),
            fontsize=7.5, fontweight="bold", color="#1a2030", zorder=7,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
        )

    # ── Map furniture (north arrow, context map, legend) ───────────────────
    _pm_north_arrow(ax)
    _pm_locator_map(ax, catches_gdf, xmin, xmax, ymin, ymax)
    _pm_legend(ax, outlet_info, rivers_gdf)

    return _fig_to_b64(fig, dpi=110)



# ── Strahler bar charts ───────────────────────────────────────────────────────
def _rpt_html_strahler_charts(riv_stats):
    if not riv_stats:
        return ""
    NAV = "#0d2440"; BLU = "#1d5c9e"; GRN = "#1a5c38"; BG = "#f8fafc"
    orders  = [str(r["order"]) for r in riv_stats]
    counts  = [r["count"]     for r in riv_stats]
    lengths = [r["total_km"]  for r in riv_stats]
    means   = [r["mean_km"]   for r in riv_stats]

    def _ax_style(ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=11, fontweight="bold", color=NAV, pad=10)
        ax.set_xlabel(xlabel, fontsize=9, color="#445566")
        ax.set_ylabel(ylabel, fontsize=9, color="#445566")
        ax.set_facecolor(BG)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]: ax.spines[sp].set_color("#ccddee")
        ax.tick_params(colors="#445566", labelsize=8)
        ax.yaxis.grid(True, color="#e0eaf4", linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor="white")

    ax = axes[0]
    shades = plt.cm.Blues(np.linspace(0.45, 0.85, len(orders)))
    bars = ax.bar(orders, counts, color=shades, width=0.6, edgecolor="white", linewidth=0.8)
    _ax_style(ax, "Stream Segments by Order", "Strahler Order", "Number of Segments")
    for b, v in zip(bars, counts):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(counts)*0.025,
                str(v), ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=NAV)

    ax = axes[1]
    bars = ax.bar(orders, lengths, color=BLU, width=0.6, edgecolor="white", linewidth=0.8, alpha=0.85)
    _ax_style(ax, "Total Length by Order", "Strahler Order", "Total Length (km)")
    for b, v in zip(bars, lengths):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(lengths)*0.025,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=NAV)

    ax = axes[2]
    bars = ax.bar(orders, means, color=GRN, width=0.6, edgecolor="white", linewidth=0.8, alpha=0.85)
    _ax_style(ax, "Mean Segment Length by Order", "Strahler Order", "Mean Length (km)")
    for b, v in zip(bars, means):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(means)*0.025,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=GRN)

    plt.tight_layout(pad=2.5)
    return _fig_to_b64(fig, dpi=100)


# ── Horton log-linear charts ──────────────────────────────────────────────────
def _rpt_html_horton_chart(riv_stats):
    if len(riv_stats) < 2:
        return ""
    NAV = "#0d2440"; BLU = "#1d5c9e"
    orders  = [r["order"]    for r in riv_stats]
    counts  = [r["count"]    for r in riv_stats]
    lengths = [r["total_km"] for r in riv_stats]

    def _style(ax, title):
        ax.set_title(title, fontsize=11, fontweight="bold", color=NAV, pad=10)
        ax.set_facecolor("#f8fafc")
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]: ax.spines[sp].set_color("#ccddee")
        ax.tick_params(colors="#445566", labelsize=8)
        ax.yaxis.grid(True, color="#e0eaf4", linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), facecolor="white")

    ax1.semilogy(orders, counts, "o-", color=BLU, linewidth=2, markersize=7,
                 markerfacecolor="white", markeredgewidth=2)
    _style(ax1, "Horton's Law — Stream Numbers")
    ax1.set_xlabel("Strahler Order", fontsize=9, color="#445566")
    ax1.set_ylabel("Number of Streams (log scale)", fontsize=9, color="#445566")
    ax1.set_xticks(orders)

    ax2.semilogy(orders, lengths, "s-", color="#1a5c38", linewidth=2, markersize=7,
                 markerfacecolor="white", markeredgewidth=2)
    _style(ax2, "Horton's Law — Stream Lengths")
    ax2.set_xlabel("Strahler Order", fontsize=9, color="#445566")
    ax2.set_ylabel("Total Length (km, log scale)", fontsize=9, color="#445566")
    ax2.set_xticks(orders)

    plt.tight_layout(pad=2.5)
    return _fig_to_b64(fig, dpi=100)


# ── Elevation box plot ───────────────────────────────────────────────────────
def _rpt_html_elev_chart(catch_stats):
    """Scientific box plot of elevation distribution per catchment."""
    # Collect raw elevation values masked to each catchment
    stats_e = [s for s in catch_stats if s.get("elev")]
    if not stats_e or not _cache or "dem_arr" not in _cache:
        return ""

    dem_arr = _cache["dem_arr"]
    affine  = _cache["affine"]
    nr, nc  = dem_arr.shape
    all_catches = [feat for r in _all_results for feat in r["catchment"]["features"]]
    if not all_catches:
        return ""

    box_data = []
    labels   = []
    colors   = []
    try:
        cg = gpd.GeoDataFrame.from_features(all_catches, crs=4326)
        ids = sorted(cg["outlet_id"].unique()) if "outlet_id" in cg.columns else []
        for oid in ids:
            sub   = cg[cg["outlet_id"] == oid]
            name  = str(sub.iloc[0].get("name", f"Catchment {oid}"))[:20]
            color = str(sub.iloc[0].get("color", "#3399ff"))
            geom  = sub.geometry.unary_union
            mask  = rasterio.features.rasterize(
                [(geom, 1)], out_shape=(nr, nc),
                transform=affine, fill=0, dtype="uint8",
            ).astype(bool)
            vals = dem_arr[mask & np.isfinite(dem_arr)]
            if len(vals) > 10:
                # subsample to 50 k points for speed
                if len(vals) > 50000:
                    idx  = np.random.choice(len(vals), 50000, replace=False)
                    vals = vals[idx]
                box_data.append(vals)
                labels.append(name)
                colors.append(color)
    except Exception:
        import traceback; traceback.print_exc()
        return ""

    if not box_data:
        return ""

    NAV = "#0d2440"
    n   = len(box_data)
    fig, ax = plt.subplots(figsize=(max(5, n * 2.2 + 1.5), 5), facecolor="white")

    bp = ax.boxplot(
        box_data,
        patch_artist=True,
        notch=False,
        vert=True,
        widths=0.5,
        medianprops=dict(color="#e53935", linewidth=2),
        whiskerprops=dict(color="#445566", linewidth=1.2),
        capprops=dict(color="#445566", linewidth=1.5),
        flierprops=dict(marker=".", color="#aabbcc",
                        markersize=2, alpha=0.4, linestyle="none"),
        boxprops=dict(linewidth=1.2),
    )
    for patch, col in zip(bp["boxes"], colors):
        import matplotlib.colors as mc
        rgba = list(mc.to_rgba(col))
        rgba[3] = 0.25
        patch.set_facecolor(rgba)
        patch.set_edgecolor(col)

    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels(labels, rotation=15 if n > 3 else 0, ha="right", fontsize=9)
    ax.set_ylabel("Elevation (m a.s.l.)", fontsize=10)
    ax.set_title("Catchment Elevation Distribution", fontsize=12,
                 fontweight="bold", color=NAV, pad=10)
    ax.set_facecolor("#f8fafc")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: ax.spines[sp].set_color("#ccddee")
    ax.yaxis.grid(True, color="#e0eaf4", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#445566", labelsize=9)
    plt.tight_layout()
    return _fig_to_b64(fig, dpi=100)


# ── HTML helpers

# ── HTML helpers ──────────────────────────────────────────────────────────────
def _he(s):
    import html as _hm
    return _hm.escape(str(s))


def _build_html_report(title, author, date_str, catch_stats, riv_stats,
                        dem_source, res_m, acc_thresh, map_b64,
                        strahler_b64, elev_b64):

    # aggregates
    total_area = sum(s["area"]     for s in catch_stats)
    total_riv  = sum(r["total_km"] for r in riv_stats)
    max_order  = max((r["order"] for r in riv_stats), default=0)
    n_catch    = len(catch_stats)
    total_segs = sum(r["count"] for r in riv_stats)
    dd  = round(total_riv / total_area, 3) if total_area else 0
    fs  = round(total_segs / total_area, 3) if total_area else 0
    lo  = round(1 / (2 * dd), 3) if dd else 0

    # bifurcation ratios
    bif_rows = []
    rb_vals  = []
    for i in range(len(riv_stats) - 1):
        n1, n2 = riv_stats[i]["count"], riv_stats[i+1]["count"]
        rb = round(n1 / n2, 2) if n2 else None
        bif_rows.append((riv_stats[i]["order"], riv_stats[i+1]["order"], n1, n2, rb))
        if rb:
            rb_vals.append(rb)
    mean_rb = round(sum(rb_vals) / len(rb_vals), 2) if rb_vals else None

    def _interp_circ(rc):
        if rc > 0.75: return "Near-circular — high runoff potential"
        if rc > 0.50: return "Moderately elongated — moderate runoff"
        return "Elongated — lower flood peaks, longer lag time"

    def _interp_dd(d):
        if d < 0.5:  return "Very coarse — permeable soils, low relief"
        if d < 2.0:  return "Moderate — typical mixed terrain"
        if d < 4.0:  return "Fine — impermeable soils, higher runoff"
        return "Very fine — highly impermeable, flash-flood prone"

    def _interp_rb(rb):
        if rb is None: return "—"
        if rb < 3:    return "< 3 — unusually low (check data)"
        if rb <= 5:   return "3–5 — normal, homogeneous geology"
        return "> 5 — structural / geological control"

    def embed(b64, cls="chart-img"):
        if not b64: return ""
        return f'<img src="data:image/png;base64,{b64}" class="{cls}" alt="">'

    # page chrome helper
    def phdr(section, pn, tp):
        return (f'<div class="page-header">'
                f'<span class="report-title">{_he(title)}</span>'
                f'<span class="page-label">{_he(section)}'
                f' &nbsp;|&nbsp; Page {pn} of {tp}</span></div>')

    def kpi(val, unit, label):
        return (f'<div class="kpi-card">'
                f'<div class="kpi-value">{_he(val)}'
                f'<span class="kpi-unit">{_he(unit)}</span></div>'
                f'<div class="kpi-label">{_he(label)}</div></div>')

    TP = 5  # total pages

    # ── PAGE 1: COVER ─────────────────────────────────────────────────────────
    cover_rows = ""
    for s in catch_stats:
        dot     = f'<span class="color-dot" style="background:{_he(s["color"])}"></span>'
        e       = s.get("elev", {})
        elev_s  = f'{e["min"]:.0f}–{e["max"]:.0f} m' if e else "—"
        cover_rows += (
            f'<tr><td>{dot}{_he(s["name"])}</td>'
            f'<td class="num">{s["area"]:,.2f}</td>'
            f'<td class="num">{s["perim"]:,.1f}</td>'
            f'<td class="num">{s["circ"]:.3f}</td>'
            f'<td class="num">{s["elong"]:.3f}</td>'
            f'<td>{elev_s}</td></tr>'
        )

    p1 = f"""
<div class="page">
  <div class="cover-hero">
    <div class="cover-eyebrow">Hydrological Catchment Analysis Report</div>
    <div class="cover-h1">{_he(title)}</div>
    <div class="cover-meta">
      <span>{_he(author) if author else ""}</span>
      <span>{_he(date_str)}</span>
      <span>{n_catch} catchment{"s" if n_catch != 1 else ""} delineated</span>
    </div>
  </div>
  <div class="kpi-row">
    {kpi(f"{total_area:,.1f}", " km²", "Total Area")}
    {kpi(f"{total_riv:,.1f}", " km", "Total River Length")}
    {kpi(str(max_order), "", "Max Strahler Order")}
    {kpi(str(total_segs), "", "Stream Segments")}
    {kpi(f"{dd}", " km/km²", "Drainage Density")}
    {kpi(f"{res_m}", " m", "DEM Resolution")}
  </div>
  <div class="page-body">
    <div class="section-title">Summary</div>
    <div class="info-box">
      <strong>DEM Source:</strong> {_he(dem_source)} &nbsp;·&nbsp;
      <strong>Resolution:</strong> {res_m} m &nbsp;·&nbsp;
      <strong>Drainage Density:</strong> {dd} km/km² &nbsp;·&nbsp;
      <strong>Stream Frequency:</strong> {fs} /km² &nbsp;·&nbsp;
      <strong>Mean Bifurcation Ratio:</strong> {mean_rb if mean_rb else "—"}
    </div>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>Catchment</th><th>Area (km²)</th><th>Perimeter (km)</th>
          <th>Circularity Rc</th><th>Elongation Re</th><th>Elevation Range</th>
        </tr></thead>
        <tbody>{cover_rows}</tbody>
      </table>
    </div>
    <p style="font-size:12px;color:#9aa;margin-top:12px">
      Generated by Catchment Delineation Tool &nbsp;·&nbsp; {_he(date_str)}
    </p>
  </div>
</div>"""

    # ── PAGE 2: MAP ────────────────────────────────────────────────────────────
    outlet_rows = ""
    for s in catch_stats:
        dot   = f'<span class="color-dot" style="background:{_he(s["color"])}"></span>'
        lat_s = f'{s["lat"]:.5f}' if s.get("lat") is not None else "—"
        lon_s = f'{s["lon"]:.5f}' if s.get("lon") is not None else "—"
        outlet_rows += (
            f'<tr><td>{dot}{_he(s["name"])}</td>'
            f'<td class="num">{lat_s}</td>'
            f'<td class="num">{lon_s}</td>'
            f'<td class="num">{s["area"]:,.2f}</td></tr>'
        )

    p2 = f"""
<div class="page">
  {phdr("Study Area Map", 2, TP)}
  <div class="page-body">
    <div class="section-title">Study Area Map</div>
    <div class="section-subtitle">DEM hillshade · delineated catchment boundaries · stream network</div>
    {embed(map_b64, "map-img")}
    <p class="fig-caption">
      Figure 1. Delineated catchment(s) overlaid on DEM hillshade (CartoDB Positron basemap).
      Red circles mark outlet locations. River network line weight scales with Strahler order.
      Source: {_he(dem_source)}.
    </p>
    <div class="tbl-wrap" style="margin-top:20px">
      <table>
        <thead><tr>
          <th>Catchment</th><th>Outlet Latitude</th><th>Outlet Longitude</th><th>Area (km²)</th>
        </tr></thead>
        <tbody>{outlet_rows}</tbody>
      </table>
    </div>
  </div>
</div>"""

    # ── PAGE 3: MORPHOMETRICS ──────────────────────────────────────────────────
    morph_rows = ""
    for s in catch_stats:
        dot = f'<span class="color-dot" style="background:{_he(s["color"])}"></span>'
        lb  = s["perim"] / math.pi if s["perim"] else 1
        ff  = round(s["area"] / (lb ** 2), 4) if lb else 0
        morph_rows += (
            f'<tr><td>{dot}{_he(s["name"])}</td>'
            f'<td class="num">{s["area"]:,.2f}</td>'
            f'<td class="num">{s["perim"]:,.2f}</td>'
            f'<td class="num">{s["circ"]:.3f}</td>'
            f'<td class="num">{s["elong"]:.3f}</td>'
            f'<td class="num">{ff:.4f}</td>'
            f'<td style="font-size:11px;color:#5a6a7a">{_interp_circ(s["circ"])}</td></tr>'
        )

    elev_rows = ""
    for s in catch_stats:
        dot = f'<span class="color-dot" style="background:{_he(s["color"])}"></span>'
        e   = s.get("elev", {})
        if e:
            rng = round(e["max"] - e["min"], 1)
            elev_rows += (
                f'<tr><td>{dot}{_he(s["name"])}</td>'
                f'<td class="num">{e["min"]:.1f}</td>'
                f'<td class="num">{e["mean"]:.1f}</td>'
                f'<td class="num">{e["max"]:.1f}</td>'
                f'<td class="num">{e.get("std", 0):.1f}</td>'
                f'<td class="num">{rng:.1f}</td></tr>'
            )
        else:
            elev_rows += (
                f'<tr><td>{dot}{_he(s["name"])}</td>'
                f'<td class="num" colspan="5">No elevation data</td></tr>'
            )

    p3 = f"""
<div class="page">
  {phdr("Morphometric Analysis", 3, TP)}
  <div class="page-body">
    <div class="section-title">Catchment Morphometric Analysis</div>
    <div class="section-subtitle">
      Areal, linear and relief parameters following Horton (1945), Miller (1953) and Schumm (1956)
    </div>

    <h3 class="sub-h">Shape &amp; Areal Parameters</h3>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>Catchment</th><th>Area (km²)</th><th>Perimeter (km)</th>
          <th>Circularity R<sub>c</sub></th><th>Elongation R<sub>e</sub></th>
          <th>Form Factor F<sub>f</sub></th><th>Interpretation</th>
        </tr></thead>
        <tbody>{morph_rows}</tbody>
      </table>
    </div>
    <div class="info-box">
      R<sub>c</sub> = 4π·A/P² &nbsp;|&nbsp; R<sub>e</sub> = (2/L)·√(A/π) &nbsp;|&nbsp; F<sub>f</sub> = A/L²<br>
      <span style="font-size:12px;color:#3a5a7a">
        R<sub>c</sub> &gt; 0.75 = circular, high runoff &nbsp;|&nbsp;
        R<sub>e</sub> &gt; 0.9 = circular &nbsp;|&nbsp;
        R<sub>e</sub> 0.5–0.7 = elongated &nbsp;|&nbsp;
        High F<sub>f</sub> = compact basin, higher peak discharge
      </span>
    </div>

    <h3 class="sub-h">Basin-wide Linear Parameters</h3>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>Parameter</th><th>Symbol</th><th class="num">Value</th><th>Unit</th><th>Interpretation</th>
        </tr></thead>
        <tbody>
          <tr><td>Drainage Density</td><td><i>D<sub>d</sub></i></td>
              <td class="num">{dd}</td><td>km/km²</td>
              <td style="font-size:11px;color:#5a6a7a">{_interp_dd(dd)}</td></tr>
          <tr><td>Stream Frequency</td><td><i>F<sub>s</sub></i></td>
              <td class="num">{fs}</td><td>/km²</td>
              <td style="font-size:11px;color:#5a6a7a">Higher → greater basin dissection &amp; surface runoff</td></tr>
          <tr><td>Length of Overland Flow</td><td><i>L<sub>o</sub></i></td>
              <td class="num">{lo}</td><td>km</td>
              <td style="font-size:11px;color:#5a6a7a">Mean travel distance before entering a channel ≈ 1/(2D<sub>d</sub>)</td></tr>
          <tr><td>Mean Bifurcation Ratio</td><td><i>R&#773;<sub>b</sub></i></td>
              <td class="num">{mean_rb if mean_rb else "—"}</td><td>—</td>
              <td style="font-size:11px;color:#5a6a7a">{_interp_rb(mean_rb)}</td></tr>
        </tbody>
      </table>
    </div>

    <h3 class="sub-h">Elevation Statistics</h3>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>Catchment</th><th>Min (m)</th><th>Mean (m)</th>
          <th>Max (m)</th><th>Std Dev (m)</th><th>Relief (m)</th>
        </tr></thead>
        <tbody>{elev_rows}</tbody>
      </table>
    </div>
    {embed(elev_b64)}
    <p class="fig-caption">Figure 2. Catchment elevation range — diamonds = mean; bars = min/max.</p>
  </div>
</div>"""

    # ── PAGE 4: RIVER NETWORK ──────────────────────────────────────────────────
    strahler_rows = ""
    for i, r in enumerate(riv_stats):
        rb_str = str(bif_rows[i][4]) if i < len(bif_rows) and bif_rows[i][4] else "—"
        strahler_rows += (
            f'<tr><td style="font-weight:700;color:#0d2440">Order {r["order"]}</td>'
            f'<td class="num">{r["count"]}</td>'
            f'<td class="num">{r["total_km"]:,.2f}</td>'
            f'<td class="num">{r["mean_km"]:.3f}</td>'
            f'<td class="num">{rb_str}</td></tr>'
        )
    strahler_rows += (
        f'<tr style="font-weight:700;background:#e8f0fb">'
        f'<td>Total</td><td class="num">{total_segs}</td>'
        f'<td class="num">{total_riv:,.2f}</td><td class="num">—</td>'
        f'<td class="num">{mean_rb if mean_rb else "—"} (mean)</td></tr>'
    )

    if mean_rb:
        if mean_rb < 3:
            bif_interp = "The mean bifurcation ratio is below 3, which may indicate flat terrain or a disturbed drainage network."
        elif mean_rb <= 5:
            bif_interp = (f"A mean R&#773;<sub>b</sub> of {mean_rb} is within the normal range (3–5), "
                          "indicating homogeneous geology with limited structural control on drainage.")
        else:
            bif_interp = (f"A mean R&#773;<sub>b</sub> of {mean_rb} exceeds 5, suggesting strong structural "
                          "or geological control, or an elongated basin geometry.")
    else:
        bif_interp = "Insufficient orders to compute bifurcation ratio."

    p4 = f"""
<div class="page">
  {phdr("River Network Analysis", 4, TP)}
  <div class="page-body">
    <div class="section-title">River Network Analysis</div>
    <div class="section-subtitle">Strahler ordering · bifurcation ratios · Horton's laws of drainage composition</div>

    <p style="font-size:13px;margin-bottom:18px">
      Stream ordering follows the <strong>Strahler (1952)</strong> system. First-order streams are
      fingertip tributaries carrying no tributaries. When two streams of equal order meet, the
      downstream reach takes the next higher order. The Strahler order at the outlet defines the
      overall order of the catchment.
    </p>

    <h3 class="sub-h">Stream Network Statistics by Order</h3>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>Stream Order</th><th>Segment Count</th><th>Total Length (km)</th>
          <th>Mean Length (km)</th><th>Bifurcation Ratio R<sub>b</sub></th>
        </tr></thead>
        <tbody>{strahler_rows}</tbody>
      </table>
    </div>
    <p style="font-size:11px;color:#7a8a9a;margin-bottom:18px">
      R<sub>b</sub>(n) = N<sub>n</sub> / N<sub>n+1</sub>.
      Normal range 3–5 (Strahler 1957). Values &gt; 5 indicate structural control or elongated basin.
    </p>

    {embed(strahler_b64)}
    <p class="fig-caption">Figure 3. Stream segment count, total length, and mean segment length by Strahler order.</p>

    <div class="info-box" style="margin-top:24px">
      <strong>Bifurcation analysis:</strong> {bif_interp}<br>
      <strong>Drainage density D<sub>d</sub></strong> = {dd} km/km² — {_interp_dd(dd).lower()}.<br>
      <strong>Stream frequency F<sub>s</sub></strong> = {fs} /km²
      {'— high dissection, significant surface runoff.' if fs > 2 else '— moderate to low dissection.'}
    </div>


  </div>
</div>"""

    # ── PAGE 5: METHODOLOGY ────────────────────────────────────────────────────
    p5 = f"""
<div class="page">
  {phdr("Methodology & References", 5, TP)}
  <div class="page-body">
    <div class="section-title">Methodology, Parameters &amp; Data Sources</div>
    <div class="section-subtitle">
      Complete processing pipeline with all hyperparameters, thresholds, and design decisions
    </div>

    <h3 class="sub-h">Data Sources</h3>
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>Dataset</th><th>Source</th><th>Resolution / Scale</th><th>Use</th></tr></thead>
        <tbody>
          <tr><td>Digital Elevation Model</td><td>{_he(dem_source)}</td><td>{res_m} m</td><td>Terrain, flow routing, morphometry</td></tr>
          <tr><td>Basemap tiles</td><td>CartoDB Positron (via contextily)</td><td>Web tiles</td><td>Cartographic background</td></tr>
          <tr><td>Country boundaries</td><td>Natural Earth 110 m</td><td>1:110,000,000</td><td>Context map inset</td></tr>
        </tbody>
      </table>
    </div>

    <h3 class="sub-h">Run Parameters (this analysis)</h3>
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>Parameter</th><th>Value Used</th><th>Effect / Purpose</th></tr></thead>
        <tbody>
          <tr><td>DEM resolution</td><td><strong>{res_m} m</strong></td>
              <td>Pixel size of the elevation grid. Finer = more detail but slower conditioning.</td></tr>
          <tr><td>Flow accumulation threshold</td><td><strong>{acc_thresh} cells</strong>
              &nbsp;≈ {round(acc_thresh * res_m * res_m / 1e6, 3)} km²</td>
              <td>Minimum upstream area for a cell to be classified as a stream channel.
                  Lower → denser network; higher → only main channels. Typical range: 200–2000 cells.</td></tr>
          <tr><td>Walling elevation offset</td><td><strong>max(DEM) + 5 000 m</strong></td>
              <td>Cells outside the user-drawn polygon are raised to this value so flow cannot
                  leak across the boundary during depression filling.</td></tr>
          <tr><td>D8 direction mapping</td><td><strong>ESRI / TauDEM convention</strong>
              (1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE)</td>
              <td>Each cell is assigned one of 8 cardinal/diagonal flow directions.</td></tr>
          <tr><td>Chaikin smoothing iterations</td><td><strong>3 passes</strong></td>
              <td>Corner-cutting algorithm applied to river polylines to remove staircase
                  artefacts from raster skeletonisation. Each pass cuts 25/75 % points.</td></tr>
          <tr><td>Catchment extent padding</td><td><strong>12 % on each side</strong></td>
              <td>Map extent extended beyond the catchment bounding box for cartographic context.</td></tr>
          <tr><td>Morphometric projection</td><td><strong>Cylindrical Equal-Area (+proj=cea +datum=WGS84)</strong></td>
              <td>Equal-area projection for accurate km² / km measurements independent of latitude.</td></tr>
          <tr><td>Elevation percentile range</td><td><strong>2nd – 98th percentile</strong></td>
              <td>Colour stretch limits for DEM hillshade to suppress outliers at cloud/void edges.</td></tr>
          <tr><td>DEM hillshade azimuth / altitude</td><td><strong>315° / 40°</strong></td>
              <td>Illumination angle (NW sun, standard cartographic convention) and sun elevation.</td></tr>
          <tr><td>Hillshade blend opacity</td><td><strong>65 %</strong></td>
              <td>Alpha of DEM layer over the CartoDB basemap — enough to show relief without obscuring labels.</td></tr>
          <tr><td>Vertical exaggeration</td><td><strong>1.5×</strong></td>
              <td>Applied to the hillshade surface for visual relief; does not affect metric calculations.</td></tr>
          <tr><td>Outlet snapping</td><td><strong>Snap to highest-accumulation cell within search radius</strong></td>
              <td>User-placed outlet is moved to the nearest high-accumulation cell to ensure
                  it lies on the modelled channel and not on an inter-fluve.</td></tr>
        </tbody>
      </table>
    </div>

    <h3 class="sub-h">Processing Pipeline</h3>
    <ol class="steps-list">
      <li>
        <strong>DEM acquisition &amp; normalisation</strong> —
        Elevation data downloaded from {_he(dem_source)} for the user-defined polygon extent.
        NoData values (≤ −9990, ≤ −32767, &gt; 9000 m) and any tagged nodata pixels are masked to NaN.
      </li>
      <li>
        <strong>CRS assignment</strong> —
        If the input raster has no embedded coordinate reference system, WGS84 (EPSG:4326)
        is assumed and written into the file metadata.
      </li>
      <li>
        <strong>Boundary walling</strong> —
        Cells outside the drawn polygon are set to
        <em>max(DEM) + 5 000 m</em>, creating an impenetrable wall.
        This prevents the depression-filling algorithm from routing flow across the
        study boundary and ensures that all delineated catchments remain
        entirely within the user-specified area of interest.
      </li>
      <li>
        <strong>Pit filling</strong> (<code>pysheds.Grid.fill_pits</code>) —
        Single-cell depressions (pits) — isolated pixels lower than all 8 neighbours — are
        raised to the minimum neighbouring elevation so D8 routing can proceed
        without terminating at isolated sinks.
      </li>
      <li>
        <strong>Depression filling</strong> (<code>pysheds.Grid.fill_depressions</code>) —
        Multi-cell enclosed basins are resolved using the <strong>priority-flood algorithm</strong>
        (Planchon &amp; Darboux 2001). The algorithm uses a min-heap priority queue to flood
        depressions from their lowest pour point upward, guaranteeing a continuous flow path
        from every cell to the boundary. This is the most computationally expensive step
        (O(N log N), visits every cell at least once); for a {res_m} m DEM at this extent
        it typically dominates total processing time.
      </li>
      <li>
        <strong>Flat resolution</strong> (<code>pysheds.Grid.resolve_flats</code>) —
        After depression filling, large plateau areas have no elevation gradient and cannot
        be assigned a unique D8 direction. A small artificial gradient is added following the
        Barnes et al. (2014) approach, ensuring flow is directed away from high ground
        and towards pour points.
      </li>
      <li>
        <strong>D8 flow direction</strong> (<code>pysheds.Grid.flowdir</code>) —
        Each raster cell is assigned a flow direction to the steepest of its 8 neighbouring
        cells. Power-of-two encoding is used (1, 2, 4, 8, 16, 32, 64, 128 for E→NE clockwise).
        Ties are broken by the first direction encountered in scanning order.
      </li>
      <li>
        <strong>Flow accumulation</strong> (<code>pysheds.Grid.accumulation</code>) —
        A topological traversal (from highest to lowest cell) counts the number of upstream
        contributing cells for every pixel. The resulting raster is the primary input for
        both stream network extraction and outlet snapping.
      </li>
      <li>
        <strong>Outlet snapping</strong> —
        The user-placed outlet point is relocated to the cell with the highest flow
        accumulation within a local search window, ensuring it sits on the modelled channel
        rather than on an adjacent hillslope cell.
      </li>
      <li>
        <strong>Catchment delineation</strong> (<code>pysheds.Grid.catchment</code>) —
        Starting from the snapped outlet, the flow-direction grid is traced upstream
        (breadth-first search) to identify all contributing cells.
        The resulting binary raster is vectorised to a polygon.
      </li>
      <li>
        <strong>Stream network extraction</strong> —
        Cells with flow accumulation ≥ <strong>{acc_thresh} cells</strong>
        (≈ {round(acc_thresh * res_m * res_m / 1e6, 3)} km²) are classified as stream channels.
        Connected groups of channel cells are skeletonised and traced into polyline segments
        by following the highest-accumulation neighbour at each junction.
        Segments are then smoothed using <strong>Chaikin corner-cutting (3 iterations)</strong>
        to remove staircase artefacts from the raster skeleton.
      </li>
      <li>
        <strong>Strahler stream ordering</strong> —
        Order 1 is assigned to all first-order tributaries (segments with no upstream tributaries).
        When two segments of equal order n meet, the downstream segment receives order n+1.
        When two segments of unequal order meet, the downstream segment inherits the higher order.
        This is computed as a recursive traversal of the extracted network graph.
      </li>
      <li>
        <strong>Morphometric computation</strong> —
        Basin area and perimeter are computed in the
        <strong>Cylindrical Equal-Area projection</strong>
        (<code>+proj=cea +datum=WGS84</code>) to ensure accurate measurements at any latitude.
        Elevation statistics are derived from the raw DEM pixel values masked to the
        catchment polygon, using the 2nd and 98th percentile to exclude voids.
      </li>
    </ol>

    <h3 class="sub-h">Morphometric Formulas Reference</h3>
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>Parameter</th><th>Symbol</th><th>Formula</th><th>Typical range</th><th>Reference</th></tr></thead>
        <tbody>
          <tr><td>Circularity ratio</td><td>R<sub>c</sub></td><td>4π·A / P²</td><td>0–1 (1 = circle)</td><td>Miller (1953)</td></tr>
          <tr><td>Elongation ratio</td><td>R<sub>e</sub></td><td>(2/L<sub>b</sub>)·√(A/π)</td><td>0.6–1.0</td><td>Schumm (1956)</td></tr>
          <tr><td>Form factor</td><td>F<sub>f</sub></td><td>A / L<sub>b</sub>²</td><td>0–0.79</td><td>Horton (1932)</td></tr>
          <tr><td>Drainage density</td><td>D<sub>d</sub></td><td>Σ L<sub>u</sub> / A</td><td>0.5–10 km/km²</td><td>Horton (1945)</td></tr>
          <tr><td>Stream frequency</td><td>F<sub>s</sub></td><td>Σ N<sub>u</sub> / A</td><td>varies</td><td>Horton (1945)</td></tr>
          <tr><td>Length of overland flow</td><td>L<sub>o</sub></td><td>1 / (2 D<sub>d</sub>)</td><td>—</td><td>Horton (1945)</td></tr>
          <tr><td>Bifurcation ratio</td><td>R<sub>b</sub></td><td>N<sub>u</sub> / N<sub>u+1</sub></td><td>3–5 (normal)</td><td>Strahler (1957)</td></tr>
        </tbody>
      </table>
    </div>

    <h3 class="sub-h">Software &amp; Libraries</h3>
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>Library</th><th>Role</th></tr></thead>
        <tbody>
          <tr><td>pysheds</td><td>DEM conditioning, D8 flow direction, accumulation, catchment delineation</td></tr>
          <tr><td>rasterio</td><td>Raster I/O, coordinate transforms, polygon rasterisation</td></tr>
          <tr><td>geopandas / shapely</td><td>Vector operations, area/perimeter in equal-area projection, river clipping</td></tr>
          <tr><td>numpy</td><td>Array-level DEM manipulation and statistics</td></tr>
          <tr><td>matplotlib</td><td>Map rendering, chart generation, PDF/PNG output</td></tr>
          <tr><td>contextily</td><td>Web map tile retrieval (CartoDB Positron basemap)</td></tr>
          <tr><td>Flask</td><td>Web application server and API endpoints</td></tr>
        </tbody>
      </table>
    </div>

    <h3 class="sub-h">References</h3>
    <ul class="refs">
      <li>Barnes, R., Lehman, C. &amp; Mulla, D. (2014). Priority-flood: An optimal depression-filling and watershed-labeling algorithm for digital elevation models. <em>Computers &amp; Geosciences</em>, 62, 117–127.</li>
      <li>Horton, R.E. (1932). Drainage basin characteristics. <em>Trans. Am. Geophys. Union</em>, 13, 350–361.</li>
      <li>Horton, R.E. (1945). Erosional development of streams and their drainage basins. <em>Bull. Geol. Soc. Am.</em>, 56, 275–370.</li>
      <li>Miller, V.C. (1953). A quantitative geomorphic study of drainage basin characteristics in the Clinch Mountain area, Virginia and Tennessee. <em>Technical Report</em>, Columbia University.</li>
      <li>Planchon, O. &amp; Darboux, F. (2001). A fast, simple and versatile algorithm to fill the depressions of digital elevation models. <em>Catena</em>, 46(2–3), 159–176.</li>
      <li>Schumm, S.A. (1956). Evolution of drainage systems and slopes in badlands at Perth Amboy, New Jersey. <em>Bull. Geol. Soc. Am.</em>, 67, 597–646.</li>
      <li>Strahler, A.N. (1952). Hypsometric (area-altitude) analysis of erosional topography. <em>Bull. Geol. Soc. Am.</em>, 63, 1117–1142.</li>
      <li>Strahler, A.N. (1957). Quantitative analysis of watershed geomorphology. <em>Trans. Am. Geophys. Union</em>, 38, 913–920.</li>
    </ul>

    <p style="font-size:11px;color:#aab;margin-top:28px;text-align:center">
      Generated by Catchment Delineation Tool &nbsp;·&nbsp; {_he(date_str)}
    </p>
  </div>
</div>"""


    # ── CSS ────────────────────────────────────────────────────────────────────
    css = """
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
      background: #eef2f7;
      color: #1a2030;
      font-size: 14px;
      line-height: 1.65;
    }
    .page {
      max-width: 980px;
      margin: 32px auto;
      background: #ffffff;
      box-shadow: 0 2px 24px rgba(0,0,0,.11);
      border-radius: 5px;
      overflow: hidden;
    }
    .page-header {
      background: #0d2440;
      color: #fff;
      padding: 14px 36px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .page-header .report-title { font-size: 12.5px; font-weight: 600; opacity: .9; }
    .page-header .page-label   { font-size: 11.5px; color: #7ab0d0; }
    .page-body { padding: 36px 40px; }
    .section-title {
      font-size: 21px; font-weight: 800; color: #0d2440;
      padding-bottom: 8px; border-bottom: 3px solid #1d5c9e;
      margin-bottom: 4px;
    }
    .section-subtitle { font-size: 13px; color: #7a8a9a; margin-bottom: 24px; }
    .sub-h { font-size: 14px; font-weight: 700; color: #0d2440; margin: 22px 0 10px; }

    /* cover */
    .cover-hero {
      background: linear-gradient(135deg, #0d2440 0%, #1d4a8a 100%);
      color: #fff; padding: 52px 48px 44px;
    }
    .cover-eyebrow { font-size: 11px; letter-spacing: .13em; text-transform: uppercase; color: #7ab0d0; margin-bottom: 12px; }
    .cover-h1 { font-size: 30px; font-weight: 800; line-height: 1.2; margin-bottom: 14px; }
    .cover-meta { font-size: 13px; color: #b0c8e4; }
    .cover-meta span { margin-right: 22px; }

    /* KPI row */
    .kpi-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 14px;
      padding: 24px 40px 28px;
      background: #f4f7fb;
    }
    .kpi-card { background: #fff; border: 1px solid #dde4f0; border-radius: 6px; padding: 16px 18px; text-align: center; }
    .kpi-value { font-size: 24px; font-weight: 800; color: #0d2440; }
    .kpi-unit  { font-size: 11px; color: #7a8a9a; margin-left: 2px; }
    .kpi-label { font-size: 10.5px; color: #7a8a9a; margin-top: 4px; text-transform: uppercase; letter-spacing: .06em; }

    /* tables */
    .tbl-wrap { overflow-x: auto; margin: 14px 0 22px; border-radius: 5px; border: 1px solid #dde4f0; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    thead th { background: #0d2440; color: #fff; padding: 9px 13px; text-align: left; font-weight: 600; font-size: 11.5px; letter-spacing: .04em; white-space: nowrap; }
    tbody tr:nth-child(even) { background: #f4f7fb; }
    tbody tr:hover           { background: #e8f0fb; }
    tbody td { padding: 8px 13px; border-bottom: 1px solid #edf0f6; vertical-align: middle; }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    .color-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }

    /* callouts */
    .info-box { background: #edf4ff; border-left: 4px solid #1d5c9e; padding: 13px 17px; border-radius: 0 4px 4px 0; font-size: 13px; color: #1a3050; margin: 14px 0 22px; }

    /* images */
    .chart-img { width: 100%; height: auto; display: block; border-radius: 4px; margin: 14px 0; }
    .map-img   { width: 100%; height: auto; display: block; border-radius: 4px; }
    .fig-caption { font-size: 11.5px; color: #7a8a9a; text-align: center; margin-top: 6px; margin-bottom: 22px; font-style: italic; }

    /* steps & refs */
    .steps-list { padding-left: 22px; font-size: 13px; }
    .steps-list li { margin: 7px 0; }
    .steps-list code { background: #f0f4f8; padding: 1px 5px; border-radius: 3px; font-size: 12px; font-family: Consolas, monospace; }
    .refs { padding-left: 20px; font-size: 12.5px; color: #445566; line-height: 2; }

    /* top bar */
    .top-bar { text-align: center; padding: 12px; background: #0d2440; color: #fff; font-size: 12.5px; }
    .top-bar strong { color: #7ab0d0; }

    /* print */
    @media print {
      body { background: #fff; }
      .top-bar { display: none; }
      .page { box-shadow: none; margin: 0; border-radius: 0; page-break-after: always; max-width: 100%; }
      .page:last-child { page-break-after: avoid; }
    }
    """

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_he(title)} — Catchment Report</title>
<style>{css}</style>
</head>
<body>
<div class="top-bar">
  &#128196; &nbsp;<strong>Catchment Analysis Report</strong> &nbsp;·&nbsp;
  Use your browser's <strong>Print</strong> (Ctrl+P / Cmd+P) to save as PDF
</div>
{p1}
{p2}
{p3}
{p4}
{p5}
</body>
</html>"""


# ── Flask route ────────────────────────────────────────────────────────────────
@app.route("/api/report", methods=["POST"])
def api_report():
    """Generate a full HTML report and return it as a downloadable file."""
    from flask import Response
    data = request.json or {}
    if not _all_results:
        return jsonify({"error": "No delineation results to report."}), 400

    title       = data.get("title",  "Catchment Analysis Report")
    author      = data.get("author", "")
    outlet_info = data.get("outlets", [])
    # Slider value sent by JS is the source of truth; fall back to stored
    # threshold from the last delineation if the request omits it
    _req_thresh = data.get("acc_threshold")
    acc_thresh  = int(_req_thresh) if _req_thresh is not None else (
        _all_results[-1]["stats"].get("acc_threshold") or 500)
    dem_source  = _dem_progress.get("source", "SRTM GL1 (30 m)")
    res_m       = round(abs(_cache["affine"].a) * 111320, 1) if _cache else 30
    date_str    = datetime.datetime.now().strftime("%d %B %Y")

    try:
        all_catches, all_outlet_feats = [], []
        lookup = {int(o["id"]): o for o in outlet_info}
        for i, r in enumerate(_all_results):
            oid   = i + 1
            info  = lookup.get(oid, {})
            name  = (info.get("name") or "").strip() or f"Catchment {oid}"
            color = info.get("color", "#3399ff")
            for feat in r["catchment"]["features"]:
                f2 = dict(feat)
                f2["properties"] = {**f2.get("properties", {}),
                                     "outlet_id": oid, "name": name, "color": color}
                all_catches.append(f2)
            for feat in r["outlet"]["features"]:
                f2 = dict(feat)
                f2["properties"] = {**f2.get("properties", {}),
                                     "outlet_id": oid, "name": name, "color": color}
                all_outlet_feats.append(f2)

        catches_gdf      = gpd.GeoDataFrame.from_features(all_catches,      crs=4326)
        outlets_gdf      = gpd.GeoDataFrame.from_features(all_outlet_feats, crs=4326)
        catchment_union  = catches_gdf.unary_union

        rivers_fc        = extract_rivers_global(acc_thresh)
        rivers_gdf_full  = (gpd.GeoDataFrame.from_features(rivers_fc["features"], crs=4326)
                            if rivers_fc.get("features") else gpd.GeoDataFrame())
        # Clip to catchment so stats + map only reflect rivers within the watershed
        rivers_gdf       = (rivers_gdf_full.clip(catchment_union)
                            if not rivers_gdf_full.empty else gpd.GeoDataFrame())

        catch_stats  = _rpt_compute_stats(catches_gdf, outlets_gdf)
        riv_stats    = _rpt_compute_river_stats(rivers_gdf)
        # Pass already-clipped rivers; _rpt_html_map will use them directly
        map_b64      = _rpt_html_map(outlet_info, acc_thresh, rivers_gdf_prebuilt=rivers_gdf)
        strahler_b64 = _rpt_html_strahler_charts(riv_stats)
        elev_b64     = _rpt_html_elev_chart(catch_stats)

        html_str = _build_html_report(
            title, author, date_str,
            catch_stats, riv_stats,
            dem_source, res_m, acc_thresh,
            map_b64, strahler_b64, elev_b64,
        )

        safe = "".join(c for c in title if c.isalnum() or c in " _-")[:40].strip()
        return Response(
            html_str,
            mimetype="text/html",
            headers={"Content-Disposition": f'attachment; filename="{safe}_report.html"'}
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload_shapefile", methods=["POST"])
def api_upload_shapefile():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected"}), 400
    fname = f.filename.lower()
    try:
        with tempfile.TemporaryDirectory() as td:
            if fname.endswith(".zip"):
                zip_path = os.path.join(td, "upload.zip")
                f.save(zip_path)
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(td)
                shp_files = [os.path.join(td, n) for n in os.listdir(td)
                             if n.lower().endswith(".shp")]
                if not shp_files:
                    return jsonify({"error": "No .shp file found in ZIP"}), 400
                read_path = shp_files[0]
                layer_name = os.path.splitext(os.path.basename(read_path))[0]
            elif fname.endswith((".shp", ".geojson", ".json")):
                safe_name = os.path.basename(f.filename).replace(" ", "_")
                read_path = os.path.join(td, safe_name)
                f.save(read_path)
                layer_name = os.path.splitext(safe_name)[0]
            else:
                return jsonify({"error": "Upload a ZIP (containing .shp) or a .shp / .geojson file"}), 400
            os.environ['SHAPE_RESTORE_SHX'] = 'YES'
            gdf = gpd.read_file(read_path)
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(4326)
            return jsonify({"geojson": json.loads(gdf.to_json()), "name": layer_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT API  —  /api/v1/   (session-isolated, CORS-enabled)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Typical agent workflow
#  ──────────────────────
#  1. POST /api/v1/session                 → {session_id}
#  2. POST /api/v1/dem/fetch               → {bounds, res_m, default_acc, …}
#     GET  /api/v1/dem/status?session_id=… → poll {pct, done}   (optional)
#  3. POST /api/v1/delineate  (×N)         → {catchment, outlet, stats}
#  4. POST /api/v1/report                  → {html, filename}
#  5. POST /api/v1/export                  → {download_url, token, filename}
#     GET  /api/v1/export/download/<token> → ZIP binary (single-use)
#  6. DELETE /api/v1/session/<id>          → cleanup
#
#  Pass session_id in the JSON body, as ?session_id= query param,
#  or in the X-Session-Id request header.
# ═══════════════════════════════════════════════════════════════════════════════

# ── CORS ──────────────────────────────────────────────────────────────────────

@app.after_request
def _v1_cors(response):
    if request.path.startswith("/api/v1"):
        response.headers["Access-Control-Allow-Origin"]  = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Session-Id"
    return response


@app.route("/api/v1/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/api/v1/<path:path>",              methods=["OPTIONS"])
def _v1_preflight(path):
    from flask import Response as _FR
    r = _FR()
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Session-Id"
    return r, 204


# ── Discovery ─────────────────────────────────────────────────────────────────

@app.route("/api/v1")
@app.route("/api/v1/")
def v1_index():
    """Structured API reference — every endpoint, field, type, and example."""
    return jsonify({
        "name":        "Catchment Delineation Agent API",
        "version":     "1.0",
        "base_url":    request.host_url.rstrip("/"),
        "description": (
            "Delineates river catchments from a 30 m SRTM DEM. "
            "Given a bounding box or polygon, downloads the elevation data, "
            "computes D8 flow direction and accumulation, snaps an outlet point "
            "to the nearest stream, traces the upstream catchment boundary, and "
            "produces a GeoJSON result set, a self-contained HTML report, and a "
            "ZIP export (map PNG + GeoJSON layers)."
        ),
        "auth": "None. No API key required.",
        "session_id_passing": (
            "Every endpoint requires a session_id. Pass it in ONE of: "
            "(a) JSON body field 'session_id', "
            "(b) query param ?session_id=…, "
            "(c) request header X-Session-Id."
        ),
        "error_format": {
            "description": "All errors return HTTP 4xx/5xx with JSON body.",
            "schema":      {"error": "string — human-readable message",
                            "trace": "string — Python traceback (only on 500)"},
            "example":     {"error": "No DEM loaded. Call /api/v1/dem/fetch first."},
        },
        "workflow": [
            "Step 1 — Create session:   POST /api/v1/session",
            "Step 2 — Load DEM:         POST /api/v1/dem/fetch  (30-120 s, synchronous)",
            "Step 3 — Delineate:        POST /api/v1/delineate  (repeat for each outlet)",
            "Step 4 — Report:           POST /api/v1/report",
            "Step 5 — Export ZIP:       POST /api/v1/export  then  GET download_url",
            "Step 6 — Cleanup:          DELETE /api/v1/session/<id>",
        ],
        "endpoints": {
            "POST /api/v1/session": {
                "description": "Create a new isolated analysis session.",
                "request_body": "Empty (no fields required).",
                "response": {
                    "session_id":  "string — use this in all subsequent calls",
                    "created_at":  "string — ISO-8601 UTC timestamp",
                },
                "example_response": {"session_id": "a3f9…", "created_at": "2025-01-01T00:00:00Z"},
            },
            "DELETE /api/v1/session/<session_id>": {
                "description": "Free a session and release its cached DEM data from memory.",
                "response": {"ok": "true"},
            },
            "POST /api/v1/dem/fetch": {
                "description": (
                    "Download a 30 m SRTM DEM for the study area, compute flow direction "
                    "and accumulation, and extract the river network. "
                    "This is synchronous and may take 30-120 seconds depending on area size. "
                    "Store default_acc — you will pass it as acc_threshold to /delineate."
                ),
                "request_body": {
                    "session_id":  "string REQUIRED",
                    "bbox":        "object — {south, north, west, east} in decimal degrees WGS84. Use this OR polygon.",
                    "polygon":     "object — GeoJSON Polygon geometry (type+coordinates). Use this OR bbox.",
                },
                "response": {
                    "bounds":           "object — {south, north, west, east} actual DEM extent",
                    "res_m":            "number — DEM cell size in metres (typically ~30)",
                    "shape":            "array  — [rows, cols] of the DEM grid",
                    "acc_p95":          "number — 95th-percentile flow accumulation value",
                    "default_acc":      "integer — recommended acc_threshold for stream extraction (use this in /delineate)",
                    "n_river_segments": "integer — number of river segments at default threshold",
                    "dem_source":       "string — data source label (e.g. 'OpenTopography SRTM GL1 (30 m)')",
                },
                "example_request":  {"session_id": "a3f9…", "bbox": {"south": 51.80, "north": 51.88, "west": -3.55, "east": -3.40}},
                "example_response": {"bounds": {"south": 51.80, "north": 51.88, "west": -3.55, "east": -3.40},
                                     "res_m": 30.9, "shape": [288, 540], "acc_p95": 130.0,
                                     "default_acc": 150, "n_river_segments": 582,
                                     "dem_source": "OpenTopography SRTM GL1 (30 m)"},
            },
            "GET /api/v1/dem/status": {
                "description": "Poll DEM processing progress. Optional — /dem/fetch is synchronous so this is only needed if you call it in a background thread.",
                "query_params": {"session_id": "string REQUIRED"},
                "response": {
                    "stage":   "string — current step label",
                    "detail":  "string — step detail",
                    "pct":     "integer — 0-100 progress percentage",
                    "done":    "boolean — true when processing is complete",
                    "error":   "string|null — error message if failed",
                    "source":  "string — DEM data source label",
                },
            },
            "POST /api/v1/snap": {
                "description": (
                    "Snap a lat/lon point to the nearest stream cell. "
                    "Useful to confirm the outlet location before delineating. "
                    "/delineate does this automatically, so calling snap first is optional."
                ),
                "request_body": {
                    "session_id":    "string REQUIRED",
                    "lat":           "number REQUIRED — latitude in decimal degrees",
                    "lon":           "number REQUIRED — longitude in decimal degrees",
                    "acc_threshold": "integer OPTIONAL — stream threshold in cells (default 500; use default_acc from /dem/fetch)",
                },
                "response": {
                    "lat":     "number — snapped latitude",
                    "lon":     "number — snapped longitude",
                    "snapped": "boolean — true if the point was moved to a stream cell",
                    "acc":     "integer — flow accumulation at the snapped cell",
                },
            },
            "POST /api/v1/delineate": {
                "description": (
                    "Delineate the upstream catchment for one outlet point. "
                    "Automatically snaps the point to the nearest stream cell. "
                    "Call multiple times to accumulate multiple catchments — "
                    "all results are used together by /report and /export."
                ),
                "request_body": {
                    "session_id":    "string  REQUIRED",
                    "lat":           "number  REQUIRED — outlet latitude (decimal degrees)",
                    "lon":           "number  REQUIRED — outlet longitude (decimal degrees)",
                    "acc_threshold": "integer OPTIONAL — stream threshold (default 500; use default_acc from /dem/fetch)",
                },
                "response": {
                    "catchment": "GeoJSON FeatureCollection — polygon of the catchment boundary. Properties: {cells, area_km2}",
                    "outlet":    "GeoJSON FeatureCollection — snapped outlet point. Properties: {acc}",
                    "stats": {
                        "area_km2":       "number  — catchment area in km²",
                        "catchment_cells":"integer — number of DEM cells in catchment",
                        "max_strahler":   "integer — highest Strahler stream order within catchment",
                        "outlet_lat":     "number  — snapped outlet latitude",
                        "outlet_lon":     "number  — snapped outlet longitude",
                        "snapped":        "boolean — whether outlet was moved to nearest stream",
                        "acc_threshold":  "integer — threshold used",
                        "dem_res_m":      "number  — DEM resolution in metres",
                    },
                },
                "example_request":  {"session_id": "a3f9…", "lat": 51.83, "lon": -3.47, "acc_threshold": 150},
                "example_response": {
                    "catchment": {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": ["…"]}, "properties": {"cells": 321, "area_km2": 0.31}}]},
                    "outlet":    {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [-3.470556, 51.828611]}, "properties": {"acc": 321}}]},
                    "stats":     {"area_km2": 0.31, "catchment_cells": 321, "max_strahler": 1, "outlet_lat": 51.828611, "outlet_lon": -3.470556, "snapped": True, "acc_threshold": 150, "dem_res_m": 30.9},
                },
            },
            "GET /api/v1/results": {
                "description": "List all catchments delineated so far in this session.",
                "query_params": {"session_id": "string REQUIRED"},
                "response": {
                    "count":   "integer — number of delineated catchments",
                    "results": "array   — each item has {index, stats, catchment:GeoJSON, outlet:GeoJSON}",
                },
            },
            "POST /api/v1/clear": {
                "description": "Clear all delineation results but keep the DEM loaded. Useful to start over without re-downloading the DEM.",
                "request_body": {"session_id": "string REQUIRED"},
                "response": {"ok": "true"},
            },
            "POST /api/v1/rivers": {
                "description": "Extract the river network GeoJSON at a given flow-accumulation threshold.",
                "request_body": {
                    "session_id":    "string  REQUIRED",
                    "acc_threshold": "integer OPTIONAL — cells threshold (default 500; lower = more streams)",
                },
                "response": {
                    "rivers":     "GeoJSON FeatureCollection of LineStrings. Each feature: {order:int (Strahler), acc:int}",
                    "n_segments": "integer — number of river segments returned",
                },
            },
            "POST /api/v1/report": {
                "description": (
                    "Generate a full catchment analysis HTML report. "
                    "The returned html string is a completely self-contained document "
                    "(inline CSS, base64 images) — save it directly as a .html file."
                ),
                "request_body": {
                    "session_id":    "string REQUIRED",
                    "title":         "string OPTIONAL — report title (default 'Catchment Analysis Report')",
                    "author":        "string OPTIONAL — author name shown on report",
                    "acc_threshold": "integer OPTIONAL — river threshold for the map (default: value used in last /delineate call)",
                    "outlets":       "array OPTIONAL — [{id:int, name:str, color:str}] to label each catchment. id is 1-based index.",
                },
                "response": {
                    "html":     "string — full self-contained HTML document",
                    "filename": "string — suggested save filename (e.g. 'My Report_report.html')",
                },
                "example_request": {"session_id": "a3f9…", "title": "Wye Catchment Study", "author": "GIS Agent",
                                    "outlets": [{"id": 1, "name": "Main Outlet", "color": "#3399ff"}]},
            },
            "POST /api/v1/export": {
                "description": (
                    "Build a ZIP containing a print-quality map PNG and GeoJSON layers. "
                    "Returns a single-use download token — GET the download_url once to retrieve the file. "
                    "ZIP contents: catchment_map.png, catchments.geojson, outlets.geojson, rivers.geojson."
                ),
                "request_body": {
                    "session_id":   "string  REQUIRED",
                    "title":        "string  OPTIONAL — map title and ZIP filename stem",
                    "acc_threshold":"integer OPTIONAL — river threshold for the map",
                    "paper":        "string  OPTIONAL — 'A4' | 'A3' | 'Letter'  (default 'A4')",
                    "orientation":  "string  OPTIONAL — 'landscape' | 'portrait'  (default 'landscape')",
                    "outlets":      "array   OPTIONAL — [{id:int, name:str, color:str}]",
                },
                "response": {
                    "token":        "string — single-use download token",
                    "filename":     "string — suggested ZIP filename",
                    "download_url": "string — full URL; GET this once to download the ZIP",
                },
                "important": "The download token is consumed on first use. Call /api/v1/export again to get a new token.",
            },
            "GET /api/v1/export/download/<token>": {
                "description": "Download the pre-built export ZIP. Single-use — token is deleted after download.",
                "response": "application/zip binary stream",
            },
        },
    })


# ── Session ───────────────────────────────────────────────────────────────────

@app.route("/api/v1/session", methods=["POST"])
def v1_create_session():
    """Create a new isolated analysis session. Returns session_id."""
    sid = uuid.uuid4().hex
    _sessions[sid] = {
        "cache":        {},
        "results":      [],
        "dem_progress": {
            "stage": "", "detail": "", "pct": 0,
            "source": "", "fallback_reason": "", "error": None, "done": False,
        },
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    return jsonify({"session_id": sid, "created_at": _sessions[sid]["created_at"]})


@app.route("/api/v1/session/<sid>", methods=["DELETE"])
def v1_delete_session(sid):
    """Free a session and release its cached DEM data."""
    _sessions.pop(sid, None)
    return jsonify({"ok": True})


# ── DEM ───────────────────────────────────────────────────────────────────────

@app.route("/api/v1/dem/fetch", methods=["POST"])
def v1_fetch_dem():
    """
    Download + process a DEM for a study area (synchronous — may take 30-120 s).

    Body (JSON):
      session_id   string    required
      polygon      object    GeoJSON Polygon geometry  ─┐ one of
      bbox         object    {south, north, west, east} ─┘ these two

    Returns:
      bounds           {south, north, west, east}
      res_m            DEM resolution in metres
      shape            [rows, cols]
      acc_p95          95th-percentile flow-accumulation value
      default_acc      recommended stream threshold (cells)
      n_river_segments number of river segments extracted
      dem_source       data source label
    """
    data = request.json or {}
    sid  = _v1_sid(data)
    if not sid or sid not in _sessions:
        return jsonify({"error": "Invalid or missing session_id. "
                                 "Create one via POST /api/v1/session"}), 400

    polygon = data.get("polygon")
    if not polygon and "bbox" in data:
        b = data["bbox"]
        s = float(b["south"]); n = float(b["north"])
        w = float(b["west"]);  e = float(b["east"])
        polygon = {"type": "Polygon",
                   "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}
    if not polygon:
        return jsonify({"error": "Provide 'polygon' (GeoJSON Polygon geometry) or "
                                 "'bbox': {south, north, west, east}"}), 400

    try:
        with _use_session(sid):
            coords = polygon["coordinates"][0]
            lats   = [c[1] for c in coords]
            lons   = [c[0] for c in coords]
            south, north = min(lats), max(lats)
            west,  east  = min(lons), max(lons)
            _dem_progress.update({"stage": "Starting…", "detail": "", "pct": 0,
                                   "source": "", "fallback_reason": "",
                                   "error": None, "done": False})
            _emit("Contacting OpenTopography…",
                  f"Bbox: {south:.3f}°–{north:.3f}°N, {west:.3f}°–{east:.3f}°E", pct=1)
            dem_path = fetch_dem_bbox(south, north, west, east)
            _emit("Clipping to study area…", "Masking cells outside polygon", pct=20)
            dem_path = clip_dem_to_polygon(dem_path, polygon)
            _all_results.clear()
            load_and_condition(dem_path)

            wb  = _cache["bounds_wgs"]
            acc = _cache["acc_arr"]
            p95 = float(np.percentile(acc[acc > 0], 95)) if (acc > 0).any() else 500
            default_acc = int(round(p95 / 50) * 50)
            _emit("Extracting river network…",
                  f"Auto-selecting threshold ≈{default_acc} cells", pct=93)
            default_acc, rivers = _auto_threshold(default_acc)
            n_segs = len(rivers.get("features", []))
            _emit_done(f"DEM ready · {_cache['shape'][0]}×{_cache['shape'][1]} cells "
                       f"· {n_segs} river segments")
            return jsonify({
                "bounds":           {"south": wb[1], "west": wb[0],
                                     "north": wb[3], "east": wb[2]},
                "res_m":            round(abs(_cache["affine"].a) * 111320, 1),
                "shape":            list(_cache["shape"]),
                "acc_p95":          p95,
                "default_acc":      default_acc,
                "n_river_segments": n_segs,
                "dem_source":       _dem_progress.get("source",
                                                      "OpenTopography SRTM GL1 (30 m)"),
            })
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/v1/dem/status")
def v1_dem_status():
    """
    Poll DEM processing progress.
    Query param or body: session_id
    Returns: {stage, detail, pct, done, error, source}
    """
    sid = _v1_sid()
    if not sid or sid not in _sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400
    return jsonify(_sessions[sid]["dem_progress"])


# ── Snap & Delineate ──────────────────────────────────────────────────────────

@app.route("/api/v1/snap", methods=["POST"])
def v1_snap():
    """
    Snap a lat/lon to the nearest stream cell.

    Body: {session_id, lat, lon, acc_threshold?:int (default 500)}
    Returns: {lat, lon, snapped:bool, acc:int}
    """
    data = request.json or {}
    sid  = _v1_sid(data)
    if not sid or sid not in _sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400
    try:
        with _use_session(sid):
            if not _cache:
                return jsonify({"error": "No DEM loaded. "
                                         "Call /api/v1/dem/fetch first."}), 400
            lat           = float(data["lat"])
            lon           = float(data["lon"])
            acc_threshold = int(data.get("acc_threshold", 500))
            row, col      = latlon_to_rowcol(lat, lon)
            s_row, s_col  = snap_to_stream(row, col, acc_threshold)
            ox, oy        = rio_transform.xy(_cache["affine"], s_row, s_col)
            return jsonify({
                "lat":     float(oy),
                "lon":     float(ox),
                "snapped": (s_row != row or s_col != col),
                "acc":     int(_cache["acc_arr"][s_row, s_col]),
            })
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/v1/delineate", methods=["POST"])
def v1_delineate():
    """
    Delineate a catchment at an outlet point.

    Body: {session_id, lat, lon, acc_threshold?:int (default 500)}
    Returns: {catchment:GeoJSON_FeatureCollection,
              outlet:GeoJSON_FeatureCollection,
              stats:{area_km2, catchment_cells, max_strahler,
                     outlet_lat, outlet_lon, snapped, acc_threshold}}

    Call multiple times to accumulate results for multi-outlet analysis.
    All accumulated results are used by /api/v1/report and /api/v1/export.
    """
    data = request.json or {}
    sid  = _v1_sid(data)
    if not sid or sid not in _sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400
    try:
        with _use_session(sid):
            if not _cache:
                return jsonify({"error": "No DEM loaded. "
                                         "Call /api/v1/dem/fetch first."}), 400
            result = delineate(
                lat=float(data["lat"]),
                lon=float(data["lon"]),
                acc_threshold=int(data.get("acc_threshold", 500)),
            )
            if "error" not in result:
                _all_results.append(result)
            return jsonify(result)
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ── Results ───────────────────────────────────────────────────────────────────

@app.route("/api/v1/results")
def v1_results():
    """List all delineated catchments for this session (query: session_id)."""
    sid = _v1_sid()
    if not sid or sid not in _sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400
    results = _sessions[sid]["results"]
    return jsonify({
        "count":   len(results),
        "results": [
            {"index":    i,
             "stats":    r["stats"],
             "catchment": r["catchment"],
             "outlet":   r["outlet"]}
            for i, r in enumerate(results)
        ],
    })


@app.route("/api/v1/clear", methods=["POST"])
def v1_clear():
    """Clear delineation results but keep the DEM loaded. Body: {session_id}"""
    data = request.json or {}
    sid  = _v1_sid(data)
    if not sid or sid not in _sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400
    _sessions[sid]["results"].clear()
    return jsonify({"ok": True})


# ── Rivers ────────────────────────────────────────────────────────────────────

@app.route("/api/v1/rivers", methods=["POST"])
def v1_rivers():
    """
    Extract river-network GeoJSON at a given accumulation threshold.

    Body: {session_id, acc_threshold?:int}
      Use the default_acc value returned by /api/v1/dem/fetch as a starting point.
    Returns: {rivers:GeoJSON_FeatureCollection, n_segments:int}
      Each feature has properties: {order:int (Strahler), acc:int}
    """
    data = request.json or {}
    sid  = _v1_sid(data)
    if not sid or sid not in _sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400
    try:
        with _use_session(sid):
            if not _cache:
                return jsonify({"error": "No DEM loaded. "
                                         "Call /api/v1/dem/fetch first."}), 400
            acc_threshold = int(data.get("acc_threshold", 500))
            rivers = extract_rivers_global(acc_threshold)
            return jsonify({"rivers":     rivers,
                             "n_segments": len(rivers.get("features", []))})
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ── Report ────────────────────────────────────────────────────────────────────

@app.route("/api/v1/report", methods=["POST"])
def v1_report():
    """
    Generate a full HTML catchment analysis report.

    Body: {session_id,
           title?:str,
           author?:str,
           acc_threshold?:int,
           outlets?:[{id:int, name:str, color:str}]}
    Returns: {html:str, filename:str}
      html is a fully self-contained HTML document (inline CSS + base64 images).
      Save it directly as a .html file or display it in a webview.
    """
    data = request.json or {}
    sid  = _v1_sid(data)
    if not sid or sid not in _sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400
    try:
        with _use_session(sid):
            if not _all_results:
                return jsonify({"error": "No delineation results. "
                                         "Call /api/v1/delineate first."}), 400
            title       = data.get("title",  "Catchment Analysis Report")
            author      = data.get("author", "")
            outlet_info = data.get("outlets", [])
            _req_thresh = data.get("acc_threshold")
            acc_thresh  = (int(_req_thresh) if _req_thresh is not None
                           else (_all_results[-1]["stats"].get("acc_threshold") or 500))
            dem_source  = _dem_progress.get("source", "SRTM GL1 (30 m)")
            res_m       = round(abs(_cache["affine"].a) * 111320, 1) if _cache else 30
            date_str    = datetime.datetime.now().strftime("%d %B %Y")

            all_catches, all_outlet_feats = [], []
            lookup = {int(o["id"]): o for o in outlet_info}
            for i, r in enumerate(_all_results):
                oid   = i + 1
                info  = lookup.get(oid, {})
                name  = (info.get("name") or "").strip() or f"Catchment {oid}"
                color = info.get("color", "#3399ff")
                for feat in r["catchment"]["features"]:
                    f2 = dict(feat)
                    f2["properties"] = {**f2.get("properties", {}),
                                         "outlet_id": oid, "name": name, "color": color}
                    all_catches.append(f2)
                for feat in r["outlet"]["features"]:
                    f2 = dict(feat)
                    f2["properties"] = {**f2.get("properties", {}),
                                         "outlet_id": oid, "name": name, "color": color}
                    all_outlet_feats.append(f2)

            catches_gdf     = gpd.GeoDataFrame.from_features(all_catches,      crs=4326)
            outlets_gdf     = gpd.GeoDataFrame.from_features(all_outlet_feats, crs=4326)
            catchment_union = catches_gdf.unary_union

            rivers_fc       = extract_rivers_global(acc_thresh)
            rivers_gdf_full = (gpd.GeoDataFrame.from_features(rivers_fc["features"], crs=4326)
                               if rivers_fc.get("features") else gpd.GeoDataFrame())
            rivers_gdf      = (rivers_gdf_full.clip(catchment_union)
                               if not rivers_gdf_full.empty else gpd.GeoDataFrame())

            catch_stats  = _rpt_compute_stats(catches_gdf, outlets_gdf)
            riv_stats    = _rpt_compute_river_stats(rivers_gdf)
            map_b64      = _rpt_html_map(outlet_info, acc_thresh,
                                          rivers_gdf_prebuilt=rivers_gdf)
            strahler_b64 = _rpt_html_strahler_charts(riv_stats)
            elev_b64     = _rpt_html_elev_chart(catch_stats)

            html_str = _build_html_report(
                title, author, date_str,
                catch_stats, riv_stats,
                dem_source, res_m, acc_thresh,
                map_b64, strahler_b64, elev_b64,
            )
            safe = "".join(c for c in title if c.isalnum() or c in " _-")[:40].strip()
            return jsonify({"html": html_str, "filename": f"{safe}_report.html"})
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ── Export ────────────────────────────────────────────────────────────────────

@app.route("/api/v1/export", methods=["POST"])
def v1_export():
    """
    Build a ZIP export (map PNG + GeoJSON layers) and return a single-use download token.

    Body: {session_id,
           title?:str,
           acc_threshold?:int,
           paper?:"A4"|"A3"|"Letter"  (default "A4"),
           orientation?:"landscape"|"portrait"  (default "landscape"),
           outlets?:[{id:int, name:str, color:str}]}
    Returns: {token:str, filename:str, download_url:str}
      GET the download_url once to retrieve the ZIP.  The token is consumed on use.
    """
    data = request.json or {}
    sid  = _v1_sid(data)
    if not sid or sid not in _sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400
    try:
        with _use_session(sid):
            if not _all_results:
                return jsonify({"error": "No delineation results. "
                                         "Call /api/v1/delineate first."}), 400
            zip_buf, safe_title = _build_export_zip(data)
            zip_bytes = zip_buf.read()

        token = uuid.uuid4().hex
        _v1_export_tokens[token] = (zip_bytes, safe_title)
        base = request.host_url.rstrip("/")
        return jsonify({
            "token":        token,
            "filename":     f"{safe_title}.zip",
            "download_url": f"{base}/api/v1/export/download/{token}",
        })
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/v1/export/download/<token>")
def v1_export_download(token):
    """Download a pre-built export ZIP. Token is single-use."""
    entry = _v1_export_tokens.get(token)
    if not entry:
        return jsonify({"error": "Token not found or already used. "
                                 "Re-call POST /api/v1/export for a new token."}), 404
    zip_bytes, safe_title = entry
    _v1_export_tokens.pop(token, None)
    return send_file(
        io.BytesIO(zip_bytes),
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{safe_title}.zip",
    )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Catchment Delineation — SRTM GL1 (30 m) via OpenTopography")
    print("  Open your browser at:  http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
