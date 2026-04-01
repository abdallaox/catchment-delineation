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

import os, sys, io, json, math, zipfile, tempfile, warnings, traceback, shutil
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
NODATA = -9999.0

# ── D8 lookup ──────────────────────────────────────────────────────────────────
D8_DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)
D8_DELTAS = {64:(-1,0), 128:(-1,1), 1:(0,1), 2:(1,1),
             4:(1,0),   8:(1,-1),   16:(0,-1), 32:(-1,-1)}
DELTA_TO_DIR = {v: k for k, v in D8_DELTAS.items()}

# ── Global state ───────────────────────────────────────────────────────────────
_cache       = {}   # Conditioned DEM + fdir/acc arrays
_all_results = []   # All delineation results, one per outlet placed


# ═══════════════════════════════════════════════════════════════════════════════
#  DEM DOWNLOAD  —  SRTM GL1 (30 m) via OpenTopography
# ═══════════════════════════════════════════════════════════════════════════════

def _download_srtm_chunk(south, north, west, east):
    key  = f"srtm_{south:.5f}_{north:.5f}_{west:.5f}_{east:.5f}"
    path = os.path.join(TILE_DIR, f"{key}.tif")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
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
            return path
        print(f"  OpenTopography error {r.status_code}: {r.text[:300]}")
    except Exception as e:
        print(f"  SRTM chunk download error: {e}")
    return None


def fetch_dem_bbox(south, north, west, east):
    """Download + mosaic SRTM GL1 tiles. Falls back to AWS Terrain Tiles if OpenTopography fails."""
    # ── Try OpenTopography first ──────────────────────────────────────────────
    CHUNK = 1.0
    lat_edges = list(np.arange(south, north, CHUNK)) + [north]
    lon_edges = list(np.arange(west,  east,  CHUNK)) + [east]
    chunks = [(lat_edges[i], lat_edges[i+1], lon_edges[j], lon_edges[j+1])
              for i in range(len(lat_edges)-1) for j in range(len(lon_edges)-1)]
    print(f"  Trying OpenTopography SRTM GL1 — {len(chunks)} chunk(s)…")
    paths = [_download_srtm_chunk(s, n, w, e) for (s, n, w, e) in chunks]
    paths = [p for p in paths if p]

    if paths:
        # OpenTopography succeeded
        if len(paths) == 1:
            return paths[0]
        print(f"  Mosaicing {len(paths)} chunks…")
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
    print("  OpenTopography unavailable — falling back to AWS Terrain Tiles…")
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
    print(f"  AWS tiles: zoom={zoom}, grid={x1-x0+1}×{y1-y0+1} ({n_tiles} tiles)…")

    tile_paths = []
    for xi in range(x0, x1 + 1):
        for yi in range(y0, y1 + 1):
            p = _download_aws_tile(zoom, xi, yi)
            if p:
                tile_paths.append(p)

    if not tile_paths:
        raise RuntimeError("Failed to download DEM from both OpenTopography and AWS Terrain Tiles. "
                           "Check your internet connection.")

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
    print(f"  Conditioning: {os.path.basename(dem_path)}")

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
    print(f"  Valid cells: {n_valid}/{raw.size}  "
          f"elev {np.nanmin(raw):.1f}–{np.nanmax(raw):.1f} m")

    # Wall DEM: outside polygon cells → very high elevation so pysheds
    # can't route flow through them, preventing catchment leakage.
    wall_elev = float(np.nanmax(raw)) + 5000.0 if n_valid > 0 else 5000.0
    dem_walled = np.where(valid_mask, raw, wall_elev)

    walled_path = dem_path.replace(".tif", "_walled.tif")
    with rasterio.open(dem_path) as src:
        meta_w = src.meta.copy()
    meta_w.update(nodata=None, dtype="float32")
    with rasterio.open(walled_path, "w", **meta_w) as dst:
        dst.write(dem_walled.astype("float32")[np.newaxis])

    print("  Running pysheds: pit fill → depression fill → flat resolve → fdir → acc…")
    grid    = Grid.from_raster(walled_path)
    dem_ps  = grid.read_raster(walled_path)
    pit     = grid.fill_pits(dem_ps)
    dep     = grid.fill_depressions(pit)
    flat    = grid.resolve_flats(dep)
    fdir_ps = grid.flowdir(flat, dirmap=D8_DIRMAP)
    acc_ps  = grid.accumulation(fdir_ps, dirmap=D8_DIRMAP)

    fdir_arr = np.array(fdir_ps).astype(np.int32)
    acc_arr  = np.array(acc_ps).astype(float)

    # Hard-zero outside polygon — belt-and-suspenders
    fdir_arr[~valid_mask] = 0
    acc_arr[~valid_mask]  = 0

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
MAX_DEM_CELLS = 8_000_000  # ~2800×2800 px — larger DEMs are too slow for Python loops

def _estimate_threshold(target_stream_cells=45_000):
    """Binary-search for a threshold that gives ~target_stream_cells stream cells.
    Much faster than building the full network repeatedly."""
    acc = _cache["acc_arr"]
    valid = _cache.get("valid_mask")
    lo, hi = 1, int(acc.max())
    for _ in range(20):
        mid = (lo + hi) // 2
        count = int((acc[valid] >= mid).sum()) if valid is not None else int((acc >= mid).sum())
        if count > target_stream_cells:
            lo = mid + 1
        else:
            hi = mid
    return max(lo, 50)

def _auto_threshold(initial):
    """Pick a threshold that keeps segment count under MAX_RIVER_SEGMENTS."""
    nrows, ncols = _cache["shape"]
    n_cells = nrows * ncols
    if n_cells > MAX_DEM_CELLS:
        # For very large DEMs estimate threshold from stream-cell count alone —
        # avoids building the network (slow Python loop) multiple times.
        threshold = _estimate_threshold(target_stream_cells=40_000)
        print(f"  Large DEM ({n_cells} cells) — estimated threshold={threshold}")
    else:
        threshold = initial
    rivers = extract_rivers_global(threshold)
    # If still over budget, double until within limit (fast for small DEMs)
    for _ in range(8):
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
        },
    }


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


@app.route("/api/fetch_dem", methods=["POST"])
def api_fetch_dem():
    """Download + clip + condition DEM. Body: {polygon} GeoJSON geometry."""
    data = request.json or {}
    try:
        if "polygon" in data:
            polygon = data["polygon"]
            coords  = polygon["coordinates"][0]
            lats = [c[1] for c in coords]; lons = [c[0] for c in coords]
            south, north = min(lats), max(lats)
            west,  east  = min(lons), max(lons)
            print(f"[DEM] Polygon bbox: S={south:.4f} N={north:.4f} W={west:.4f} E={east:.4f}")
            dem_path = fetch_dem_bbox(south, north, west, east)
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
        default_acc, rivers = _auto_threshold(default_acc)
        return jsonify({
            "bounds":      {"south": w[1], "west": w[0], "north": w[3], "east": w[2]},
            "res_m":       round(abs(_cache["affine"].a) * 111320, 1),
            "shape":       list(_cache["shape"]),
            "acc_p95":     p95,
            "default_acc": default_acc,
            "rivers":      rivers,
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


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

    try:
        with tempfile.TemporaryDirectory() as td:
            raw_path = os.path.join(td, "upload.tif")
            f.save(raw_path)

            with rasterio.open(raw_path) as src:
                src_crs    = src.crs
                src_nodata = src.nodata
                src_bounds = src.bounds
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
            print(f"  Normalising → WGS84 float32 …")
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
        default_acc, rivers = _auto_threshold(default_acc)
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
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


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


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Catchment Delineation — SRTM GL1 (30 m) via OpenTopography")
    print("  Open your browser at:  http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
