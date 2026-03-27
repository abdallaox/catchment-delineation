"""
Catchment Delineation Web App  —  Global Edition
==================================================
Click anywhere on the world map. The tool downloads a DEM from
AWS Terrain Tiles (no API key required), conditions it, delineates
the catchment + river network + subbasins, and lets you export results.

Usage:
    python web_app.py
    Open http://localhost:5000 in your browser.
"""

import os, sys, io, json, math, zipfile, tempfile, warnings, traceback
import numpy as np
import rasterio
import rasterio.features
import rasterio.transform as rio_transform
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
from scipy.ndimage import distance_transform_edt
from pysheds.grid import Grid
from collections import deque
import requests

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# AWS Terrain Tiles (public, no auth)
TILE_URL  = "https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif"
TILE_DIR  = os.path.join(tempfile.gettempdir(), "catchment_dem_tiles")
os.makedirs(TILE_DIR, exist_ok=True)

# ── D8 lookup ─────────────────────────────────────────────────────────────────
D8_DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)
D8_DELTAS = {64:(-1,0), 128:(-1,1), 1:(0,1), 2:(1,1),
             4:(1,0),   8:(1,-1),   16:(0,-1), 32:(-1,-1)}
DELTA_TO_DIR = {v: k for k, v in D8_DELTAS.items()}

# ── Global state ──────────────────────────────────────────────────────────────
_cache = {}          # Conditioned DEM + fdir/acc
_last_result = {}    # Most recent delineation


# ═══════════════════════════════════════════════════════════════════════════════
#  DEM DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def _wgs84_to_tile(lat, lon, zoom):
    """WGS84 → XYZ tile indices."""
    n   = 2 ** zoom
    x   = int((lon + 180.0) / 360.0 * n)
    lr  = math.radians(lat)
    y   = int((1.0 - math.asinh(math.tan(lr)) / math.pi) / 2.0 * n)
    return x, y


def _select_zoom(radius_km, lat):
    """Choose zoom level that gives ~3-4 tiles across the requested area."""
    # Each tile at zoom Z covers ≈ 40075*cos(lat)/2^Z km wide
    circ_km = 40075.0 * abs(math.cos(math.radians(lat)))
    target  = radius_km * 2.0 / 3.5      # want ~3.5 tiles across diameter
    zoom    = round(math.log2(max(circ_km / target, 1)))
    return int(max(8, min(13, zoom)))


def _download_tile(z, x, y):
    """Download one AWS terrain GeoTIFF tile. Returns local path or None."""
    path = os.path.join(TILE_DIR, f"z{z}_x{x}_y{y}.tif")
    if os.path.exists(path) and os.path.getsize(path) > 100:
        return path
    url = TILE_URL.format(z=z, x=x, y=y)
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            return path
    except Exception as e:
        print(f"  Tile download error ({z}/{x}/{y}): {e}")
    return None


def fetch_dem(lat, lon, radius_km=20):
    """
    Download, merge and reproject DEM tiles for the area around (lat, lon).
    Returns path to a WGS84 GeoTIFF.  Tiles are cached locally.
    """
    zoom   = _select_zoom(radius_km, lat)
    dlat   = radius_km / 111.0
    dlon   = radius_km / (111.0 * abs(math.cos(math.radians(lat))) + 1e-9)

    west, east   = lon - dlon, lon + dlon
    south, north = lat - dlat, lat + dlat

    # Tile range (y increases southward)
    x0, y0 = _wgs84_to_tile(north, west,  zoom)
    x1, y1 = _wgs84_to_tile(south, east,  zoom)

    # Cap to avoid runaway downloads
    if (x1-x0+1)*(y1-y0+1) > 25:
        print(f"  Warning: clamping tile grid ({x1-x0+1}×{y1-y0+1}) to 5×5")
        x1 = min(x1, x0+4); y1 = min(y1, y0+4)

    print(f"  Downloading {(x1-x0+1)*(y1-y0+1)} tiles "
          f"(zoom={zoom}, grid={x1-x0+1}×{y1-y0+1})…")

    paths = []
    for xi in range(x0, x1+1):
        for yi in range(y0, y1+1):
            p = _download_tile(zoom, xi, yi)
            if p:
                paths.append(p)

    if not paths:
        raise RuntimeError("Failed to download any DEM tiles. "
                           "Check internet connection.")

    NODATA = -9999.0   # explicit nodata value throughout pipeline

    # Merge tiles
    datasets = [rasterio.open(p) for p in paths]
    mosaic, mosaic_tf = rio_merge(datasets, nodata=NODATA)
    src_crs = datasets[0].crs
    meta = datasets[0].meta.copy()
    for ds in datasets:
        ds.close()

    # AWS int16 tiles use -32768 as nodata — replace with our sentinel
    arr = mosaic[0].astype(float)
    arr[arr <= -32767] = NODATA
    arr[arr > 9000]    = NODATA    # above Everest = tile garbage
    mosaic[0] = arr

    meta.update(height=mosaic.shape[1], width=mosaic.shape[2],
                transform=mosaic_tf, nodata=NODATA, dtype="float32")

    tmp_merged = os.path.join(TILE_DIR, f"merged_{zoom}_{x0}_{y0}_{x1}_{y1}.tif")
    with rasterio.open(tmp_merged, "w", **meta) as dst:
        dst.write(mosaic[0:1].astype("float32"))

    # Reproject to WGS84 if tiles are in Web Mercator
    dst_crs = CRS.from_epsg(4326)
    if src_crs and src_crs.to_epsg() == 4326:
        return tmp_merged

    print("  Reprojecting to WGS84…")
    tmp_wgs = os.path.join(TILE_DIR,
                           f"wgs84_{zoom}_{x0}_{y0}_{x1}_{y1}.tif")
    with rasterio.open(tmp_merged) as src:
        tf_dst, w_dst, h_dst = calculate_default_transform(
            src_crs or CRS.from_epsg(3857), dst_crs,
            src.width, src.height, *src.bounds
        )
        meta2 = src.meta.copy()
        meta2.update(crs=dst_crs, transform=tf_dst,
                     width=w_dst, height=h_dst, nodata=NODATA)
        with rasterio.open(tmp_wgs, "w", **meta2) as dst:
            reproject(source=rasterio.band(src, 1),
                      destination=rasterio.band(dst, 1),
                      src_crs=src_crs or CRS.from_epsg(3857),
                      dst_crs=dst_crs,
                      src_nodata=NODATA, dst_nodata=NODATA,
                      resampling=Resampling.bilinear)
    return tmp_wgs


# ═══════════════════════════════════════════════════════════════════════════════
#  DEM CONDITIONING
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_condition(dem_path):
    """Condition DEM and compute flow routing. Populates _cache."""
    global _cache
    print(f"  Conditioning DEM: {os.path.basename(dem_path)}")

    with rasterio.open(dem_path) as src:
        raw    = src.read(1).astype(float)
        nodata = src.nodata
        affine = src.transform
        crs    = src.crs
        nrows, ncols = src.height, src.width

    # Mask nodata: explicit tag + AWS int16 sentinel + impossible elevations
    if nodata is not None:
        raw[raw == nodata] = np.nan
    raw[raw <= -9990]  = np.nan   # our sentinel
    raw[raw <= -32767] = np.nan   # AWS int16 nodata
    raw[raw >   9000]  = np.nan   # above Everest = garbage
    print(f"  Valid cells: {np.isfinite(raw).sum()} / {raw.size}  "
          f"elev range: {np.nanmin(raw):.1f}–{np.nanmax(raw):.1f} m")

    left   = affine.c
    top    = affine.f
    right  = left + affine.a * ncols
    bottom = top  + affine.e * nrows
    bounds_wgs = (left, bottom, right, top)

    # Assign WGS84 if no CRS
    if crs is None:
        tmp2 = dem_path.replace(".tif", "_crs.tif")
        with rasterio.open(dem_path) as s:
            profile = s.profile.copy()
        profile["crs"] = CRS.from_epsg(4326)
        with rasterio.open(dem_path) as s:
            data = s.read()
        with rasterio.open(tmp2, "w", **profile) as d:
            d.write(data)
        dem_path = tmp2
        crs = CRS.from_epsg(4326)

    grid    = Grid.from_raster(dem_path)
    dem_ps  = grid.read_raster(dem_path)
    pit     = grid.fill_pits(dem_ps)
    dep     = grid.fill_depressions(pit)
    flat    = grid.resolve_flats(dep)
    fdir_ps = grid.flowdir(flat, dirmap=D8_DIRMAP)
    acc_ps  = grid.accumulation(fdir_ps, dirmap=D8_DIRMAP)

    fdir_arr = np.array(fdir_ps).astype(np.int32)
    acc_arr  = np.array(acc_ps).astype(float)

    _cache.clear()
    _cache.update(
        dem_arr=raw, affine=affine, crs=crs,
        fdir_arr=fdir_arr, acc_arr=acc_arr,
        bounds_wgs=bounds_wgs, shape=(nrows, ncols),
    )
    print(f"  Ready. Shape={nrows}×{ncols}  "
          f"bounds={[round(v,4) for v in bounds_wgs]}")
    return _cache


def _dem_covers(lat, lon, margin=0.005):
    """True if current cached DEM covers this point (with margin)."""
    if not _cache:
        return False
    w = _cache["bounds_wgs"]
    return (w[0]+margin <= lon <= w[2]-margin and
            w[1]+margin <= lat <= w[3]-margin)


# ═══════════════════════════════════════════════════════════════════════════════
#  HYDROLOGICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def latlon_to_rowcol(lat, lon):
    affine = _cache["affine"]
    nrows, ncols = _cache["shape"]
    row, col = rio_transform.rowcol(affine, lon, lat)
    return int(np.clip(row, 0, nrows-1)), int(np.clip(col, 0, ncols-1))


def snap_to_stream(row, col, threshold, radius=15):
    acc = _cache["acc_arr"]
    nrows, ncols = _cache["shape"]
    best_r, best_c, best_a = row, col, -1
    for dr in range(-radius, radius+1):
        for dc in range(-radius, radius+1):
            nr, nc = row+dr, col+dc
            if 0 <= nr < nrows and 0 <= nc < ncols:
                a = acc[nr, nc]
                if a >= threshold and a > best_a:
                    best_a, best_r, best_c = a, nr, nc
    return best_r, best_c


def bfs_upstream(start_r, start_c):
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


def compute_strahler(stream_mask):
    fdir_arr = _cache["fdir_arr"]
    acc_arr  = _cache["acc_arr"]
    nrows, ncols = _cache["shape"]
    order = np.zeros((nrows, ncols), dtype=np.int32)
    for r, c in sorted(zip(*np.where(stream_mask)),
                       key=lambda rc: acc_arr[rc[0], rc[1]]):
        up = [order[r+dr, c+dc]
              for (dr, dc) in D8_DELTAS.values()
              if (0 <= r+dr < nrows and 0 <= c+dc < ncols
                  and stream_mask[r+dr, c+dc]
                  and fdir_arr[r+dr, c+dc] == DELTA_TO_DIR.get((-dr, -dc)))]
        if not up:
            order[r, c] = 1
        else:
            mx = max(up)
            order[r, c] = mx+1 if up.count(mx) >= 2 else mx
    return order


def _no_holes(geom):
    """Strip interior rings (holes) from a Polygon or MultiPolygon."""
    from shapely.geometry import Polygon, MultiPolygon
    if geom is None:
        return None
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    return geom


def mask_to_polygon(mask):
    """Vectorise a binary mask. Returns hole-free polygon or None."""
    affine = _cache["affine"]
    geoms = [shape(s) for s, v in
             rasterio.features.shapes(mask.astype(np.uint8),
                                      mask=mask.astype(np.uint8),
                                      transform=affine) if v == 1]
    if not geoms:
        return None
    return _no_holes(unary_union(geoms))


def vectorize_labels(label_arr, catch_mask, basin_meta_list):
    """
    Vectorise all subbasin labels in one pass so adjacent polygons
    share exact boundaries (no gaps).  Returns dict {basin_id: polygon}.
    """
    affine = _cache["affine"]
    polys  = {}
    for geom_dict, val in rasterio.features.shapes(
            label_arr.astype(np.int32),
            mask=catch_mask.astype(np.uint8),
            transform=affine):
        val = int(val)
        if val == 0:
            continue
        g = shape(geom_dict)
        polys[val] = unary_union([polys[val], g]) if val in polys else g

    # Remove holes from every subbasin polygon
    return {k: _no_holes(v) for k, v in polys.items()}


def build_river_network(stream_mask, strahler):
    """Node-based polyline tracing — fully connected network."""
    fdir_arr = _cache["fdir_arr"]
    acc_arr  = _cache["acc_arr"]
    affine   = _cache["affine"]
    nrows, ncols = _cache["shape"]

    n_up = np.zeros((nrows, ncols), dtype=np.int8)
    n_dn = np.zeros((nrows, ncols), dtype=np.int8)
    for r, c in zip(*np.where(stream_mask)):
        for (dr, dc) in D8_DELTAS.values():
            nr, nc = r+dr, c+dc
            if 0 <= nr < nrows and 0 <= nc < ncols and stream_mask[nr, nc]:
                needed = DELTA_TO_DIR.get((-dr, -dc))
                if needed and fdir_arr[nr, nc] == needed:
                    n_up[r, c] += 1
        d = fdir_arr[r, c]
        if d in D8_DELTAS:
            dr, dc = D8_DELTAS[d]
            nr, nc = r+dr, c+dc
            if 0 <= nr < nrows and 0 <= nc < ncols and stream_mask[nr, nc]:
                n_dn[r, c] = 1

    nodes = {(r, c) for r, c in zip(*np.where(stream_mask))
             if n_up[r, c] == 0 or n_up[r, c] >= 2 or n_dn[r, c] == 0}

    features = []
    for (sr, sc) in nodes:
        coords, cells = [], []
        r, c = sr, sc
        while True:
            x, y = rio_transform.xy(affine, r, c)
            coords.append([x, y]); cells.append((r, c))
            d = fdir_arr[r, c]
            if d not in D8_DELTAS: break
            dr, dc = D8_DELTAS[d]
            nr, nc = r+dr, c+dc
            if not (0 <= nr < nrows and 0 <= nc < ncols) or not stream_mask[nr, nc]: break
            r, c = nr, nc
            if (r, c) in nodes:
                x2, y2 = rio_transform.xy(affine, r, c)
                coords.append([x2, y2]); cells.append((r, c))
                break
        if len(coords) < 2: continue
        ords = [int(strahler[rr, cc]) for rr, cc in cells if strahler[rr, cc] > 0]
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"order": max(ords) if ords else 1,
                           "acc": int(acc_arr[sr, sc]),
                           "max_acc": int(max(acc_arr[rr, cc] for rr, cc in cells))}
        })
    return {"type": "FeatureCollection", "features": features}


def delineate(lat, lon, radius_km, acc_threshold, min_order, min_basin_pct):
    """Full pipeline: fetch DEM (if needed) + delineate."""
    # ── 1. Ensure DEM covers this point ──────────────────────────────────────
    if not _dem_covers(lat, lon):
        print(f"[DEM] Fetching tiles for ({lat:.4f}, {lon:.4f}) "
              f"radius={radius_km} km…")
        dem_path = fetch_dem(lat, lon, radius_km)
        load_and_condition(dem_path)

    nrows, ncols = _cache["shape"]

    # ── 2. Click → pixel → snap ───────────────────────────────────────────────
    row, col   = latlon_to_rowcol(lat, lon)
    s_row, s_col = snap_to_stream(row, col, acc_threshold)

    # ── 3. Catchment ──────────────────────────────────────────────────────────
    catch_mask = bfs_upstream(s_row, s_col)
    n_catch = int(catch_mask.sum())
    if n_catch < 20:
        return {"error": "Catchment too small. Try a lower accumulation "
                "threshold or click closer to a stream."}

    # ── 4. Stream network ─────────────────────────────────────────────────────
    stream_mask = (_cache["acc_arr"] >= acc_threshold) & catch_mask
    strahler    = compute_strahler(stream_mask)

    # ── 5. Subbasin pour points ───────────────────────────────────────────────
    pour_points = [(s_row, s_col)]
    fdir_arr    = _cache["fdir_arr"]
    for r, co in zip(*np.where(stream_mask)):
        if strahler[r, co] < min_order:
            continue
        sig_up = [(nr, nc)
                  for (dr, dc) in D8_DELTAS.values()
                  for nr, nc in [(r+dr, co+dc)]
                  if (0 <= nr < nrows and 0 <= nc < ncols
                      and stream_mask[nr, nc]
                      and fdir_arr[nr, nc] == DELTA_TO_DIR.get((-dr, -dc))
                      and strahler[nr, nc] >= min_order)]
        if len(sig_up) >= 2:
            pour_points.append((r, co))

    # ── 6. Subbasins ─────────────────────────────────────────────────────────
    min_cells = max(1, int(n_catch * min_basin_pct / 100))
    label     = np.zeros((nrows, ncols), dtype=np.int32)
    remaining = catch_mask.copy()
    basin_meta = []
    acc_arr    = _cache["acc_arr"]

    for lid, (pr, pc) in enumerate(
            sorted(pour_points, key=lambda rc: acc_arr[rc[0], rc[1]]), 1):
        if not remaining[pr, pc]: continue
        sub = bfs_upstream(pr, pc) & remaining
        if sub.sum() < min_cells: continue
        label[sub] = lid
        remaining[sub] = False
        basin_meta.append({"id": lid, "pr": int(pr), "pc": int(pc)})

    # Gap-fill
    unlabeled = catch_mask & (label == 0)
    if unlabeled.any() and label.max() > 0:
        labeled = label > 0
        _, nearest = distance_transform_edt(~labeled, return_indices=True)
        label[unlabeled] = label[nearest[0][unlabeled], nearest[1][unlabeled]]

    # ── 7. Vectorise ──────────────────────────────────────────────────────────
    affine = _cache["affine"]

    catch_poly = mask_to_polygon(catch_mask)
    catch_fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(catch_poly),
         "properties": {"cells": n_catch}}
    ]} if catch_poly else {"type": "FeatureCollection", "features": []}

    # Vectorise all subbasins at once — shared boundaries, no gaps, no holes
    label_polys = vectorize_labels(label, catch_mask, basin_meta)

    sub_features = []
    for m in basin_meta:
        poly = label_polys.get(m["id"])
        if poly:
            ox, oy = rio_transform.xy(affine, m["pr"], m["pc"])
            sub_features.append({
                "type": "Feature", "geometry": mapping(poly),
                "properties": {"id": m["id"], "cells": int((label == m["id"]).sum()),
                               "out_x": round(ox, 6), "out_y": round(oy, 6)}
            })

    riv_fc = build_river_network(stream_mask, strahler)

    ox, oy = rio_transform.xy(affine, s_row, s_col)
    out_fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature",
         "geometry": {"type": "Point", "coordinates": [ox, oy]},
         "properties": {"acc": int(acc_arr[s_row, s_col])}}
    ]}

    result = {
        "catchment": catch_fc,
        "subbasins":  {"type": "FeatureCollection", "features": sub_features},
        "rivers":     riv_fc,
        "outlet":     out_fc,
        "stats": {
            "catchment_cells": n_catch,
            "n_subbasins":     len(sub_features),
            "n_rivers":        len(riv_fc["features"]),
            "max_strahler":    int(strahler[stream_mask].max()) if stream_mask.any() else 0,
            "outlet_lat":      round(oy, 6),
            "outlet_lon":      round(ox, 6),
            "snapped":         (s_row != row or s_col != col),
            "dem_res_m":       round(abs(affine.a) * 111320, 1),
        },
    }
    _last_result.clear()
    _last_result.update(result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/fetch_dem", methods=["POST"])
def api_fetch_dem():
    """Download + condition DEM for a location. Returns bounds & stats."""
    data = request.json or {}
    try:
        lat       = float(data["lat"])
        lon       = float(data["lon"])
        radius_km = float(data.get("radius_km", 20))
        print(f"[DEM] Fetching ({lat:.4f}, {lon:.4f}) radius={radius_km} km…")
        dem_path = fetch_dem(lat, lon, radius_km)
        load_and_condition(dem_path)
        w   = _cache["bounds_wgs"]
        acc = _cache["acc_arr"]
        p95 = float(np.percentile(acc[acc > 0], 95)) if (acc > 0).any() else 500
        return jsonify({
            "bounds": {"south": w[1], "west": w[0], "north": w[3], "east": w[2]},
            "res_m":  round(abs(_cache["affine"].a) * 111320, 1),
            "shape":  list(_cache["shape"]),
            "acc_p95": p95,
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/dem_info")
def api_dem_info():
    if not _cache:
        return jsonify({"loaded": False})
    w   = _cache["bounds_wgs"]
    acc = _cache["acc_arr"]
    return jsonify({
        "loaded": True,
        "bounds": {"south": w[1], "west": w[0], "north": w[3], "east": w[2]},
        "shape":  list(_cache["shape"]),
        "acc": {
            "max": float(acc.max()),
            "p95": float(np.percentile(acc, 95)),
            "p99": float(np.percentile(acc, 99)),
        },
        "res_m": round(abs(_cache["affine"].a) * 111320, 1),
    })


@app.route("/api/dem_preview.png")
def api_dem_preview():
    if not _cache:
        import base64
        px = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9Q"
            "DwADhgGAWjR9awAAAABJRU5ErkJggg==")
        return send_file(io.BytesIO(px), mimetype="image/png")

    dem   = _cache["dem_arr"].copy()   # already NaN-masked in load_and_condition
    valid = np.isfinite(dem)

    if not valid.any():
        import base64
        px = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9Q"
            "DwADhgGAWjR9awAAAABJRU5ErkJggg==")
        return send_file(io.BytesIO(px), mimetype="image/png")

    valid_vals = dem[valid]
    # Use 2–98th percentile so edge outliers don't collapse the colour range
    vmin = float(np.percentile(valid_vals, 2))
    vmax = float(np.percentile(valid_vals, 98))
    if vmax - vmin < 1.0:
        vmax = vmin + 1.0
    print(f"  DEM preview: vmin={vmin:.1f}  vmax={vmax:.1f}  "
          f"valid={valid.sum()}")

    dem_f = np.where(valid, np.clip(dem, vmin, vmax), vmin)

    # Hillshade for relief shading
    relief    = vmax - vmin
    vert_exag = float(np.clip(500.0 / max(relief, 1.0), 1.0, 15.0))
    ls = LightSource(azdeg=315, altdeg=40)
    hs = ls.hillshade(dem_f, vert_exag=vert_exag)

    # Standard hypsometric tint: green (low) → brown → white (high)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = plt.cm.terrain(norm(dem_f))

    # Blend with hillshade for 3-D relief
    rgba[..., :3] = rgba[..., :3] * (0.55 + 0.45 * hs[..., None])

    # Fully opaque where valid, transparent outside
    rgba[..., 3]  = np.where(valid, 1.0, 0.0)

    h, w = dem.shape
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    ax.imshow(rgba, origin="upper", interpolation="bilinear")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight",
                pad_inches=0, transparent=True)
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/delineate", methods=["POST"])
def api_delineate():
    data = request.json or {}
    try:
        result = delineate(
            lat          = float(data["lat"]),
            lon          = float(data["lon"]),
            radius_km    = float(data.get("radius_km", 20)),
            acc_threshold= int(data.get("acc_threshold", 500)),
            min_order    = int(data.get("min_order", 2)),
            min_basin_pct= float(data.get("min_basin_pct", 2.0)),
        )
        # Include updated DEM bounds so frontend can refresh overlay
        if _cache and "error" not in result:
            w = _cache["bounds_wgs"]
            result["dem_bounds"] = {
                "south": w[1], "west": w[0], "north": w[3], "east": w[2]
            }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e),
                        "trace": traceback.format_exc()}), 500


@app.route("/api/export/<fmt>")
def api_export(fmt):
    if not _last_result.get("catchment"):
        return jsonify({"error": "No results to export."}), 400

    layers = [
        ("catchment", gpd.GeoDataFrame.from_features(
            _last_result["catchment"]["features"], crs=4326)),
        ("subbasins",  gpd.GeoDataFrame.from_features(
            _last_result["subbasins"]["features"],  crs=4326)),
        ("rivers",     gpd.GeoDataFrame.from_features(
            _last_result["rivers"]["features"],     crs=4326)),
        ("outlet",     gpd.GeoDataFrame.from_features(
            _last_result["outlet"]["features"],     crs=4326)),
    ]

    if fmt == "gpkg":
        tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        tmp.close()
        for name, gdf in layers:
            if not gdf.empty:
                gdf.to_file(tmp.name, layer=name, driver="GPKG")
        return send_file(tmp.name,
                         mimetype="application/geopackage+sqlite3",
                         as_attachment=True,
                         download_name="catchment_results.gpkg")

    elif fmt == "shp":
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, gdf in layers:
                if gdf.empty: continue
                with tempfile.TemporaryDirectory() as td:
                    p = os.path.join(td, f"{name}.shp")
                    gdf.to_file(p)
                    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                        fp = p.replace(".shp", ext)
                        if os.path.exists(fp):
                            zf.write(fp, f"{name}{ext}")
        buf.seek(0)
        return send_file(buf, mimetype="application/zip",
                         as_attachment=True,
                         download_name="catchment_results.zip")

    return jsonify({"error": f"Unknown format: {fmt}"}), 400


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Global DEM mode — tiles downloaded on demand from AWS Terrain Tiles.")
    print("  Open your browser at:  http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
