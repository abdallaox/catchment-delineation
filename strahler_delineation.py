"""
Strahler-Order Watershed Delineation
=====================================
Logic:
  1. Extract stream network from flow accumulation (threshold).
  2. Compute Strahler stream order for every stream cell using BFS from headwaters.
  3. Overall catchment  = delineated from the outlet of the MAIN (highest-order) river.
  4. Major subbasins    = one per confluence where two tributaries of order >= min_order
                          join (Strahler order increases at that cell).
  5. Non-overlapping labelling covers the entire DEM.

Usage:
    python strahler_delineation.py --dem dem.tif
    python strahler_delineation.py --dem dem.tif --threshold 400 --min_order 2
"""

import sys, os, argparse, warnings, uuid, tempfile
from collections import deque
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from pysheds.grid import Grid
from shapely.geometry import shape, Point, LineString
from shapely.ops import unary_union
from scipy.ndimage import distance_transform_edt

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# D8 encoding used by pysheds
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)
D8 = {64:(-1,0), 128:(-1,1), 1:(0,1), 2:(1,1),
       4:(1,0),   8:(1,-1), 16:(0,-1), 32:(-1,-1)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_crs(path, fallback="EPSG:4326"):
    with rasterio.open(path) as src:
        if src.crs and src.crs.to_string():
            return path, src.crs
        print(f"[INFO] No CRS — assigning {fallback}.")
        meta = src.meta.copy(); meta["crs"] = fallback
        tmp  = os.path.join(tempfile.gettempdir(), f"dem_{uuid.uuid4().hex}.tif")
        with rasterio.open(tmp, "w", **meta) as dst:
            dst.write(src.read())
        return tmp, rasterio.crs.CRS.from_user_input(fallback)

def rc_to_xy(t, r, c):
    return float(t.c + (c+0.5)*t.a), float(t.f + (r+0.5)*t.e)


# ── 1. Load & condition ───────────────────────────────────────────────────────

def load_condition(dem_path):
    print("[1/7] Loading and conditioning DEM...")
    dem_path, crs = ensure_crs(dem_path)
    grid = Grid.from_raster(dem_path); grid.crs = crs
    dem  = grid.read_raster(dem_path)
    dem  = grid.fill_pits(dem)
    dem  = grid.fill_depressions(dem)
    dem  = grid.resolve_flats(dem)
    fdir = grid.flowdir(dem, dirmap=DIRMAP)
    n_sinks = int(np.sum(np.array(fdir) == 0))
    print(f"      Sinks after conditioning: {n_sinks}")
    return grid, dem, crs


# ── 2. Flow routing ───────────────────────────────────────────────────────────

def flow_routing(grid, dem):
    print("[2/7] Flow direction + accumulation...")
    fdir = grid.flowdir(dem, dirmap=DIRMAP)
    acc  = grid.accumulation(fdir, dirmap=DIRMAP)
    return fdir, acc


# ── 3. Strahler order ─────────────────────────────────────────────────────────

def compute_strahler(fdir_arr, acc_arr, threshold):
    """
    Compute Strahler stream order for all stream cells.
    Rules:
      - Headwaters (no upstream stream cells) = order 1.
      - Where two streams of the SAME order N merge -> order N+1.
      - Where streams of different orders merge -> max(orders) unchanged.
    """
    print("[3/7] Computing Strahler stream order...")
    stream = acc_arr >= threshold
    rows, cols = fdir_arr.shape

    # Count directed inflow for every stream cell
    inflow = np.zeros((rows, cols), dtype=np.int32)
    for dval, (dr, dc) in D8.items():
        sr, sc = np.where((fdir_arr == dval) & stream)
        dr2, dc2 = sr + dr, sc + dc
        ok = (dr2 >= 0)&(dr2 < rows)&(dc2 >= 0)&(dc2 < cols)
        dr2, dc2 = dr2[ok], dc2[ok]
        is_str = stream[dr2, dc2]
        np.add.at(inflow, (dr2[is_str], dc2[is_str]), 1)

    headwaters = stream & (inflow == 0)
    order  = np.zeros((rows, cols), dtype=np.int32)
    order[headwaters] = 1

    # BFS from headwaters downstream
    remaining = inflow.copy()
    upstream_orders = {}            # (r,c) -> [list of upstream Strahler orders]

    queue = deque(zip(*np.where(headwaters)))
    while queue:
        r, c = queue.popleft()
        cur_order = order[r, c]
        if cur_order == 0:
            continue
        dval = fdir_arr[r, c]
        if dval == 0 or dval not in D8:
            continue
        dr, dc = D8[dval]
        nr, nc = r+dr, c+dc
        if not (0 <= nr < rows and 0 <= nc < cols):
            continue
        if not stream[nr, nc]:
            continue
        upstream_orders.setdefault((nr, nc), []).append(cur_order)
        remaining[nr, nc] -= 1
        if remaining[nr, nc] <= 0:
            ups   = upstream_orders.get((nr, nc), [cur_order])
            mx    = max(ups)
            order[nr, nc] = mx + 1 if ups.count(mx) >= 2 else mx
            queue.append((nr, nc))

    max_order = int(order[stream].max()) if stream.any() else 1
    print(f"      Max Strahler order: {max_order}")
    for o in range(1, max_order+1):
        print(f"        Order {o}: {int(np.sum(order == o)):,} cells")
    return order, stream, max_order


# ── 4. Main outlet (highest-order stream, nearest to DEM exit) ────────────────

def find_main_outlet(grid, fdir_arr, acc_arr, order_arr, stream):
    """
    The main outlet is the highest-Strahler stream cell that is immediately
    upstream of a DEM boundary exit (or the cell that flows off the edge).
    We find all boundary-exit stream cells and pick the one with max accumulation.
    """
    print("[4/7] Finding main outlet...")
    rows, cols = fdir_arr.shape
    t = grid.affine

    # Cells whose flow direction exits the DEM boundary
    exit_cells = []
    for dval, (dr, dc) in D8.items():
        sr, sc = np.where((fdir_arr == dval) & stream)
        dst_r, dst_c = sr+dr, sc+dc
        exits = (dst_r<0)|(dst_r>=rows)|(dst_c<0)|(dst_c>=cols)
        for r2, c2 in zip(sr[exits], sc[exits]):
            exit_cells.append((r2, c2, int(acc_arr[r2, c2])))

    if exit_cells:
        # Pick the exit cell with the highest accumulation
        exit_cells.sort(key=lambda x: x[2], reverse=True)
        r0, c0, _ = exit_cells[0]
    else:
        # Fallback: global max accumulation
        r0, c0 = np.unravel_index(np.argmax(acc_arr), acc_arr.shape)

    x0, y0 = rc_to_xy(t, r0, c0)
    print(f"      Main outlet: ({x0:.5f}, {y0:.5f})  "
          f"acc={int(acc_arr[r0,c0]):,}  order={int(order_arr[r0,c0])}")
    return r0, c0, x0, y0


# ── 5. Overall catchment ──────────────────────────────────────────────────────

def delineate_main_catchment(grid, fdir, acc_arr, order_arr, stream, r0, c0, crs):
    """
    Delineate the catchment from the main outlet.
    Try direct row/col first; fall back to tracing upstream via fdir array.
    """
    print("[5/7] Delineating overall catchment...")
    rows, cols = acc_arr.shape
    t = grid.affine
    x0, y0 = rc_to_xy(t, r0, c0)

    # Always use our own BFS upstream trace — more reliable than pysheds
    # for boundary-exit outlet cells where pysheds snapping goes wrong.
    print("      Tracing upstream via BFS (reliable for boundary outlets)...")
    catch_arr = _trace_upstream(np.array(fdir), r0, c0, rows, cols)
    n   = int(catch_arr.sum())
    pct = 100 * n / acc_arr.size
    print(f"      Catchment: {n:,} cells ({pct:.1f}% of DEM)")

    # Cross-check: if BFS gives far fewer cells than acc value, warn
    expected = int(acc_arr[r0, c0])
    if n < expected * 0.5:
        print(f"      [WARN] Traced {n:,} cells but acc={expected:,}. "
              f"DEM may be a clip of a larger watershed.")

    polys = [shape(g) for g, v in
              shapes(catch_arr.astype(np.uint8), transform=grid.affine) if v == 1]
    poly = unary_union(polys) if polys else None
    gdf  = gpd.GeoDataFrame({"id":[1], "n_cells":[int(catch_arr.sum())]},
                              geometry=[poly] if poly else [], crs=crs)
    return gdf, catch_arr


def _trace_upstream(fdir_arr, r0, c0, rows, cols):
    """BFS upstream from (r0,c0) using reverse D8."""
    # Reverse D8: what direction value points TO (dr,dc)?
    # If a source cell at (r,c) has fdir=dval, it flows to (r+dr, c+dc).
    # To go upstream from (nr,nc), find cells (r,c) s.t. (r+dr,c+dc)==(nr,nc)
    # i.e. r=nr-dr, c=nc-dc and fdir[r,c]==dval
    visited = np.zeros((rows, cols), dtype=bool)
    visited[r0, c0] = True
    queue = deque([(r0, c0)])
    while queue:
        nr, nc = queue.popleft()
        for dval, (dr, dc) in D8.items():
            pr, pc = nr-dr, nc-dc
            if 0 <= pr < rows and 0 <= pc < cols:
                if not visited[pr, pc] and fdir_arr[pr, pc] == dval:
                    visited[pr, pc] = True
                    queue.append((pr, pc))
    return visited


# ── 6. Major confluences (Strahler order increases) ───────────────────────────

def find_major_confluences(grid, fdir_arr, acc_arr, order_arr, stream,
                            min_order, catch_arr):
    """
    A major confluence is a stream cell where:
      - Strahler order is > min_order  (order increases here)
      - At least 2 upstream tributaries of the same order flow in
      - It lies within the overall catchment
    These are exactly the cells where the order steps up.
    """
    print(f"[6/7] Finding major confluences (Strahler order >= {min_order})...")
    rows, cols = fdir_arr.shape
    t = grid.affine

    # For each stream cell, find the orders of all upstream stream cells
    # A cell where two same-order tributaries merge will have order > each upstream
    inflow_orders = {}   # (r,c) -> list of upstream orders
    for dval, (dr, dc) in D8.items():
        sr, sc = np.where((fdir_arr == dval) & stream)
        dst_r, dst_c = sr+dr, sc+dc
        ok = (dst_r>=0)&(dst_r<rows)&(dst_c>=0)&(dst_c<cols)
        dst_r, dst_c = dst_r[ok], dst_c[ok]
        sr_ok, sc_ok = sr[ok], sc[ok]
        is_str = stream[dst_r, dst_c]
        for r2, c2, r1, c1 in zip(dst_r[is_str], dst_c[is_str],
                                    sr_ok[is_str], sc_ok[is_str]):
            inflow_orders.setdefault((r2, c2), []).append(int(order_arr[r1, c1]))

    pour_pts = []
    for (r, c), ups in inflow_orders.items():
        cur_ord = int(order_arr[r, c])
        if cur_ord < min_order:
            continue
        # Order increases here → at least two tributaries of the same lower order
        max_up = max(ups) if ups else 0
        n_max  = ups.count(max_up)
        if n_max < 2:
            continue
        # Must be inside overall catchment
        if not catch_arr[r, c]:
            continue
        x, y = rc_to_xy(t, r, c)
        pour_pts.append({"x": x, "y": y, "acc": float(acc_arr[r, c]),
                          "order": cur_ord, "type": "confluence"})

    # Add main outlet
    r0, c0 = np.unravel_index(
        np.argmax(np.where(catch_arr, acc_arr, 0)), acc_arr.shape)
    x0, y0 = rc_to_xy(t, r0, c0)
    pour_pts.append({"x": x0, "y": y0, "acc": float(acc_arr[r0,c0]),
                      "order": int(order_arr[r0,c0]), "type": "outlet"})

    pour_pts.sort(key=lambda p: p["acc"], reverse=True)
    print(f"      {len(pour_pts)-1} major confluences + 1 main outlet.")
    return pour_pts


# ── 7. Non-overlapping subbasin labelling ─────────────────────────────────────

def build_subbasins(grid, fdir, fdir_arr, acc_arr, pour_pts, catch_arr, crs):
    print(f"[7/7] Delineating {len(pour_pts)} subbasins...")
    rows, cols = fdir_arr.shape
    t = grid.affine

    # Delineate full catchment for each pour point using BFS upstream trace.
    # We always use _trace_upstream (not pysheds grid.catchment) because
    # pysheds snaps boundary-exit cells incorrectly, producing tiny catchments.
    # The BFS trace follows the flow-direction array directly — always correct.
    cat_list = []
    for i, pt in enumerate(pour_pts):
        rx = int(round((pt["y"] - t.f) / t.e))
        cx = int(round((pt["x"] - t.c) / t.a))
        # Clamp to grid
        rx = max(0, min(rows-1, rx))
        cx = max(0, min(cols-1, cx))
        # Snap to the stream cell with the highest accumulation within 3 pixels
        best_acc, best_r, best_c = -1, rx, cx
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                nr, nc = rx+dr, cx+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if acc_arr[nr, nc] >= pt["acc"] * 0.8 and acc_arr[nr, nc] > best_acc:
                        best_acc, best_r, best_c = acc_arr[nr, nc], nr, nc
        arr = _trace_upstream(fdir_arr, best_r, best_c, rows, cols) & catch_arr
        n   = int(arr.sum())
        cat_list.append((i, float(acc_arr[best_r, best_c]), arr))
        print(f"      Pour point {i+1}/{len(pour_pts)}: "
              f"acc={int(pt['acc']):,}  traced={n:,} cells")

    # Sort upstream-first (ascending accumulation)
    cat_list.sort(key=lambda x: x[1])

    # Label: each cell claimed by its most-upstream pour point
    labels = np.zeros((rows, cols), dtype=np.int32)
    for i, acc_val, arr in cat_list:
        labels[arr & (labels == 0)] = i + 1

    # Propagate labels to uncovered cells inside catchment via downstream walk
    for r0, c0 in zip(*np.where((labels == 0) & catch_arr)):
        if labels[r0, c0] != 0:
            continue
        chain, r, c = [(r0, c0)], r0, c0
        found = 0
        for _ in range(rows + cols):
            dval = fdir_arr[r, c]
            if dval not in D8:
                break
            dr, dc = D8[dval]
            nr, nc = r+dr, c+dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                break
            if labels[nr, nc]:
                found = labels[nr, nc]; break
            chain.append((nr, nc)); r, c = nr, nc
        if found:
            for cr, cc in chain:
                labels[cr, cc] = found

    # Nearest-label fill for any remaining gaps
    if np.any((labels == 0) & catch_arr):
        mask = (labels == 0) & catch_arr
        _, idx = distance_transform_edt(labels == 0, return_indices=True)
        labels[mask] = labels[idx[0][mask], idx[1][mask]]

    print(f"      Unlabeled cells inside catchment: "
          f"{int(np.sum((labels==0) & catch_arr))}")

    # Vectorise
    acc_lu = {i+1: acc for i, acc, _ in cat_list}
    subbasins = []
    for uid in sorted(set(labels[labels > 0])):
        mask = (labels == uid).astype(np.uint8)
        polys = [shape(g) for g,v in shapes(mask, transform=t) if v==1]
        if not polys:
            continue
        poly = unary_union(polys)
        if poly.is_empty:
            continue
        pi = uid - 1
        subbasins.append({
            "id":    uid,
            "order": pour_pts[pi]["order"] if pi < len(pour_pts) else 0,
            "acc":   acc_lu.get(uid, 0),
            "type":  pour_pts[pi]["type"]  if pi < len(pour_pts) else "",
            "geometry": poly,
        })

    gdf_sub = gpd.GeoDataFrame(subbasins, crs=crs)

    # Outlet points
    outlets = [{"id":i,"order":p["order"],"acc":p["acc"],"type":p["type"],
                 "geometry":Point(p["x"],p["y"])} for i,p in enumerate(pour_pts)]
    gdf_out = gpd.GeoDataFrame(outlets, crs=crs)

    print(f"      {len(gdf_sub)} subbasins built.")
    return gdf_sub, gdf_out


# ── Export + map ──────────────────────────────────────────────────────────────

def export_and_map(gdf_sub, gdf_out, gdf_rivers, gdf_catch,
                    order_arr, stream, grid, max_order, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    gdf_sub.to_file(os.path.join(out_dir, "subbasins.shp"))
    gdf_out.to_file(os.path.join(out_dir, "outlets.shp"))
    gdf_catch.to_file(os.path.join(out_dir, "catchment.shp"))
    if not gdf_rivers.empty:
        gdf_rivers.to_file(os.path.join(out_dir, "rivers.shp"))

    print(f"      subbasins.shp  ({len(gdf_sub)})")
    print(f"      outlets.shp    ({len(gdf_out)})")
    print(f"      catchment.shp  (overall boundary)")
    print(f"      rivers.shp     ({len(gdf_rivers)} segments)")

    # ── Map ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.set_aspect("equal")
    ax.set_facecolor("#d4e8f5")
    ax.set_title(
        f"Strahler-Order Watershed  |  {len(gdf_sub)} subbasins  |  "
        f"Max Strahler order: {max_order}",
        fontsize=13, fontweight="bold", pad=12
    )

    # Subbasins
    n    = len(gdf_sub)
    cmap = plt.cm.get_cmap("Pastel1", max(n, 2))
    for idx, (_, row) in enumerate(gdf_sub.iterrows()):
        gpd.GeoDataFrame([row], crs=gdf_sub.crs).plot(
            ax=ax, color=cmap(idx % 9), edgecolor="white",
            linewidth=0.5, alpha=0.70)
    gdf_sub.boundary.plot(ax=ax, edgecolor="#333", linewidth=0.8, zorder=3)

    # Overall catchment boundary
    if not gdf_catch.empty:
        gdf_catch.boundary.plot(ax=ax, edgecolor="black",
                                 linewidth=2.5, zorder=5,
                                 label="Overall catchment")

    # Rivers coloured by Strahler order
    if not gdf_rivers.empty:
        order_colors = plt.cm.get_cmap("cool", max_order + 1)
        mx = float(gdf_rivers["max_acc"].max()) or 1.0
        for _, rrow in gdf_rivers.iterrows():
            rel = rrow["max_acc"] / mx
            o   = min(int(rrow.get("order", 1)), max_order)
            lw  = 0.3 + 3.0 * rel
            col = order_colors(o / max_order)
            gpd.GeoDataFrame([rrow], crs=gdf_rivers.crs).plot(
                ax=ax, color=col, linewidth=lw, zorder=4)

    # Outlet dots scaled by acc + coloured by Strahler order
    conf = gdf_out[gdf_out["type"] == "confluence"]
    main = gdf_out[gdf_out["type"] == "outlet"]
    mx_acc = float(gdf_out["acc"].max()) or 1.0

    if not conf.empty:
        order_colors2 = plt.cm.get_cmap("autumn_r", max_order + 1)
        for _, row in conf.iterrows():
            o   = min(int(row["order"]), max_order)
            sz  = 30 + 250 * (row["acc"] / mx_acc) ** 0.5
            col = order_colors2(o / max_order)
            ax.scatter(row.geometry.x, row.geometry.y, s=sz,
                       c=[col], edgecolors="white", linewidths=0.7, zorder=7)

    if not main.empty:
        ax.scatter(main.geometry.x, main.geometry.y, s=400,
                   c="red", edgecolors="darkred", marker="*",
                   linewidths=1.2, zorder=9,
                   label=f"Main outlet  (acc={int(main['acc'].iloc[0]):,})")

    # Legend
    stream_patches = [
        plt.Line2D([0],[0], color=plt.cm.cool(o/max_order),
                   lw=0.8+2.0*o/max_order,
                   label=f"Order {o} stream")
        for o in range(1, max_order+1)
    ]
    other_patches = [
        mpatches.Patch(fc="#d9d9d9", ec="#333", label="Subbasins"),
        plt.Line2D([0],[0], color="black", lw=2.5, label="Overall catchment"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="orange",
                   markersize=8, label="Major confluence outlet"),
        plt.Line2D([0],[0], marker="*", color="w", markerfacecolor="red",
                   markersize=14, label="Main outlet"),
    ]
    ax.legend(handles=stream_patches + other_patches,
              loc="upper right", fontsize=8, framealpha=0.9, edgecolor="gray",
              ncol=2)

    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "strahler_map.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"      strahler_map.png")


# ── Rivers GDF ────────────────────────────────────────────────────────────────

def extract_rivers(grid, fdir, acc, order_arr, threshold, crs):
    print("[3b] Extracting river network...")
    branches = grid.extract_river_network(fdir, acc >= threshold, dirmap=DIRMAP)
    acc_arr  = np.array(acc)
    t        = grid.affine
    rows, cols = acc_arr.shape
    lines, acc_vals, orders = [], [], []
    for feat in branches["features"]:
        coords = feat["geometry"]["coordinates"]
        if len(coords) < 2:
            continue
        lines.append(LineString(coords))
        seg_acc, seg_ord = [], []
        for x, y in coords:
            c = int((x - t.c) / t.a)
            r = int((y - t.f) / t.e)
            if 0 <= r < rows and 0 <= c < cols:
                seg_acc.append(float(acc_arr[r, c]))
                seg_ord.append(int(order_arr[r, c]))
        acc_vals.append(max(seg_acc) if seg_acc else 0.0)
        orders.append(max(seg_ord) if seg_ord else 1)
    gdf = gpd.GeoDataFrame(
        {"id": range(len(lines)), "max_acc": acc_vals, "order": orders},
        geometry=lines, crs=crs
    )
    print(f"      {len(gdf)} river segments.")
    return gdf


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem",       required=True)
    parser.add_argument("--threshold", type=int, default=None,
                        help="Flow accumulation threshold  (default: auto ~0.5%%)")
    parser.add_argument("--min_order", type=int, default=2,
                        help="Minimum Strahler order for subbasin confluences (default: 2)")
    parser.add_argument("--out_dir",   default="strahler_output")
    args = parser.parse_args()

    grid, dem, crs     = load_condition(args.dem)
    fdir, acc          = flow_routing(grid, dem)
    acc_arr            = np.array(acc)
    fdir_arr           = np.array(fdir)

    threshold = args.threshold or max(30, int(acc_arr.size * 0.005))
    print(f"      Stream threshold: {threshold:,} cells  "
          f"({100*np.sum(acc_arr>=threshold)/acc_arr.size:.1f}% of DEM)")

    order_arr, stream, max_order = compute_strahler(fdir_arr, acc_arr, threshold)
    gdf_rivers  = extract_rivers(grid, fdir, acc, order_arr, threshold, crs)

    r0, c0, x0, y0     = find_main_outlet(grid, fdir_arr, acc_arr, order_arr, stream)
    gdf_catch, catch_arr = delineate_main_catchment(grid, fdir, acc_arr, order_arr,
                                                      stream, r0, c0, crs)
    pour_pts            = find_major_confluences(grid, fdir_arr, acc_arr, order_arr,
                                                  stream, args.min_order, catch_arr)
    gdf_sub, gdf_out    = build_subbasins(grid, fdir, fdir_arr, acc_arr,
                                           pour_pts, catch_arr, crs)

    out_dir = args.out_dir
    if os.path.exists(out_dir):
        out_dir = f"{out_dir}_{uuid.uuid4().hex[:6]}"

    print(f"\nExporting to {out_dir} ...")
    export_and_map(gdf_sub, gdf_out, gdf_rivers, gdf_catch,
                   order_arr, stream, grid, max_order, out_dir)

    print("\n-- Summary --------------------------------------------------")
    print(f"   Max Strahler order : {max_order}")
    print(f"   Min order filter   : {args.min_order}")
    print(f"   Subbasins          : {len(gdf_sub)}")
    print(f"   Major confluences  : {len(pour_pts)-1}")
    print(f"   River segments     : {len(gdf_rivers)}")
    print(f"   Catchment cells    : {int(catch_arr.sum()):,} "
          f"({100*catch_arr.sum()/acc_arr.size:.1f}% of DEM)")
    print(f"   Output             : {os.path.abspath(out_dir)}")
    print("-------------------------------------------------------------")
    print("Done!")


if __name__ == "__main__":
    main()
