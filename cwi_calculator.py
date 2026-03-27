"""
Composite Wetness Index (CWI) Calculator
==========================================
Downloads Environment Agency flood risk data for the study area, calculates
the CWI for each subbasin, and exports QGIS-ready outputs.

CWI = 0.3*fz3 + 0.2*fz2 + 0.25*sw_high + 0.15*sw_med + 0.1*sw_low

All variables are fractions (0-1) of the subbasin area covered by each class.
CWI ranges from 0 (very dry) to 1 (very wet / high flood risk).

Outputs (in outputs/cwi_results/):
  cwi_results.gpkg   — GeoPackage with all layers (open directly in QGIS)
  cwi_style.qml      — QGIS graduated colour style for CWI layer
  cwi_map.png        — Publication-quality map

Usage:
    python cwi_calculator.py
"""

import sys, os, warnings, json, time
import numpy as np
import geopandas as gpd
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from shapely.geometry import shape
from shapely.ops import unary_union

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

SUBBASINS_SHP = "outputs/strahler_results/subbasins.shp"
CATCHMENT_SHP = "outputs/strahler_results/catchment.shp"
RIVERS_SHP    = "outputs/strahler_results/rivers.shp"
OUT_DIR       = "outputs/cwi_results"

CWI_WEIGHTS = {
    "fz3":     0.30,
    "fz2":     0.20,
    "sw_high": 0.25,
    "sw_med":  0.15,
    "sw_low":  0.10,
}

# EA WFS endpoints
EA_FZ_WFS   = ("https://environment.data.gov.uk/spatialdata/"
                "flood-map-for-planning-flood-zones/wfs")
# Combined FZ2+FZ3 layer — filter by flood_zone attribute ('FZ2' or 'FZ3')
EA_FZ_LAYER = ("dataset-04532375-a198-476e-985e-0579a0a11b47:"
               "Flood_Zones_2_3_Rivers_and_Sea")

# Surface water: EA only provides WMS (not WFS). Place a local shapefile/gpkg
# here if you have one downloaded from:
# https://environment.data.gov.uk/dataset/b5aaa28d-6eb9-460e-8d6f-43caa71fbe0e
SW_LOCAL_PATH = "data/surface_water_risk.shp"   # or .gpkg
SW_RISK_COL   = "Risk"     # column name for risk class (High/Medium/Low)


# ── WFS download ──────────────────────────────────────────────────────────────

def wfs_download(base_url, layer_name, bbox, retries=3, timeout=60):
    """
    Download a WFS layer as GeoDataFrame for the given bbox.
    bbox = (minx, miny, maxx, maxy) in EPSG:4326.
    Returns GeoDataFrame or None on failure.
    """
    minx, miny, maxx, maxy = bbox
    # Add small buffer
    buf = 0.002
    minx -= buf; miny -= buf; maxx += buf; maxy += buf

    params = {
        "service":      "WFS",
        "version":      "2.0.0",
        "request":      "GetFeature",
        "typeNames":    layer_name,
        "bbox":         f"{minx},{miny},{maxx},{maxy},EPSG:4326",
        "srsName":      "EPSG:4326",
        "outputFormat": "application/json",
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(base_url, params=params, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("features"):
                    print(f"      [WARN] No features returned for {layer_name}.")
                    return gpd.GeoDataFrame()
                gdf = gpd.GeoDataFrame.from_features(data["features"],
                                                      crs="EPSG:4326")
                print(f"      Downloaded {len(gdf):,} features for {layer_name}.")
                return gdf
            else:
                print(f"      HTTP {resp.status_code} for {layer_name} "
                      f"(attempt {attempt}/{retries})")
        except Exception as e:
            print(f"      Error ({e}) for {layer_name} (attempt {attempt}/{retries})")
        if attempt < retries:
            time.sleep(2)

    return None


def get_capabilities(base_url):
    """Fetch WFS GetCapabilities and return available layer names."""
    try:
        resp = requests.get(base_url, params={
            "service": "WFS", "version": "2.0.0", "request": "GetCapabilities"
        }, timeout=30)
        if resp.status_code == 200:
            # Simple text search for FeatureType Names
            import re
            names = re.findall(r'<Name>(.*?)</Name>', resp.text)
            return [n for n in names if n and ':' not in n]
    except Exception:
        pass
    return []


# ── Fetch flood layers ────────────────────────────────────────────────────────

def fetch_flood_zones(bbox):
    """
    Download FZ2 + FZ3 from the EA unified WFS.
    Both zones are in a single layer; we split by the 'flood_zone' attribute.
    """
    print("[2/5] Downloading EA Flood Zone data (FZ2 + FZ3)...")
    gdf = wfs_download(EA_FZ_WFS, EA_FZ_LAYER, bbox)

    if gdf is None or gdf.empty:
        print("      [FAIL] Could not download flood zones.")
        return gpd.GeoDataFrame(), gpd.GeoDataFrame()

    # Normalise the flood_zone column (may be 'FZ2'/'FZ3' or 'Flood Zone 2' etc.)
    fz_col = None
    for col in gdf.columns:
        if "zone" in col.lower() or "fz" in col.lower():
            fz_col = col; break
    if fz_col is None:
        print(f"      [WARN] No flood_zone column found. Columns: {list(gdf.columns)}")
        return gdf, gpd.GeoDataFrame()

    print(f"      Flood zone column: '{fz_col}'  "
          f"values: {list(gdf[fz_col].unique())}")

    fz3 = gdf[gdf[fz_col].str.contains("3", na=False)].copy()
    fz2 = gdf[gdf[fz_col].str.contains("2", na=False)].copy()
    print(f"      FZ3: {len(fz3):,} features  |  FZ2: {len(fz2):,} features")
    return fz3, fz2


def fetch_surface_water(bbox):
    """
    Load surface water risk data.
    EA only provides this as WMS (not WFS), so we expect a locally placed file.
    If not found, return None with instructions.
    """
    print("[3/5] Loading surface water flood risk data...")

    # Check for local file first
    for path in [SW_LOCAL_PATH,
                 "data/surface_water_risk.gpkg",
                 "data/RoFSW.shp",
                 "data/surface_water.shp"]:
        if os.path.exists(path):
            gdf = gpd.read_file(path)
            # Clip to bbox
            minx, miny, maxx, maxy = bbox
            gdf = gdf.cx[minx:maxx, miny:maxy]
            print(f"      Loaded {len(gdf):,} features from {path}")
            return gdf

    print("      [INFO] No local surface water file found.")
    print("      To include surface water risk:")
    print("        1. Go to: https://environment.data.gov.uk/dataset/"
          "b5aaa28d-6eb9-460e-8d6f-43caa71fbe0e")
    print("        2. Download Shapefile for your area")
    print(f"        3. Save as: {os.path.abspath(SW_LOCAL_PATH)}")
    print("      Continuing with FZ2/FZ3 only (sw_* terms = 0)...")
    return None


# ── % area calculation ────────────────────────────────────────────────────────

def pct_overlap(subbasin_geom, flood_gdf):
    """
    Return fraction (0-1) of subbasin_geom covered by flood_gdf polygons.
    """
    if flood_gdf is None or flood_gdf.empty:
        return 0.0
    try:
        sub_area = subbasin_geom.area
        if sub_area == 0:
            return 0.0
        clipped = flood_gdf.clip(subbasin_geom)
        if clipped.empty:
            return 0.0
        overlap_area = clipped.geometry.area.sum()
        return min(1.0, overlap_area / sub_area)
    except Exception:
        return 0.0


def pct_overlap_by_class(subbasin_geom, flood_gdf, class_col, class_val):
    """
    Return fraction of subbasin covered by flood_gdf rows where
    class_col == class_val.
    """
    if flood_gdf is None or flood_gdf.empty:
        return 0.0
    subset = flood_gdf[flood_gdf[class_col].str.lower() == class_val.lower()]
    return pct_overlap(subbasin_geom, subset)


# ── CWI calculation ───────────────────────────────────────────────────────────

def calculate_cwi(gdf_sub, fz3, fz2, sw):
    print("[4/5] Calculating CWI per subbasin...")

    # Detect surface water risk column
    sw_col = None
    if sw is not None and not sw.empty:
        for col in sw.columns:
            vals = sw[col].dropna().unique()
            vals_l = [str(v).lower() for v in vals]
            if any("high" in v for v in vals_l):
                sw_col = col
                print(f"      Surface water risk column: '{sw_col}'  "
                      f"values: {list(vals)[:8]}")
                break
        if sw_col is None:
            print(f"      [WARN] Could not identify risk column. Columns: {list(sw.columns)}")

    rows = []
    for idx, sub in gdf_sub.iterrows():
        geom = sub.geometry
        basin_id = sub.get("id", idx)

        # Flood zones (binary: in zone or not)
        fz3_pct = pct_overlap(geom, fz3)
        fz2_pct = pct_overlap(geom, fz2)

        # Surface water (by risk class)
        if sw is not None and not sw.empty and sw_col:
            sw_high = pct_overlap_by_class(geom, sw, sw_col, "high")
            sw_med  = pct_overlap_by_class(geom, sw, sw_col, "medium")
            sw_low  = pct_overlap_by_class(geom, sw, sw_col, "low")
        else:
            sw_high = sw_med = sw_low = 0.0

        # Convert fractions to percentages for formula inputs
        fz3_p  = fz3_pct  * 100
        fz2_p  = fz2_pct  * 100
        swh_p  = sw_high  * 100
        swm_p  = sw_med   * 100
        swl_p  = sw_low   * 100

        # Raw CWI = weighted sum of percentages (theoretical max = 100)
        cwi_raw = (CWI_WEIGHTS["fz3"]     * fz3_p +
                   CWI_WEIGHTS["fz2"]     * fz2_p +
                   CWI_WEIGHTS["sw_high"] * swh_p +
                   CWI_WEIGHTS["sw_med"]  * swm_p +
                   CWI_WEIGHTS["sw_low"]  * swl_p)

        rows.append({
            "id":       basin_id,
            "fz3_pct":  round(fz3_p, 2),
            "fz2_pct":  round(fz2_p, 2),
            "swh_pct":  round(swh_p, 2),
            "swm_pct":  round(swm_p, 2),
            "swl_pct":  round(swl_p, 2),
            "CWI_raw":  cwi_raw,
            "geometry": geom,
        })

    # Min-max normalise to 0–5 so values spread across the full range
    gdf = gpd.GeoDataFrame(rows, crs=gdf_sub.crs)
    raw_min = gdf["CWI_raw"].min()
    raw_max = gdf["CWI_raw"].max()
    if raw_max > raw_min:
        gdf["CWI"] = ((gdf["CWI_raw"] - raw_min) / (raw_max - raw_min) * 5).round(3)
    else:
        gdf["CWI"] = 0.0
    gdf = gdf.drop(columns="CWI_raw")

    for _, row in gdf.iterrows():
        print(f"      Basin {row['id']:>3}: FZ3={row['fz3_pct']:5.1f}%  "
              f"FZ2={row['fz2_pct']:5.1f}%  "
              f"SW_H={row['swh_pct']:5.1f}%  "
              f"SW_M={row['swm_pct']:5.1f}%  "
              f"SW_L={row['swl_pct']:5.1f}%  "
              f"CWI={row['CWI']:.3f}")

    return gdf


# ── QGIS export ───────────────────────────────────────────────────────────────

CWI_CLASSES = [
    ("Very Dry",  "#f7fcf5"),   # near-white green
    ("Dry",       "#a1d99b"),   # light green
    ("Moderate",  "#41ab5d"),   # mid green
    ("Wet",       "#006d2c"),   # dark green
    ("Very Wet",  "#00250f"),   # very dark green
]


def write_qml_style(gdf_cwi, path):
    """
    Write a QGIS QML graduated colour style for the CWI field.
    5 categories (Very Dry → Very Wet) using a green ramp.
    Breaks are computed from the actual data distribution.
    """
    vals = gdf_cwi["CWI"].dropna().sort_values().values
    if len(vals) == 0:
        return

    breaks = np.quantile(vals, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    labels = [c[0] for c in CWI_CLASSES]
    colors = [c[1] for c in CWI_CLASSES]

    ranges_xml = ""
    for i in range(5):
        lo = breaks[i]
        hi = breaks[i+1]
        r, g, b, _ = [int(x*255) for x in mcolors.to_rgba(colors[i])]
        ranges_xml += f"""
    <range lower="{lo:.4f}" upper="{hi:.4f}" label="{labels[i]} ({lo:.3f}–{hi:.3f})"
           symbol="{i}" render="true"/>"""

    symbols_xml = ""
    for i, col in enumerate(colors):
        r, g, b, _ = [int(x*255) for x in mcolors.to_rgba(col)]
        symbols_xml += f"""
    <symbol alpha="1" clip_to_extent="1" name="{i}" type="fill" force_rhr="0">
      <data_defined_properties/>
      <layer class="SimpleFill" enabled="1" locked="0" pass="0">
        <Option type="Map">
          <Option name="color" value="{r},{g},{b},200" type="QString"/>
          <Option name="outline_color" value="255,255,255,255" type="QString"/>
          <Option name="outline_width" value="0.4" type="QString"/>
        </Option>
      </layer>
    </symbol>"""

    qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="AllStyleCategories">
  <renderer-v2 attr="CWI" type="graduatedSymbol" graduatedMethod="GraduatedColor"
               enableorderby="0" symbollevels="0" forceraster="0">
    <ranges>{ranges_xml}
    </ranges>
    <symbols>{symbols_xml}
    </symbols>
    <source-symbol>
      <symbol alpha="1" clip_to_extent="1" name="0" type="fill" force_rhr="0">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="color" value="125,125,125,200" type="QString"/>
          </Option>
        </layer>
      </symbol>
    </source-symbol>
    <colorramp name="[source]" type="gradient">
      <Option type="Map">
        <Option name="color1" value="33,102,172,255" type="QString"/>
        <Option name="color2" value="215,48,39,255"  type="QString"/>
      </Option>
    </colorramp>
    <classificationMethod id="Jenks"/>
    <labelformat trimtrailingzeroes="false" format="%1 - %2" decimals="4"/>
  </renderer-v2>
  <labeling type="simple">
    <settings calloutType="simple">
      <text-style fieldName="CWI" fontSize="7" textColor="0,0,0,255"
                  fontWeight="75"/>
    </settings>
  </labeling>
</qgis>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(qml)
    print(f"      QGIS style -> {path}")


def export_geopackage(gdf_cwi, fz3, fz2, sw,
                       gdf_catch, gdf_rivers, gdf_out, out_dir):
    base = "cwi_results"
    gpkg = os.path.join(out_dir, f"{base}.gpkg")
    # If the file is locked (e.g. open in QGIS), write to a timestamped fallback
    if os.path.exists(gpkg):
        try:
            import tempfile, shutil
            with open(gpkg, "a+b"):
                pass
        except PermissionError:
            import datetime
            ts = datetime.datetime.now().strftime("%H%M%S")
            gpkg = os.path.join(out_dir, f"{base}_{ts}.gpkg")
            print(f"      [WARN] cwi_results.gpkg locked — writing to {os.path.basename(gpkg)}")

    gdf_cwi.to_file(gpkg, layer="subbasins_cwi",   driver="GPKG")
    gdf_catch.to_file(gpkg, layer="catchment",      driver="GPKG")
    gdf_rivers.to_file(gpkg, layer="rivers",         driver="GPKG")
    gdf_out.to_file(gpkg, layer="outlets",           driver="GPKG")

    if fz3 is not None and not fz3.empty:
        fz3.to_file(gpkg, layer="flood_zone_3",      driver="GPKG")
    if fz2 is not None and not fz2.empty:
        fz2.to_file(gpkg, layer="flood_zone_2",      driver="GPKG")
    if sw is not None and not sw.empty:
        sw.to_file(gpkg, layer="surface_water_risk", driver="GPKG")

    print(f"      GeoPackage -> {gpkg}")
    return gpkg


# ── Map ───────────────────────────────────────────────────────────────────────

def make_map(gdf_cwi, fz3, fz2, sw, gdf_catch, gdf_rivers, out_dir):
    # Build custom green colormap from CWI_CLASSES
    green_cmap = mcolors.LinearSegmentedColormap.from_list(
        "cwi_green", [c[1] for c in CWI_CLASSES], N=256
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.patch.set_facecolor("#f0f4f8")
    ax.set_aspect("equal")
    ax.set_facecolor("#cde0f0")
    ax.set_title("Composite Wetness Index (CWI)", fontsize=14, fontweight="bold", pad=12)

    cwi_max = float(gdf_cwi["CWI"].max()) if not gdf_cwi.empty else 0.0

    if not gdf_cwi.empty and cwi_max > 0:
        gdf_cwi.plot(column="CWI", ax=ax,
                     cmap=green_cmap, vmin=0, vmax=5,
                     edgecolor="white", linewidth=0.8, alpha=0.9,
                     legend=False)
    else:
        gdf_cwi.plot(ax=ax, color="#cccccc", edgecolor="white", linewidth=0.8)
        ax.text(0.5, 0.5, "No flood data\nreturned by EA API",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="#555", style="italic")

    # Label basins with CWI value
    for _, row in gdf_cwi.iterrows():
        cwi_val = row["CWI"]
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        # Use dark text on light classes, white on dark
        txt_color = "white" if cwi_val > 1.5 else "#333333"
        ax.text(cx, cy, f"{cwi_val:.3f}", fontsize=7,
                ha="center", va="center", fontweight="bold", color=txt_color)

    if not gdf_catch.empty:
        gdf_catch.boundary.plot(ax=ax, edgecolor="black", linewidth=1.8, zorder=5)
    if not gdf_rivers.empty:
        mx = float(gdf_rivers["max_acc"].max()) or 1.0
        for _, r in gdf_rivers.iterrows():
            rel = r["max_acc"] / mx
            gpd.GeoDataFrame([r], crs=gdf_rivers.crs).plot(
                ax=ax, color=plt.cm.Blues(0.4 + 0.6 * rel),
                linewidth=0.4 + 2.0 * rel, zorder=4)

    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.3)

    # 5-class legend patches
    breaks = np.quantile(gdf_cwi["CWI"].dropna().values, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) \
        if not gdf_cwi.empty else np.linspace(0, 1, 6)
    legend_patches = []
    for i, (label, color) in enumerate(CWI_CLASSES):
        lo, hi = breaks[i], breaks[i + 1]
        patch = mpatches.Patch(
            facecolor=color, edgecolor="white", linewidth=0.6,
            label=f"{label}  ({lo:.3f} – {hi:.3f})"
        )
        legend_patches.append(patch)
    ax.legend(handles=legend_patches, title="Wetness Class",
              loc="lower left", fontsize=8, title_fontsize=9,
              framealpha=0.9, edgecolor="#aaa")

    plt.tight_layout()
    path = os.path.join(out_dir, "cwi_map.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"      Map -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load subbasins
    print("[1/5] Loading subbasins...")
    gdf_sub   = gpd.read_file(SUBBASINS_SHP)
    gdf_catch = gpd.read_file(CATCHMENT_SHP)
    gdf_riv   = gpd.read_file(RIVERS_SHP)
    gdf_out   = gpd.read_file("outputs/strahler_results/outlets.shp")
    print(f"      {len(gdf_sub)} subbasins loaded.")

    bbox = tuple(gdf_sub.total_bounds)   # (minx, miny, maxx, maxy)
    print(f"      Study area bbox: {[round(v,4) for v in bbox]}")

    # Download EA data
    fz3, fz2 = fetch_flood_zones(bbox)
    sw        = fetch_surface_water(bbox)

    # Report what we got
    print(f"\n      EA data summary:")
    print(f"        FZ3 features : {len(fz3) if fz3 is not None else 'FAILED':>6}")
    print(f"        FZ2 features : {len(fz2) if fz2 is not None else 'FAILED':>6}")
    print(f"        SW  features : {len(sw)  if sw  is not None else 'FAILED':>6}\n")

    # CWI
    gdf_cwi = calculate_cwi(gdf_sub, fz3, fz2, sw)

    # Export
    print("\n[5/5] Exporting outputs...")
    gpkg = export_geopackage(gdf_cwi, fz3, fz2, sw,
                              gdf_catch, gdf_riv, gdf_out, OUT_DIR)
    write_qml_style(gdf_cwi, os.path.join(OUT_DIR, "cwi_style.qml"))
    make_map(gdf_cwi, fz3, fz2, sw, gdf_catch, gdf_riv, OUT_DIR)

    # Also save CSV summary
    csv_path = os.path.join(OUT_DIR, "cwi_summary.csv")
    gdf_cwi.drop(columns="geometry").to_csv(csv_path, index=False)
    print(f"      Summary CSV -> {csv_path}")

    print("\n-- CWI Results ------------------------------------------")
    print(gdf_cwi[["id","fz3_pct","fz2_pct","swh_pct","swm_pct",
                    "swl_pct","CWI"]].to_string(index=False))
    print("\n-- QGIS Instructions ------------------------------------")
    print(f"  1. Open QGIS")
    print(f"  2. Drag & drop:  {os.path.abspath(gpkg)}")
    print(f"  3. Right-click 'subbasins_cwi' layer -> Properties -> Style")
    print(f"     -> Load Style -> {os.path.abspath(OUT_DIR)}/cwi_style.qml")
    print("---------------------------------------------------------")
    print(f"\nDone! Outputs in: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
