import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def load_and_interpolate_data(json_path: str):
    with open(json_path) as f:
        data = json.load(f)

    vectors = data["vectors"]
    lons = np.array([v["lon"] for v in vectors])
    lats = np.array([v["lat"] for v in vectors])
    speeds = np.array([v["speed"] for v in vectors])
    dirs_rad = np.radians(np.array([v["direction"] for v in vectors]))
    
    u_arr = speeds * np.sin(dirs_rad)
    v_arr = speeds * np.cos(dirs_rad)
    
    valid = (speeds >= 1.0) & (speeds <= 20.0)
    lons = lons[valid]
    lats = lats[valid]
    speeds = speeds[valid]
    u_arr = u_arr[valid]
    v_arr = v_arr[valid]

    if len(lons) < 4:
        return None

    pad = 0.1
    boundaries = {
        "lon_min": lons.min() - pad,
        "lon_max": lons.max() + pad,
        "lat_min": lats.min() - pad,
        "lat_max": lats.max() + pad
    }

    grid_lon_fine = np.linspace(boundaries["lon_min"], boundaries["lon_max"], 120)
    grid_lat_fine = np.linspace(boundaries["lat_min"], boundaries["lat_max"], 100)
    GL, GL2 = np.meshgrid(grid_lon_fine, grid_lat_fine)
    points = np.column_stack([lons, lats])

    speed_grid = griddata(points, speeds, (GL, GL2), method = "cubic")
    u_grid = griddata(points, u_arr, (GL, GL2), method = "cubic")
    v_grid = griddata(points, v_arr, (GL, GL2), method = "cubic")
    
    speed_nn = griddata(points, speeds, (GL, GL2), method = "nearest")
    u_nn = griddata(points, u_arr, (GL, GL2), method = "nearest")
    v_nn = griddata(points, v_arr, (GL, GL2), method = "nearest")
    
    speed_grid = np.where(np.isnan(speed_grid), speed_nn, speed_grid)
    u_grid = np.where(np.isnan(u_grid), u_nn, u_grid)
    v_grid = np.where(np.isnan(v_grid), v_nn, v_grid)

    grid_lon_coarse = np.linspace(boundaries["lon_min"] + 0.1, boundaries["lon_max"] - 0.1, 18)
    grid_lat_coarse = np.linspace(boundaries["lat_min"] + 0.1, boundaries["lat_max"] - 0.1, 14)
    GC, GC2 = np.meshgrid(grid_lon_coarse, grid_lat_coarse)
    
    u_coarse = griddata(points, u_arr, (GC, GC2), method = "nearest")
    v_coarse = griddata(points, v_arr, (GC, GC2), method = "nearest")
    spd_coarse = griddata(points, speeds, (GC, GC2), method = "nearest")
    
    mag_c = np.sqrt(u_coarse ** 2 + v_coarse ** 2) + 1e-8
    u_c_norm = u_coarse / mag_c
    v_c_norm = v_coarse / mag_c

    return {
        "metadata": data,
        "boundaries": boundaries,
        "fine_mesh": (GL, GL2),
        "coarse_mesh": (GC, GC2),
        "grids": (speed_grid, u_c_norm, v_c_norm, spd_coarse),
        "stats": (speeds.mean(), speeds.max(), speeds.min(), len(lons))
    }

def generate_wind_map(payload: dict, output_path: str):
    if payload is None:
        return

    data = payload["metadata"]
    boundaries = payload["boundaries"]
    GL, GL2 = payload["fine_mesh"]
    GC, GC2 = payload["coarse_mesh"]
    speed_grid, u_c_norm, v_c_norm, spd_coarse = payload["grids"]
    mean_spd, max_spd, min_spd, total_vectors = payload["stats"]

    fig, ax = plt.subplots(figsize = (13, 9), facecolor = "#f0f0f0", subplot_kw = {'projection': ccrs.PlateCarree()})
    ax.set_facecolor("#d0e8f0")
    
    ax.add_feature(cfeature.COASTLINE, linewidth = 1.5, zorder = 4) # type: ignore
    ax.add_feature(cfeature.LAND, facecolor = '#d9d9d9', edgecolor = 'black', zorder = 3) # type: ignore
    ax.add_feature(cfeature.OCEAN, facecolor = "#d0e8f0", zorder = 0) # type: ignore
    
    cmap = plt.cm.RdYlBu_r # type: ignore
    norm = mcolors.Normalize(vmin = 2, vmax = max_spd)
    
    cf = ax.contourf(GL, GL2, speed_grid, levels = np.linspace(2, max_spd, 20), cmap = cmap, norm = norm, alpha = 0.85, zorder = 1)
    ax.contour(GL, GL2, speed_grid, levels = np.linspace(2, max_spd, 8), colors = "white", linewidths = 0.4, alpha = 0.3, zorder = 2)
    
    ax.quiver(GC, GC2, u_c_norm, v_c_norm, spd_coarse, cmap = cmap, norm = norm, scale = 22, width = 0.003, headwidth = 4, headlength = 5, pivot = "mid", zorder = 4, edgecolor = "black", linewidth = 0.3)
    
    cbar = fig.colorbar(cf, ax = ax, orientation = "vertical", fraction = 0.025, pad = 0.02)
    cbar.set_label("Wind Speed (m/s)", color = "black", fontsize = 12)
    
    ax.set_xlim(boundaries["lon_min"], boundaries["lon_max"])
    ax.set_ylim(boundaries["lat_min"], boundaries["lat_max"])
    
    gl = ax.gridlines(draw_labels = True, color = "white", linestyle = "--", linewidth = 0.5, alpha = 0.6, zorder = 0) # type: ignore
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'color': 'black', 'size': 10}
    gl.ylabel_style = {'color': 'black', 'size': 10}

    for sp in ax.spines.values():
        sp.set_edgecolor("black")
        sp.set_linewidth(1.2)

    scene = data.get("scene_date", data.get("date", ""))
    ax.set_title(f"SAR-Derived Ocean Wind Field · Gujarat Coast\nSentinel-1 | {scene} | {total_vectors} vectors", color = "black", fontsize = 13, pad = 12, fontweight = "bold")
    
    stats_text = f"Mean : {mean_spd:.1f} m/s\nMax  : {max_spd:.1f} m/s\nMin  : {min_spd:.1f} m/s\nN    : {total_vectors}"
    ax.text(0.015, 0.975, stats_text, transform = ax.transAxes, fontsize = 9, color = "black", va = "top", family = "monospace", bbox = dict(boxstyle = "round,pad=0.5", facecolor = "white", edgecolor = "#aaa", alpha = 0.85), zorder = 5)
    ax.annotate("N", xy = (0.97, 0.96), xytext = (0.97, 0.88), xycoords = "axes fraction", arrowprops = dict(arrowstyle = "->", color = "black", lw = 2), ha = "center", color = "black", fontsize = 14, fontweight = "bold", zorder = 5)
    ax.text((boundaries["lon_min"] + boundaries["lon_max"]) / 2, boundaries["lat_min"] + 0.08, "Gulf of Kutch / Arabian Sea", color = "#1a3a5c", fontsize = 10, alpha = 0.8, style = "italic", ha = "center", zorder = 5)

    plt.tight_layout()
    plt.savefig(output_path, dpi = 150, bbox_inches = "tight", facecolor = fig.get_facecolor())
    plt.close()

if __name__ == "__main__":
    json_file = sys.argv[1] if len(sys.argv) > 1 else "result.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "wind_field_gujarat.png"
    
    processed_payload = load_and_interpolate_data(json_file)
    generate_wind_map(processed_payload, output_file) # type: ignore
