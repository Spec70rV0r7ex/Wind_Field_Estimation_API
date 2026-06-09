import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_wind_field(vectors: list, title: str = "ResNet SAR Wind Field", save_path: str = "wind_map.png"):
    lats = np.array([v.lat for v in vectors])
    lons = np.array([v.lon for v in vectors])
    u = np.array([v.u for v in vectors])
    v = np.array([v.v for v in vectors])
    speed = np.array([v.speed for v in vectors])

    fig = plt.figure(figsize = (10, 10), facecolor = 'white')
    ax = plt.axes(projection = ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE.with_scale('10m'),linewidth = 1.2, edgecolor = 'black', zorder = 4) # type: ignore
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor = '#f9f9f9', zorder = 1) # type: ignore
    step = max(1, len(lats) // 400)

    quiv = ax.quiver(
                        lons[::step], lats[::step], u[::step], v[::step], speed[::step],
                        cmap = 'jet', scale = 500, width = 0.003, headwidth = 4, headlength = 5, headaxislength = 4,
                        pivot = 'middle', transform = ccrs.PlateCarree(), zorder = 3
                )

    cbar = fig.colorbar(quiv, ax = ax, orientation = 'horizontal', pad = 0.06, fraction = 0.04)
    cbar.set_label('ResNet wind speed (m/s)', fontsize = 11, fontweight = 'bold')
    cbar.ax.tick_params(labelsize = 9)

    ax.set_title(title, fontsize = 14, pad = 20)
    
    gl = ax.gridlines(draw_labels = True, linewidth = 0.5, color = 'gray', alpha = 0.8, linestyle = ':', zorder = 2) # type: ignore
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}

    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs = ccrs.PlateCarree()) # type: ignore

    plt.tight_layout()
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight', facecolor = 'white')
    print(f"Map saved successfully to {save_path}")
    plt.close()
