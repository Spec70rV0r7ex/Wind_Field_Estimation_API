import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from scipy.ndimage import binary_dilation

def apply_hard_target_mask(sar_patch: np.ndarray) -> bool:
    """
    Detects ships, oil rigs, and extreme convective rain cells.
    Returns True if the patch is pure ocean, False if it contains bright anomalies.
    """

    bright_pixel_count = np.sum(sar_patch > -1.5)

    if bright_pixel_count >= 2:
        return False 
    return True 

def apply_land_mask(sar_patch: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    """
    Masks out land pixels and applies a 1-pixel safety buffer to remove
    bright coastal edge artifacts (e.g., wave breaking, mudflats).
    """
    land_percentage = np.sum(land_mask) / land_mask.size
    if land_percentage > 0.25:
        return None  # type: ignore

    structure = np.ones((3, 3), dtype=bool)
    buffered_land_mask = binary_dilation(land_mask, structure=structure) # type: ignore

    ocean_pixels = sar_patch[~buffered_land_mask] # type: ignore

    if len(ocean_pixels) == 0:
        return None  # type: ignore

    ocean_mean = np.mean(ocean_pixels)
    ocean_std = np.std(ocean_pixels)

    processed_patch = sar_patch.copy()
    noise = np.random.normal(ocean_mean, ocean_std, size=np.sum(buffered_land_mask)) # type: ignore
    processed_patch[buffered_land_mask] = noise

    return processed_patch

def filter_outliers_20deg(wind_directions: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Discards vectors differing > 20° from spatial nearest neighbors."""

    coords = np.column_stack((lats, lons))
    tree = KDTree(coords)

    _, indices = tree.query(coords, k = 5)

    filtered = np.copy(wind_directions)
    for i in range(len(wind_directions)):
        neighbor_idx = indices[i, 1:]
        neighbor_dirs = wind_directions[neighbor_idx]

        sin_med = np.median(np.sin(np.radians(neighbor_dirs)))
        cos_med = np.median(np.cos(np.radians(neighbor_dirs)))
        consensus = np.degrees(np.arctan2(sin_med, cos_med)) % 360

        diff = np.abs(wind_directions[i] - consensus)
        diff = np.minimum(diff, 360 - diff)

        if diff > 20.0:
            filtered[i] = np.nan
    
    return filtered

def apply_180_dealiasing(filtered_directions: np.ndarray, ecmwf_reference_angle: float) -> np.ndarray:
    """180-degree De-aliasing using NWP model average."""
    dealiased = np.copy(filtered_directions)

    diff = np.abs(dealiased - ecmwf_reference_angle)
    diff = np.minimum(diff, 360 - diff)
    needs_flip = diff > 90

    dealiased[needs_flip] = (dealiased[needs_flip] + 180) % 360
    return dealiased

def interpolate_missing_vectors(wind_field: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Replaces discarded (NaN) vectors by geographically interpolating neighbors."""

    invalid = np.isnan(wind_field)
    if not np.any(invalid):
        return wind_field

    valid_coords = np.column_stack((lons[~invalid], lats[~invalid]))
    valid_values = wind_field[~invalid]
    invalid_coords = np.column_stack((lons[invalid], lats[invalid]))

    sin_vals = np.sin(np.radians(valid_values))
    cos_vals = np.cos(np.radians(valid_values))

    interp_sin = griddata(valid_coords, sin_vals, invalid_coords, method = 'linear')
    interp_cos = griddata(valid_coords, cos_vals, invalid_coords, method = 'linear')

    nan_interp = np.isnan(interp_sin)
    if np.any(nan_interp):
        interp_sin[nan_interp] = griddata(valid_coords, sin_vals, invalid_coords[nan_interp], method = 'nearest')
        interp_cos[nan_interp] = griddata(valid_coords, cos_vals, invalid_coords[nan_interp], method = 'nearest')

    interpolated_angles = np.degrees(np.arctan2(interp_sin, interp_cos)) % 360

    final_field = np.copy(wind_field)
    final_field[invalid] = interpolated_angles

    return final_field
