import ee
import numpy as np
import math
from config.settings import settings

def init_gee():
    try:
        ee.Initialize(project = settings.GEE_PROJECT_ID)
        print(f"Earth Engine initialized successfully with project: {settings.GEE_PROJECT_ID}")
    except Exception as e:
        print(f"Initialization failed: {e}")
        raise RuntimeError(
            "Missing local credentials. Run this in your terminal: "
            "gcloud auth application-default login --scopes = https://www.googleapis.com/auth/earthengine,https://www.googleapis.com/auth/cloud-platform"
        )

def fetch_inference_data(polygon_coords: list, date_str: str):
    init_gee()
    aoi = ee.Geometry.Polygon(polygon_coords)
    start_date = ee.Date(date_str)

    collection = (ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(aoi).filterDate(start_date, start_date.advance(1, 'day')).filter(ee.Filter.eq('instrumentMode', 'IW')).select(['VV', 'angle'])) # type: ignore  
    if collection.size().getInfo() == 0:
        raise ValueError("No Sentinel-1 IW GRD data found for this date/location.")
    s1_image = collection.mosaic().clip(aoi)

    era5_collection = ee.ImageCollection("ECMWF/ERA5/HOURLY").filterDate(date_str, ee.Date(date_str).advance(1, 'day')) # type: ignore
    if era5_collection.size().getInfo() == 0:
        raise ValueError(f"ERA5 weather data is not yet available for {date_str}. Google Earth Engine has a 2-3 month latency for ERA5. Please try an older historical date.")
    era5 = era5_collection.mean()
    era5_stats = era5.reduceRegion(reducer = ee.Reducer.mean(), geometry = aoi, scale = 10000).getInfo() # type: ignore
    
    u_avg = era5_stats.get('u_component_of_wind_10m', 0.0)
    v_avg = era5_stats.get('v_component_of_wind_10m', 0.0)
    era5_ref_angle = math.degrees(math.atan2(u_avg, v_avg)) % 360

    era5_u = era5.select('u_component_of_wind_10m').rename('era5_u')
    era5_v = era5.select('v_component_of_wind_10m').rename('era5_v')

    elevation = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic() # type: ignore
    land_mask = elevation.gt(0).rename('land_mask') 

    patch_size = 24
    patch_list = s1_image.select('VV').neighborhoodToArray(ee.Kernel.square(patch_size, 'pixels')) # type: ignore
    mask_list = land_mask.neighborhoodToArray(ee.Kernel.square(patch_size, 'pixels')) # type: ignore

    combined = (patch_list.addBands(mask_list).addBands(s1_image.select('angle')).addBands(era5_u).addBands(era5_v))
    latlon = ee.Image.pixelLonLat().addBands(combined) # type: ignore

    data = latlon.sample(region = aoi, scale = 2000, geometries = False).getInfo()

    lats, lons, sar_patches, land_masks, inc_angles, vv_db = [], [], [], [], [], []
    era5_u_array, era5_v_array = [], []

    for f in data.get('features', []):
        props = f['properties']

        vv_matrix = np.array(props.get('VV', []))
        mask_matrix = np.array(props.get('land_mask', []))

        if vv_matrix.shape == (49, 49) and mask_matrix.shape == (49, 49):
            lats.append(props['latitude'])
            lons.append(props['longitude'])
            sar_patches.append(vv_matrix)
            land_masks.append(mask_matrix)
            inc_angles.append(props['angle'])
            vv_db.append(np.mean(vv_matrix))

            era5_u_array.append(props.get('era5_u', 0.0))
            era5_v_array.append(props.get('era5_v', 0.0))

    return (
            np.array(lats),
            np.array(lons),
            np.array(sar_patches),
            np.array(land_masks),
            np.array(inc_angles),
            np.array(vv_db),
            era5_ref_angle,
            np.array(era5_u_array),
            np.array(era5_v_array)
        )
