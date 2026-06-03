import ee
import pandas as pd
import numpy as np

def init_gee():
    """Initializes Google Earth Engine. Falls back to auth if needed."""
    try:
        ee.Initialize()
    except Exception:
        print("GEE not initialized. Please run `earthengine authenticate` in terminal.")
        ee.Authenticate()
        ee.Initialize()

def get_sar_and_era5(polygon_coords, date_str, scale=5000):
    """
    Fetches S1 GRD VV backscatter and ERA5 background wind fields.
    polygon_coords: A list of [lon, lat] coordinate pairs defining a polygon.
    scale: spatial resolution in meters to sample the data
    """
    # Create an Earth Engine Polygon. 
    # The extra bracket [polygon_coords] defines the outer ring of the polygon.
    geom = ee.Geometry.Polygon([polygon_coords])
    date = ee.Date(date_str)
    
    # Sentinel-1 GRD
    s1_col = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(geom) \
        .filterDate(date, date.advance(1, 'day')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
    
    if s1_col.size().getInfo() == 0:
        return pd.DataFrame() # No imagery
    
    s1_img = s1_col.first()
    
    # Calculate Radar Look Direction based on pass heading
    pass_dir = s1_img.get('orbitProperties_pass').getInfo()
    heading = 12 if pass_dir == 'ASCENDING' else 192
    radar_look_dir = (heading + 90) % 360
    
    # ERA5 Reanalysis
    era5_col = ee.ImageCollection("ECMWF/ERA5/HOURLY") \
        .filterBounds(geom) \
        .filterDate(date, date.advance(1, 'day'))
    
    era5_img = era5_col.first()
    
    # Stack Bands and Sample
    combined = s1_img.select(['VV', 'angle']).addBands(era5_img.select(['u_component_of_wind_10m', 'v_component_of_wind_10m']))
    
    # Extract points over the Polygon geometry
    samples = combined.sample(region=geom, scale=scale, numPixels=10000, geometries=True)
    features = samples.getInfo().get('features', [])
    
    data = []
    for f in features:
        props = f['properties']
        coords = f['geometry']['coordinates']
        if all(k in props for k in ('VV', 'angle', 'u_component_of_wind_10m')):
            data.append({
                'lon': coords[0],
                'lat': coords[1],
                'vv_db': props['VV'],
                'incidence': props['angle'],
                'u_10': props['u_component_of_wind_10m'],
                'v_10': props['v_component_of_wind_10m'],
                'radar_look_dir': radar_look_dir
            })
            
    df = pd.DataFrame(data)
    if not df.empty:
        df['sigma0_linear'] = 10 ** (df['vv_db'] / 10.0)
    return df