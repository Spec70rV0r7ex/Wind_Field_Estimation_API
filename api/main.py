from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
import numpy as np
import math
import os
import json

from api.schemas import WindRequest, WindResponse, VectorPoint
from gee.fetch_sentinel1 import fetch_inference_data 
from model.predict import load_model, predict_wind_directions
from utils.windspeed import invert_cmod5
from model.postprocess import apply_land_mask, filter_outliers_20deg, apply_180_dealiasing, interpolate_missing_vectors

app = FastAPI(
    title = "SAR Coastal Wind API : Coast of Gujarat",
    description = "High Resolution ocean wind field vectors using ResNet and Sentinel-1 SAR imagery."
)

print("Loading M64RN4 Model Weights...")
resnet_model, device = load_model()
print("Model loaded successfully!")


@app.post("/api/v1/wind-field", response_model = WindResponse)
async def get_wind_field(request: WindRequest):
    try:
        date_str = request.date.strftime("%Y-%m-%d")
        
        (lats, lons, sar_patches, land_masks, inc_angles, vv_db, era5_ref_angle, era5_u_array, era5_v_array) = fetch_inference_data(request.polygon, date_str)
        
        valid_indices = []
        for i in range(len(sar_patches)):
            processed_patch = apply_land_mask(sar_patches[i], land_masks[i])
            sar_patches[i] = processed_patch
            valid_indices.append(i)

        sar_patches = sar_patches[valid_indices]
        lats = lats[valid_indices]
        lons = lons[valid_indices]
        inc_angles = inc_angles[valid_indices]
        vv_db = vv_db[valid_indices]
        era5_u_array = era5_u_array[valid_indices]
        era5_v_array = era5_v_array[valid_indices]
        
        true_speeds = np.sqrt(era5_u_array**2 + era5_v_array**2)
        true_dirs = np.degrees(np.arctan2(era5_u_array, era5_v_array)) % 360
        
        raw_angles = predict_wind_directions(resnet_model, device, sar_patches)
        
        filtered_angles = filter_outliers_20deg(raw_angles, lats, lons)
        dealiased_angles = apply_180_dealiasing(filtered_angles, era5_ref_angle)
        final_directions = interpolate_missing_vectors(dealiased_angles, lats, lons)
        
        CALIBRATION_OFFSET = 4.0
        calibarated_vv_db = vv_db - CALIBRATION_OFFSET
        wind_speeds = invert_cmod5(calibarated_vv_db, inc_angles, final_directions)
        
        vectors = []
        for i in range(len(lats)):
            if np.isnan(final_directions[i]): 
                continue 
                
            rad = math.radians(final_directions[i])
            u = -wind_speeds[i] * math.sin(rad)
            v = -wind_speeds[i] * math.cos(rad)
            
            vectors.append(VectorPoint(
                lat = round(float(lats[i]), 4), 
                lon = round(float(lons[i]), 4), 
                u = round(float(u), 2), 
                v = round(float(v), 2), 
                speed = round(float(wind_speeds[i]), 2), 
                direction = round(float(final_directions[i]), 2),
                true_speed = round(float(true_speeds[i]), 2),
                true_dir = round(float(true_dirs[i]), 2)
            ))
        final_response = WindResponse(date = request.date, vectors = vectors)
        os.makedirs("data", exist_ok=True)
        safe_date = date_str.replace("-", "")
        file_path = os.path.join("data", f"gujarat_wind_{safe_date}.json")
        with open(file_path, "w") as f:
            json.dump(jsonable_encoder(final_response), f, indent=4)
        return WindResponse(date = request.date, vectors = vectors)
        
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Pipeline Error: {str(e)}")
