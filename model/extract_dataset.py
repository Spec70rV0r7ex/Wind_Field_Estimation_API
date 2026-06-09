import ee
import numpy as np
import os
import math

try:
    ee.Initialize(project = 'vocal-raceway-470111-k8')
    print("Earth Engine Initialized successfully.")
except Exception as e:
    print("Earth Engine not authenticated. Triggering authentication...")
    ee.Authenticate()
    ee.Initialize()

SAVE_DIR = './data/SAR_Wind_Dataset_2024'
os.makedirs(SAVE_DIR, exist_ok = True)
print(f"Dataset will be saved locally to: {SAVE_DIR}")

gujarat_coords = [
    [67.79, 23.12], [70.68, 19.94], [72.78, 20.01], [73.04, 20.48],
    [72.74, 21.23], [73.05, 22.24], [72.41, 22.45], [72.16, 22.15],
    [71.97, 21.21], [70.68, 20.86], [69.33, 22.17], [70.24, 22.38],
    [70.57, 23.14], [69.30, 22.90], [68.63, 23.43], [68.86, 23.91],
    [68.36, 23.94], [67.95, 23.51], [67.79, 23.12]
]
roi = ee.Geometry.Polygon(gujarat_coords)

START_DATE = '2024-01-01'
END_DATE = '2024-12-31'
SCALE = 100

def extract_dataset():
    print("Querying Sentinel-1 data for 100x100 patches...")

    s1_collection = (ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(roi).filterDate(START_DATE, END_DATE).filter(ee.Filter.eq('instrumentMode', 'IW')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).select(['VV'])) # type: ignore

    s1_list = s1_collection.toList(s1_collection.size())
    num_images = s1_list.size().getInfo()

    X_data = []
    Y_data = []

    for i in range(num_images):
        try:
            s1_image = ee.Image(s1_list.get(i))
            timestamp = s1_image.date().millis() # type: ignore
            date_ee = ee.Date(timestamp)

            wind_data = (ee.ImageCollection("ECMWF/ERA5/HOURLY").filterDate(date_ee.advance(-2, 'hour'), date_ee.advance(2, 'hour')).select(['u_component_of_wind_10m', 'v_component_of_wind_10m']).first()) # type: ignore

            if not wind_data:
                continue

            patch_list = s1_image.neighborhoodToArray(ee.Kernel.square(24, 'pixels')) # type: ignore
            patch_combined = patch_list.addBands(wind_data)

            image_footprint = s1_image.geometry().intersection(roi) # type: ignore

            samples = patch_combined.sample(
                region = image_footprint,
                scale = SCALE,
                numPixels = 150,
                geometries = False
            ).getInfo()

            for feature in samples.get('features', []):
                props = feature['properties']

                if 'VV' not in props or 'u_component_of_wind_10m' not in props:
                    continue

                vv_matrix = np.array(props['VV'])

                if vv_matrix.shape[0] >= 49 and vv_matrix.shape[1] >= 49:
                    vv_matrix = vv_matrix[:49, :49]
                else:
                    continue

                u = props['u_component_of_wind_10m']
                v = props['v_component_of_wind_10m']

                direction_rad = math.atan2(u, v)
                label = [math.sin(direction_rad), math.cos(direction_rad)]

                X_data.append(vv_matrix)
                Y_data.append(label)

            print(f"Processed image {i + 1} / {num_images} | Accumulated 49x49 Patches: {len(X_data)}")

        except Exception as e:
            print(f"Error processing image {i+1}: {e}")

    X_array = np.array(X_data).astype(np.float32)
    Y_array = np.array(Y_data).astype(np.float32)

    if len(X_array) > 0:
        X_array = np.expand_dims(X_array, axis = 1)
        print("\nExtraction Complete!")
        print(f"Final Dataset Shape - X (SAR): {X_array.shape}, Y (Labels): {Y_array.shape}")
        np.save(os.path.join(SAVE_DIR, 'SAR_X_49_2024.npy'), X_array)
        np.save(os.path.join(SAVE_DIR, 'WIND_Y_49_2024.npy'), Y_array)
        print(f"Saved successfully to {SAVE_DIR}")

if __name__ == '__main__':
    extract_dataset()
