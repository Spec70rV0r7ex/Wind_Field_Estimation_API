import json
import pandas as pd
import numpy as np
import sys
import os

from api.schemas import VectorPoint
from utils.visualization import plot_wind_field

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from utils.validation_math import calculate_metrics
from utils.visualization import plot_wind_field
from api.schemas import VectorPoint

file_path = '../data/gujarat_wind_20240209.json'

with open(file_path, 'r') as f:
    data = json.load(f)
print(f"Successfully loaded data for {data['date']} containing the vectors.")

pred_speeds = np.array([v['speed'] for v in data['vectors']])
pred_dirs = np.array([v['direction'] for v in data['vectors']])
true_speeds = np.array([v['true_speed'] for v in data['vectors']])
true_dirs = np.array([v['true_dir'] for v in data['vectors']])

metrics = calculate_metrics(pred_speeds, pred_dirs, true_speeds, true_dirs)

validation_df = pd.DataFrame({
    'Metric': [
        'Bias (β)', 
        'MAE (Mean Abs Error)', 
        'cRMSd / RMSE', 
        'Linear Correlation'
    ],
    'Wind Speed (m/s)': [
        f"{metrics['speed_bias']:+.2f}", 
        f"{metrics['speed_mae']:.2f}", 
        f"{metrics['speed_rmse']:.2f}",
        f"{metrics['speed_corr']:.2f}"
    ],
    'Wind Direction (°)': [
        f"{metrics['dir_bias']:+.1f}°", 
        f"{metrics['dir_mae']:.1f}°", 
        f"{metrics['dir_rmse']:.1f}°",
        f"{metrics['dir_corr']:.2f}"
    ]
})

print(f"SAR WIND PIPELINE VALIDATION REPORT ({data['date']})")
print(f"   Sample Size (N) = {metrics['sample_size']} valid ocean patches")
display(validation_df)

vectors = [VectorPoint(**v) for v in data['vectors']]
plot_wind_field(vectors, title = f"Gujarat Coastal Wind Field ({data['date']})", save_path = "gujarat_wind_map.png")
