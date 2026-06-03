import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def calculate_metrics(wind_data):
    if not wind_data:
        return {"error": "No data retrieved to validate."}
        
    y_true = np.array([d['era5_wind_speed'] for d in wind_data])
    y_pred = np.array([d['sar_wind_speed'] for d in wind_data])
    
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) < 2:
        return {"error": "Insufficient valid data points for calculation."}
        
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "rmse_m_s": round(rmse, 3),
        "bias_m_s": round(bias, 3),
        "r2_score": round(r2, 3),
        "samples_count": len(y_true)
    }