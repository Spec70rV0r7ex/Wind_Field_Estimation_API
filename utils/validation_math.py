import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

def circular_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Calculates the shortest angular difference on a 360-degree circle."""
    diff = np.abs(pred - true)
    return np.minimum(diff, 360 - diff)

def circular_correlation(alpha: np.ndarray, beta: np.ndarray) -> float:
    """Calculates the Circular Pearson Correlation Coefficient for angular variables."""
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)

    mean_a = np.arctan2(np.mean(np.sin(alpha_rad)), np.mean(np.cos(alpha_rad)))
    mean_b = np.arctan2(np.mean(np.sin(beta_rad)), np.mean(np.cos(beta_rad)))

    num = np.sum(np.sin(alpha_rad - mean_a) * np.sin(beta_rad - mean_b))
    den = np.sqrt(np.sum(np.sin(alpha_rad - mean_a)**2) * np.sum(np.sin(beta_rad - mean_b)**2))

    if den == 0: return 0.0
    return float(num / den)

def calculate_metrics(pred_speeds: np.ndarray, pred_dirs: np.ndarray, 
                      true_speeds: np.ndarray, true_dirs: np.ndarray) -> dict:
    """
    Computes Bias, MAE, RMSE, and Correlation for your final report.
    """
    speed_bias = float(np.mean(pred_speeds - true_speeds))
    speed_mae = float(mean_absolute_error(true_speeds, pred_speeds))
    speed_rmse = float(np.sqrt(mean_squared_error(true_speeds, pred_speeds)))

    if np.std(true_speeds) == 0 or np.std(pred_speeds) == 0:
        speed_corr = 0.0
    else:
        speed_corr, _ = pearsonr(true_speeds, pred_speeds)

    dir_errors = circular_error(pred_dirs, true_dirs)

    dir_bias = float(np.degrees(np.arctan2(
        np.mean(np.sin(np.radians(pred_dirs - true_dirs))),
        np.mean(np.cos(np.radians(pred_dirs - true_dirs)))
    )))
    dir_mae = float(np.mean(dir_errors))
    dir_rmse = float(np.sqrt(np.mean(dir_errors**2)))

    dir_corr = circular_correlation(pred_dirs, true_dirs)

    return {
        "speed_bias": round(speed_bias, 3),
        "speed_mae": round(speed_mae, 3),
        "speed_rmse": round(speed_rmse, 3),
        "speed_corr": round(speed_corr, 3), # type: ignore
        "dir_bias": round(dir_bias, 3),
        "dir_mae": round(dir_mae, 3),
        "dir_rmse": round(dir_rmse, 3),
        "dir_corr": round(dir_corr, 3),
        "sample_size": len(pred_speeds)
    }