import numpy as np
from .gee_utils import get_sar_and_era5
from .cmod5n import cmod5n_inverse

def retrieve_wind_field(polygon_coords, date_str, scale=5000):
    # Pass the series of coordinates directly to GEE
    df = get_sar_and_era5(polygon_coords, date_str, scale)
    if df.empty:
        return []
    
    df['era5_wind_speed'] = np.sqrt(df['u_10']**2 + df['v_10']**2)
    wind_dir_to = (np.degrees(np.arctan2(df['u_10'], df['v_10'])) + 360) % 360
    df['phi'] = (wind_dir_to - df['radar_look_dir']) % 360
    
    df['sar_wind_speed'] = cmod5n_inverse(
        sigma0_obs=df['sigma0_linear'].values, 
        phi=df['phi'].values, 
        theta=df['incidence'].values
    )
    
    df = df.dropna(subset=['sar_wind_speed'])
    cols_to_round = ['lon', 'lat', 'vv_db', 'sar_wind_speed', 'era5_wind_speed']
    df[cols_to_round] = df[cols_to_round].round(3)
    
    return df.to_dict(orient='records')