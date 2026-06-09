import numpy as np
from scipy.optimize import minimize

def cmod_empirical_forward(v, phi, theta):
    gamma = 1.4 - 0.015 * (theta - 40.0)
    a_theta_db = -12.0 - 0.3 * (theta - 35.0) 
    a_theta_lin = 10 ** (a_theta_db / 10.0)
    b0 = a_theta_lin * ((v / 10.0) ** gamma)
    b1 = 0.05 + 0.005 * v
    b2 = 0.15 + 0.002 * v
    phi_rad = np.radians(phi)

    return b0 * (1.0 + b1 * np.cos(phi_rad) + b2 * np.cos(2.0 * phi_rad))

def invert_cmod5(vv_db: np.ndarray, incidence_angles: np.ndarray, wind_directions: np.ndarray) -> np.ndarray:
    wind_speeds = np.zeros_like(vv_db)
    SATELLITE_LOOK_DIR = 80.0 

    for i in range(len(vv_db)):
        s0_db_true = vv_db[i]
        inc_angle = incidence_angles[i]

        if np.isnan(inc_angle) or inc_angle < 10.0 or inc_angle > 80.0:
            inc_angle = 35.0 
        phi = (wind_directions[i] - SATELLITE_LOOK_DIR) % 360 

        def objective(u):
            if u[0] <= 0.1: return 1e6 
            sim_s0_lin = cmod_empirical_forward(u[0], phi, inc_angle)
            sim_s0_db = 10 * np.log10(sim_s0_lin + 1e-8) 
            return (sim_s0_db - s0_db_true) ** 2

        res = minimize(objective, [8.0], bounds = [(0.1, 35.0)], method = 'L-BFGS-B')
        wind_speeds[i] = res.x[0]

    return wind_speeds
