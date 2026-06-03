import numpy as np
from scipy.optimize import brentq

def cmod5n_forward(v, phi, theta):
    """
    CMOD5.N forward model for equivalent neutral wind.
    Inputs:
        v: wind speed [m/s]
        phi: relative angle between radar look direction and wind direction [degrees]
        theta: incidence angle [degrees]
    Returns:
        sigma0 in linear units
    """
    DTOR = 57.29577951
    THETM = 40.
    THETHR = 25.
    ZPOW = 1.6
    
    # 28 Coefficients for CMOD5.N
    C = [0, -0.6878, -0.7957,  0.3380, -0.1728,  0.0000, 
            0.0040,  0.1103,  0.0159,  6.7329,  2.7713, 
           -2.2885,  0.4971, -0.7250,  0.0450,  0.0066, 
            0.3222,  0.0120, 22.7000,  2.0813,  3.0000, 
            8.3659, -3.3428,  1.3236,  6.2437,  2.3893, 
            0.3249,  4.1590,  1.6930]
    
    Y0 = C[19]
    PN = C[20]
    A = C[19] - (C[19]-1)/C[20]
    B = 1. / (C[20] * (C[19]-1.)**(3-1))

    FI = phi / DTOR
    CSFI = np.cos(FI)
    CS2FI = 2.00 * CSFI * CSFI - 1.00
    
    X = (theta - THETM) / THETHR
    XX = X * X
    
    A0 = C[1] + C[2]*X + C[3]*XX + C[4]*X*XX
    A1 = C[5] + C[6]*X
    A2 = C[7] + C[8]*X
    GAM = C[9] + C[10]*X + C[11]*XX
    S0 = C[12] + C[13]*X
    
    V = v
    S = A2 * V
    
    # Handle vectors/arrays cleanly for constraints
    S_vec = np.atleast_1d(S).copy()
    S0_vec = np.atleast_1d(S0).copy()
    if S0_vec.size == 1 and S_vec.size > 1:
        S0_vec = np.full_like(S_vec, S0_vec[0])
        
    SlS0 = S_vec < S0_vec
    S_vec[SlS0] = S0_vec[SlS0]
    
    A3 = 1. / (1. + np.exp(-S_vec))
    S_orig = np.atleast_1d(S)
    A3[SlS0] = A3[SlS0] * (S_orig[SlS0] / S0_vec[SlS0])**(S0_vec[SlS0] * (1. - A3[SlS0]))
    
    B0 = (A3**GAM) * 10.**(A0 + A1*V)
    
    B1 = C[15]*V * (0.5 + X - np.tanh(4.*(X + C[16] + C[17]*V)))
    B1 = C[14]*(1. + X) - B1
    B1 = B1 / (np.exp(0.34*(V - C[18])) + 1.)
    
    V0 = C[21] + C[22]*X + C[23]*XX
    D1 = C[24] + C[25]*X + C[26]*XX
    D2 = C[27] + C[28]*X
    
    V2 = (V / V0 + 1.)
    V2 = np.atleast_1d(V2)
    V2ltY0 = V2 < Y0
    V2[V2ltY0] = A + B * (V2[V2ltY0] - 1.)**PN
    
    B2 = (-D1 + D2*V2) * np.exp(-V2)
    
    CMOD5_N = B0 * (1.0 + B1*CSFI + B2*CS2FI)**ZPOW
    return np.squeeze(CMOD5_N)

def cmod5n_inverse(sigma0_obs, phi, theta):
    """
    Inverts the CMOD5.N GMF to solve for wind speed.
    """
    def objective(v_guess, s0_obs, p, t):
        return cmod5n_forward(v_guess, p, t) - s0_obs
    
    s0 = np.atleast_1d(sigma0_obs)
    p = np.atleast_1d(phi)
    t = np.atleast_1d(theta)
    
    v_out = np.zeros_like(s0)
    for i in range(len(s0)):
        try:
            # Search for root between 0.1 m/s and 50.0 m/s (Hurricane force cap)
            v_out[i] = brentq(objective, 0.1, 50.0, args=(s0[i], p[i], t[i]))
        except ValueError:
            v_out[i] = np.nan
            
    return np.squeeze(v_out)