import gmrf
import numpy as np

def ivgmrf_denoise(data, 
                   _lambda = 1e-11, 
                   _alpha = 1e-8, 
                   _sigma2 = 1000, 
                   _eps = 1e-09, 
                   _lambda_rate = 1e-13, 
                   _alpha_rate = 1e-06
):
    m = gmrf.ivgmrf.ivgmrf()
    m.set_lambda(_lambda)
    m.alpha = _alpha
    m.sigma2 = _sigma2
    m.set_eps(_eps)
    m.lambda_rate = _lambda_rate
    m.alpha_rate = _alpha_rate

    result = m.denoise(data)

    return result