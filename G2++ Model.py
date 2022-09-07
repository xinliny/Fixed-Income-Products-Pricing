import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy
import scipy.stats
from scipy.stats import norm
from scipy.linalg import solve
from numpy.linalg import inv
import math


# G2++ MODEL

def G2(T, dt, r0, x0, y0, sigma1, sigma2, a, b, rho, phi):
    n = int(round(T/dt))
    x = np.zeros(n)
    y = np.zeros(n)
    r = np.zeros(n)
    r[0] = r0
    x[0] = x0
    y[0] = y0
    for i in range(1,n):
        z = np.random.normal(0,1,2)
        w1 = z[0]
        w2 = rho*z[0] + np.sqrt(1-rho**2)*z[1]
        x[i] = x[i-1] - a*x[i-1]*dt + sigma1*np.sqrt(dt)*w1
        y[i] = y[i-1] - b*x[i-1]*dt + sigma2*np.sqrt(dt)*w2
        r[i] = x[i]+y[i]+phi
    
    R = dt*sum(r)  # discount rate
    rt = r[n-1]  # interest rate at T
    xt = x[n-1]
    yt = y[n-1]
    return R, rt, xt, yt

  
# EUROPEAN PUT OPTION PRICE ON A PURE DISCOUNT BOND 

def ep_g2_im(T_opt, K, T, dt, r0, x0, y0, sigma1, sigma2, a, b, rho, phi):
    put = []
    for i in range(0, 1000):
        R, rt, xt, yt = G2(T_opt, dt, r0, x0, y0, sigma1, sigma2, a, b, rho, phi)
    
        p = []
        for i in range(0,100):
            p.append(np.exp(-G2(T-T_opt, dt, rt, xt, yt, sigma1, sigma2, a, b, rho, phi)[0]))
    
        bond_price = np.mean(p)*1000
        payoff = max(K-bond_price, 0)
        put.append(payoff * np.exp(-R))
    
    return np.mean(put)
