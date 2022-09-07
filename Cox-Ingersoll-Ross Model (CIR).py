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


# CIR MODEL

def CIR(T,dt,r0,sigma,kappa,r_mu):
    n = int(round(T/dt))
    r = np.zeros(n)
    r[0] = r0
    z = np.random.normal(0,1,n)
    for i in range(1,n):
        r[i] = r[i-1] + (kappa*r_mu - kappa*r[i-1])*dt + \
               sigma*np.sqrt(dt*r[i-1])*z[i]
        # r[i] = r[i-1] + (kappa*r_mu - kappa*r[i-1])*dt + sigma*np.sqrt(dt*r[i-1])*np.random.normal(0,1,1)[0]
    R = dt*sum(r)  # discount rate
    rt = r[n-1]  # interest rate at T
    return R, rt

  
# PURE DISCOUNT BOND PRICING

def ec_pdb_cir_im(T_opt, K, T,dt,r0,sigma,kappa,r_mu,fv):
    # T_opt: maturity of european call
    # T: maturity of pure discount bond

    call = []
    for i in range(0,500):
        
        R, rt = CIR(T_opt,dt,r0,sigma,kappa,r_mu)
        p = []
        for i in range(0,100):
            p.append(np.exp(-CIR(T-T_opt, dt, rt, sigma, kappa, r_mu)[0]))
        bond_price = np.mean(p)*fv
        payoff = max(bond_price-K, 0)
        call.append(payoff * np.exp(-R))
        
    return np.mean(call)
  
  
# EUROPEAN CALL OPTION ON PURE DISCOUNT BOND (IMPLICIT FINITE-DIFFERENCE METHOD)

h1 = np.sqrt(kappa**2+2*sigma**2)
h2 = (kappa+h1)/2
h3 = 2*kappa*r_mu/(sigma**2)
b = (np.exp(h1*(T-T_opt))-1) / (h2*(np.exp(h1*(T-T_opt))-1) + h1)
a = ( h1*np.exp(h2*(T-T_opt)) / (h2*(np.exp(h1*(T-T_opt))-1) + h1) )**h3

def IFD_call(K, T_opt, dt, sigma, kappa, r_mu, dr, r_l, r_h):
    
    m = int(round(T_opt/dt))   # m columns: time steps
    n = int(round((r_h-r_l)/dr))   # n rows: n different prices based on different r0
    
    R = np.linspace(r_h, r_l, n+1).reshape(n+1,1)
    bond_price = 1000 * a * np.exp(-b*R)
    
    F = np.where(bond_price-K>0, bond_price-K, 0)  # call option values at T
    P = np.repeat(bond_price, m+1).reshape(n+1, m+1)  # create bond_price table
    
    A = np.zeros((n+1)**2).reshape(n+1, n+1)
    A[0,0] = 1
    A[0,1] = -1
    A[n,n-1] = 1
    A[n,n] = -1
    
    for i in range(1, n):
        pu = -0.5 * dt * ((sigma**2)*R[i]/(dr**2) + kappa*(r_mu-R[i])/dr)
        pm = 1 + dt*((sigma**2)*R[i])/(dr**2) + R[i]*dt
        pd = -0.5 * dt * ((sigma**2)*R[i]/(dr**2) - kappa*(r_mu-R[i])/dr)
        A[i, i-1] = pu
        A[i, i] = pm
        A[i, i+1] = pd
    
    for k in np.linspace(m-1, 0, m):
        k = int(k)
        B = F
        B[0] = P[n-1,k] - P[n-2,k]
        B[n] = 0
        F = np.dot(inv(A), B)
    
    return F,R
  
dr = 0.01
r_l = 0
r_h = 0.2
P_2b = pd.DataFrame(columns=['r','price'])

P_2b['r'] = pd.Series(map(lambda x: x[0], IFD_call(K, T_opt, dt, sigma, kappa, r_mu, dr, r_l, r_h)[1]))
P_2b['price'] = pd.Series(map(lambda x: x[0], IFD_call(K, T_opt, dt, sigma, kappa, r_mu, dr, r_l, r_h)[0]))
print(P_2b)


# EUROPEAN CALL ON A PURE DISCOUNT BOND (EXPLICIT METHOD)

def ec_pdb_cir_ex(T_opt, K, T,dt,r0,sigma,kappa,r_mu,fv):
    h1 = np.sqrt(kappa**2+2*sigma**2)
    h2 = (kappa+h1)/2
    h3 = 2*kappa*r_mu/(sigma**2)
    B = (np.exp(h1*(T-T_opt))-1) / (h2*(np.exp(h1*(T-T_opt))-1) + h1)
    A = ( h1*np.exp(h2*(T-T_opt)) / (h2*(np.exp(h1*(T-T_opt))-1) + h1) )**h3
    
    call = []
    for i in range(0,10000):
        R, rt = CIR(T_opt,dt,r0,sigma,kappa,r_mu)   # refer to option maturity
        bond_price = A*np.exp(-B*rt)*fv
        payoff = max(bond_price-K, 0)
        call.append(np.exp(-R) * payoff)
   
    return np.mean(call)


  
