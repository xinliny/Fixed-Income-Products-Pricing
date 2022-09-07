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


# PURE DISCOUNT BOND PRICING VIA MONTE-CARLO SIMULATION

def vasicek(T, dt, r0, sigma, kappa, r_mu):
    n = int(round(T/dt))
    r = np.zeros(n)
    r[0] = r0
    for i in range(1,n):
        r[i] = r[i-1] + (kappa*r_mu - kappa*r[i-1])*dt + sigma*np.sqrt(dt)*np.random.normal(0,1,1)[0]
    R = dt*sum(r)  # discount rate
    rt = r[n-1]  # interest rate at T
    return R, rt


def pdb_vasicek(T,dt,r0,sigma,kappa,r_mu,fv):
    p = []
    for i in range(0,100):
        p.append(np.exp(-vasicek(T,dt,r0,sigma,kappa,r_mu)[0]))
    price = np.mean(p)*fv
    return price


# COUPON-PAYING BOND PRICING

# time list
t = list(range(5,45,5))
all_T = []
for i in range(len(t)):
    all_T.append(t[i]*0.1)

# coupon & face value list
fv = 1000
cp = 30
c = [cp for i in range(8)]
c[-1] = cp+fv

prc = []
for i in range(0, len(all_T)):
    prc.append(pdb_vasicek(all_T[i], dt, r0, sigma, kappa, r_mu, c[i]))
    
coupon_bond_price = sum(p2)



# EUROPEAN CALL OPTION PRICE ON PURE DISCOUNT BOND

# 1. EXPLICIT METHOD
def ec_pdb_vsk_ex(T_opt, K, T, dt, r0, sigma, kappa, r_mu, fv):
    # T_opt: maturity of european call
    # T: maturity of pure discount bond
    B = 1/kappa*(1-np.exp(-kappa*(T-T_opt)))
    A = np.exp((r_mu-sigma**2/(2*kappa**2))*(B-T+T_opt)) - (sigma**2/(4*kappa))*B**2
    
    call = []
    for i in range(0,1000):
        R, rt = vasicek(T_opt,dt,r0,sigma,kappa,r_mu)
        bond_price = A*np.exp(-B*rt)*fv
        payoff = max(bond_price-K, 0)
        call.append(payoff * np.exp(-R))
        
    return np.mean(call)


# 2. IMPLICIT METHOD
p4 = []

for m in tqdm(range(0,10)):
    
    R, rt = vasicek(T,dt,r0,sigma,kappa,r_mu)
    
    p = []
    for i in range(0,len(all_T)):
        p.append(pdb_vasicek(all_T[i]-T_opt, dt, r0, sigma, kappa, r_mu, c[i])) # remember to minus option maturity
    bond_price = sum(p)
    payoff = max(bond_price-K, 0)
    p4.append(payoff*np.exp(-R))
    
price4 = np.mean(p4)


# EUROPEAN CALL OPTION PRICE ON COUPON-PAYING BOND

p5 = []

for m in tqdm(range(0,10000)):
    
    R, rt = vasicek(T_opt,dt,r0,sigma,kappa,r_mu) # remember T refers to call maturity, rather than bond maturity
    
    p = []
    for i in range(0,len(all_T)):
        B = 1/kappa*(1-np.exp(-kappa*(all_T[i]-T_opt)))
        A = np.exp((r_mu-sigma**2/(2*kappa**2))*(B-all_T[i]+T_opt)) - (sigma**2/(4*kappa))*B**2
        p.append(A*np.exp(-B*rt)*c[i])
    bond_price = sum(p)
    payoff = max(bond_price-K, 0)
    p5.append(payoff*np.exp(-R))
    
price5 = np.mean(p5)
price5



