# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 19:44:33 2021

@author: mrgna
"""

import os
import sys
sys.path.insert(1,os.path.dirname(os.getcwd()))

import numpy as np
import matplotlib.pyplot as plt

from BlackScholesModel import BScallprice


S0 = 1
sigma = 0.3
r = 0.01
mu = 0.0
T = 0.01
T2 = 1

N_sim = int(1e6)
normals = np.random.normal(size = N_sim)

get_S1 = lambda T_: S0 * np.exp((mu - sigma**2/2)*T_ + np.sqrt(T_)*sigma*normals)
S1 = get_S1(T)

print(np.mean(S1), S0 * np.exp(mu))
print(np.var(S1), (np.exp(T*sigma**2) - 1) * (np.exp(2*(mu - sigma**2/2)*T + T*sigma**2)))

strike = 1.1
#option_payoff = lambda spot: np.maximum(spot - strike,0)
option_payoff = lambda spot, T: BScallprice(s0 = spot, sigma = sigma, r = r, T = T2, K = strike)

option_payoffs = option_payoff(S1, T)
print(np.exp(-r*T)*np.mean(option_payoffs))

option_price = BScallprice(s0 = S0, sigma = sigma, r = r, T = T + T2, K = strike)

def calculate_PF(s0, s1, r, h, T, p0):
    return h * (s1 - s0 * np.exp(r*T)) + p0*np.exp(r*T)

p0 = option_price
calculate_PF_h = lambda h: calculate_PF(S0, S1, r, h, T, p0)

MSE = lambda h: np.mean((calculate_PF_h(h) - option_payoffs)**2)

# =============================================================================
# hs = np.linspace(0,1,1000)
# MSEs = np.array([MSE(h) for h in hs])
# 
# 
# plt.plot(hs,MSEs)
# plt.show()
# 
# min_h = hs[np.argmin(MSEs)]
# print("Min h:",min_h)
# =============================================================================

delta = BScallprice(s0 = S0, sigma = sigma, r = r, T = T + T2, K = strike, greek = "delta")
print("Delta:",delta)

def get_analytical_h(T, p0 = None):
    if p0 == None:
        p0 = BScallprice(s0 = S0, sigma = sigma, r = r, T = T + T2, K = strike)
    S1 = get_S1(T)
    option_payoffs = option_payoff(S1, T)
    return np.mean((option_payoffs - p0*np.exp(r*T))*(S1 - S0*np.exp(r*T))) / np.mean((S1 - S0*np.exp(r*T))**2)

analytical_h = get_analytical_h(T)
print("Analytical h:",analytical_h)


Ts = np.concatenate((np.linspace(0.0001,0.009,100),np.linspace(0.01,1,100)))
analytical_hs = [get_analytical_h(T_) for T_ in Ts]
deltas = [BScallprice(s0 = S0, sigma = sigma, r = r, T = T_ + T2, K = strike, greek = "delta") for T_ in Ts]

plt.plot(Ts,deltas)
plt.plot(Ts,analytical_hs)
plt.show()

######
tmp_T = 0.1
tmp_p0 = BScallprice(s0 = S0, sigma = sigma, r = r, T = tmp_T + T2, K = strike)
p0s = np.linspace(-5.*tmp_p0,10*tmp_p0,200)
hs_p0 = [get_analytical_h(tmp_T, p0) for p0 in p0s]
plt.plot(p0s,hs_p0)
