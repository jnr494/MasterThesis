# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 15:58:09 2021

@author: mrgna
"""

import numpy as np
import scipy

def HestonCar(phi, x, tau, rho, sigma, kappa, theta, r, nu, type_):
    if type_ == 1:
        u = 0.5
        b = kappa - rho * sigma
    elif type_ == 2:
        u = -0.5
        b = kappa
    pass

    d = np.sqrt((rho * sigma * phi * 1j - b)**2 - sigma**2 * (2 * u * phi * 1j - phi**2))
    g = (b - rho * sigma * phi * 1j + d) / (b - rho * sigma * 1j * phi - d)
    c = 1 / g
    C = r * phi * 1j * tau + kappa * theta / sigma**2 \
        * ((b - rho * sigma * phi * 1j + d) * tau - 2 * np.log((1 - g * np.exp(d * tau)) / (1 - g)))
    D = (b - rho * sigma * phi * 1j + d) / sigma**2 * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau)))
  
    f = np.exp(C+D * nu + 1j * phi * x)
    return f


def HestonIntegrant(phi, x, tau, rho, sigma, kappa, theta, r, nu, K, type_):
  Char = HestonCar(phi,x,tau,rho,sigma,kappa,theta,r,nu,type_)
  integrant = ((np.exp(-1j * phi * np.log(K)) * Char) / (1j * phi)).real
  return integrant


def HestonProb(x, tau, rho, sigma, kappa, theta, r, nu, K, type_):
    int_func = lambda phi: HestonIntegrant(phi, x, tau, rho, sigma, kappa, theta, r, nu, K, type_)
    integral = scipy.integrate.quad(int_func, 0.0001, 5000)
    P = 0.5 + 1 / np.pi * integral[0]
    return P

def HestonCallPrice(S, tau, rho, sigma, kappa, theta, r, nu, K):
    x = np.log(S)
    Prob1 = HestonProb(x, tau, rho, sigma, kappa, theta, r, nu, K, type_ = 1)
    Prob2 = HestonProb(x,tau,rho,sigma,kappa,theta,r,nu,K,type_ = 2)
    Price = np.exp(x) * Prob1 - K * np.exp(-r * tau) * Prob2
    return Price


###### Lipton implementering af Heston call-priser fra Finkont2 #######
def heston_lipton_callprice(spot, timetoexp, strike, r, divyield, V, theta, kappa, epsilon, rho, greek=1):
    #print(spot, timetoexp, strike, r, divyield, V, theta, kappa, epsilon, rho, greek)
    X = np.log(spot / strike) + (r - divyield) * timetoexp
    kappahat = kappa - 0.5 * rho * epsilon
    xiDummy = kappahat**2 + 0.25 * epsilon**2
  
    def integrand(k):
        xi = np.sqrt(k**2 * epsilon**2 * (1 - rho**2) + 2j * k * epsilon * rho * kappahat + xiDummy)
        Psi_P = -(1j * k * rho * epsilon + kappahat) + xi
        Psi_M = (1j * k * rho * epsilon + kappahat) + xi
        alpha = - kappa * theta * (Psi_P * timetoexp + 2 * np.log((Psi_M + Psi_P * np.exp(-xi * timetoexp)) / (2 * xi))) / epsilon**2
        beta = -(1 - np.exp(-xi * timetoexp)) / (Psi_M + Psi_P * np.exp(-xi * timetoexp))
        numerator = np.exp((-1j * k + 0.5) * X + alpha + (k**2 + 0.25) * beta * V)
    
        if greek==1: #price
            dummy = (numerator / (k**2 + 0.25)).real
        elif greek==2: #delta
            dummy = ((0.5 - 1j * k) * numerator / (spot * (k**2 + 0.25))).real
        if greek==3: #
            dummy = -(numerator / spot**2).real
        if greek==4: #dif varians
            dummy = (numerator * beta).real
        
        return dummy
    
    dummy = scipy.integrate.quad(integrand, -100, 100)[0]

    if greek==1: #price
        dummy = np.exp(-divyield * timetoexp) * spot - strike * np.exp(-r * timetoexp) * dummy / (2 * np.pi)
        
    elif greek==2: # delta
        dummy = np.exp(-divyield * timetoexp) - strike * np.exp(-r * timetoexp) * dummy / (2 * np.pi)
    
    elif greek==3: 
        dummy = -strike * np.exp(-r * timetoexp) * dummy / (2 * np.pi)
    
    elif greek==4: #dif varians
        dummy = -strike * np.exp(-r * timetoexp) * dummy / (2 * np.pi)
    
    return dummy


if __name__ == "__main__":
    print(HestonCallPrice(S = 100, tau = 1, rho = -0.9, sigma = 0.3, kappa = 2, theta = 0.09, r = 0.03, nu = 0.09, K = 100))
    print(heston_lipton_callprice(spot = 100, timetoexp = 1, strike = 100, r = 0.03, divyield =0., 
                             V = 0.09, theta = 0.09, kappa = 2, epsilon = 0.3, rho = -0.9, greek = 2))
    
    p_v = lambda v: heston_lipton_callprice(spot = 100, timetoexp = 1, strike = 100, r = 0.03, divyield =0., 
                                            V = v, theta = 0.09, kappa = 2, epsilon = 0.3, rho = -0.9, greek = 1)
    
    h = 0.001
    (p_v(0.09 + h) - p_v(0.09)) / h

    heston_lipton_callprice(spot = 100, timetoexp = 1, strike = 100, r = 0.03, divyield =0., 
                            V = 0.09, theta = 0.09, kappa = 2, epsilon = 0.3, rho = -0.9, greek = 4)