# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 12:01:39 2021

@author: Magnus Frandsen
"""

import numpy as np
from scipy.stats import norm

def BScallprice(s0, sigma, r, T, K, greek = None):
    d1 = (np.log(s0 / K) + (r + 0.5 * sigma **2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    
    if greek is None:
        return s0 * Nd1 - np.exp(-r * T) * K * Nd2 #return price
    elif greek == "delta":
        return Nd1 #return delta
    

class BlackScholesModel():
    def __init__(self,S0, mu, sigma, rate, dt, n = 1):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        
        self.rate0 = rate
        
        self.dt = dt
        self.n = n
       
        self.reset_model()
        
    def evolve_s_b(self):
        #Evolve bank
        self.bank *= np.exp(self.rate * self.dt)
        
        #Evovle s 
        normals = np.random.normal(size = self.n)
        self.spot *= np.exp((self.mu - 0.5 * self.sigma **2) * self.dt + self.sigma * np.sqrt(self.dt) * normals)
                           
        #Evovle rate
        self.rate = self.rate
        
        #Evovle time
        self.time += self.dt
        
        #Save bank and spot
        self.spot_hist = np.append(self.spot_hist, self.spot[:,np.newaxis], 1)
        self.bank_hist = np.append(self.bank_hist, self.bank[:,np.newaxis], 1)
        self.rate_hist = np.append(self.rate_hist, self.rate[:,np.newaxis], 1)
        
        return self.spot, self.bank, self.time

    def reset_model(self, n = None):
         if not n is None:
             self.n = n
        
         self.time = 0
         self.spot = self.S0 * np.ones(self.n)
         self.bank = 1 * np.ones(self.n)
         self.rate = self.rate0 * np.ones(self.n)
         
         self.spot_hist = np.array(self.spot)[:,np.newaxis] #Applying np.array to make the arrays independent
         self.bank_hist = np.array(self.bank)[:,np.newaxis]
         self.rate_hist = np.array(self.rate)[:,np.newaxis]


    def init_option(self, name, params):
        if name == "call":
            strike = float(params[0])
            maturity = float(params[1])
            
            self.get_optimal_hs = lambda time, spot: BScallprice(spot, self.sigma, self.rate0, maturity - time, strike, "delta")
            
            return BScallprice(self.S0, self.sigma, self.rate0, maturity, strike)
            
        return None
    
    def get_current_optimal_hs(self):
        return self.get_optimal_hs(self.time, self.spot)
    
if __name__ == "__main__":
    model = BlackScholesModel(100, 0.05, 0.2, 0.02, 0.1, 10)
    model.init_option("call", [95, 2])
    model.get_current_optimal_hs()
    
    model.evolve_s_b()
    model.get_current_optimal_hs()
