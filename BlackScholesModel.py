# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 12:01:39 2021

@author: Magnus Frandsen
"""

import numpy as np
from scipy.stats import norm
import OptionClass
from BlackScholesFunctions import BScallprice, BSputprice, BSdocallprice, BSdoputprice

class BlackScholesModel():
    def __init__(self,S0, mu, sigma, corr, rate, dt, n = 1, n_assets = 1):
        self.S0 = S0 * np.ones(n_assets)
        self.mu = mu
        self.sigma = sigma
        self.corr = corr
        
        self.rate0 = rate
        
        self.dt = dt
        self.n = n
        self.n_assets = n_assets
       
        self.reset_model()
        
    def evolve_s_b(self):
        #Evolve bank
        self.bank *= np.exp(self.rate * self.dt)
        
        #Evovle s 
        normals = np.random.multivariate_normal(mean = np.zeros(self.n_assets), 
                                                      cov = self.corr, size = (self.n))
        
        self.spot *= np.exp((self.mu - 0.5 * self.sigma **2) * self.dt + self.sigma * np.sqrt(self.dt) * normals)
                           
        #Evovle rate
        self.rate = self.rate
        
        #Evovle time
        self.time += self.dt
        
        #Save bank and spot
        self.spot_hist = np.append(self.spot_hist, self.spot[...,np.newaxis], -1)
        self.bank_hist = np.append(self.bank_hist, self.bank[...,np.newaxis], -1)
        self.rate_hist = np.append(self.rate_hist, self.rate[...,np.newaxis], -1)
        
        #update min max
        self.min_spot = np.minimum(self.min_spot, self.spot)
        self.max_spot = np.maximum(self.max_spot, self.spot)
        
        #update spot return
        self.spot_return = (self.spot_hist[...,-1] / (self.spot_hist[...,-2]+1e-8))
        
        return self.spot, self.bank, self.time

    def reset_model(self, n = None):
         if not n is None:
             self.n = n
        
         self.time = 0
         self.spot = self.S0 * np.ones((self.n, self.n_assets))
         self.bank = 1 * np.ones(self.n)
         self.rate = self.rate0 * np.ones(self.n)
         
         self.spot_hist = np.array(self.spot)[...,np.newaxis] #Applying np.array to make the arrays independent
         self.bank_hist = np.array(self.bank)[...,np.newaxis]
         self.rate_hist = np.array(self.rate)[...,np.newaxis]
         
         #min max
         self.min_spot = np.array(self.spot)
         self.max_spot = np.array(self.spot)
         
         #spot return
         self.spot_return = np.ones_like(self.spot)
         
    def init_option(self, option_por):
        self.option_por = option_por
        
        price = 0
        for option, units in zip(self.option_por.options, self.option_por.units):
            tmp_name = option.name
            tmp_under = option.underlying
            
            if tmp_name == "call":
                strike = option.params[0]
                maturity = option.params[1]
                
                tmp_price = BScallprice(self.S0[tmp_under], self.sigma[tmp_under], self.rate0, 
                                     maturity, strike)
                
                price += tmp_price * units
            
            if tmp_name == "put":
                strike = option.params[0]
                maturity = option.params[1]
                
                tmp_price = BSputprice(self.S0[tmp_under], self.sigma[tmp_under], self.rate0, 
                                     maturity, strike)
                
                price += tmp_price * units
            
            if tmp_name == "docall":
                strike = option.params[0]
                maturity = option.params[1]
                boundary = option.params[2]
                
                tmp_price = BSdocallprice(self.S0[tmp_under], self.sigma[tmp_under], self.rate0, 
                                     maturity, strike, boundary)
                
                price += tmp_price * units
            
            if tmp_name == "doput":
                strike = option.params[0]
                maturity = option.params[1]
                boundary = option.params[2]
                
                tmp_price = BSdoputprice(self.S0[tmp_under], self.sigma[tmp_under], self.rate0, 
                                     maturity, strike, boundary)
                
                price += tmp_price * units
            
        return price
# =============================================================================
#         if name == "call":
#             strike = float(params[0])
#             maturity = float(params[1])
#             
#             self.get_optimal_hs = lambda time, spot: BScallprice(spot, self.sigma, self.rate0, maturity - time, strike, "delta")
#             
#             return BScallprice(self.S0, self.sigma, self.rate0, maturity, strike)
# =============================================================================
            
        return None
    
    def get_optimal_hs(self, time, spot_hist):
        if len(spot_hist.shape) == 2:
            spot_hist = spot_hist[...,np.newaxis]
        
        spot = spot_hist[...,-1]
        
        hs = np.zeros((len(spot),self.n_assets))
        for option, units in zip(self.option_por.options, self.option_por.units):
            tmp_name = option.name
            tmp_under = option.underlying
            
            if tmp_name == "call":
                strike = option.params[0]
                maturity = option.params[1]
                
                tmp_hs = BScallprice(spot[:,tmp_under], self.sigma[tmp_under], self.rate0, 
                                     maturity - time, strike, "delta")
                
                hs[:,tmp_under] += tmp_hs * units
            
            if tmp_name == "put":
                strike = option.params[0]
                maturity = option.params[1]
                
                tmp_hs = BSputprice(spot[:,tmp_under], self.sigma[tmp_under], self.rate0, 
                                     maturity - time, strike, "delta")
                
                hs[:,tmp_under] += tmp_hs * units
            
            if tmp_name == "docall":
                strike = option.params[0]
                maturity = option.params[1]
                boundary = option.params[2]
                
                indicator = np.min(spot_hist[:,tmp_under,:], axis = -1) > boundary
                
                tmp_hs = BSdocallprice(spot[:,tmp_under], self.sigma[tmp_under], self.rate0, 
                                     maturity - time, strike, boundary, "delta")
                
                hs[:,tmp_under] += indicator * tmp_hs * units
            
            if tmp_name == "doput":
                strike = option.params[0]
                maturity = option.params[1]
                boundary = option.params[2]
                
                indicator = np.min(spot_hist[:,tmp_under,:], axis = -1) > boundary
                
                tmp_hs = BSdoputprice(spot[:,tmp_under], self.sigma[tmp_under], self.rate0, 
                                     maturity - time, strike, boundary, "delta")
                
                hs[:,tmp_under] += indicator * tmp_hs * units
            
        return hs
        
    def get_current_optimal_hs(self, *args):
        return self.get_optimal_hs(self.time, self.spot_hist)

class BShedge():
    def __init__(self, bs_model):
        self.bs_model = bs_model
            
    def get_current_optimal_hs(self, model, current_hs, current_pf):
        return self.bs_model.get_optimal_hs(model.time, model.spot_hist)


if __name__ == "__main__":
    mu = np.array([0.02, 0.03, 0.01])
    sigma = np.array([0.2,0.15, 0.1])
    corr = np.array([[1,0.95, 0.9], [0.95,1, 0.8], [0.9,0.8,1]])
    
    model = BlackScholesModel(1, mu, sigma, corr, 0.02, 0.1, 10, 3)
    options = OptionClass.OptionPortfolio([OptionClass.Option("call",[0.95,1],0),
                                           OptionClass.Option("call",[1,1],1)],
                                          [1,-2])
    model.init_option(options)
    print(model.get_current_optimal_hs())
    model.evolve_s_b()
    print(model.get_current_optimal_hs())
    #model.init_option("call", [95, 2])
    #model.get_current_optimal_hs()
    
    #model.evolve_s_b()
    #model.get_current_optimal_hs()
    
    #a = np.random.multivariate_normal(mean = np.array([0,0,0]), cov = corr, size = (10))
