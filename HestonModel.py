# -*- coding: utf-8 -*-
"""
Created on Sun Mar 7 2021

@author: Magnus Frandsen
"""

import numpy as np
from scipy.stats import norm
from scipy.interpolate import griddata
import OptionClass
import HestonLiptonPrice

def hestoncallprice(s0, v0, kappa, theta, sigma, rho, r, T, K, greek = None):
    if len(np.array(s0).shape) == 0:
        s0 = np.array([s0])
        v0 = np.array([v0])
    
    if greek is None:
        tmp_greek = 1
    elif greek == "delta":
        tmp_greek = 2
    elif greek == "vega":
        tmp_greek = 4
    
    grid_size = 20
    if len(s0) > 2*grid_size**2:
        min_s = np.min(s0)
        max_s = np.max(s0)
        min_v = np.min(v0)
        max_v = np.max(v0)
        
        if abs(max_s - min_s) < 1e-6:
            tmp_call = HestonLiptonPrice.heston_lipton_callprice(spot = s0[0], timetoexp = T, strike = K, 
                                                                r = r[0], divyield = 0, V = v0[0], 
                                                                theta = theta, kappa = kappa, 
                                                                epsilon = sigma, rho = rho, greek = tmp_greek)
            call = np.ones(len(s0)) * tmp_call
        else:
            data = np.zeros((int(grid_size**2),2))
            calls = np.zeros(int(grid_size**2))
            count = 0
            for i in range(grid_size):
                tmp_s = min_s + i * (max_s - min_s) / (grid_size - 1)
                for j in range(grid_size):
                    tmp_v = min_v + j * (max_v - min_v) / (grid_size - 1)
                    tmp_call = HestonLiptonPrice.heston_lipton_callprice(spot = tmp_s, timetoexp = T, strike = K, 
                                                                        r = r[0], divyield = 0, V = tmp_v, 
                                                                        theta = theta, kappa = kappa, 
                                                                        epsilon = sigma, rho = rho, greek = tmp_greek)
                    
                    data[count,0] = tmp_s
                    data[count,1] = tmp_v
                    calls[count] = tmp_call
                    count += 1
            
            #print(data[0:5,:],calls[0:5])
            call = griddata(data, calls, np.hstack((s0[:,np.newaxis],v0[:,np.newaxis])))
            #print(np.hstack((s0[:,np.newaxis],v0[:,np.newaxis])))
        
    else:
        call = [HestonLiptonPrice.heston_lipton_callprice(spot = s, timetoexp = T, strike = K, 
                                                                r = rate, divyield = 0, V = v, 
                                                                theta = theta, kappa = kappa, 
                                                                epsilon = sigma, rho = rho, greek = tmp_greek)
                for s, v, rate in zip(s0,v0, r)]
    
    return np.array(call)

def hestonputprice(s0, v0, kappa, theta, sigma, rho, r, T, K, greek = None):
    call = hestoncallprice(s0, v0, kappa, theta, sigma, rho, r, T, K, greek = greek)
    
    if greek is None:
        return K * np.exp(-r * T) + call - s0
    elif greek == "delta":
        return call - 1
    elif greek == "vega":
        return call

class HestonModel():
    def __init__(self,S0, mu, v0, kappa, theta, sigma, rho, rate, dt, ddt, n = 1):
        self.S0 = S0
        self.mu = mu
        
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta 
        self.sigma = sigma
        self.rho = rho
        
        self.corr = np.array([[1,rho],[rho,1]])
        
        self.rate0 = rate
        
        self.dt = dt
        self.ddt = dt / np.ceil(dt / ddt)
        self.ddt_steps = int(np.round(self.dt / self.ddt,6))
        
        self.n = n
        self.n_true_assets = 1
        self.n_assets = self.n_true_assets * 2
       
        self.reset_model()
        
    def evolve_s_b(self):
        #Evolve bank
        self.bank *= np.exp(self.rate * self.dt)
        
        for _ in range(self.ddt_steps):
            sqrt_v = np.sqrt(self.v)
            #Evovle s 
            normals1 = np.random.normal(size = (self.n,1))
            normals2 = np.random.normal(size = (self.n,1))
            
            self.spot1 *= np.exp((self.mu - 0.5 * self.sigma **2) * self.ddt + np.sqrt(self.v * self.ddt) * normals1)
            
            #Evovle v
            self.v += self.kappa * (self.theta - self.v) * self.ddt + self.sigma * sqrt_v * np.sqrt(self.ddt) * normals2                   
            self.v = np.abs(self.v)
            
            self.time2 += self.ddt
        
        #combine spot1 and v
        self.spot = np.hstack((self.spot1, self.v))
        
        #Evovle rate
        self.rate = self.rate
        
        #Evovle time
        self.time += self.dt
        
        #Save bank and spot
        self.spot_hist = np.append(self.spot_hist, self.spot[...,np.newaxis], -1)
        self.bank_hist = np.append(self.bank_hist, self.bank[...,np.newaxis], -1)
        self.rate_hist = np.append(self.rate_hist, self.rate[...,np.newaxis], -1)
        
        return self.spot, self.bank, self.time

    def reset_model(self, n = None):
         if not n is None:
             self.n = n
        
         self.time = 0
         self.time2 = 0
         
         self.spot1 = self.S0 * np.ones((self.n, self.n_true_assets))
         self.v = self.v0 * np.ones((self.n,self.n_true_assets))      
         self.spot = np.hstack((self.spot1,self.v))
         
         self.bank = 1 * np.ones(self.n)
         self.rate = self.rate0 * np.ones(self.n)
         
         self.spot_hist = np.array(self.spot)[...,np.newaxis] #Applying np.array to make the arrays independent
         self.v_hist = np.array(self.v)[...,np.newaxis]
         self.bank_hist = np.array(self.bank)[...,np.newaxis]
         self.rate_hist = np.array(self.rate)[...,np.newaxis]


    def init_option(self, option_por):
        self.option_por = option_por
        
        price = 0
        for option, units in zip(self.option_por.options, self.option_por.units):
            tmp_name = option.name
            tmp_under = option.underlying
            
            if tmp_name == "call":
                strike = option.params[0]
                maturity = option.params[1]
                
                tmp_price = hestoncallprice(s0 = self.S0, v0 = self.v0, kappa = self.kappa, 
                                            theta = self.theta, sigma = self.sigma, rho = self.rho, 
                                            r = self.rate, T = maturity, K = strike, greek = None)
                
                price += tmp_price * units
            
            if tmp_name == "put":
                strike = option.params[0]
                maturity = option.params[1]
                
                tmp_price = hestonputprice(s0 = self.S0, v0 = self.v0, kappa = self.kappa, 
                                            theta = self.theta, sigma = self.sigma, rho = self.rho, 
                                            r = self.rate, T = maturity, K = strike, greek = None)
                
                price += tmp_price * units
            
        return price

    
    def get_optimal_hs(self, time, spot):
        hs = np.zeros((len(spot),self.n_assets))
        for option, units in zip(self.option_por.options, self.option_por.units):
            tmp_name = option.name
            tmp_under = option.underlying
            
            if tmp_name == "call":
                strike = option.params[0]
                maturity = option.params[1]
                
                tmp_hs_s = hestoncallprice(s0 = spot[:,tmp_under], v0 = spot[:,2*tmp_under+1], kappa = self.kappa, 
                                            theta = self.theta, sigma = self.sigma, rho = self.rho, 
                                            r = self.rate, T = maturity - time, K = strike, greek = "delta")
                
                tmp_hs_v = hestoncallprice(s0 = spot[:,tmp_under], v0 = spot[:,2*tmp_under+1], kappa = self.kappa, 
                                            theta = self.theta, sigma = self.sigma, rho = self.rho, 
                                            r = self.rate, T = maturity - time, K = strike, greek = "vega")
                
                hs[:,tmp_under] += tmp_hs_s * units
                hs[:,self.n_true_assets + tmp_under] += tmp_hs_v * units
            
            if tmp_name == "put":
                strike = option.params[0]
                maturity = option.params[1]
                
                tmp_hs_s = hestonputprice(s0 = spot[:,tmp_under], v0 = spot[:,self.n_true_assets + tmp_under], 
                                          kappa = self.kappa, theta = self.theta, sigma = self.sigma, rho = self.rho, 
                                          r = self.rate, T = maturity - time, K = strike, greek = "delta")
                
                tmp_hs_v = hestonputprice(s0 = spot[:,tmp_under], v0 = spot[:,self.n_true_assets + tmp_under], 
                                          kappa = self.kappa, theta = self.theta, sigma = self.sigma, rho = self.rho, 
                                          r = self.rate, T = maturity - time, K = strike, greek = "vega")
                
                hs[:,tmp_under] += tmp_hs_s * units
                hs[:,self.n_true_assets + tmp_under] += tmp_hs_v * units
            
        return hs
        
    def get_current_optimal_hs(self, *args):
        print("Get hs", self.time)
        return self.get_optimal_hs(self.time, self.spot)
    
if __name__ == "__main__":
    model = HestonModel(100, mu = 0.03, v0 = 0.09, kappa = 2, theta = 0.09, 
                        sigma = 0.3, rho = -0.5, rate = 0.03, dt = 1/200)
    
    model.reset_model(int(1e5))
    for i in range(400):
        model.evolve_s_b()

    option_payoff = np.exp(- 0.03 * 2) * np.maximum(model.spot[:,0] - 50,0)
    option_price = np.mean(option_payoff)
    
