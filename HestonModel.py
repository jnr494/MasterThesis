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
            #print("call1",call[0:5], tmp_greek)
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
            
            #print(data.shape,calls[0:5])
            call = griddata(data, calls, np.hstack((s0,v0)))
            #print(np.hstack((s0[:,np.newaxis],v0[:,np.newaxis])))
            #print("call2",call[0:5], tmp_greek)
            #print(data[0:5,:], calls[0:5], np.hstack((s0,v0))[0:5,:])
        call = call[:,np.newaxis]
        
    else:
        call = [HestonLiptonPrice.heston_lipton_callprice(spot = s, timetoexp = T, strike = K, 
                                                                r = rate, divyield = 0, V = v, 
                                                                theta = theta, kappa = kappa, 
                                                                epsilon = sigma, rho = rho, greek = tmp_greek)
                for s, v, rate in zip(s0,v0, r)]
    
        #print(np.array(call).shape)
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
    def __init__(self, S0, mu, v0, kappa, theta, sigma, rho, rate, dt, ddt, lambda_ = 0, n = 1, ignore_sa = False):
        self.S0 = S0
        self.mu = mu
        
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta 
        self.sigma = sigma
        self.rho = rho
        
        self.corr = np.array([[1,rho],[rho,1]])
        
        self.rate0 = rate
        
        #Q params
        self.lambda_ = lambda_
        self.kappaQ = self.kappa + lambda_ 
        self.thetaQ = self.theta * self.kappa / (self.kappa + lambda_)
        
        #second asset
        self.sa_maturity = 1
        self.sa_strike = S0
        self.use_v = False
        self.ignore_sa = ignore_sa
        
        self.dt = dt
        self.ddt = dt / np.ceil(dt / ddt)
        self.ddt_steps = int(np.round(self.dt / self.ddt,6))
        
        self.n = n
        self.n_true_assets = 1
        self.n_assets = self.n_true_assets * 2
       
        self.reset_model()
    
    def second_asset_value(self):
        if self.ignore_sa is True:
            return np.zeros_like(self.v)
        
        if self.use_v is True:
            return self.v
        else:
            maturity = self.sa_maturity - self.time
            call_prices = hestoncallprice(s0 = self.spot1, v0 = self.v, kappa = self.kappaQ, 
                                            theta = self.thetaQ, sigma = self.sigma, rho = self.rho, 
                                            r = self.rate, T = maturity, K = self.sa_strike, greek = None)
            return call_prices
    
    def calculate_greeks_second_asset(self, time, spot, v):
        if self.ignore_sa is True:
            return np.zeros((self.n,2))
        
        if self.use_v is True:
            tmp_greek = np.zeros((self.n,2))
            tmp_greek[:,-1] = 1
            return tmp_greek
        else:
            maturity = self.sa_maturity - time
            delta = hestoncallprice(s0 = spot, v0 = v, kappa = self.kappaQ, 
                                            theta = self.thetaQ, sigma = self.sigma, rho = self.rho, 
                                            r = self.rate, T = maturity, K = self.sa_strike, greek = "delta")
            vega = hestoncallprice(s0 = self.spot1, v0 = self.v, kappa = self.kappaQ, 
                                            theta = self.thetaQ, sigma = self.sigma, rho = self.rho, 
                                            r = self.rate, T = maturity, K = self.sa_strike, greek = "vega")
            
            return np.hstack((delta[:,np.newaxis],vega[:,np.newaxis]))
            
            
            
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
        
        #Evovle rate
        self.rate = self.rate
        
        #Evovle time
        self.time += self.dt
        
        #combine spot1 and second asset
        second_asset_value = self.second_asset_value()
        self.spot = np.hstack((self.spot1, second_asset_value))
        
        #Save bank and spot
        self.spot_hist = np.append(self.spot_hist, self.spot[...,np.newaxis], -1)
        self.bank_hist = np.append(self.bank_hist, self.bank[...,np.newaxis], -1)
        self.rate_hist = np.append(self.rate_hist, self.rate[...,np.newaxis], -1)
        
        #update min max
        self.min_spot = np.minimum(self.min_spot, self.spot)
        self.max_spot = np.maximum(self.max_spot, self.spot)
        
        return self.spot, self.bank, self.time

    def reset_model(self, n = None):
         if not n is None:
             self.n = n
        
         self.time = 0
         self.time2 = 0
         
         self.spot1 = self.S0 * np.ones((self.n, self.n_true_assets))
         self.v = self.v0 * np.ones((self.n,self.n_true_assets))      
         
         self.bank = 1 * np.ones(self.n)
         self.rate = self.rate0 * np.ones(self.n)

         second_asset_value = self.second_asset_value()
         #print(self.spot1.shape, second_asset_value.shape)
         self.spot = np.hstack((self.spot1, second_asset_value))
         
         #hist
         self.spot_hist = np.array(self.spot)[...,np.newaxis] #Applying np.array to make the arrays independent
         self.v_hist = np.array(self.v)[...,np.newaxis]
         self.bank_hist = np.array(self.bank)[...,np.newaxis]
         self.rate_hist = np.array(self.rate)[...,np.newaxis]
         
         #min max
         self.min_spot = np.array(self.spot)
         self.max_spot = np.array(self.spot)

    def init_option(self, option_por):
        self.option_por = option_por
        
        price = 0
        for option, units in zip(self.option_por.options, self.option_por.units):
            tmp_name = option.name
            tmp_under = option.underlying
            
            if tmp_name == "call":
                strike = option.params[0]
                maturity = option.params[1]
                
                tmp_price = hestoncallprice(s0 = self.S0, v0 = self.v0, kappa = self.kappaQ, 
                                            theta = self.thetaQ, sigma = self.sigma, rho = self.rho, 
                                            r = self.rate, T = maturity, K = strike, greek = None)
                
                price += tmp_price * units
            
            if tmp_name == "put":
                strike = option.params[0]
                maturity = option.params[1]
                
                tmp_price = hestonputprice(s0 = self.S0, v0 = self.v0, kappa = self.kappaQ, 
                                            theta = self.thetaQ, sigma = self.sigma, rho = self.rho, 
                                            r = self.rate, T = maturity, K = strike, greek = None)
                
                price += tmp_price * units
            
        return price

    
    def get_optimal_hs(self, time, spot, v):
        hs = np.zeros((len(spot),self.n_assets))
        for option, units in zip(self.option_por.options, self.option_por.units):
            tmp_name = option.name
            tmp_under = option.underlying
            
            if tmp_name == "call":
                strike = option.params[0]
                maturity = option.params[1]
                
                tmp_delta = hestoncallprice(s0 = spot, v0 = v, kappa = self.kappaQ, 
                                            theta = self.thetaQ, sigma = self.sigma, rho = self.rho, 
                                            r = self.rate, T = maturity - time, K = strike, greek = "delta")
                
                tmp_vega = hestoncallprice(s0 = spot, v0 = v, kappa = self.kappaQ, 
                                            theta = self.thetaQ, sigma = self.sigma, rho = self.rho, 
                                            r = self.rate, T = maturity - time, K = strike, greek = "vega")
                
            
            if tmp_name == "put":
                strike = option.params[0]
                maturity = option.params[1]
                
                tmp_delta = hestonputprice(s0 = spot, v0 = v, 
                                          kappa = self.kappa, theta = self.theta, sigma = self.sigma, rho = self.rho, 
                                          r = self.rate, T = maturity - time, K = strike, greek = "delta")
                
                tmp_vega = hestonputprice(s0 = spot, v0 = v, 
                                          kappa = self.kappaQ, theta = self.thetaQ, sigma = self.sigma, rho = self.rho, 
                                          r = self.rate, T = maturity - time, K = strike, greek = "vega")
                
            tmp_delta = np.squeeze(tmp_delta)
            tmp_vega = np.squeeze(tmp_vega)
                
            sa_greeks = np.squeeze(self.calculate_greeks_second_asset(time, spot, v))
            
            print(sa_greeks.shape)
            
            #print(tmp_delta[0:5,:],tmp_vega[0:5,:])
            if self.ignore_sa is True:
                no_second_asset = np.squeeze(np.zeros_like(tmp_vega))
            else:
                print(tmp_vega.shape,  sa_greeks[:,-1].shape)
                no_second_asset = tmp_vega / sa_greeks[:,-1]
            
            print(tmp_delta.shape, no_second_asset.shape, (sa_greeks[:,0]).shape, (no_second_asset * sa_greeks[:,0]).shape)
            no_primary_asset = tmp_delta - no_second_asset * sa_greeks[:,0] 

            hs[:,tmp_under] += np.squeeze(no_primary_asset * units)
            hs[:,self.n_true_assets + tmp_under] += np.squeeze(no_second_asset * units)
            
        return hs
        
    def get_current_optimal_hs(self, *args):
        print("Get hs", self.time)
        return self.get_optimal_hs(self.time, self.spot1, self.v)
    
if __name__ == "__main__":
    model = HestonModel(100, mu = 0.03, v0 = 0.09, kappa = 2, theta = 0.09, 
                        sigma = 0.3, rho = -0.5, rate = 0.03, dt = 1/200)
    
    model.reset_model(int(1e5))
    for i in range(400):
        model.evolve_s_b()

    option_payoff = np.exp(- 0.03 * 2) * np.maximum(model.spot[:,0] - 50,0)
    option_price = np.mean(option_payoff)
    
