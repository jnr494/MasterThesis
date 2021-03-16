# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 15:15:36 2021

@author: Magnus Frandsen
"""

import numpy as np
from PortfolioClass import Portfolio

class Binom_model():
    def __init__(self,S0, alpha, sigma, rate, p, dt, rate_change = 0., n = 1):
        self.n_assets = 1
        
        self.S0 = S0
        self.alpha = alpha
        self.sigma = sigma
        
        self.rate0 = rate
        self.rate_change = rate_change
        
        self.p = p  
        self.dt = dt
        self.n = n
       
        self.reset_model()
        
    def evolve_s_b(self):
        #Evolve bank
        self.bank *= np.exp(self.rate * self.dt)
        
        #Evovle s 
        sign = np.random.choice([1,-1], p = [self.p, 1-self.p], size = self.spot.shape)
        self.spot *= np.exp(self.alpha * self.dt + sign * self.sigma * np.sqrt(self.dt))
        
        self.no_u += np.maximum(sign,0)  
        self.no_d += np.maximum(-sign,0)
        
        #Evovle rate
        #self.rate += self.rate_change * (sign)
        
        rho = 0.5
        sign2 = np.random.choice([1,-1], p = [self.p, 1-self.p], size = self.n)
        self.rate = rho * self.rate + (1 - rho) * self.rate0 + self.rate_change * sign2
        
        #Evovle time
        self.time += self.dt
        
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
         self.spot = self.S0 * np.ones((self.n,1))
         self.bank = 1 * np.ones(self.n)
         self.rate = self.rate0 * np.ones(self.n)
         
         self.spot_hist = np.array(self.spot)[...,np.newaxis] #Applying np.array to make the arrays independent
         self.bank_hist = np.array(self.bank)[...,np.newaxis]
         self.rate_hist = np.array(self.rate)[...,np.newaxis]
         
         self.no_d = np.zeros(self.spot.shape)
         self.no_u = np.zeros(self.spot.shape)
         
         #min max
         self.min_spot = np.array(self.spot)
         self.max_spot = np.array(self.spot)
         
    def init_option(self, option_por):
        maturity = option_por.options[0].params[1]
        self.optimal_hedge_setup(maturity)
        
        self.mat_spots = np.array([self.S0*self.optimal_hedge.u**(self.optimal_hedge.n-i) * self.optimal_hedge.d**(i) 
                                   for i in range(self.optimal_hedge.n+1)])[:,np.newaxis]
        
        tmp_option_payoffs = option_por.get_portfolio_payoff(self.mat_spots[...,np.newaxis])
        
        option_price = self.optimal_hedge.calculate_hedge_price(tmp_option_payoffs)
        
        return option_price
        
    def optimal_hedge_setup(self, T):
         self.optimal_hedge = Optimal_hedge(T, self)
    
    def get_optimal_hs(self, time, spot):
        time_idx = int(np.round(time / self.dt , 6))
        hs = []
        for s in spot:
            tmp_bool = abs(self.optimal_hedge.St[:,time_idx] - s) <= 1e-6
            spot_place = np.argmax(tmp_bool)
            if np.count_nonzero(tmp_bool) != 1:
                print("FUCK", time, s, time_idx)
            hs.append(self.optimal_hedge.hs[spot_place, time_idx])
            
        return np.array(hs)[:,np.newaxis]
    
    def get_current_optimal_hs(self, *args):
        return self.get_optimal_hs(self.time, self.spot)
    
class Optimal_hedge():
    def __init__(self, T, bin_model):
        self.n = int(T / bin_model.dt)
        self.T = T
        self.bin_model = bin_model
    
        self.u = np.exp(bin_model.alpha * bin_model.dt + bin_model.sigma * np.sqrt(bin_model.dt))
        self.d = np.exp(bin_model.alpha * bin_model.dt - bin_model.sigma * np.sqrt(bin_model.dt))
        
    def calculate_hedge_price(self, payoffs):
        self.payoffs = payoffs
        self.hs = np.zeros((self.n+1,self.n+1))
        self.hb = np.zeros((self.n+1,self.n+1))
        self.price = np.zeros((self.n+1,self.n+1)) -1e6
        
        self.price[:,-1] = self.payoffs
        
        self.St = np.zeros((self.n+1,self.n+1))
        self.Bt = np.zeros(self.n+1)
        
        for i in range(self.n,-1,-1):
            self.Bt[i] = np.exp(self.bin_model.rate0 * (i/self.n * self.T))
            for j in range(i+1):
                self.St[j,i] = self.bin_model.S0 * self.u ** (i-j) * self.d ** (j)
                
                if i < self.n:
                    #print(i,j,self.Bt[i],self.Bt[i+1])
                    self.hs[j,i] = (self.price[j,i+1] - self.price[j+1,i+1]) / (self.St[j,i] * (self.u - self.d))
                    self.hb[j,i] = (self.price[j+1,i+1]*self.u - self.price[j,i+1]*self.d) \
                        / ((self.u - self.d) * self.Bt[i+1])
            
                    self.price[j,i] = self.hs[j,i] * self.St[j,i] + self.hb[j,i] * self.Bt[i]
        
        return self.price[0,0]
    

if __name__ == "__main__":
    model = Binom_model(1, 0.055, 0.2, 0.035, 0.6, 1/10, n= 10)
    model.optimal_hedge_setup(1)
    
    option = lambda spot: max([spot - 1.1,0])
    tmp_option_payoffs = [option(1*model.optimal_hedge.u**(model.optimal_hedge.n-i) * model.optimal_hedge.d**(i)) 
                          for i in range(model.optimal_hedge.n+1)]
    model.optimal_hedge.calculate_hedge_price(tmp_option_payoffs)
    
    print(model.get_current_optimal_hs())
    
    port = Portfolio(0,10,model, 0)
    
    port.rebalance(np.ones(10) * 0.5)
    port.update_pf_value()
    
    for i in range(10):
        model.evolve_s_b()
        print(model.get_current_optimal_hs())
        port.update_pf_value()
        port.rebalance(np.ones(10) * 0.5)

