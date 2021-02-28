# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 15:15:36 2021

@author: Magnus Frandsen
"""

import numpy as np
import copy

class Binom_model():
   def __init__(self,S0, alpha, sigma, rate, p, dt, rate_change = 0.):
       self.S0 = S0
       self.alpha = alpha
       self.sigma = sigma
       
       self.rate0 = rate
       self.rate_change = rate_change
       
       self.p = p  
       self.dt = dt
      
       self.reset_model()
       
   def evolve_s_b(self):
       #Evolve bank
       self.bank *= np.exp(self.rate * self.dt)
       
       #Evovle s 
       sign = np.random.choice([1,-1], p = [self.p, 1-self.p])
       self.spot *= np.exp(self.alpha * self.dt + sign * self.sigma * np.sqrt(self.dt))
       
       if sign == 1:
           self.no_u += 1
       else:
           self.no_d += 1
       
       #Evovle rate
       self.rate += self.rate_change * (sign)
       
       rho = 0
       self.rate = rho * self.rate + (1 - rho) * self.rate0 + self.rate_change * np.random.normal()
       
       #Evovle time
       self.time += self.dt
       
       #Save bank and spot
       self.spot_hist.append(copy.deepcopy(self.spot))
       self.bank_hist.append(copy.deepcopy(self.bank))
       self.rate_hist.append(copy.deepcopy(self.rate))
       
       return self.spot, self.bank, self.time

   def reset_model(self):
        self.time = 0
        self.spot = self.S0
        self.bank = 1
        self.rate = self.rate0
        
        self.spot_hist = [copy.deepcopy(self.spot)]
        self.bank_hist = [copy.deepcopy(self.bank)]
        self.rate_hist = [copy.deepcopy(self.rate0)]
        
        self.no_d = 0
        self.no_u = 0

        
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
            self.Bt[i] = np.exp(self.bin_model.rate * (i/self.n * self.T))
            for j in range(i+1):
                self.St[j,i] = self.bin_model.S0 * self.u ** (i-j) * self.d ** (j)
                
                if i < self.n:
                    #print(i,j,self.Bt[i],self.Bt[i+1])
                    self.hs[j,i] = (self.price[j,i+1] - self.price[j+1,i+1]) / (self.St[j,i] * (self.u - self.d))
                    self.hb[j,i] = (self.price[j+1,i+1]*self.u - self.price[j,i+1]*self.d) \
                        / ((self.u - self.d) * self.Bt[i+1])
            
                    self.price[j,i] = self.hs[j,i] * self.St[j,i] + self.hb[j,i] * self.Bt[i]             
    
class Portfolio():
    def __init__(self, init_hs, init_value, bin_model, transaction_cost = 0):
        self.pf_value = init_value
        self.bin_model = bin_model
        
        self.hs_hist = []
        self.hb_hist = []
        self.pf_value_hist = []
        
        self.transaction_cost = transaction_cost
        self.tc_hist = []
        
        self.hs = init_hs
        self.rebalance(init_hs)
        self.update_pf_value()
        
        
    def rebalance(self, new_hs):
        hs_change = abs(self.hs - new_hs)
        tmp_tc = hs_change * self.bin_model.spot * self.transaction_cost
        self.tc_hist.append(tmp_tc)
        
        self.hs = new_hs
        self.hb = (self.pf_value - self.hs * self.bin_model.spot - tmp_tc) / self.bin_model.bank
    
        self.hs_hist.append(self.hs)
        self.hb_hist.append(self.hb)
        
    def update_pf_value(self):
        self.pf_value = self.hs * self.bin_model.spot + self.hb * self.bin_model.bank
        
        self.pf_value_hist.append(self.pf_value)
    
    
if __name__ == "__main__":
    model = Binom_model(100, 0.055, 0.2, 0.035, 0.6, 1/10)
    opt_hedge = Optimal_hedge(2, model)
    
    option = lambda spot: max([spot - 105,0])
    option_payoffs = [option(100*opt_hedge.u**(opt_hedge.n-i) * opt_hedge.d**(i)) for i in range(opt_hedge.n+1)]
    opt_hedge.calculate_hedge_price(option_payoffs)
    
    por = Portfolio(opt_hedge.hs[0,0],opt_hedge.price[0,0],model)
    
    for i in range(0,opt_hedge.n):
        model.evolve_s_b()
        por.update_pf_value()
        por.rebalance(opt_hedge.hs[model.no_d, i+1])
    
    print("Option payoff:",option(model.spot))
    print("Por value:",por.pf_value)
    print("Hedge error:", por.pf_value - option(model.spot) )
