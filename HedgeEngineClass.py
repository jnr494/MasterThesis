# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 15:20:25 2021

@author: mrgna
"""

import numpy as np
import PortfolioClass

class HedgeEngineClass():
    def __init__(self, n, s_model, models, option_por):
        self.s_model = s_model
        self.models = models
        self.option_por = option_por
        
        self.n_assets = self.s_model.n_assets
        self.n = n
        
    def run_hedge_experiment(self, N_samples, init_pf, transaction_cost):    
        self.s_model.reset_model(N_samples)
                
        self.ports = []
        
        #matrices to store investment in underlying
        self.hs_matrix = []
        
        for m in self.models:
            self.ports.append(PortfolioClass.Portfolio(0, init_pf, self.s_model, transaction_cost = transaction_cost))
            self.hs_matrix.append(np.zeros((N_samples, self.n_assets, self.n)))
        
        #init rebalance
        for por, m in zip(self.ports, self.models):
            por.rebalance(m.get_current_optimal_hs(self.s_model, por.hs))
        
        for i in range(self.n):
            #Save hs and time
            for por, hs_m in zip(self.ports, self.hs_matrix):
                hs_m[...,i] = por.hs 
            
            self.s_model.evolve_s_b()
            
            for por in self.ports:
                por.update_pf_value()
            
            if i < self.n - 1:
                for por, m in zip(self.ports, self.models):
                    por.rebalance(m.get_current_optimal_hs(self.s_model, por.hs)) 
        
        self.pf_values = [por.pf_value for por in self.ports]
        
        self.hedge_spots = self.s_model.spot
        self.spot_hist = self.s_model.spot_hist
        self.option_values = self.option_por.get_portfolio_payoff(self.spot_hist)
        
        self.Pnl = [np.array(pf_val) - np.array(self.option_values) for pf_val in self.pf_values]
        self.Pnl_disc = np.array(self.Pnl) / self.s_model.bank