# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 15:20:25 2021

@author: mrgna
"""

import numpy as np
import PortfolioClass
import StandardNNModel
import helper_functions

class HedgeEngineClass():
    def __init__(self, n, s_model, models, option_por):
        self.s_model = s_model
        self.models = models
        self.option_por = option_por
        
        self.n_assets = self.s_model.n_assets
        self.n = n
        
        self.quick_models_idx = [type(model) == StandardNNModel.NN_simple_hedge for model in models]
        self.quick_models = [models[i] for i in range(len(models)) if self.quick_models_idx[i]]
        self.slow_models = [models[i] for i in range(len(models)) if not self.quick_models_idx[i]]
        
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
            por.rebalance(m.get_current_optimal_hs(self.s_model, por.hs, por.pf_value[:,np.newaxis]))
        
        for i in range(self.n):
            #Save hs and time
            for por, hs_m in zip(self.ports, self.hs_matrix):
                hs_m[...,i] = por.hs 
            
            self.s_model.evolve_s_b()
            
            for por in self.ports:
                por.update_pf_value()
            
            if i < self.n - 1:
                for por, m in zip(self.ports, self.models):
                    por.rebalance(m.get_current_optimal_hs(self.s_model, por.hs, por.pf_value[:,np.newaxis])) 
        
        self.pf_values = [por.pf_value for por in self.ports]
        
        self.hedge_spots = self.s_model.spot
        self.spot_hist = self.s_model.spot_hist
        self.option_values = self.option_por.get_portfolio_payoff(self.spot_hist)
        
        self.Pnl = [np.array(pf_val) - np.array(self.option_values) for pf_val in self.pf_values]
        self.Pnl_disc = np.array(self.Pnl) / self.s_model.bank
    
    def run_quick_hedge_exp(self, N_samples, init_pf, transaction_cost):
        self.s_model.reset_model(N_samples)
                
        self.ports = []
        
        #matrices to store investment in underlying
        self.hs_matrix = []
        
        for m in self.slow_models:
            self.ports.append(PortfolioClass.Portfolio(0, init_pf, self.s_model, transaction_cost = transaction_cost))
            self.hs_matrix.append(np.zeros((N_samples, self.n_assets, self.n)))
        
        #init rebalance
        for por, m in zip(self.ports, self.slow_models):
            por.rebalance(m.get_current_optimal_hs(self.s_model, por.hs, por.pf_value[:,np.newaxis]))
        
        for i in range(self.n):
            #Save hs and time
            for por, hs_m in zip(self.ports, self.hs_matrix):
                hs_m[...,i] = por.hs 
            
            self.s_model.evolve_s_b()
            
            for por in self.ports:
                por.update_pf_value()
            
            if i < self.n - 1:
                for por, m in zip(self.ports, self.slow_models):
                    por.rebalance(m.get_current_optimal_hs(self.s_model, por.hs, por.pf_value[:,np.newaxis])) 
        
        self.pf_values_slow = [por.pf_value for por in self.ports]
        
        ######get quick pf_values
        #setup data
        x, y, banks = helper_functions.generate_dataset(self.s_model, n_steps = self.n, n_samples = N_samples, 
                                                        option_por = self.option_por, new = False)
        
        #get quick pf values
        self.PnL_quick = [np.squeeze(model.model.predict(x)) + (init_pf - model.get_init_pf()) * self.s_model.bank
                                for model in self.quick_models]
                            
        self.hedge_spots = self.s_model.spot
        self.spot_hist = self.s_model.spot_hist
        self.option_values = self.option_por.get_portfolio_payoff(self.spot_hist)
        
        self.Pnl_slow = [np.array(pf_val) - np.array(self.option_values) for pf_val in self.pf_values_slow]
        
        #put together PnL quick and slow
        self.Pnl = []
        
        count_slow = 0
        count_quick = 0
        
        for i in range(len(self.models)):
            if self.quick_models_idx[i] is True:
                self.Pnl.append(self.PnL_quick[count_quick])
                count_quick += 1
            else:
                self.Pnl.append(self.Pnl_slow[count_slow])
                count_slow += 1
        
        self.Pnl_disc = np.array(self.Pnl) / self.s_model.bank