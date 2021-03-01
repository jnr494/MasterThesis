# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:36:13 2021

@author: Magnus Frandsen
"""

import numpy as np

class Portfolio():
    def __init__(self, init_hs, init_value, model, transaction_cost = 0):
        self.model = model
        
        self.n = self.model.n
        
        self.pf_value = np.ones(self.n) * init_value
        
        self.hs_hist = np.zeros((self.n,1))
        self.hb_hist = np.zeros((self.n,1))
        self.pf_value_hist = np.zeros((self.n,1))
        
        self.transaction_cost = transaction_cost
        self.tc_hist = np.zeros((self.n,1))
        
        self.hs = np.ones(self.n) * init_hs
        self.rebalance(self.hs)
        self.update_pf_value()
        
        #Remove zeros from hs_hist, hb_hist and pf_hist
        self.hs_hist = self.hs_hist[:,1:]
        self.hb_hist = self.hb_hist[:,1:]
        self.pf_value_hist = self.pf_value_hist[:,1:]
        self.tc_hist = self.tc_hist[:,1:]
        
    def rebalance(self, new_hs):
        hs_change = abs(self.hs - new_hs)
        tmp_tc = hs_change * self.model.spot * self.transaction_cost
        self.tc_hist = np.append(self.tc_hist, tmp_tc[:,np.newaxis], 1)
        
        self.hs = new_hs
        self.hb = (self.pf_value - self.hs * self.model.spot - tmp_tc) / self.model.bank
        
        
        self.hs_hist = np.append(self.hs_hist, self.hs[:,np.newaxis], 1)
        self.hb_hist = np.append(self.hb_hist, self.hb[:,np.newaxis], 1)
        
    def update_pf_value(self):
        self.pf_value = self.hs * self.model.spot + self.hb * self.model.bank
        
        self.pf_value_hist = np.append(self.pf_value_hist, self.pf_value[:,np.newaxis], 1)