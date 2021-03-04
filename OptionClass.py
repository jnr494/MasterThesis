# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:06:55 2021

@author: Magnus Frandsen
"""

import numpy as np

def call_option_payoff(spot,strike):
    return np.maximum(spot - strike, 0)

def put_option_payoff(spot, strike):
    return np.maximum(strike - spot, 0)

class Option():
    def __init__(self, name, params, underlying):
        self.name = name
        self.params = params
        self.underlying = underlying
        
        if self.name == "call":
            self.payoff = lambda spot: call_option_payoff(spot, self.params[0])
        
        if self.name == "put":
            self.payoff = lambda spot: put_option_payoff(spot, self.params[0])
        
class OptionPortfolio():
    def __init__(self, options, units):
        self.options = options
        self.units = units
    
    def get_portfolio_payoff(self,spots):
        payoff = 0
        for option, units in zip(self.options, self.units):
            tmp_underlying = option.underlying
            payoff += option.payoff(spots[:,tmp_underlying]) * units
        
        return payoff