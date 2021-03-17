# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:09:16 2021

@author: mrgna
"""

import numpy as np

def generate_dataset(s_model, n_steps, n_samples, option_por, new = True):
    if new is True:
        s_model.reset_model(n_samples)
        for j in range(n_steps):
            s_model.evolve_s_b()
    
    spots = s_model.spot_hist
    banks = s_model.bank_hist
    rates = s_model.rate_hist
    
    spots_tilde = spots / banks[:,np.newaxis,:]
    
    #extra infomation (for pathdependence)
    min_spots = np.minimum.accumulate(spots,axis = -1)
    
    #Get option payoffs from samples
    option_payoffs = option_por.get_portfolio_payoff(spots)[:,np.newaxis]
    
    #Setup x and y
    x = [spots_tilde[...,i]  for i in range(n_steps+1)] \
        + [rates[:,i:(i+1)]  for i in range(n_steps)] \
        + [min_spots[...,i] for i in range(n_steps)] \
        + [banks[:,-1:],option_payoffs]
        
    x = np.column_stack(x)
        
    y = np.zeros(shape = (n_samples,1))
    
    return x,y, banks

def binary_search(func,a,b, delta = 1e-6):
    a = a
    b = b
    tmp_delta = b - a

    while tmp_delta > delta:
        m = (b-a) / 2 + a
        f_m = func(m)
        
        #print(a,b,m,f_m)
        
        if f_m < 0:
            b = m
        else:
            a = m
        
        tmp_delta= b - a
        
    return (b-a)/2 + a