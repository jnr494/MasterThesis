# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 16:12:27 2021

@author: mrgna
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import BlackScholesModel


def generate_data_for_MG(model, n, N, plot = True, overlap = False):
    print(overlap)
    model.reset_model(1)
    if overlap is True:
        for _ in range(n+N-1):
            model.evolve_s_b()
    else:
        for _ in range(n*N):
            model.evolve_s_b()
        
    spots = model.spot_hist[0,0,:]
    plt.plot(spots)
    plt.show()
        
    log_returns = np.log(spots[1:] / spots[:-1])
    
    if overlap is True:
        log_returns_data = np.zeros((N,n))
        for i in range(N):
            log_returns_data[i,:] = log_returns[i:(i+n)]
    else:
        log_returns_data = log_returns.reshape((N,n))
    
    return log_returns_data

def transform_data(data, minmax = True):
    if minmax is True:
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    scaler.fit(data)
    
    data_norm = scaler.transform(data)
    return data_norm, scaler
    
def sample_from_vae(decoder, n_samples, s0 = 1):
    latent_dim = decoder.layers[0].input_shape[0][1]
    latent_sample = np.random.normal(size = (n_samples, latent_dim))
    latent_sample_decoded = decoder.predict(latent_sample)
        
    return latent_sample_decoded
    
def convert_log_returns_to_paths(s0, log_returns):
    returns = np.exp(log_returns)
    cum_returns = np.cumprod(returns, axis = 1)    
    
    paths = s0 * cum_returns
    paths = np.concatenate((s0 * np.ones((len(returns),1)), paths), axis = 1)
    
    return paths
    
    

if __name__ == '__main__':
    n = 20
    N = 100
    T = 1/12
    s0, mu, sigma, rate, dt = (1, 0.03, 0.2, 0.01, T/n)
    model = BlackScholesModel.BlackScholesModel(s0,mu, sigma, np.ones((1,1)), rate, dt)
    
    log_returns = generate_data_for_MG(model, n, N,overlap=True)
    log_returns_norm, scaler = transform_data(log_returns)
    
    
    