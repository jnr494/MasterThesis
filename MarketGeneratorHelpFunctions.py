# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 16:12:27 2021

@author: mrgna
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import iisignature

import BlackScholesModel


def generate_data_for_MG(model, n, N, plot = True, overlap = False, seed = None, cheat = False, cond_n = 0, cond_type = 0):
    if cheat is True:
        return generate_data_for_MG_cheat(model, n, N, plot, overlap, seed)
    
    real_cond_n = cond_n
    
    if cond_type == 1:
        cond_n = 0
        
    print(overlap)
    np.random.seed(seed)
    model.reset_model(1)
    if overlap is True:
        for _ in range(cond_n + n + N-1):
            model.evolve_s_b()
    else:
        for _ in range((cond_n + n) * N):
            model.evolve_s_b()
        
    spots = model.spot_hist[0,0,:]
    plt.plot(spots)
    plt.show()
        
    returns = spots[1:] / spots[:-1] - 1
    
    if overlap is True:
        all_returns_data = np.zeros((N,cond_n+n))
        for i in range(N):
            all_returns_data[i,:] = returns[i:(i+cond_n+n)]
    else:
        all_returns_data = returns.reshape((N,cond_n+n))
    
    returns_data = all_returns_data[:,cond_n:]
    
    if real_cond_n > 0:
        if cond_type == 0:
            cond_data = all_returns_data[:,:cond_n]
        elif cond_type == 1:
            vols = model.spot_hist[0,1,:]
            if overlap is True:
                cond_data = vols[N:][:,np.newaxis]
            else:
                cond_data = vols[:-1].reshape((N,n))[:,:1]
    else:
        cond_data = None
        
    return returns_data, cond_data

def generate_data_for_MG_cheat(model, n, N, plot = True, overlap = False, seed = None):
    print(overlap)
    np.random.seed(seed)
    model.reset_model(N)
    for _ in range(n):
        model.evolve_s_b()
        
    spots = model.spot_hist[:,0,:]
    plt.plot(spots[:100,:].T)
    plt.show()
        
    returns = spots[:,1:] / spots[:,:-1] - 1
        
    return returns, None

def transform_data(data, minmax = True):
    if minmax is True:
        scaler = MinMaxScaler((-1,1))
    else:
        scaler = StandardScaler()
    
    scaler.fit(data)
    
    data_norm = scaler.transform(data)
    return data_norm, scaler
    
def sample_from_vae(vae, n_samples, s0 = 1, cond = None, seed = None, std = 1, mean = 0, real_decoder = False):
    np.random.seed(seed)
    
    latent_dim = vae.decoder.input_shape[1] - vae.cond_n
    latent_sample = np.random.normal(size = (n_samples, latent_dim)) * std + mean
    
    if not cond is None:
        decoder_input = np.hstack((latent_sample,cond))
    else:
        decoder_input = latent_sample
    
    latent_sample_decoded = vae.decoder.predict(decoder_input)

    if real_decoder is True:
        decoder_normals = np.random.normal(size = latent_sample_decoded.shape)
        alpha = vae.alpha
        c = (1-alpha) / alpha
        latent_sample_decoded += np.sqrt(c) * decoder_normals
    
    return latent_sample_decoded
    
def convert_log_returns_to_paths(s0, log_returns):
    #returns = np.exp(log_returns)
    returns = log_returns + 1
    cum_returns = np.cumprod(returns, axis = 1)    
    
    paths = s0 * cum_returns
    paths = np.concatenate((s0 * np.ones((len(returns),1)), paths), axis = 1)
    
    return paths
    
###Signatures
def leadlag(path):
    repeat = np.repeat(path,2)
    lead = repeat[1:]
    lag = repeat[:-1]
    
    leadlag = np.vstack((lead,lag)).T
    return leadlag

class LogSignature():
    def __init__(self, order):
        d = 2
        self.prep = iisignature.prepare(d, order)
        self.logsig_size = iisignature.logsiglength(d, order)
    def __call__(self, paths):
        path_dims = paths.ndim
        if path_dims == 1:
            ll_path = leadlag(paths)
            return iisignature.logsig(ll_path, self.prep)
        if path_dims == 2:
            logsigs = np.zeros((len(paths),self.logsig_size))
            for idx, path in enumerate(paths):
                tmp_ll_path = leadlag(path)
                logsigs[idx,:] = iisignature.logsig(tmp_ll_path, self.prep)
            return logsigs
        else:
            print("Some mistake with dimensions of paths. Expected 1 og 2, but got", path_dims)
            return None

if __name__ == '__main__':
    n = 20
    N = 100
    T = 1/12
    s0, mu, sigma, rate, dt = (1, 0.03, 0.2, 0.01, T/n)
    model = BlackScholesModel.BlackScholesModel(s0,mu, sigma, np.ones((1,1)), rate, dt)
    
    log_returns = generate_data_for_MG(model, n, N,overlap=True)
    log_returns_norm, scaler = transform_data(log_returns)
    
    
    