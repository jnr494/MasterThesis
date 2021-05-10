# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 11:37:39 2021

@author: mrgna
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import VAE
import MarketGeneratorHelpFunctions


class MarketGenerator():
    def __init__(self, s_model, n, cond_n = 0, cond_type = 0):
        self.s_model = s_model
        self.n = n
        self.cond_n = cond_n
        self.cond_type = cond_type #0: returns, 1: v(t), 2: ? but maybe realized vol
        self.train_history = []
        
        if self.cond_n > 0:
            self.mm2 = VAE.create_moment_model(cond_n,n)
        else:
            self.mm2 = None
        
    def create_vae(self, latent_dim = 10, layers_units = [20,10], alpha = 0.9, beta = 0, gamma = 0, 
                   real_decoder = False):
        
        self.encoder = VAE.create_encoder(self.n, layers_units, latent_dim = latent_dim, cond_dim = self.cond_n)
        self.decoder = VAE.create_decoder(self.n, latent_dim = latent_dim, layers_units = layers_units[::-1], 
                                          final_activation = None, cond_dim = self.cond_n)
        
        if real_decoder is True:
            self.real_decoder = VAE.create_real_decoder(self.decoder, (1-alpha)/alpha)
        else:
            self.real_decoder = self.decoder
        
        self.vae = VAE.VAE(self.encoder, self.decoder, self.real_decoder, alpha = alpha, 
                           beta = beta, gamma = gamma, cond_type = self.cond_type)
        
        #compile model
        VAE.compile_vae(self.vae)
    
    def create_training_path(self, N, overlap = False, seed = None, cheat = False):
        self.log_returns, self.cond = MarketGeneratorHelpFunctions.generate_data_for_MG(self.s_model, self.n, N, 
                                                                                        overlap = overlap, seed = seed, 
                                                                                        cheat = cheat, cond_n = self.cond_n,
                                                                                        cond_type = self.cond_type)
        self.training_paths = MarketGeneratorHelpFunctions.convert_log_returns_to_paths(self.s_model.S0, self.log_returns)
        
        #self.PCA = PCA(self.n)
        #self.PCA.fit(self.training_paths)
        #self.log_returns = self.PCA.transform(self.training_paths)
        
        self.log_returns_norm, self.scaler = MarketGeneratorHelpFunctions.transform_data(self.log_returns, minmax = False)
        
        if self.cond_n > 0:
            self.cond_norm, self.cond_scaler = MarketGeneratorHelpFunctions.transform_data(self.cond, minmax = False)        

    def train_vae(self, epochs = 2500, batch_size = 32, lrs = [0.001,0.001,0.0001,0.0001], verbose =2 , 
                  best_model_name = "best_vae_model.hdf5"):
        data_y = self.log_returns_norm
        if self.cond_n == 0: 
            data_x = None
        else:
            data_x = self.cond_norm
        
        for lr in lrs:
            tmp_history = VAE.train_vae(self.vae, data_y, data_x, epochs = epochs, batch_size = batch_size, 
                                    learning_rate = lr, patience = [epochs,epochs], verbose = verbose,
                                    best_model_name=best_model_name)
            self.train_history.append(tmp_history)
    
    def generate_paths(self, n_samples, cond = None, seed = None, save = True, return_returns = False, std = 1, mean = 0, real_decoder = False):
        cond_norm = self.cond_scaler.transform(cond) if not cond is None else None
        sample_vae = MarketGeneratorHelpFunctions.sample_from_vae(self.vae, n_samples, cond = cond_norm, 
                                                                  seed = seed, std = std, mean = mean, real_decoder = real_decoder)
        log_return_vae = self.scaler.inverse_transform(sample_vae)       
        
        if return_returns is True:
            return log_return_vae
        
        generated_paths = MarketGeneratorHelpFunctions.convert_log_returns_to_paths(self.s_model.S0, log_return_vae)
        #generated_paths = self.PCA.inverse_transform(log_return_vae)
        
        if save is True:
            self.generated_paths = generated_paths
        
        return generated_paths
    
    #does not work for cond_type 1
    def generate_longer_paths(self, n_samples, length, cond = None, seed = None, save = True, std = 1, mean = 0, real_decoder = False):
        n_segments = int(np.ceil(length / self.n))
        
        
        seeds = [None] * n_segments if seed is None else seed + np.arange(n_segments)
        
        #first run
        returns = self.generate_paths(n_samples, cond = cond, seed = seeds[0], return_returns = True, 
                                      std = std, mean = mean, real_decoder = real_decoder)
        
        for i in range(1,n_segments):
            if cond is None:
                tmp_cond = None
            else:
                tmp_cond = np.hstack((cond, returns))[:,-cond.shape[1]:]
            
            new_returns = self.generate_paths(n_samples, cond = tmp_cond, seed = seeds[i], return_returns = True, std = std, mean = mean)
            returns = np.hstack((returns,new_returns))

        returns = returns[:,:length]
        
        generated_paths = MarketGeneratorHelpFunctions.convert_log_returns_to_paths(self.s_model.S0, returns)
        
        if save is True:
            self.generated_paths = generated_paths
        return generated_paths
        
    
    def qq_plot_fit(self):
        N = len(self.training_paths)
        n_samples = 100000
        sampled_paths = self.generate_paths(n_samples)
        
        percentiles = np.linspace(0.01,0.99,N)
        i = self.n
        actual_quantiles = np.quantile(self.training_paths[:,i], percentiles)
        sampled_quantiles = np.quantile(sampled_paths[:,i], percentiles)
        
        #qq plot with theoretical quantiles of simulated bs paths    
        plt.scatter(actual_quantiles, sampled_quantiles, s = 5)
        plt.plot(actual_quantiles[[0,-1]],actual_quantiles[[0,-1]],c='k')
        plt.show()
        
    def plot_generated_paths(self, n_samples, cond = None, length = None):
        if length is None:
            paths = self.generate_paths(n_samples, cond = cond, save = False)
        else:
            paths = self.generate_longer_paths(n_samples, length, cond = cond, save = False)
        plt.plot(paths.T)
        plt.show()