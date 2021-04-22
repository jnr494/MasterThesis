# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 11:37:39 2021

@author: mrgna
"""

import matplotlib.pyplot as plt
import numpy as np

import VAE
import MarketGeneratorHelpFunctions


class MarketGenerator():
    def __init__(self, s_model, n):
        self.s_model = s_model
        self.n = n
        self.train_history = []
        
    def create_vae(self, latent_dim = 10, layers_units = [20,10], alpha = 0.2, beta = 0):
        self.encoder = VAE.create_encoder(self.n, layers_units, latent_dim = latent_dim)
        self.decoder = VAE.create_decoder(self.n, latent_dim = latent_dim, layers_units = layers_units[::-1], final_activation = None)
        self.vae = VAE.VAE(self.encoder, self.decoder, alpha = alpha, beta = beta)
        
        #compile model
        VAE.compile_vae(self.vae)
    
    def create_training_path(self, N, overlap = False, seed = None, cheat = False):
        self.log_returns = MarketGeneratorHelpFunctions.generate_data_for_MG(self.s_model, self.n, N, overlap = overlap, 
                                                                             seed = seed, cheat = cheat)
        self.log_returns_norm, self.scaler = MarketGeneratorHelpFunctions.transform_data(self.log_returns, minmax = False)
        self.training_paths = MarketGeneratorHelpFunctions.convert_log_returns_to_paths(self.s_model.S0, self.log_returns)

    def train_vae(self, epochs = 2500, batch_size = 128, lrs = [0.01,0.001,0.0001,0.00001], verbose =2 , 
                  best_model_name = "best_vae_model.hdf5"):
        for lr in lrs:
            tmp_history = VAE.train_vae(self.vae, self.log_returns_norm, epochs = epochs, batch_size = batch_size, 
                                    learning_rate = lr, patience = [epochs,epochs], verbose = verbose,
                                    best_model_name=best_model_name)
            self.train_history.append(tmp_history)
    
    def generate_paths(self, n_samples, seed = None, save = True, return_returns = False):
        sample_vae = MarketGeneratorHelpFunctions.sample_from_vae(self.decoder, n_samples, seed = seed)
        log_return_vae = self.scaler.inverse_transform(sample_vae)       
        
        if return_returns is True:
            return log_return_vae
        
        generated_paths = MarketGeneratorHelpFunctions.convert_log_returns_to_paths(self.s_model.S0, log_return_vae)
        
        if save is True:
            self.generated_paths = generated_paths
        
        return generated_paths
    
    def generate_longer_paths(self, n_samples, length, seed = None, save = True):
        n_segments = int(np.ceil(length / self.n))
        
        seeds = [None] * n_segments if seed is None else seed + np.arange(n_segments)
        returns = [self.generate_paths(n_samples, seed = seed, return_returns = True) for seed in seeds]
        returns = np.hstack(returns)
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
        
    def plot_generated_paths(self, n_samples, length = None):
        if length is None:
            paths = self.generate_paths(100, save = False)
        else:
            paths = self.generate_longer_paths(n_samples, length, save = False)
        plt.plot(paths.T)
        plt.show()