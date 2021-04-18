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
    
    def create_training_path(self, N, overlap = False, seed = None):
        self.log_returns = MarketGeneratorHelpFunctions.generate_data_for_MG(self.s_model, self.n, N, overlap = overlap, seed = seed)
        self.log_returns_norm, self.scaler = MarketGeneratorHelpFunctions.transform_data(self.log_returns, minmax = False)
        self.training_paths = MarketGeneratorHelpFunctions.convert_log_returns_to_paths(self.s_model.S0, self.log_returns)

    def train_vae(self, epochs = 2500, batch_size = 128, lrs = [0.01,0.001,0.0001,0.00001], verbose =2 , 
                  best_model_name = "best_vae_model.hdf5"):
        for lr in lrs:
            tmp_history = VAE.train_vae(self.vae, self.log_returns_norm, epochs = epochs, batch_size = batch_size, 
                                    learning_rate = lr, patience = [epochs,epochs], verbose = verbose,
                                    best_model_name=best_model_name)
            self.train_history.append(tmp_history)
    
    def generate_paths(self, n_samples, seed = None, save = True):
        sample_vae = MarketGeneratorHelpFunctions.sample_from_vae(self.decoder, n_samples, seed = seed)
        log_return_vae = self.scaler.inverse_transform(sample_vae)
        
        generated_paths = MarketGeneratorHelpFunctions.convert_log_returns_to_paths(self.s_model.S0, log_return_vae)
        
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
        
    def plot_generated_paths(self, n_samples):
        paths = self.generate_paths(100, False)
        plt.plot(paths.T)
        plt.show()