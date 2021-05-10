# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 12:20:28 2021

@author: mrgna
"""

import numpy as np
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tf.config.set_visible_devices([], 'GPU')

####### Comparisson of paths
class PathComparator():
    def __init__(self,paths1,paths2, T, names = ["paths1","paths2"]):
        self.n = paths1.shape[1] - 1
        self.T = T
        
        self.paths = [paths1,paths2]
        self.returns = [p[:,1:] / p[:,:-1] - 1 for p in self.paths]
        self.times = np.linspace(0,self.T,self.n+1)
        self.names = names
        
    def run_comparrison(self, plot = True):   
        #Compare mean and std of paths
        
        self.means = [np.mean(p, 0) for p in self.paths]
        self.stds = [np.std(p, 0) for p in self.paths]
        
        if plot is True:
            [plt.plot(self.times,m) for m in self.means]
            plt.title(("Plot of means"))
            plt.show() 
            
            [plt.plot(self.times,s) for s in self.stds]
            plt.show() 
        
        
        ### QQ plot and ECDF 
        self.idxs = [1,-1]
        self.ecdf_saved = []
        self.ecdf_ranges = []
        
        for idx in self.idxs: 
            # QQ Plot
            
            percentiles = np.linspace(0.01,0.99,99)
            self.quantiles = [np.quantile(p[:,idx], percentiles) for p in self.paths]
            
            #qq plot with theoretical quantiles of simulated bs paths   
            if plot is True:
                plt.scatter(self.quantiles[0], self.quantiles[1], s = 5)
                plt.plot(self.quantiles[0][[0,-1]],self.quantiles[0][[0,-1]],c='k')
                plt.show()
            
            # ECDF
            ecdfs = [ECDF(p[:,idx]) for p in self.paths]
        
            range1 = np.linspace(np.min(self.paths[0][:,idx]),np.max(self.paths[0][:,idx]),1000)
            
            tmp_ecdf_dist = np.abs(ecdfs[0](range1) - ecdfs[1](range1))
            max_dif_x = np.argmax(tmp_ecdf_dist)
            max_dif = np.max(np.abs(tmp_ecdf_dist))
            
            if plot is True:
                plt.plot(range1, ecdfs[0](range1))
                plt.plot(range1, ecdfs[1](range1))
                #plt.plot([range1[max_dif_x]]*2,[0,1])
                plt.show()
            
            self.ecdf_saved.append([ecdf(range1) for ecdf in ecdfs])
            self.ecdf_ranges.append(range1)
            
        #### Kolmogorov-Smirnov across time
        
        self.pvalues = [stats.ks_2samp(self.paths[0][:,i], self.paths[1][:,i], mode = 'exact')[1] for i in range(1,self.n+1)]
        
        if plot is True:
            plt.plot(self.times[1:], self.pvalues)
            plt.show()
                
        #### corr comparrison
        
        def calculate_cor_lag(x,lag):
            lead_x = x[:,lag:]
            lag_x = x[:,:-lag]
            cov_x = tfp.stats.correlation(lead_x,lag_x, 0, event_axis = None)
            return cov_x.numpy()
        
        self.lags = range(1,self.n)
        self.abs_corrs = [[np.mean(calculate_cor_lag(np.abs(lr), l)) for l in self.lags] for lr in self.returns]
        
        if plot is True:
            plt.plot(self.lags,self.abs_corrs[0])
            plt.plot(self.lags,self.abs_corrs[1])
            plt.show()
        
        self.corrs = [[np.mean(calculate_cor_lag(lr, l)) for l in self.lags] for lr in self.returns]
        
        if plot is True:
            plt.plot(self.lags,self.corrs[0])
            plt.plot(self.lags,self.corrs[1])
            plt.show()

if __name__ == '__main__':
    import BlackScholesModel
    import HestonModel
    import MarketGenerator
    
    BS = False
    
    real_n = 20
    n = 20
    real_T = 1/12
    dt = real_T / real_n
    cond_n = 0
    
    if BS:
        ########## Black shocles
        n = n
        T = dt * n
        
        s0 = 1
        mu = 0.05
        sigma1 = 0.3
        sigma2 = 0.3
        rate = 0.01
        
        corr = np.array([1])
        
        model1 = BlackScholesModel.BlackScholesModel(s0, mu, sigma1, np.ones((1,1)), rate, dt)
        model2 = BlackScholesModel.BlackScholesModel(s0, mu, sigma2, np.ones((1,1)), rate, dt)
        
        n_paths1 = 1000
        n_paths2 = 1000
        
        model1.reset_model(n_paths1)
        model2.reset_model(n_paths2)
        
        for _ in range(n):
            model1.evolve_s_b()
            model2.evolve_s_b()
            
        paths1 = model1.spot_hist[:,0,:]
        paths2 = model2.spot_hist[:,0,:]
        
        PathComparator(paths1,paths2,T).run_comparrison()
     
    ###### Heston 
    else:
        n = n
        T = dt * n
        
        s0 = 1
        mu = 0.05
        v0 = 0.1
        kappa = 10
        theta = 0.1
        sigma = 1
        rho = -0.9
        rate = 0.01
        dt = T/n
        ddt = 0.01
        
        model1 = HestonModel.HestonModel(s0,mu,v0,kappa,theta,sigma,rho,rate,dt,ddt)
        model2 = HestonModel.HestonModel(s0,mu,v0,kappa,theta,sigma,rho,rate,dt,ddt)
        
        model1.use_v = True
        model2.use_v = True
        
        n_paths1 = 1000
        n_paths2 = 1000
        
        model1.reset_model(n_paths1)
        model2.reset_model(n_paths2)
        
        for _ in range(n):
            model1.evolve_s_b()
            model2.evolve_s_b()
            
        paths1 = model1.spot_hist[:,0,:]
        paths2 = model2.spot_hist[:,0,:]
        
        PathComparator(paths1,paths2,T).run_comparrison()

     ####### MG
    n_samples = 1000
    real_decoder = True
    
    MG = MarketGenerator.MarketGenerator(model1, n, cond_n)
    MG.create_vae(latent_dim = n, layers_units=[40], alpha = 0.95, beta = 100, real_decoder= real_decoder)
    MG.create_training_path(n_samples, overlap = True, seed = None, cheat = True)
    MG.train_vae(epochs = 500, batch_size = 128, lrs = [0.01,0.0001,0.00001])
    
    if BS is False:
        plt.plot(model1.spot_hist[:,1,:].T)
        plt.show()
    
    training_paths = MG.training_paths
    
    #get cond
    cond = MG.cond
    
    k = 20000 // n_samples
    tmp_cond = np.repeat(cond,k,axis = 0) if cond is not None else None 
    mg_paths = MG.generate_paths(n_samples*k, cond = tmp_cond, save = False, 
                                 std = 1, mean = 0, real_decoder = real_decoder)
    
    MG.plot_generated_paths(100, cond[:100,:] if cond_n > 0 else None)
    plt.plot(training_paths[:100,:].T)
    plt.show()

    
    pc = PathComparator(training_paths,mg_paths,T)
    pc.run_comparrison()
    
    
    ###### Test on longer 
    if real_n > n:
        n_long_samples = 20000
        if cond_n == 0:
            cond = None
        else:    
            if BS:
                    model1.reset_model(1)
                    for i in range(cond_n):
                        model1.evolve_s_b()
                    spots = model1.spot_hist[:,0,:]
                    cond = spots[:,1:] / spots[:,:-1] - 1
                    cond = np.repeat(cond,n_long_samples,axis = 0)
    
            else:
                if cond_n > 0:
                    new_v0 = v0
                    model1.reset_model(1000)
                    for i in range(100):
                        model1.evolve_s_b()
                    idx = np.argmin(np.abs(model1.v - new_v0)) #v0
                    spots = model1.spot_hist[:,0,:]
                    cond = spots[idx:idx+1,1:] / spots[idx:idx+1,:-1] - 1
                    cond = cond[:,-cond_n:]
                    cond = np.repeat(cond,n_long_samples,axis = 0)
                                
                    model1.v0 = new_v0
                else:
                    cond = None

                
        long_mg_paths = MG.generate_longer_paths(n_long_samples, length = real_n, cond = cond, save = False, 
                                                 std = 1, mean = 0, real_decoder = real_decoder)
        
        MG.plot_generated_paths(100, cond[:100,:] if cond_n > 0 else None, real_n)
        
        model1.reset_model(20000)
        for i in range(real_n):
            model1.evolve_s_b()
        long_model_paths = model1.spot_hist[:,0,:]
        
        pc_long = PathComparator(long_model_paths,long_mg_paths,T)
        pc_long.run_comparrison()
        
        if BS is False:
            model1.v0 = v0



