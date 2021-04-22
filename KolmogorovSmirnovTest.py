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
        self.log_returns = [np.log(p[:,1:] / p[:,:-1]) for p in self.paths]
        self.times = np.linspace(0,self.T,self.n+1)
        self.names = names
        
    def run_comparrison(self):   
        #Compare mean and std of paths
        
        self.means = [np.mean(p, 0) for p in self.paths]
        print(self.means)
        self.stds = [np.std(p, 0) for p in self.paths]
        
        [plt.plot(self.times,m) for m in self.means]
        plt.title(("Plot of means"))
        plt.show() 
        
        [plt.plot(self.times,s) for s in self.stds]
        plt.show() 
        
        
        ### QQ plot and ECDF 
        idx = -1
        
        # QQ Plot
        
        percentiles = np.linspace(0.01,0.99,99)
        quantiles = [np.quantile(p[:,idx], percentiles) for p in self.paths]
        
        #qq plot with theoretical quantiles of simulated bs paths    
        plt.scatter(quantiles[0], quantiles[1], s = 5)
        plt.plot(quantiles[0][[0,-1]],quantiles[0][[0,-1]],c='k')
        plt.show()
        
        # ECDF
        ecdfs = [ECDF(p[:,idx]) for p in self.paths]
        
        range1 = np.linspace(np.min(self.paths[0][:,idx]),np.max(self.paths[0][:,idx]),1000)
        
        tmp_ecdf_dist = np.abs(ecdfs[0](range1) - ecdfs[1](range1))
        max_dif_x = np.argmax(tmp_ecdf_dist)
        max_dif = np.max(np.abs(tmp_ecdf_dist))
        
        plt.plot(range1, ecdfs[0](range1))
        plt.plot(range1, ecdfs[1](range1))
        #plt.plot([range1[max_dif_x]]*2,[0,1])
        plt.show()
        
        #### Kolmogorov-Smirnov across time
        
        pvalues = [stats.ks_2samp(self.paths[0][:,i], self.paths[1][:,i], mode = 'exact')[1] for i in range(1,self.n+1)]
        
        plt.plot(self.times[1:], pvalues)
        plt.show()
        
        # =============================================================================
        # stat, pvalue = stats.ks_2samp(spots1, spots2, mode = 'exact')
        # print("Stat",stat,"p-value",pvalue)
        # 
        # print(ks_stastistic_threshold(pvalue, n_paths1, n_paths2))
        # 
        # =============================================================================
        
        #### corr comparrison
        
        def calculate_cor_lag(x,lag):
            lead_x = x[:,lag:]
            lag_x = x[:,:-lag]
            cov_x = tfp.stats.correlation(lead_x,lag_x, 0, event_axis = None)
            return cov_x.numpy()
        
        lags = range(1,self.n)
        abs_corrs = [[np.mean(calculate_cor_lag(np.abs(lr), l)) for l in lags] for lr in self.log_returns]
        
        plt.plot(lags,abs_corrs[0])
        plt.plot(lags,abs_corrs[1])
        plt.show()
        
        corrs = [[np.mean(calculate_cor_lag(lr, l)) for l in lags] for lr in self.log_returns]
        
        plt.plot(lags,corrs[0])
        plt.plot(lags,corrs[1])
        plt.show()

if __name__ == '__main__':
    import BlackScholesModel
    import HestonModel
    import MarketGenerator
    
    BS = False
    
    real_n = 20
    n = 5
    real_T = 1/12
    dt = real_T / real_n
    
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
        
        n_paths1 = 5000
        n_paths2 = 5000
        
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
        kappa = 5
        theta = 0.1
        sigma = 1
        rho = -0.9
        rate = 0.01
        dt = T/n
        ddt = 0.01
        
        model1 = HestonModel.HestonModel(s0,mu,v0,kappa,theta,sigma,rho,rate,dt,ddt)
        model2 = HestonModel.HestonModel(s0,mu,v0,kappa,theta,sigma*0.01,rho,rate,dt,ddt)
        
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
    
    MG = MarketGenerator.MarketGenerator(model1, n)
    MG.create_vae(latent_dim = 10, layers_units=[60], alpha = 0.2, beta = 10)
    MG.create_training_path(1000, overlap = True, seed = 69, cheat = True)
    MG.train_vae()
    
    training_paths = MG.training_paths
    mg_paths = MG.generate_paths(10000,save = False)
    
    MG.plot_generated_paths(100)
    plt.plot(training_paths[:100,:].T)
    plt.show()

    
    pc = PathComparator(training_paths,mg_paths,T)
    pc.run_comparrison()
    
    ###### Test on longer 
    long_mg_paths = MG.generate_longer_paths(10000, length = real_n, save = False)
    MG.plot_generated_paths(100, real_n)
    
    model1.reset_model(1000)
    for i in range(real_n):
        model1.evolve_s_b()
    long_model_paths = model1.spot_hist[:,0,:]
    
    pc_long = PathComparator(long_model_paths,long_mg_paths,T)
    pc_long.run_comparrison()