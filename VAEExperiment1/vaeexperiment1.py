# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:55:27 2021

@author: mrgna
"""

import os
import sys
sys.path.insert(1,os.path.dirname(os.getcwd()))

import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import gc

import MarketGenerator
import BlackScholesModel
import KolmogorovSmirnovTest


n = 20
T = 1/12
dt = T / n
cond_n = 0

s0 = 1
mu = 0.05
sigma1 = 0.3
sigma2 = 0.3
rate = 0.01

bs_model = BlackScholesModel.BlackScholesModel(s0, mu, sigma1, np.ones((1,1)), rate, dt)

n_samples = 1000
real_decoder = True
alpha, beta = (0.9, 0)
layers_units = [40]
cheat = False

MG = MarketGenerator.MarketGenerator(bs_model, n, cond_n)
MG.create_vae(latent_dim = n, layers_units=layers_units, alpha = alpha, beta = beta, real_decoder= real_decoder)
MG.create_training_path(n_samples, overlap = True, seed = None, cheat = cheat)
MG.train_vae(epochs = 500, batch_size = 128, lrs = [0.01,0.0001,0.00001], best_model_name = None)

if cheat is False:
    original_train_path = np.squeeze(bs_model.spot_hist)
    p_len = len(original_train_path)
    times = np.linspace(0, (p_len - 1) * dt, p_len)
    
    plt.plot(times,original_train_path, label = "Black-Scholes path")
    plt.xlabel("t")
    plt.legend()
    plt.savefig("black_scholes_path_nocheat.eps", bbox_inches='tight')
    plt.show()
    
    MG.create_training_path(n_samples, overlap = True, seed = None, cheat = True)

training_paths = MG.training_paths

#get cond
cond = MG.cond

#create MG paths
k = 20000 // n_samples
tmp_cond = np.repeat(cond,k,axis = 0) if cond is not None else None 
mg_paths = MG.generate_paths(n_samples*k, cond = tmp_cond, save = False, 
                             std = 1, mean = 0, real_decoder = real_decoder)

###############################Run 1st comparrison    
pc = KolmogorovSmirnovTest.PathComparator(training_paths,mg_paths,T, ['Black Scholes','VAE'])
pc.run_comparrison(plot = False)

####Plots

#plot training paths and MG paths
no_of_plotted_paths = 100

plt.plot(pc.times, mg_paths[:no_of_plotted_paths,:].T)
plt.ylim((0.75,1.35))
plt.xlabel("t")
plt.savefig("example_paths_MG_{}_{}.eps".format(cheat, beta), bbox_inches='tight')
plt.show()

plt.plot(pc.times, training_paths[:no_of_plotted_paths,:].T)
plt.ylim((0.75,1.35))
plt.xlabel("t")
plt.savefig("example_paths_BS_{}_{}.eps".format(cheat, beta), bbox_inches='tight')
plt.show()

#Plot means
[plt.plot(pc.times,m, ls, label = l) for m, l, ls in zip(pc.means, pc.names,['-','--'])]
plt.legend()
plt.xlabel("t")
plt.savefig("vae1_means_comp_{}_{}.eps".format(cheat, beta), bbox_inches = "tight")
plt.show()  

#Plot std
[plt.plot(pc.times,s, ls, label = l) for s, l, ls in zip(pc.stds, pc.names,['-','--'])]
plt.legend()
plt.xlabel("t")
plt.savefig("vae1_stds_comp_{}_{}.eps".format(cheat, beta), bbox_inches = "tight")
plt.show() 

#Plot correlation
c_min, c_max = (-0.1,0.1)
[plt.plot(pc.lags, c, ls, label = l) for c, l, ls in zip(pc.corrs, pc.names,['-','--'])]
plt.legend()
plt.xlabel("Lag")
plt.ylim((c_min,c_max))
plt.savefig("vae1_corrs_comp_{}_{}.eps".format(cheat, beta), bbox_inches = "tight")
plt.show() 

#Plot abs correlation
c_min, c_max = (-0.1,0.1)
[plt.plot(pc.lags,c, ls, label = l) for c, l, ls in zip(pc.abs_corrs, pc.names,['-','--'])]
plt.legend()
plt.xlabel("Lag")
plt.ylim((c_min,c_max))
plt.savefig("vae1_abs_corrs_comp_{}_{}.eps".format(cheat, beta), bbox_inches = "tight")
plt.show() 

#Plot qq
plt.scatter(pc.quantiles[0], pc.quantiles[1], s = 5)
plt.plot(pc.quantiles[0][[0,-1]],pc.quantiles[0][[0,-1]],c='k')
plt.xlabel("BS quantiles")
plt.ylabel("VAE quantiles")
plt.savefig("vae1_qq_{}_{}.eps".format(cheat, beta), bbox_inches = "tight")
plt.show()

#Plot ecdf for T
[plt.plot(pc.ecdf_ranges[-1],e, ls, label = l) for e, l, ls in zip(pc.ecdf_saved[-1], pc.names,['-','--'])]
plt.legend()
plt.xlabel("spot")
plt.savefig("vae1_ecdf_T_{}_{}.eps".format(cheat, beta), bbox_inches = "tight")
plt.show()

#Plot KS pvalues
plt.plot(pc.times[1:], pc.pvalues, label = "Kolmogorov-Smirnov p-values")
plt.legend()
plt.ylim((0,1))
plt.xlabel("t")
plt.savefig("vae1_ks_pvalues_{}_{}.eps".format(cheat, beta), bbox_inches = "tight")
plt.show()

####################### averages runs
from tqdm import tqdm

n_vaes = 10

pcss = []

n_samples = 1000
real_decoder = True
alpha, beta = (0.9, 0)
layers_units = [40]
var_name2 = "beta"
#var_name2 = "alpha"
var_name = r"$\{}$".format(var_name2)

#var_name2 = "n_samples"
#var_name = "training samples"

betas = [0,10,100]
alphas = [0.8,0.9,0.99]
n_sampless = [100,250,1000]
indep_test_paths = True
cheat = False

var_copy = betas

#for n_samples in n_sampless:
#for alpha in alphas:
for beta in betas:
    tmp_pcs = []
    for i in tqdm(range(n_vaes)):
        print("alpha, beta, samples:",alpha,beta, n_samples)
        #Creat MG and train
        tmp_MG = MarketGenerator.MarketGenerator(bs_model, n, cond_n)
        tmp_MG.create_vae(latent_dim = n, layers_units=layers_units, alpha = alpha, beta = beta, real_decoder= real_decoder)
        tmp_MG.create_training_path(n_samples, overlap = True, seed = None, cheat = True)
        tmp_MG.train_vae(epochs = 500, batch_size = 128, lrs = [0.01,0.0001,0.00001], verbose = 0, best_model_name = None)
        
        #Get paths
        if indep_test_paths is True:
            tmp_MG.create_training_path(1000, overlap = True, seed = None, cheat = cheat)
            
        tmp_training_paths = tmp_MG.training_paths
        
        tmp_mg_paths = tmp_MG.generate_paths(20000, cond = None, save = False, 
                                     std = 1, mean = 0, real_decoder = real_decoder)
        
        tmp_pc = KolmogorovSmirnovTest.PathComparator(tmp_training_paths,tmp_mg_paths,T, ['Black Scholes','VAE'])
        tmp_pc.run_comparrison(plot = False)
        tmp_pcs.append(tmp_pc)
        
        del tmp_MG
        del tmp_training_paths
        del tmp_mg_paths
        
        gc.collect()
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
    
    pcss.append(tmp_pcs)
    


    
#KS plot
for i, pcs in enumerate(pcss):
    pvalues = [pc.pvalues for pc in pcs]
    pvalues = np.vstack(pvalues)
    
    pvalues_mean = np.mean(pvalues,axis = 0)
    pvalues_std = np.std(pvalues, axis = 0)
    
    conf_interval = np.vstack([pvalues_mean + c * 1.96 * pvalues_std / np.sqrt(n_vaes) for c in [-1.,1.]])
    
    plt.plot(pc.times[1:], pvalues_mean, label = r"{}={}".format(var_name, var_copy[i]), c = 'C{}'.format(i))
    plt.plot(np.tile(pc.times[1:],(2,1)).T, conf_interval.T ,'--', c = 'C{}'.format(i))
    
plt.ylim((0,1))
plt.legend()
plt.xlabel("t")
plt.savefig("vae1_ks_avg_{}.eps".format(var_name2), bbox_inches = "tight" )
plt.show()

#corr plot
for i, pcs in enumerate(pcss):
    bs_corrs = np.mean(np.vstack([pc.corrs[0] for pc in pcs]), axis = 0)
    mg_corrs = np.mean(np.vstack([pc.corrs[1] for pc in pcs]), axis = 0)
    mg_corrs_std = np.std(np.vstack([pc.corrs[1] for pc in pcs]), axis = 0)
    
    conf_interval = np.vstack([mg_corrs + c * 1.96 * mg_corrs_std / np.sqrt(n_vaes) for c in [-1.,1.]])
    
    #plt.plot(pc.lags, bs_corrs)
    plt.plot(pc.lags, mg_corrs, c = 'C{}'.format(i+1))
    plt.plot(np.tile(pc.lags,(2,1)).T, conf_interval.T ,'--', c = 'C{}'.format(i+1))
plt.ylim((c_min,c_max))
plt.show()

#abs corr plot
for i, pcs in enumerate(pcss):
    bs_abs_corrs = np.mean(np.vstack([pc.abs_corrs[0] for pc in pcs]), axis = 0)
    mg_abs_corrs = np.mean(np.vstack([pc.abs_corrs[1] for pc in pcs]), axis = 0)
    mg_abs_corrs_std = np.std(np.vstack([pc.corrs[1] for pc in pcs]), axis = 0)
    
    conf_interval = np.vstack([mg_abs_corrs + c * 1.96 * mg_abs_corrs_std / np.sqrt(n_vaes) for c in [-1.,1.]])
    
    #plt.plot(pc.lags, bs_abs_corrs)
    plt.plot(pc.lags, mg_abs_corrs, c = 'C{}'.format(i+1))
    plt.plot(np.tile(pc.lags,(2,1)).T, conf_interval.T ,'--', c = 'C{}'.format(i+1))
plt.ylim((c_min,c_max))
plt.xlabel("Lag")
plt.show()


