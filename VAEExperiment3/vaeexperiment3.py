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
#tf.config.set_visible_devices([], 'GPU')
import gc

import MarketGenerator
import HestonModel
import KolmogorovSmirnovTest


n = 20
T = 1/12
dt = T / n
cond_n = 1
cond_type = 1

s0 = 1
mu = 0.05
v0 = 0.05
kappa = 4
theta = 0.05
sigma = 0.25
rho = 0
rate = 0.01
dt = T/n
ddt = 0.01

heston_model = HestonModel.HestonModel(s0,mu,v0,kappa,theta,sigma,rho,rate,dt,ddt)
heston_model.use_v = True

n_samples = 1000
real_decoder = True
alpha, beta, gamma = (0.9, 0, 1)
layers_units = [60]
cheat = False

MG = MarketGenerator.MarketGenerator(heston_model, n, cond_n, cond_type)
MG.create_vae(latent_dim = int(n), layers_units=layers_units, alpha = alpha, 
              gamma = gamma, beta = beta, real_decoder = real_decoder)
MG.create_training_path(n_samples, overlap = False, seed = None, cheat = cheat)
MG.train_vae(epochs = 500, batch_size = 128, lrs = [0.01,0.0001,0.00001], best_model_name = None)

a = MG.cond
b = MG.log_returns[:,0:1]
plt.scatter(a,np.abs(b))
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(a, np.abs(b))  # perform linear regression
Y_pred = linear_regressor.predict(a)  # make predictions
plt.plot(a,Y_pred, c = 'r')
plt.show()

#get test paths
original_train_path = np.squeeze(heston_model.spot_hist[0,0,:])
p_len = len(original_train_path)
times = np.linspace(0, (p_len - 1) * dt, p_len)

plt.plot(times,original_train_path, label = "Heston path")
plt.xlabel("t")
plt.legend()
plt.savefig("heston_path_nocheat.eps", bbox_inches='tight')
plt.show()


k = 20000
org_training_paths = np.array(MG.training_paths)
org_cond = np.array(MG.cond)
training_paths = org_training_paths
tmp_cond = org_cond
tmp_cond = np.repeat(tmp_cond,k // n_samples,axis = 0) if tmp_cond is not None else None 

init_var = np.quantile(org_cond, 0.5)
MG.cond_n = 0
heston_model.v0 = init_var
new_samples = 1000
MG.create_training_path(new_samples, overlap = False, seed = None, cheat = True)
heston_model.v0 = v0
MG_cond_n = cond_n
tmp_cond = np.ones((k,1)) * init_var
training_paths = MG.training_paths

#get cond


#create MG paths
mg_paths = MG.generate_paths(k, cond = tmp_cond, save = False, 
                             std = 1, mean = 0, real_decoder = real_decoder)

###############################Run 1st comparrison    
pc = KolmogorovSmirnovTest.PathComparator(training_paths,mg_paths,T, ['Heston','VAE'])
pc.run_comparrison(plot = False)

####Plots

#plot training paths and MG paths
no_of_plotted_paths = 100

plt.plot(pc.times, mg_paths[:no_of_plotted_paths,:].T)
plt.ylim((0.75,1.35))
plt.xlabel("t")
plt.savefig("vae3_example_paths_MG_{}_{}.eps".format(cheat, beta), bbox_inches='tight')
plt.show()

plt.plot(pc.times, training_paths[:no_of_plotted_paths,:].T)
plt.ylim((0.75,1.35))
plt.xlabel("t")
plt.savefig("vae2_example_paths_Heston_{}_{}.eps".format(cheat, beta), bbox_inches='tight')
plt.show()

#Plot means
[plt.plot(pc.times,m, ls, label = l) for m, l, ls in zip(pc.means, pc.names,['-','--'])]
plt.legend()
plt.xlabel("t")
plt.savefig("vae3_means_comp_{}_{}.eps".format(cheat, beta), bbox_inches = "tight")
plt.show()  

#Plot std
[plt.plot(pc.times,s, ls, label = l) for s, l, ls in zip(pc.stds, pc.names,['-','--'])]
plt.legend()
plt.xlabel("t")
plt.savefig("vae3_stds_comp_{}_{}.eps".format(cheat, beta), bbox_inches = "tight")
plt.show() 

#Plot correlation
c_min, c_max = (-0.02,0.05)
[plt.plot(pc.lags, c, ls, label = l) for c, l, ls in zip(pc.corrs, pc.names,['-','--'])]
plt.legend()
plt.xlabel("Lag")
plt.ylim((c_min,c_max))
plt.savefig("vae3_corrs_comp_{}_{}.eps".format(cheat, beta), bbox_inches = "tight")
plt.show() 

#Plot abs correlation
c_min, c_max = (-0.02,0.05)
[plt.plot(pc.lags,c, ls, label = l) for c, l, ls in zip(pc.abs_corrs, pc.names,['-','--'])]
plt.legend()
plt.xlabel("Lag")
plt.ylim((c_min,c_max))
plt.savefig("vae3_abs_corrs_comp_{}_{}.eps".format(cheat, beta), bbox_inches = "tight")
plt.show() 

#Plot qq
plt.scatter(pc.quantiles[0], pc.quantiles[1], s = 5)
plt.plot(pc.quantiles[0][[0,-1]],pc.quantiles[0][[0,-1]],c='k')
plt.xlabel("Heston quantiles")
plt.ylabel("VAE quantiles")
plt.savefig("vae3_qq_{}_{}.eps".format(cheat, beta), bbox_inches = "tight")
plt.show()

#Plot ecdf for T
[plt.plot(pc.ecdf_ranges[-1],e, ls, label = l) for e, l, ls in zip(pc.ecdf_saved[-1], pc.names,['-','--'])]
plt.legend()
plt.xlabel("spot")
plt.savefig("vae3_ecdf_T_{}_{}.eps".format(cheat, beta), bbox_inches = "tight")
plt.show()

#Plot KS pvalues
plt.plot(pc.times[1:], pc.pvalues, label = "Kolmogorov-Smirnov p-values")
plt.legend()
plt.ylim((0,1))
plt.xlabel("t")
plt.savefig("vae3_ks_pvalues_{}_{}.eps".format(cheat, beta), bbox_inches = "tight")
plt.show()

####################### averages runs
from tqdm import tqdm

n_vaes = 5

pcss = []
init_vars = []

n_samples = 1000
real_decoder = True
alpha, beta, gamma = (0.9, 0, 0)
layers_units = [60]
var_name2 = "gamma"
#var_name2 = "alpha"
var_name = r"$\{}$".format(var_name2)

#var_name2 = "n_samples"
#var_name = "training samples"

gammas = [0,10,100]
indep_test_paths = True
quantiles = np.linspace(0.1,0.9,17)
#quantiles = [0.1,0.4,0.6,0.9]

init_vars = [[] for q in quantiles]

var_copy = gammas
for gamma in gammas:
    tmp_pcs = [[] for q in quantiles]
    
    for i in tqdm(range(n_vaes)):
        print("alpha, beta, gamma, samples:",alpha,beta, gamma, n_samples)
        #Creat MG and train
        tmp_MG = MarketGenerator.MarketGenerator(heston_model, n, cond_n, cond_type)
        tmp_MG.create_vae(latent_dim = n, layers_units=layers_units, alpha = alpha, beta = beta, gamma = gamma, 
                          real_decoder= real_decoder, summary = False)
        tmp_MG.create_training_path(n_samples, overlap = False, seed = None, cheat = False)
        tmp_MG.train_vae(epochs = 500, batch_size = 128, lrs = [0.01,0.0001,0.00001], verbose = 0, best_model_name = None)
        
        org_cond = np.array(tmp_MG.cond)
        
        for q_idx, q in enumerate(quantiles):
            #Setup
            init_var = np.quantile(org_cond, q)
            tmp_MG.cond_n = 0
            heston_model.v0 = init_var #change init var
            #Paths
            tmp_MG.create_training_path(1000, overlap = False, seed = None, cheat = True, plot = False)
            tmp_training_paths = tmp_MG.training_paths   
            #print(gamma,i, tmp_training_paths)
            if tmp_training_paths[0,-1] == tmp_training_paths[1,-1]:
                print("FAIL")
                raise Exception('fuck')
            #reset
            heston_model.v0 = v0
            tmp_MG.cond_n = cond_n
                
            k = 20000
            tmp_cond = np.ones((k,1)) * init_var
            tmp_mg_paths = tmp_MG.generate_paths(k, cond = tmp_cond, save = False, 
                                         std = 1, mean = 0, real_decoder = real_decoder)
            
            tmp_pc = KolmogorovSmirnovTest.PathComparator(np.array(tmp_training_paths),np.array(tmp_mg_paths),
                                                          T, ['Black Scholes','VAE'])
            tmp_pc.run_comparrison(plot = False)
            tmp_pcs[q_idx].append(tmp_pc)
            
            init_vars[q_idx].append(init_var)
            
        del tmp_MG
        del tmp_training_paths
        del tmp_mg_paths
        
        gc.collect()
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
    
    pcss.append(tmp_pcs)
    
#KS plot
for q_idx, q in enumerate(quantiles):
    for i, pcs in enumerate(pcss):
        pvalues = [pc.pvalues for pc in pcs[q_idx]]
        pvalues = np.vstack(pvalues)
        
        pvalues_mean = np.mean(pvalues,axis = 0)
        pvalues_std = np.std(pvalues, axis = 0)
        
        conf_interval = np.vstack([pvalues_mean + c * 1.96 * pvalues_std / np.sqrt(n_vaes) for c in [-1.,1.]])
        
        plt.plot(pc.times[1:], pvalues_mean, label = r"VAE, {}={}".format(var_name, var_copy[i]), c = 'C{}'.format(i+1))
        plt.plot(np.tile(pc.times[1:],(2,1)).T, conf_interval.T ,'--', c = 'C{}'.format(i+1))
        
    plt.ylim((0,1))
    plt.legend()
    plt.xlabel("t")
    plt.title("Quantile: {}".format(q))
    #plt.savefig("vae3_ks_avg_{}_q{}.eps".format(var_name2, int(q*100)), bbox_inches = "tight" )
    plt.show()

#Collect KS
ks_means = [[] for _ in pcss]
ks_stds = [[] for _ in pcss]
for i, pcs in enumerate(pcss):
    for q_idx, q in enumerate(quantiles):
        pvalues = [pc.pvalues for pc in pcs[q_idx]]
        pvalues = np.vstack(pvalues)
        pvalues = np.mean(pvalues, axis = 1)
        
        pvalues_mean = np.mean(pvalues)
        pvalues_std = np.std(pvalues)
        
        ks_means[i].append(pvalues_mean)
        ks_stds[i].append(pvalues_std)

for i, (m, s) in enumerate(zip(ks_means,ks_stds)):
    plt.plot(quantiles, m, label = r"VAE, {}={}".format(var_name, var_copy[i]), c = 'C{}'.format(i+1))
    
    conf_interval = np.vstack([np.array(m) + c * 1.96 * np.array(s) / np.sqrt(n_vaes) for c in [-1.,1.]])
    plt.plot(quantiles, conf_interval.T, '--', c = 'C{}'.format(i+1))
plt.legend()
plt.ylim((0,1))
plt.xlabel(r"$\nu(0)$-quantile")
#plt.savefig("vae3_ks_avg_{}_avgq.eps".format(var_name2), bbox_inches = "tight" )
plt.show()
    

#corr plot
c_min, c_max = (-0.02,0.03)
q_idx = 10
q = np.round(quantiles[q_idx],2)
bs_corrs = 0
for i, pcs in enumerate(pcss):
    bs_corrs += np.mean(np.vstack([pc.corrs[0] for pc in pcs[q_idx]]), axis = 0) / len(pcss)
    
plt.plot(pc.lags, bs_corrs, label = "Heston")

for i, pcs in enumerate(pcss):
    mg_corrs = np.mean(np.vstack([pc.corrs[1] for pc in pcs[q_idx]]), axis = 0)
    mg_corrs_std = np.std(np.vstack([pc.corrs[1] for pc in pcs[q_idx]]), axis = 0)
    
    conf_interval = np.vstack([mg_corrs + c * 1.96 * mg_corrs_std / np.sqrt(n_vaes) for c in [-1.,1.]])
    
    #plt.plot(pc.lags, bs_corrs)
    plt.plot(pc.lags, mg_corrs, c = 'C{}'.format(i+1), label = r"VAE, {}={}".format(var_name, var_copy[i]))
    #plt.plot(np.tile(pc.lags,(2,1)).T, conf_interval.T ,'--', c = 'C{}'.format(i+1))
plt.ylim((c_min,c_max))
plt.xlabel("Lag")
plt.title("Quantile:{}".format(q))
plt.legend()
#plt.savefig("vae3_corr_avg_{}_q{}.eps".format(var_name2,q), bbox_inches = "tight" )
plt.show()

#abs corr plot
bs_corrs = 0
for i, pcs in enumerate(pcss):
    bs_corrs += np.mean(np.vstack([pc.abs_corrs[0] for pc in pcs[q_idx]]), axis = 0) / len(pcss)
    
plt.plot(pc.lags,bs_corrs, label = "Heston")

for i, pcs in enumerate(pcss):
    mg_abs_corrs = np.mean(np.vstack([pc.abs_corrs[1] for pc in pcs[q_idx]]), axis = 0)
    mg_abs_corrs_std = np.std(np.vstack([pc.corrs[1] for pc in pcs[q_idx]]), axis = 0)
    
    conf_interval = np.vstack([mg_abs_corrs + c * 1.96 * mg_abs_corrs_std / np.sqrt(n_vaes) for c in [-1.,1.]])
    
    #plt.plot(pc.lags, bs_abs_corrs)
    plt.plot(pc.lags, mg_abs_corrs, c = 'C{}'.format(i+1), label = r"VAE, {}={}".format(var_name, var_copy[i]))
    #plt.plot(np.tile(pc.lags,(2,1)).T, conf_interval.T ,'--', c = 'C{}'.format(i+1))
plt.ylim((c_min,c_max))
plt.xlabel("Lag")
plt.legend()
plt.title("Quantile:{}".format(q))
#plt.savefig("vae3_abscorr_avg_{}_q{}.eps".format(var_name2,q), bbox_inches = "tight" )
plt.show()

##
var_quantiles = [np.round(np.mean(v),4) for v in init_vars]
print(var_quantiles)
