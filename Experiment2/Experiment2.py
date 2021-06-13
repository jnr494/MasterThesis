# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:07:50 2021

@author: mrgna
"""

#Experiment 2.0 Black Scholes with 1 asset w/wo transaction costs

import os
import sys
sys.path.insert(1,os.path.dirname(os.getcwd()))

import time
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import BinomialModel
import BlackScholesModel

import StandardNNModel
import OptionClass
import HedgeEngineClass
import nearPD
import helper_functions

n = 60 #60
rate = 0.02
rate_change = 0
T = 3/12

alpha = 0.95 #confidence level for CVaR
alpha1 = 0.5 #confidence level for CVaR as comparrison

#Create stock model
S0 = 1

run = "BS"
train_models = True

exp_nr = 1 ####### Set 0 for exp 2.0 and 1 for exp 2.1

#Exp 2.x settings
if exp_nr == 0:
    tc = 0
elif exp_nr == 1:
    tc = 0.005
elif exp_nr == 2:
    rate = 0
    tc = 0
elif exp_nr == 3:
    rate = 0
    tc = 0
    n = 3

if run == "BS":
    n_assets = 1
    
    np.random.seed(69)    
    
    mu = np.array([0.05])
    sigma = np.array([0.3])
    
    if exp_nr == 2:
        mu = np.array([0.])
    
    if exp_nr == 3:
        mu = np.array([0.2])
        sigma = np.array([0.3])
    
    corr = np.ones((n_assets,n_assets))
    for i in range(n_assets):
        for j in range(i,n_assets):
            if not i == j:
                #print(i,j)
                tmp_cor = np.random.uniform(0.5,1)
                corr[i,j] = tmp_cor
                corr[j,i] = tmp_cor    
    
    corr = nearPD.nearPD(corr, 1000) 
    
    s_model = BlackScholesModel.BlackScholesModel(1, mu, sigma, corr, rate, T / n, n_assets = n_assets)
    
    #Setup call option with strike 1
    units = [1]
    options = [OptionClass.Option("call",[S0,T],0)]
    option_por = OptionClass.OptionPortfolio(options,units)


#Option por
option_price = s_model.init_option(option_por)

#Create sample paths 
N = 18
n_samples = 2**N

x, y, banks = helper_functions.generate_dataset(s_model, n, n_samples, option_por)

############
## NN models
############

n_layers = 4 #4
n_units = 5 #5
tf.random.set_seed(69)

#Create NN model with MSE
model_mse = StandardNNModel.NN_simple_hedge(n_assets = n_assets, 
                                        n_layers = n_layers, n_units = n_units, 
                                        activation = 'elu', final_activation = None, 
                                        output2_dim = 1)

model_mse.create_model(n, rate = 0, dt = T / n, transaction_costs = tc, init_pf = option_price) 

#Create NN model with MSE with pf info
model_mse2 = StandardNNModel.NN_simple_hedge(n_assets = n_assets, 
                                        n_layers = n_layers, n_units = n_units, 
                                        activation = 'elu', final_activation = None, 
                                        output2_dim = 1)

model_mse2.create_model(n, rate = 0, dt = T / n, transaction_costs = tc, init_pf = option_price, 
                        ignore_pf = False, ignore_info = True) 


#Create NN model with rm
model = StandardNNModel.NN_simple_hedge(n_assets = n_assets, 
                                        n_layers = n_layers, n_units = n_units, 
                                        activation = 'elu', final_activation = None, 
                                        output2_dim = 1)

model.create_model(n, rate = 0, dt = T / n, transaction_costs = tc, init_pf = option_price) #0!!!!!!!!!!

model.create_rm_model(alpha = alpha)

#Create NN model with rm and lower confidence level
model2 = StandardNNModel.NN_simple_hedge(n_assets = n_assets, 
                                        n_layers = n_layers, n_units = n_units, 
                                        activation = 'elu', final_activation = None, 
                                        output2_dim = 1)

model2.create_model(n, rate = 0, dt = T / n, transaction_costs = tc, init_pf = option_price, #0!!!!!!!!!!!!!
                    ignore_pf = False)

model2.create_rm_model(alpha = alpha) #alpha1  

#Training models

best_model_name1 = "best_model_mse_2_{}.hdf5".format(exp_nr)
best_model_name12 = "best_model_mse2_2_{}.hdf5".format(exp_nr)
best_model_name2 = "best_model_rm_high_2_{}.hdf5".format(exp_nr)
best_model_name3 = "best_model_rm_low_2_{}.hdf5".format(exp_nr)

if train_models is True:
    #train mse model  
    model_mse.train_model(x, y, batch_size = 1024, epochs = 100, patience = [5,11], learning_rate = 0.01, best_model_name = best_model_name1)
    
    if not exp_nr == 2: 
        #train mse model2 
        model_mse2.train_model(x, y, batch_size = 1024, epochs = 100, patience = [5,11], learning_rate = 0.01, best_model_name = best_model_name12)
    
    t0 = time.time()
    #train CVaR high model   
    model.train_rm_model(x, epochs = 100, batch_size = 1024, patience = [5,11], lr = 0.01, best_model_name = best_model_name2)
    t1 = time.time()
    print("Training time:",t1-t0,"seconds")
    
    if not exp_nr == 2:
        #train CVaR low model   
        model2.train_rm_model(x, epochs = 100, batch_size = 1024, patience = [5,11], lr = 0.01, best_model_name = best_model_name3)
        

model_mse.model.load_weights(best_model_name1)
model.model_rm.load_weights(best_model_name2)
if not exp_nr == 2:
    model_mse2.model.load_weights(best_model_name12)
    model2.model_rm.load_weights(best_model_name3)  

#Look at RM-cvar prediction and empirical cvar
test = model.model_rm.predict(x)
print('model w:', test[0,1])
print('model_rm cvar:',model.get_J(x))

#Test CVAR network
test1 = - model.model.predict(x) 
print('emp cvar (in sample):',np.mean(test1[np.quantile(test1,alpha) <= test1]))

cvar = lambda w: w + np.mean(np.maximum(test1 - w,0) / (1- alpha))
xs = np.linspace(0,option_price*2,10000)
plt.plot(xs, [cvar(x) for x in xs])
plt.show()
print('emp cvar2 (in sample):',np.min([cvar(x) for x in xs]))
print('w:',xs[np.argmin([cvar(x) for x in xs])])

#Find zero cvar init_pf
init_pf_nn = model.get_init_pf() + model.get_J(x) * np.exp(-rate * T)

#Hedge simulations with fitted model
init_pf = option_price

N_hedge_samples = 50000

#create portfolios
models = [s_model, model_mse, model, model2, model_mse2]
model_names = ["Analytical","NN MSE", "NN CVaR {}".format(alpha), "NN CVaR {}".format(alpha1),
               "NN MSE w PF info"]

if exp_nr == 2:
    models = models[:3]
    model_names = model_names[:3]

#models = [s_model]
#model_names = [run]

#create hedge experiment engine
np.random.seed(420)
hedge_engine = HedgeEngineClass.HedgeEngineClass(n, s_model, models, option_por)

#run hedge experiment
np.random.seed(69)
hedge_engine.run_hedge_experiment(N_hedge_samples, init_pf, tc)

pf_values = hedge_engine.pf_values
hedge_spots = hedge_engine.hedge_spots
option_values = hedge_engine.option_values
Pnl = hedge_engine.Pnl_disc
hs_matrix = hedge_engine.hs_matrix


#################
#Plots
#################

dpi = 500

if n_assets == 1:
    #Plot of hedge accuracy
    tmp_xs = np.linspace(0.5*S0,2*S0)
    tmp_option_values = option_por.get_portfolio_payoff(tmp_xs[:,np.newaxis,np.newaxis])
    
    for pf_vals, name in zip(pf_values, model_names):
        plt.plot(tmp_xs, tmp_option_values, label = 'Option payoff')
        plt.scatter(hedge_spots,pf_vals, color = 'black', s = 2, label = "{} $PF_N$".format(name), alpha = 0.3)
        plt.xlabel("$S_N$")
        plt.legend()
        #plt.title("Hedge Errors:" + name)
        plt.savefig("ex2_{}_hedge_acc_{}.png".format(exp_nr, name), dpi = dpi, bbox_inches='tight')
        plt.show()
        plt.close()

if n_assets == 1:
    for pnl, name in zip(Pnl, model_names):
        plt.scatter(hedge_spots,pnl, color = "k", label = "{} PnL".format(name), s= 3, alpha = 0.2)
        plt.legend()
        plt.xlabel("$S_N$")
        #plt.title("Pnl scatter:" + name)
        plt.savefig("ex2_{}_model_pnl_{}.png".format(exp_nr, name), dpi = dpi, bbox_inches='tight')
        plt.show()
        plt.close()
    
#Plot hs from nn vs optimal
for i in range(2):
    for j in range(n_assets):
        times = np.arange(n)/n * T
        for hs_m, name, ls in zip(hs_matrix[:-1], model_names[:-1], ["-","--","--"]):
            plt.plot(times, hs_m[i,j,:], ls, label = name, lw = 2)
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("Units of $S$")
        plt.savefig("ex2_{}_hs_sample_{}.eps".format(exp_nr,i), bbox_inches='tight')
        plt.show()
        plt.close()
        
#Plot hs at some time over different spots
time = 0.5*T
time_idx = time_idx = int(np.round(time / s_model.dt , 6))
if n_assets == 1:
    tmp_spots = np.arange(60,180)/100*s_model.S0
    tmp_spots_tilde = tmp_spots * np.exp(- rate * time)
    
    tmp_model_hs = s_model.get_optimal_hs(time,tmp_spots[:,np.newaxis])
    plt.plot(tmp_spots, tmp_model_hs,
             label = "{}".format(model_names[0]))
     
    current_hs = np.array([0.5])
    current_pf = np.array([0.0])
    min_spot = np.array(0)
    spot_return = np.array(0)
    for m, name in zip(models[1:-2], model_names[1:-2]):
        plt.plot(np.arange(60,180)/100*s_model.S0,
                 [m.get_hs(time, current_hs, current_pf, spot_tilde, rate, min_spot, spot_return) for spot_tilde in tmp_spots_tilde], 
                 '--', label = "{}".format(name))
    
    hs_range = np.max(tmp_model_hs) - np.min(tmp_model_hs)
    plt.ylim(np.min(tmp_model_hs) - hs_range * 0.2, np.max(tmp_model_hs) + hs_range * 0.2)
    plt.legend()
    #plt.title("Holding in $S$ at time $t = {}$".format(time))
    plt.xlabel("$S({})$".format(time))
    plt.savefig("ex2_{}_hs_time_x.eps".format(exp_nr), bbox_inches='tight')
    plt.show()
    plt.close()

#Plot pnls on bar chart
for i in range(1,len(model_names)):
    plt.hist(Pnl[0], bins = 100, label=model_names[0],alpha = 0.8, density = True)
    plt.hist(Pnl[i], bins = 100, label=model_names[i], alpha = 0.5, density = True)
    #plt.title('Out of sample Pnl distribution')
    plt.legend()
    plt.xlabel("PnL")
    plt.savefig("ex2_{}_oos_pnldist_{}.png".format(exp_nr, model_names[i]), dpi = dpi, bbox_inches='tight')
    plt.show()
    plt.close()

############
#Calculations
###########

save_output = True
folder_name = ""

#Print outputs
if save_output is True:
    text_file = open(folder_name + "output exp 2_{}, hedge points {}, tc = {}, samples = 2_{}.txt".format(exp_nr,n,tc,N),"w")

def print_overload(*args):
    str_ = ''
    for x in args:
        str_ += " " + str(x)
    str_ = str_[1:]
    if save_output is True:
        text_file.write(str_ + "\n")
    print(str_)

#option price and p0
print_overload("Exp nr:",exp_nr)
print_overload("hedge points {}, tc = {}, samples = 2**{}".format(n,tc,N))
print_overload("Option price:", option_price)
print_overload("NN MSE p0:", model_mse.get_init_pf(), model_mse.get_init_pf() / option_price * 100)
tmp_p0 = model.get_init_pf() + model.get_J(x) * np.exp(-rate * T)
print_overload("NN CVaR{} p0:".format(alpha), tmp_p0, tmp_p0 / option_price * 100)
tmp_p0 = model2.get_init_pf() + model2.get_J(x) * np.exp(-rate * T)
print_overload("NN CVaR{} p0:".format(alpha1), tmp_p0, tmp_p0 / option_price * 100)


#Avg abs Pnl
for pnl, name in zip(Pnl, model_names):
    print_overload("Avg abs PnL ({}):".format(name), np.round(np.mean(abs(pnl)),5), 
      '(',np.round(np.std(abs(pnl)) / np.sqrt(N_hedge_samples),8),')',
      np.round(np.mean(abs(pnl))  / option_price * 100,5))

#Avg squared Pnl
for pnl, name in zip(Pnl, model_names):
    print_overload("Avg squared PnL ({}):".format(name), np.round(np.mean(pnl**2),8),
    '(',np.round(np.std(pnl**2) / np.sqrt(N_hedge_samples),8),')')

#Avg Pnl
for pnl, name in zip(Pnl, model_names):
    print_overload("Avg PnL ({}):".format(name), np.round(np.mean(pnl),8),
    '(', np.round(np.std(pnl),8),')',              
    '((', np.round(np.std(pnl) / np.sqrt(N_hedge_samples),8),'))',
      np.round(np.mean(pnl) / option_price * 100,5))

#Calculate CVAR high
for pnl, name in zip(Pnl, model_names):
    tmp_loss = - pnl
    tmp_cvar = np.mean(tmp_loss[np.quantile(tmp_loss, alpha) <= tmp_loss])
    print_overload('Out of sample CVAR{} ({}):'.format(alpha, name),tmp_cvar)
    
#Calculate CVAR low
for pnl, name in zip(Pnl, model_names):
    tmp_loss = - pnl
    tmp_cvar = np.mean(tmp_loss[np.quantile(tmp_loss, alpha1) <= tmp_loss])
    print_overload('Out of sample CVAR{} ({}):'.format(alpha1, name),tmp_cvar)

#Turnover
for por, name in zip(hedge_engine.ports, model_names):
    tmp_turnover = np.mean(por.turnover, axis = 0)
    tmp_turnover_std = np.std(por.turnover, axis = 0)
    print_overload('Avg. Turnover ({})'.format(name), tmp_turnover,
          '(',tmp_turnover_std / np.sqrt(N_hedge_samples),')')
    
#Avg transaction costs
for por, name in zip(hedge_engine.ports, model_names):
    tmp_tc = np.mean(np.sum(por.tc_hist, axis = 1))
    tmp_tc_std = np.std(np.sum(por.tc_hist, axis = 1))
    print_overload('Avg. Transaction Costs ({})'.format(name), tmp_tc,
          '(',tmp_tc_std / np.sqrt(N_hedge_samples),')')
    
if save_output is True:
    text_file.close()

####### Wrong sigma analysis #########
#### Change sigma
if exp_nr == 2:
    sigmas = np.linspace(0.05,0.6,150)
    N_hedge_samples = 10000
    
    new_s_model = BlackScholesModel.BlackScholesModel(1, mu, sigma, corr, rate, T / n, n_assets = n_assets)
    bs_hedger = BlackScholesModel.BShedge(s_model)
    new_models = [bs_hedger] + models[1:]
    
    #s_model.reset_model(N_hedge_samples)
    
    def get_avg_pnl(Pnl): return np.array(
        [np.round(np.mean(pnl), 5) for pnl in Pnl])
    
    
    def get_avg_abs_pnl(Pnl): return np.array(
        [np.round(np.mean(np.abs(pnl)), 5) for pnl in Pnl])
    
    
    def get_avg_sq_pnl(Pnl): return np.array(
        [np.round(np.mean(pnl**2), 8) for pnl in Pnl])
    
    
    def get_oos_cvar(Pnl): return np.array(
        [np.round(np.mean((-pnl)[np.quantile(-pnl, alpha) <= (-pnl)]), 5) for pnl in Pnl])
    
    avg_pnls = [[] for _ in range(3)] 
    avg_abs_pnls = [[] for _ in range(3)]
    avg_sq_pnls = [[] for _ in range(3)]
    cvars = [[] for _ in range(3)]   
    
    for idx, tmp_sigma in enumerate(sigmas):
        print(idx, "Runing sigma = {}".format(tmp_sigma))
        new_s_model.sigma = tmp_sigma
        
        np.random.seed(420)
        tmp_hedge_engine = HedgeEngineClass.HedgeEngineClass(n, new_s_model, new_models, option_por)
        
        #run hedge experiment
        np.random.seed(69)
        #tmp_hedge_engine.run_hedge_experiment(N_hedge_samples, init_pf, tc)
        tmp_hedge_engine.run_quick_hedge_exp(N_hedge_samples, init_pf, tc)
        
        tmp_Pnl = tmp_hedge_engine.Pnl_disc
        
        tmp_avg_pnl = get_avg_pnl(tmp_Pnl)
        tmp_avg_abs_pnl = get_avg_abs_pnl(tmp_Pnl)
        tmp_avg_sq_pnl = get_avg_sq_pnl(tmp_Pnl)
        tmp_cvar = get_oos_cvar(tmp_Pnl)
        
        for i in range(3):
    
            avg_pnls[i].append(tmp_avg_pnl[i])
            avg_abs_pnls[i].append(tmp_avg_abs_pnl[i])
            avg_sq_pnls[i].append(tmp_avg_sq_pnl[i])
            cvars[i].append(tmp_cvar[i])
    
    for tmp_measure, tmp_name in zip([avg_pnls, avg_abs_pnls, cvars], ["Avg. Pnl", "Avg. abs. Pnl", "CvaR 0.95"]):
        for idx, (name, style) in enumerate(zip(model_names,['-','--','--'])):
            plt.plot(sigmas,np.array(tmp_measure[idx]), style, label = name)
        plt.legend()
        plt.xlabel("$\sigma$")
        plt.ylabel(tmp_name)
        plt.savefig("exp2.2 " + tmp_name + ".png", dpi = dpi, bbox_inches='tight')
        plt.show()
        
        for idx, (name, style) in enumerate(zip(model_names,['-','--','--'])):
            plt.plot(sigmas,np.array(tmp_measure[idx]) - np.array(tmp_measure[0]), style, label = name)
        plt.legend()
        plt.xlabel("$\sigma$")
        plt.ylabel(tmp_name + " diff. from Analytical")
        plt.savefig("exp2.2 " + tmp_name + " diff.png", dpi = dpi, bbox_inches='tight')
        plt.show()
     
# =============================================================================
# #Mu exp
# if exp_nr == 2:
#     mus = np.linspace(-0.1,0.1,20)
#     N_hedge_samples = 10000
#     
#     new_s_model = BlackScholesModel.BlackScholesModel(1, mu, sigma, corr, rate, T / n, n_assets = n_assets)
#     bs_hedger = BlackScholesModel.BShedge(s_model)
#     new_models = [bs_hedger] + models[1:]
#     
#     #s_model.reset_model(N_hedge_samples)
#     
#     def get_avg_pnl(Pnl): return np.array(
#         [np.round(np.mean(pnl), 5) for pnl in Pnl])
#     
#     
#     def get_avg_abs_pnl(Pnl): return np.array(
#         [np.round(np.mean(np.abs(pnl)), 5) for pnl in Pnl])
#     
#     
#     def get_avg_sq_pnl(Pnl): return np.array(
#         [np.round(np.mean(pnl**2), 8) for pnl in Pnl])
#     
#     
#     def get_oos_cvar(Pnl): return np.array(
#         [np.round(np.mean((-pnl)[np.quantile(-pnl, alpha) <= (-pnl)]), 5) for pnl in Pnl])
#     
#     mu_avg_pnls = [[] for _ in range(3)] 
#     mu_avg_abs_pnls = [[] for _ in range(3)]
#     mu_avg_sq_pnls = [[] for _ in range(3)]
#     mu_cvars = [[] for _ in range(3)]   
#     
#     for idx, tmp_mu in enumerate(mus):
#         print(idx, "Runing mu = {}".format(tmp_mu))
#         new_s_model.mu = np.array(tmp_mu)
#         
#         np.random.seed(420)
#         tmp_hedge_engine = HedgeEngineClass.HedgeEngineClass(n, new_s_model, new_models, option_por)
#         
#         #run hedge experiment
#         np.random.seed(69)
#         #tmp_hedge_engine.run_hedge_experiment(N_hedge_samples, init_pf, tc)
#         tmp_hedge_engine.run_quick_hedge_exp(N_hedge_samples, init_pf, tc)
#         
#         tmp_Pnl = tmp_hedge_engine.Pnl_disc
#         
#         tmp_avg_pnl = get_avg_pnl(tmp_Pnl)
#         tmp_avg_abs_pnl = get_avg_abs_pnl(tmp_Pnl)
#         tmp_avg_sq_pnl = get_avg_sq_pnl(tmp_Pnl)
#         tmp_cvar = get_oos_cvar(tmp_Pnl)
#         
#         for i in range(3):
#     
#             mu_avg_pnls[i].append(tmp_avg_pnl[i])
#             mu_avg_abs_pnls[i].append(tmp_avg_abs_pnl[i])
#             mu_avg_sq_pnls[i].append(tmp_avg_sq_pnl[i])
#             mu_cvars[i].append(tmp_cvar[i])
#     
#     for tmp_measure, tmp_name in zip([mu_avg_pnls, mu_avg_abs_pnls, mu_cvars], ["Avg. Pnl", "Avg. abs. Pnl", "CvaR 0.95"]):
#         for idx, (name, style) in enumerate(zip(model_names,['-','--','--'])):
#             plt.plot(mus,np.array(tmp_measure[idx]), style, label = name, )
#         plt.legend()
#         plt.xlabel("$\sigma$")
#         plt.ylabel(tmp_name)
#         plt.show()
#         
#         for idx, (name, style) in enumerate(zip(model_names,['-','--','--'])):
#             plt.plot(mus,np.array(tmp_measure[idx]) - np.array(tmp_measure[0]), style, label = name)
#         plt.legend()
#         plt.xlabel("$\sigma$")
#         plt.ylabel(tmp_name + " diff. from Analytical")
#         plt.show()
# 
# 
# =============================================================================
