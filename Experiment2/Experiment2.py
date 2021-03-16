# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:07:50 2021

@author: mrgna
"""

#Experiment 2.0 Black Scholes with 1 asset and no transaction costs

import os
import sys
sys.path.insert(1,os.path.dirname(os.getcwd()))

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
#tc = 0.0 #transaction cost

alpha = 0.95 #confidence level for CVaR
alpha1 = 0.5 #confidence level for CVaR as comparrison

#Create stock model
S0 = 1

run = "BS"

exp_nr = 1 ####### Set 0 for exp 2.0 and 1 for exp 2.1

#Exp 2.x settings
if exp_nr == 0:
    tc = 0
elif exp_nr == 1:
    tc = 0.005

if run == "BS":
    n_assets = 1
    
    np.random.seed(69)    
    
    mu = np.array([0.05])
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
N = 18 #18
n_samples = 2**N

x, y, banks = helper_functions.generate_dataset(s_model, n, n_samples, option_por)

############
## NN models
############

n_layers = 4
n_units = 5
tf.random.set_seed(69)

#Create NN model
model_mse = StandardNNModel.NN_simple_hedge(n_assets = n_assets, input_dim = 1, 
                                        n_layers = n_layers, n_units = n_units, 
                                        activation = 'elu', final_activation = None, 
                                        output2_dim = 1)

model_mse.create_model(n, rate = 0, dt = T / n, transaction_costs = tc, init_pf = option_price, 
                   ignore_rates = True, ignore_minmax = True, ignore_info = True) 


#Create NN model with rm
model = StandardNNModel.NN_simple_hedge(n_assets = n_assets, input_dim = 1, 
                                        n_layers = n_layers, n_units = n_units, 
                                        activation = 'elu', final_activation = None, 
                                        output2_dim = 1)

model.create_model(n, rate = 0, dt = T / n, transaction_costs = tc, init_pf = 0, 
                    ignore_rates = True, ignore_minmax = True, ignore_info = True)

model.create_rm_model(alpha = alpha)

#Create NN model with rm and lower confidence level
model2 = StandardNNModel.NN_simple_hedge(n_assets = n_assets, input_dim = 1, 
                                        n_layers = n_layers, n_units = n_units, 
                                        activation = 'elu', final_activation = None, 
                                        output2_dim = 1)

model2.create_model(n, rate = 0, dt = T / n, transaction_costs = tc, init_pf = 0, 
                    ignore_rates = True, ignore_minmax = True, ignore_info = True)

model2.create_rm_model(alpha = alpha1)  

#Training models
train_models = True

best_model_name1 = "best_model_mse_2_{}.hdf5".format(exp_nr)
best_model_name2 = "best_model_rm_high_2_{}.hdf5".format(exp_nr)
best_model_name3 = "best_model_rm_low_2_{}.hdf5".format(exp_nr)

if train_models is True:
    #train mse model  
    model_mse.train_model(x, y, batch_size = 1024, epochs = 100, patience = [5,11], learning_rate = 0.01, best_model_name = best_model_name1)
    
    #train CVaR high model   
    model.train_rm_model(x, epochs = 100, batch_size = 1024, patience = [5,11], lr = 0.01, best_model_name = best_model_name2)
    
    #train CVaR low model   
    model2.train_rm_model(x, epochs = 100, batch_size = 1024, patience = [5,11], lr = 0.01, best_model_name = best_model_name3)
    

model_mse.model.load_weights(best_model_name1)
model.model_rm.load_weights(best_model_name2)
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
models = [s_model, model, model_mse, model2]
model_names = ["Analytical", "NN CVaR {}".format(alpha), "NN MSE", "NN CVaR {}".format(alpha1)]

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
        times = np.arange(n)/n
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
             label = "{} hs".format(model_names[0]))

   
        
    current_hs = np.array([0.5])
    min_spot = np.array(0)
    for m, name in zip(models[1:-1], model_names[1:-1]):
        plt.plot(np.arange(60,180)/100*s_model.S0,
                 [m.get_hs(time, current_hs, spot_tilde, rate, min_spot) for spot_tilde in tmp_spots_tilde], 
                 '--', label = "{} hs".format(name))
    
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

#option price and p0
print("Exp nr:",exp_nr)
print("NN MSE p0:", model_mse.get_init_pf())
print("NN CVaR{} p0:".format(alpha), model.get_init_pf() + model.get_J(x) * np.exp(-rate * T))
print("NN CVaR{} p0:".format(alpha1), model2.get_init_pf() + model2.get_J(x) * np.exp(-rate * T))
print("Option price:", option_price)

#Avg abs Pnl
for pnl, name in zip(Pnl, model_names):
    print("Avg abs PnL ({}):".format(name), np.round(np.mean(abs(pnl)),5), 
      '(',np.round(np.std(abs(pnl)),5),')',
      np.round(np.mean(abs(pnl))  / init_pf,5))

#Avg squared Pnl
for pnl, name in zip(Pnl, model_names):
    print("Avg squared PnL ({}):".format(name), np.round(np.mean(pnl**2),8))

#Avg Pbl
for pnl, name in zip(Pnl, model_names):
    print("Avg PnL ({}):".format(name), np.round(np.mean(pnl),8))

#Calculate CVAR high
for pnl, name in zip(Pnl, model_names):
    tmp_loss = - pnl
    tmp_cvar = np.mean(tmp_loss[np.quantile(tmp_loss, alpha) <= tmp_loss])
    print('Out of sample CVAR{} ({}):'.format(alpha, name),tmp_cvar)
    
#Calculate CVAR low
for pnl, name in zip(Pnl, model_names):
    tmp_loss = - pnl
    tmp_cvar = np.mean(tmp_loss[np.quantile(tmp_loss, alpha1) <= tmp_loss])
    print('Out of sample CVAR{} ({}):'.format(alpha1, name),tmp_cvar)

#Turnover
for por, name in zip(hedge_engine.ports, model_names):
    print('Avg. Turnover ({})'.format(name), np.mean(por.turnover, axis = 0))
    
#Avg transaction costs
for por, name in zip(hedge_engine.ports, model_names):
    print('Avg. Transaction Costs ({})'.format(name), np.mean(np.sum(por.tc_hist, axis = 1)))