# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 10:34:41 2021

@author: mrgna
"""

#Experiment 4.0 Heston model with 1 underlying and trading a call option

import os
import sys
sys.path.insert(1,os.path.dirname(os.getcwd()))

import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import HestonModel

import StandardNNModel
import OptionClass
import HedgeEngineClass
import nearPD
import helper_functions

n = 20
rate = 0.02
T = 1/12

alpha = 0.95 #confidence level for CVaR

#Create stock model
S0 = 1

run = "Heston"
exp_nr = 0 ####### Set 0 for exp 2.0 and 1 for exp 2.1
train_models = True

#Exp 2.x settings
if exp_nr == 0:
    tc = 0
    ignore_sa = False
elif exp_nr == 1:
    tc = 0.005
    ignore_sa = False
elif exp_nr == 2:
    tc = 0
    ignore_sa = True #ignores second asset

if run == "Heston":
    np.random.seed(69)
    
    s_model = HestonModel.HestonModel(S0, mu = 0.03, v0 = 0.09, kappa = 2, theta = 0.09, sigma = 0.3, 
                                      rho = -0.9, rate = rate, dt = T / n, ddt = 0.01, ignore_sa = ignore_sa)
    
    n_true_assets = s_model.n_true_assets
    n_assets = s_model.n_assets
    
    n_options = 1
    options = [OptionClass.Option("call",[S0,T],0)]
    units = [1]
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

n_layers = 4
n_units = 5
tf.random.set_seed(69)

#Create NN model with rm
model = StandardNNModel.NN_simple_hedge(n_assets = n_assets, input_dim = 1, 
                                        n_layers = n_layers, n_units = n_units, 
                                        activation = 'elu', final_activation = None, 
                                        output2_dim = 1)

model.create_model(n, rate = 0, dt = T / n, transaction_costs = tc, init_pf = 0, 
                    ignore_rates = True, ignore_minmax = True, ignore_info = True)

model.create_rm_model(alpha = alpha)

#Training models

best_model_name2 = "best_model_rm_4_{}.hdf5".format(exp_nr)

if train_models is True:
    #train CVaR high model   
    model.train_rm_model(x, epochs = 100, batch_size = 1024, patience = [5,11], lr = 0.01, best_model_name = best_model_name2)
    
model.model_rm.load_weights(best_model_name2)
 

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
models = [s_model, model]
model_names = ["Analytical", "NN CVaR {}".format(alpha)]

#models = [s_model]
#model_names = [run]

#create hedge experiment engine
np.random.seed(420)
hedge_engine = HedgeEngineClass.HedgeEngineClass(n, s_model, models, option_por)

#run hedge experiment
np.random.seed(69)
hedge_engine.run_hedge_experiment(N_hedge_samples, init_pf, tc)

pf_values = hedge_engine.pf_values
hedge_spots = hedge_engine.hedge_spots[:,0]
option_values = hedge_engine.option_values
Pnl = hedge_engine.Pnl_disc
hs_matrix = hedge_engine.hs_matrix


#################
#Plots
#################

dpi = 500


#Plot of hedge accuracy
tmp_xs = np.linspace(0.5*S0,2*S0)
tmp_option_values = option_por.get_portfolio_payoff(tmp_xs[:,np.newaxis,np.newaxis])

for pf_vals, name in zip(pf_values, model_names):
    plt.plot(tmp_xs, tmp_option_values, label = 'Option payoff')
    plt.scatter(hedge_spots,pf_vals, color = 'black', s = 2, label = "{} $PF_N$".format(name), alpha = 0.3)
    plt.xlabel("$S_N$")
    plt.legend()
    #plt.title("Hedge Errors:" + name)
    plt.savefig("ex4_{}_hedge_acc_{}.png".format(exp_nr, name), dpi = dpi, bbox_inches='tight')
    plt.show()
    plt.close()

#Plot pnl
for pnl, name in zip(Pnl, model_names):
    plt.scatter(hedge_spots,pnl, color = "k", label = "{} PnL".format(name), s= 3, alpha = 0.2)
    plt.legend()
    plt.xlabel("$S_N$")
    #plt.title("Pnl scatter:" + name)
    plt.savefig("ex4_{}_model_pnl_{}.png".format(exp_nr, name), dpi = dpi, bbox_inches='tight')
    plt.show()
    plt.close()
    
#Plot hs from nn vs optimal
for i in range(2):
    for j in range(n_assets):
        times = np.arange(n)/n
        for hs_m, name, ls in zip(hs_matrix, model_names, ["-","--","--"]):
            plt.plot(times, hs_m[i,j,:], ls, label = name, lw = 2)
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("Units of $S$")
        plt.savefig("ex4_{}_hs_sample_{}.eps".format(exp_nr,i), bbox_inches='tight')
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
    plt.savefig("ex4_{}_hs_time_x.eps".format(exp_nr), bbox_inches='tight')
    plt.show()
    plt.close()

#Plot pnls on bar chart
for i in range(1,len(model_names)):
    plt.hist(Pnl[0], bins = 100, label=model_names[0],alpha = 0.8, density = True)
    plt.hist(Pnl[i], bins = 100, label=model_names[i], alpha = 0.5, density = True)
    #plt.title('Out of sample Pnl distribution')
    plt.legend()
    plt.xlabel("PnL")
    plt.savefig("ex4_{}_oos_pnldist_{}.png".format(exp_nr, model_names[i]), dpi = dpi, bbox_inches='tight')
    plt.show()
    plt.close()

############
#Calculations
###########

#option price and p0
print("Exp nr:",exp_nr)
print("NN CVaR{} p0:".format(alpha), model.get_init_pf() + model.get_J(x) * np.exp(-rate * T))
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
    

#Turnover
for por, name in zip(hedge_engine.ports, model_names):
    print('Avg. Turnover ({})'.format(name), np.mean(por.turnover, axis = 0))
    
#Avg transaction costs
for por, name in zip(hedge_engine.ports, model_names):
    print('Avg. Transaction Costs ({})'.format(name), np.mean(np.sum(por.tc_hist, axis = 1)))