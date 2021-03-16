# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 14:12:33 2021

@author: mrgna
"""

import os
import sys
sys.path.insert(1,os.path.dirname(os.getcwd()))

import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')


import BinomialModel

import StandardNNModel
import OptionClass
import HedgeEngineClass
import nearPD

n = 10
rate = 0.0
rate_change = 0
T = 1
tc = 0.0 #transaction cost


#Create stock model
S0 = 100

run = "Bin"

#Setup model
np.random.seed(69)
s_model = BinomialModel.Binom_model(S0, 0.03, 0.2, rate, 0.5, T/n, rate_change)
n_assets = s_model.n_assets

#setup option
option_type = "call"
strike = 100

options = [OptionClass.Option(np.random.choice([option_type]), [strike,T],0)]
units = [1]
option_por = OptionClass.OptionPortfolio(options,units)

#Option por
option_price = s_model.init_option(option_por)

#Create sample paths 
N = 18
n_samples = 2**N

s_model.reset_model(n_samples)
for j in range(n):
    s_model.evolve_s_b()

spots = s_model.spot_hist
banks = s_model.bank_hist
rates = s_model.rate_hist

spots_tilde = spots / banks[:,np.newaxis,:]

#extra infomation (for pathdependence)
min_spots = np.minimum.accumulate(spots,axis = -1)

#Get option payoffs from samples
option_payoffs = option_por.get_portfolio_payoff(spots)[:,np.newaxis]

#Setup x and y
x = [spots_tilde[...,i]  for i in range(n+1)] \
    + [rates[:,i:(i+1)]  for i in range(n)] \
    + [min_spots[...,i] for i in range(n)] \
    + [banks[:,-1:],option_payoffs]
    
x = np.column_stack(x)
    
y = np.zeros(shape = (n_samples,1))


alpha = 0.9

#Create NN model
model_mse = StandardNNModel.NN_simple_hedge(n_assets = n_assets, input_dim = 1, 
                                        n_layers = 3, n_units = 4, 
                                        activation = 'elu', final_activation = None, 
                                        output2_dim = 1)

model_mse.create_model(n, rate = 0, dt = T / n, transaction_costs = tc, init_pf = None, 
                   ignore_rates = True, ignore_minmax = True, ignore_info = True) 

#Train model
tf.random.set_seed(69)
model_mse.train_model(x, y, batch_size = 1024, epochs = 100, patience = [5,11], learning_rate = 0.01)

model_mse.model.load_weights("best_model.hdf5")

#trained init price
model_price = model_mse.get_init_pf()

#Hedge simulations with fitted model
init_pf = option_price

N_hedge_samples = 10000

#create portfolios
models = [s_model, model_mse]
model_names = [run, "NN MSE"]

#create hedge experiment engine
hedge_engine = HedgeEngineClass.HedgeEngineClass(n, s_model, models, option_por)

#run hedge experiment
np.random.seed(420)
hedge_engine.run_hedge_experiment(N_hedge_samples, init_pf, tc)

pf_values = hedge_engine.pf_values
hedge_spots = hedge_engine.hedge_spots
option_values = hedge_engine.option_values
Pnl = hedge_engine.Pnl
hs_matrix = hedge_engine.hs_matrix

#Output
#Option price and learned price
print("Option price:", option_price, "Learned price:", model_price, "Dif:", option_price - model_price)

#Avg abs Pnl
for pnl, name in zip(Pnl, model_names):
    print("Avg abs PnL ({}):".format(name), np.round(np.mean(abs(pnl)),5), 
      '(',np.round(np.std(abs(pnl)),5),')',
      np.round(np.mean(abs(pnl))  / init_pf,5))

#Avg squared Pnl
for pnl, name in zip(Pnl, model_names):
    print("Avg squared PnL ({}):".format(name), np.round(np.mean(pnl**2),5))

#Avg Pbl
for pnl, name in zip(Pnl, model_names):
    print("Avg PnL ({}):".format(name), np.round(np.mean(pnl),5))

#Calculate CVAR
for pnl, name in zip(Pnl, model_names):
    tmp_loss = - pnl
    tmp_cvar = np.mean(tmp_loss[np.quantile(tmp_loss, alpha) <= tmp_loss])
    print('Out of sample CVAR ({}):'.format(name),tmp_cvar)

#Turnover
for por, name in zip(hedge_engine.ports, model_names):
    print('Avg. Turnover ({})'.format(name), np.mean(por.turnover, axis = 0))
    
#Avg transaction costs
for por, name in zip(hedge_engine.ports, model_names):
    print('Avg. Transaction Costs ({})'.format(name), np.mean(np.sum(por.tc_hist, axis = 1)))
    
#######
#PLOTS
######

#Plot of hedge accuracy
tmp_xs = np.linspace(0.5*S0,2*S0)
tmp_option_values = option_por.get_portfolio_payoff(tmp_xs[:,np.newaxis,np.newaxis])

for pf_vals, name in zip(pf_values[1:], model_names[1:]):
    plt.plot(tmp_xs, tmp_option_values, zorder = 0, label = "Option payoff")
    plt.scatter(hedge_spots,pf_vals, color = 'black', s = 3, zorder = 10, label = "NN Model $PF_N$")
    #plt.title("Hedge Accuracy")
    plt.xlabel("$S_N$")
    plt.legend()
    plt.savefig("ex1_hedge_acc.eps", bbox_inches='tight')
    plt.show()
    plt.close()

for pnl, name in zip(Pnl[1:], model_names[1:]):
    plt.scatter(hedge_spots,pnl, label = "NN Model PnL", s = 3, color = "black")
    #plt.title("NN Model Pnl")
    plt.legend()
    plt.xlabel("$S_N$")
    plt.savefig("ex1_model_pnl.eps", bbox_inches='tight')
    plt.show()
    plt.close()
    
#Plot hs from nn vs optimal
for i in range(2):
    for j in range(n_assets):
        times = np.arange(n)/n
        plt.scatter(times, hs_matrix[0][i,j,:], label = "Analytical")
        plt.plot(times, hs_matrix[1][i,j,:], '--', label = "NN model", color = 'k')
        plt.legend()
        #plt.title("Units in $S$, Sample: {} ".format(i))
        plt.xlabel("time")
        plt.ylabel("Units of $S$")
        plt.savefig("ex1_hs_sample_{}.eps".format(i), bbox_inches='tight')
        plt.show()
        plt.close()

#Plot hs at some time over different spots
time = 0.6*T
time_idx = time_idx = int(np.round(time / s_model.dt , 6))
if n_assets == 1:
    tmp_spots = np.arange(60,180)/100*s_model.S0
    tmp_spots_tilde = tmp_spots * np.exp(- rate * time)
    
    tmp_spots_bin = s_model.optimal_hedge.St[:(time_idx+1), time_idx]
    tmp_model_hs = s_model.get_optimal_hs(time, tmp_spots_bin)
    plt.scatter(tmp_spots_bin, tmp_model_hs,
                label = "{}".format("Analytical"))   
        
    current_hs = np.array([0.5])
    min_spot = np.array(0)
    for m, name in zip(models[1:], ["NN Model"]):
        plt.plot(np.arange(60,180)/100*s_model.S0,
                 [m.get_hs(time, current_hs, spot_tilde, rate, min_spot) for spot_tilde in tmp_spots_tilde], 
                 '--', color = 'black', label = "{}".format(name))
    
    hs_range = np.max(tmp_model_hs) - np.min(tmp_model_hs)
    plt.ylim(np.min(tmp_model_hs) - hs_range * 0.2, np.max(tmp_model_hs) + hs_range * 0.2)
    plt.legend()
    #plt.title("Holding in $S$ at time $t = {}$".format(time))
    plt.xlabel("$S({})$".format(time))
    plt.savefig("ex1_hs_time_x.eps", bbox_inches='tight')
    plt.show()
    plt.close()
