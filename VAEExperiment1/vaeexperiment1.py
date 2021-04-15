# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 12:02:29 2021

@author: mrgna
"""

#VAE Experiment 1 Black Scholes with 1 asset w/wo transaction costs

import os
import sys
sys.path.insert(1,os.path.dirname(os.getcwd()))

import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import BlackScholesModel
import MarketGenerator

import StandardNNModel
import OptionClass
import HedgeEngineClass
import helper_functions

n = 20 
rate = 0.01
rate_change = 0
T = 1/12

alpha = 0.95 #confidence level for CVaR

#Create stock model
S0 = 1

train_models = True

exp_nr = 0

#Exp 2.x settings
if exp_nr == 0:
    tc = 0

#Create model for underlying
n_assets = 1
mu = np.array([0.05])
sigma = np.array([0.3])

s_model = BlackScholesModel.BlackScholesModel(S0, mu, sigma, np.ones((1,1)), rate, T / n)

#Setup call option with strike 1
units = [1]
options = [OptionClass.Option("call",[S0,T],0)]
option_por = OptionClass.OptionPortfolio(options,units)

#Option por
option_price = s_model.init_option(option_por)

#Create sample paths 
N = 15
n_samples = 2**N

np.random.seed(69)
x, y, banks = helper_functions.generate_dataset(s_model, n, n_samples, option_por)
bank_hist, rate_hist = (s_model.bank_hist, s_model.rate_hist)

#Create function to create MG and MG-samples

def create_MG(N, plot = True):
    MG = MarketGenerator.MarketGenerator(s_model, n)
    MG.create_vae(latent_dim = n, layers_units=[2*n])
    MG.create_training_path(N, overlap = True)
    MG.train_vae()
    
    if plot is True:
        MG.qq_plot_fit()
        MG.plot_generated_paths(100)
        
    return MG

def generate_MG_data(MG, n_samples):
    MG.generate_paths(n_samples,save = True)
    x_MG, y_MG, _ = helper_functions.generate_dataset_from_MG(MG, bank_hist, rate_hist, option_por)
    
    return x_MG, y_MG

############
## NN models
############

n_layers = 4 #4
n_units = 5 #5
tf.random.set_seed(69)


#Create NN model with rm
model = StandardNNModel.NN_simple_hedge(n_assets = n_assets, 
                                        n_layers = n_layers, n_units = n_units, 
                                        activation = 'elu', final_activation = "sigmoid", 
                                        output2_dim = 1)

model.create_model(n, rate = 0, dt = T / n, transaction_costs = tc, init_pf = 0)

model.create_rm_model(alpha = alpha)

#Create NN models for MG data
N_MG_samples = int(n*(1/T)*1 - n + 1) # 2 years of data with n samples in T time.
n_MG_models = 2
MGs = []
MG_models = []
best_model_name_MGs = []
MG_model_names = ["NN CVaR MG{}".format(i) for i in range(n_MG_models)]
MG_train_losses = []

for i in range(n_MG_models):
    # create MG
    MGs.append(create_MG(N_MG_samples))
    
    #create MG model
    tmp_model_MG = StandardNNModel.NN_simple_hedge(n_assets = n_assets, 
                                            n_layers = n_layers, n_units = n_units, 
                                            activation = 'elu', final_activation = "sigmoid", 
                                            output2_dim = 1)
    tmp_model_MG.create_model(n, rate = 0, dt = T / n, transaction_costs = tc, init_pf = 0)
    tmp_model_MG.create_rm_model(alpha = alpha)
    MG_models.append(tmp_model_MG)
    
    best_model_name_MGs.append("best_model_rm_MG{}_{}.hdf5".format(i, exp_nr))
    
#Training models
best_model_name = "best_model_rm_normal_{}.hdf5".format(exp_nr)


if train_models is True:

    #train CVaR model   
    model.train_rm_model(x, epochs = 100, batch_size = 1024, patience = [5,11], lr = 0.01, best_model_name = best_model_name)

    
    #train MG CVaR model
    for MG, MG_model, name in zip(MGs, MG_models, best_model_name_MGs):
        tmp_x_MG, _ = generate_MG_data(MG, n_samples)
        MG_model.train_rm_model(tmp_x_MG, epochs = 100, batch_size = 1024, patience = [5,11], lr = 0.01, best_model_name = name)
        

model.model_rm.load_weights(best_model_name)
for model_MG, name in zip(MG_models,best_model_name_MGs):
    model_MG.model_rm.load_weights(name)  

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

N_hedge_samples = 100000

#create portfolios
models = [s_model, model] + MG_models
model_names = ["Analytical", "NN CVaR Ordinary"] + MG_model_names


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
spot_hist = hedge_engine.spot_hist
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
        plt.savefig("MGexp1_{}_hedge_acc_{}.png".format(exp_nr, name), dpi = dpi, bbox_inches='tight')
        plt.show()
        plt.close()

if n_assets == 1:
    for pnl, name in zip(Pnl, model_names):
        plt.scatter(hedge_spots,pnl, color = "k", label = "{} PnL".format(name), s= 3, alpha = 0.2)
        plt.legend()
        plt.xlabel("$S_N$")
        #plt.title("Pnl scatter:" + name)
        plt.savefig("MGexp1_{}_model_pnl_{}.png".format(exp_nr, name), dpi = dpi, bbox_inches='tight')
        plt.show()
        plt.close()
    
#Plot hs from nn vs optimal
for i in range(2):
    for j in range(n_assets):
        times = np.arange(n)/n
        for hs_m, name, ls in zip(hs_matrix, model_names, ["-","--","--","--"]):
            plt.plot(times, hs_m[i,j,:], ls, label = name, lw = 2)
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("Units of $S$")
        plt.savefig("MGexp1_{}_hs_sample_{}.eps".format(exp_nr,i), bbox_inches='tight')
        plt.show()
        plt.close()
        
#Plot hs at some time over different spots
time = 0.5*T
time_idx = int(np.round(time / s_model.dt , 6))
if n_assets == 1:
    tmp_spots = np.arange(60,180)/100*s_model.S0
    tmp_spots_tilde = tmp_spots * np.exp(- rate * time)
    
    tmp_model_hs = s_model.get_optimal_hs(time,tmp_spots[:,np.newaxis])
    plt.plot(tmp_spots, tmp_model_hs,
             label = "{} hs".format(model_names[0]))
     
    current_hs = np.array([0.5])
    current_pf = np.array([0.0])
    min_spot = np.array(0)
    spot_return = np.array(0)
    for m, name in zip(models[1:], model_names[1:]):
        plt.plot(np.arange(60,180)/100*s_model.S0,
                 [m.get_hs(time, current_hs, current_pf, spot_tilde, rate, min_spot, spot_return) for spot_tilde in tmp_spots_tilde], 
                 '--', label = "{} hs".format(name))
    
    hs_range = np.max(tmp_model_hs) - np.min(tmp_model_hs)
    plt.ylim(np.min(tmp_model_hs) - hs_range * 0.2, np.max(tmp_model_hs) + hs_range * 0.2)
    plt.legend()
    #plt.title("Holding in $S$ at time $t = {}$".format(time))
    plt.xlabel("$S({})$".format(time))
    plt.savefig("MGexp1_{}_hs_time_x.eps".format(exp_nr), bbox_inches='tight')
    plt.show()
    plt.close()

#Plot pnls on bar chart
for i in range(1,len(model_names)):
    plt.hist(Pnl[0], bins = 100, label=model_names[0],alpha = 0.8, density = True)
    plt.hist(Pnl[i], bins = 100, label=model_names[i], alpha = 0.5, density = True)
    #plt.title('Out of sample Pnl distribution')
    plt.legend()
    plt.xlabel("PnL")
    plt.savefig("MGexp1_{}_oos_pnldist_{}.png".format(exp_nr, model_names[i]), dpi = dpi, bbox_inches='tight')
    plt.show()
    plt.close()

############
#Calculations
###########

save_output = True
folder_name = ""

#Print outputs
if save_output is True:
    text_file = open(folder_name + "output MG exp 1_{}, hedge points {}, tc = {}, samples = 2_{}.txt".format(exp_nr,n,tc,N),"w")

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
tmp_p0 = model.get_init_pf() + model.get_J(x) * np.exp(-rate * T)
print_overload("NN Ordinary p0:", tmp_p0, tmp_p0 / option_price * 100)
tmp_p0 = model_MG.get_init_pf() + model_MG.get_J(x) * np.exp(-rate * T)
print_overload("NN MG p0:", tmp_p0, tmp_p0 / option_price * 100)


#Avg abs Pnl
for pnl, name in zip(Pnl, model_names):
    print_overload("Avg abs PnL ({}):".format(name), np.round(np.mean(abs(pnl)),5), 
      '(',np.round(np.std(abs(pnl)) / np.sqrt(N_hedge_samples),8),')',
      np.round(np.mean(abs(pnl))  / option_price * 100,5))

#Avg squared Pnl
for pnl, name in zip(Pnl, model_names):
    print_overload("Avg squared PnL ({}):".format(name), np.round(np.mean(pnl**2),8),
    '(',np.round(np.std(pnl**2) / np.sqrt(N_hedge_samples),8),')')

#Avg Pbl
for pnl, name in zip(Pnl, model_names):
    print_overload("Avg PnL ({}):".format(name), np.round(np.mean(pnl),8),
    '(', np.round(np.std(pnl) / np.sqrt(N_hedge_samples),8),')',
      np.round(np.mean(pnl) / option_price * 100,5))

#Calculate CVAR
for pnl, name in zip(Pnl, model_names):
    tmp_loss = - pnl
    tmp_cvar = np.mean(tmp_loss[np.quantile(tmp_loss, alpha) <= tmp_loss])
    print_overload('Out of sample CVAR{} ({}):'.format(alpha, name),tmp_cvar)
    
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
