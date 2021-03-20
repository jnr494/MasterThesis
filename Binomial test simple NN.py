# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 13:25:54 2021

@author: Magnus Frandsen
"""

import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')


import BinomialModel
import BlackScholesModel
import HestonModel

import StandardNNModel
import OptionClass
import HedgeEngineClass
import nearPD
import helper_functions

n = 5
rate = 0.0
rate_change = 0
T = 1/12
tc = 0.0 #transaction cost

alpha = 0.9 #confidence level for CVaR

#Create stock model
S0 = 1

run = "Heston"


if run == "BS":
    n_assets = 3
    
    np.random.seed(69)    
    
    mu = np.random.uniform(low = 0, high = 0.1, size = n_assets)
    sigma = np.random.uniform(0.05,0.4, n_assets)
    
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
    
    n_options = 5
    options = [OptionClass.Option(np.random.choice(["call","put"]), 
                                  [(1+0.05*T*np.random.uniform(-1,1))*S0,T,S0*np.random.uniform(0.9,0.94)],
                                  np.random.randint(0,n_assets)) for _ in range(n_options)]
    units = list(np.random.uniform(low = -5, high = 5, size = n_options))

# =============================================================================
#     units = [1]
#     options = [OptionClass.Option("docall",[S0,T,0.95*S0],0)]
# =============================================================================
    option_por = OptionClass.OptionPortfolio(options,units)

elif run == "Bin":
    np.random.seed(69)
    
    s_model = BinomialModel.Binom_model(S0, 0.05, 0.2, rate, 0.5, T/n, rate_change)
    
    n_assets = s_model.n_assets
    
    n_options = 1
    options = [OptionClass.Option(np.random.choice(["call","put"]), 
                                  [(1+0.5*T*np.random.uniform(-1,1))*S0,T],np.random.randint(0,n_assets)) for _ in range(n_options)]
    units = list(np.random.uniform(low = -5, high = 5, size = n_options))
    option_por = OptionClass.OptionPortfolio(options,units)

elif run == "Heston":
    np.random.seed(69)
    
    s_model = HestonModel.HestonModel(S0, mu = 0.03, v0 = 0.09, kappa = 2, theta = 0.09, sigma = 0.3, 
                                      rho = -0.9, rate = rate, dt = T / n, ddt = 0.01)
    
    n_true_assets = s_model.n_true_assets
    n_assets = s_model.n_assets
    
    n_options = 1
    options = [OptionClass.Option(np.random.choice(["call","put"]), 
                                  [(1+0.5*T*np.random.uniform(-1,1))*S0,T],np.random.randint(0,n_true_assets)) for _ in range(n_options)]
    units = list(np.random.uniform(low = -5, high = 5, size = n_options))
    option_por = OptionClass.OptionPortfolio(options,units)

else:
    raise ValueError("Please choose implemented model") 

#Option por
option_price = s_model.init_option(option_por)

#Create sample paths 
N = 15
n_samples = 2**N

x, y, banks = helper_functions.generate_dataset(s_model, n, n_samples, option_por)


#Create NN model
model_mse = StandardNNModel.NN_simple_hedge(n_assets = n_assets, input_dim = 1, 
                                        n_layers = 3, n_units = 6, 
                                        activation = 'elu', final_activation = None, 
                                        output2_dim = 1)

model_mse.create_model(n, rate = 0, dt = T / n, transaction_costs = tc, init_pf = option_price, 
                   ignore_rates = True, ignore_minmax = True, ignore_info = True) 

#Train model
model_mse.train_model(x, y, batch_size = 1024, epochs = 100, patience = [5,11], learning_rate = 0.01)

model_mse.model.load_weights("best_model.hdf5")
model_mse.model.trainable = False


# =============================================================================
# model_mse.create_rm_model(alpha = alpha)
# 
# for epochs, lr in zip([20,10],[1e-2]):
#     print(epochs,lr)
#     model_mse.train_rm_model(x, epochs, batch_size = 1024, lr = lr)
# =============================================================================

#Create NN model with rm
model = StandardNNModel.NN_simple_hedge(n_assets = n_assets, input_dim = 1, 
                                        n_layers = 3, n_units = 6, 
                                        activation = 'elu', final_activation = None, 
                                        output2_dim = 1)

model.create_model(n, rate = 0, dt = T / n, transaction_costs = tc, init_pf = 0, 
                    ignore_rates = True, ignore_minmax = True, ignore_info = True)

model.create_rm_model(alpha = alpha)


#Train model
model.train_rm_model(x, epochs = 100, batch_size = 1024, patience = [5,11], lr = 0.01)

model.model_rm.load_weights("best_model_rm.hdf5")

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

pnl_0p0 = model.model.predict(x) 
pnl_from_p0 = lambda p0, max_include: pnl_0p0[:max_include,:] + banks[:max_include,-1] * p0

def cvar_from_p0(p0): 
    tmp_loss = - pnl_from_p0(p0, min([n_samples, int(2**12)]))
    return np.mean(tmp_loss[np.quantile(tmp_loss,alpha) <= tmp_loss])
    
init_pf_nn2 = model.get_init_pf() + helper_functions.binary_search(cvar_from_p0,init_pf_nn * 0, init_pf_nn * 2, delta = 1e-6)

print("init_pf1:",init_pf_nn)
print("init_pf2:",init_pf_nn2)


#Hedge simulations with fitted model
init_pf = option_price

N_hedge_samples = 10000

#create portfolios
models = [s_model, model, model_mse]
model_names = [run, "NN Risk", "NN MSE"]

#models = [s_model]
#model_names = [run]

#create hedge experiment engine
hedge_engine = HedgeEngineClass.HedgeEngineClass(n, s_model, models, option_por)
#run hedge experiment
np.random.seed(69)
hedge_engine.run_hedge_experiment(N_hedge_samples, init_pf, tc)

pf_values = hedge_engine.pf_values
hedge_spots = hedge_engine.hedge_spots
option_values = hedge_engine.option_values
Pnl = hedge_engine.Pnl
hs_matrix = hedge_engine.hs_matrix

if n_assets == 1:
    #Plot of hedge accuracy
    tmp_xs = np.linspace(0.5*S0,2*S0)
    tmp_option_values = option_por.get_portfolio_payoff(tmp_xs[:,np.newaxis,np.newaxis])
    
    for pf_vals, name in zip(pf_values, model_names):
        plt.plot(tmp_xs, tmp_option_values)
        plt.scatter(hedge_spots,pf_vals, color = 'black', s = 2)
        plt.title("Hedge Errors:" + name)
        plt.show()

if n_assets == 1:
    for pnl, name in zip(Pnl, model_names):
        plt.scatter(hedge_spots,pnl)
        plt.title("Pnl scatter:" + name)
        plt.show()

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
    
#Plot hs from nn vs optimal

for i in range(3):
    for j in range(n_assets):
        times = np.arange(n)/n
        for hs_m, name in zip(hs_matrix, model_names):
            plt.plot(times, hs_m[i,j,:], label = name)
        plt.legend()
        plt.title("Sample: {} Asset: {}".format(i,j))
        plt.show()

#Plot worst
for pnl, name in zip(Pnl, model_names):
    for i in range(n_assets):
        for hs_m, name2 in zip(hs_matrix, model_names): 
            plt.plot(times, hs_m[np.argmin(pnl),i,:], label = name2)
        plt.legend()
        plt.title('Worst {} Pnl, Asset: {}'.format(name,i))
        plt.show()

#tmp plot
worst = np.argmin(Pnl[0])
Pnl[0][worst]
plt.plot(times, s_model.spot_hist[worst,0,1:])
plt.show()

#Plot hs at some time over different spots
time = 0.5*T
time_idx = time_idx = int(np.round(time / s_model.dt , 6))
if n_assets == 1:
    tmp_spots = np.arange(60,180)/100*s_model.S0
    tmp_spots_tilde = tmp_spots * np.exp(- rate * time)
    
    if type(s_model) != BinomialModel.Binom_model:
        tmp_model_hs = s_model.get_optimal_hs(time,tmp_spots[:,np.newaxis])
        plt.plot(tmp_spots, tmp_model_hs,
                 label = "{} hs".format(model_names[0]))
    else:
        tmp_spots_bin = s_model.optimal_hedge.St[:(time_idx+1), time_idx]
        tmp_model_hs = s_model.get_optimal_hs(time, tmp_spots_bin)
        plt.scatter(tmp_spots_bin, tmp_model_hs,
                    label = "{} hs".format(model_names[0]))   
        
    current_hs = np.array([0.5])
    min_spot = np.array(0)
    for m, name in zip(models[1:], model_names[1:]):
        plt.plot(np.arange(60,180)/100*s_model.S0,
                 [m.get_hs(time, current_hs, spot_tilde, rate, min_spot) for spot_tilde in tmp_spots_tilde], 
                 label = "{} hs".format(name))
    
    hs_range = np.max(tmp_model_hs) - np.min(tmp_model_hs)
    plt.ylim(np.min(tmp_model_hs) - hs_range * 0.2, np.max(tmp_model_hs) + hs_range * 0.2)
    plt.legend()
    plt.show()

#Plot pnls on bar chart
for i in range(1,len(model_names)):
    plt.hist([Pnl[0],Pnl[i]], bins = 40, histtype='bar', label=[model_names[0],model_names[i]])
    plt.title('Out of sample Pnl distribution')
    plt.legend()
    plt.show()

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