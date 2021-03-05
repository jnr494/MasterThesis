# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 13:25:54 2021

@author: Magnus Frandsen
"""

import numpy as np 
import matplotlib.pyplot as plt
import copy

import BinomialModel
import BlackScholesModel

import StandardNNModel
import PortfolioClass
import OptionClass
import HedgeEngineClass
import nearPD

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

n = 10
rate = 0.1
rate_change = 0
T = 1
tc = 0.01 #transaction cost


#Create stock model
S0 = 1

run = "BS"


if run == "BS":
    n_assets = 5
    
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
    
    s_model = BlackScholesModel.BlackScholesModel(1, mu, sigma, corr, 0.02, T / n, n_assets = n_assets)
    
    n_options = 10
    options = [OptionClass.Option(np.random.choice(["call","put"]), 
                                  [(1+0.5*T*np.random.uniform(-1,1))*S0,T],np.random.randint(0,n_assets)) for _ in range(n_options)]
    units = list(np.random.uniform(low = -5, high = 5, size = n_options))
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
    
else:
    raise ValueError("Plase choose implemented model") 

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

#Option
option_price = s_model.init_option(option_por)

#Get option payoffs from samples
option_payoffs = option_por.get_portfolio_payoff(spots[...,-1])[:,np.newaxis]

#Setup x and y
x = [spots_tilde[...,i]  for i in range(n+1)] \
    + [rates[:,i:(i+1)]  for i in range(n)] \
    + [banks[:,-1:],option_payoffs]
    
x = np.column_stack(x)
    
y = np.zeros(shape = (n_samples,1))

#Create NN model
alpha = 0.9
model_mse = StandardNNModel.NN_simple_hedge(n_assets = n_assets, input_dim = 1, 
                                        n_layers = 3, n_units = 4, 
                                        activation = 'elu', final_activation = None)
model_mse.create_model(n, rate = 0, dt = T / n, transaction_costs = tc, init_pf = option_price, 
                   ignore_rates = True) 

#Train model
for epochs, lr in zip([100],[1e-2]):
    print(epochs,lr)
    model_mse.train_model(x, y, batch_size = 1024, epochs = epochs, patience = [5,11], learning_rate = lr)

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
                                        n_layers = 3, n_units = 4, 
                                        activation = 'elu', final_activation = None)
model.create_model(n, rate = 0, dt = T / n, transaction_costs = tc, init_pf = 0, 
                    ignore_rates = True)

model.create_rm_model(alpha = alpha)

#model.model_rm.load_weights("best_model_rm.hdf5")
#Train model
for epochs, lr in zip([100],[1e-2]):
    print(epochs,lr)
    model.train_rm_model(x, epochs, batch_size = 1024, patience = [5,11], lr = lr)

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

def binary_search(func,a,b, delta = 1e-6):
    a = a
    b = b
    tmp_delta = b - a

    while tmp_delta > delta:
        m = (b-a) / 2 + a
        f_m = func(m)
        
        #print(a,b,m,f_m)
        
        if f_m < 0:
            b = m
        else:
            a = m
        
        tmp_delta= b - a
        
    return (b-a)/2 + a
    
init_pf_nn2 = model.get_init_pf() + binary_search(cvar_from_p0,init_pf_nn * 0, init_pf_nn * 2, delta = 1e-6)

print("init_pf1:",init_pf_nn)
print("init_pf2:",init_pf_nn2)


#Hedge simulations with fitted model
init_pf = option_price

N_hedge_samples = 20000

#create portfolios
models = [s_model, model, model_mse]
model_names = [run, "NN Risk", "NN MSE"]

# =============================================================================
# s_model.reset_model(N_hedge_samples)
# ports = []
# 
# #matrices to store investment in underlying for nn and optimal
# hs_matrix = []
# 
# for m in models:
#     ports.append(PortfolioClass.Portfolio(0, init_pf, s_model, transaction_cost = tc))
#     hs_matrix.append(np.zeros((N_hedge_samples, n_assets, n)))
# 
# #init rebalance
# for por, m in zip(ports,models):
#     por.rebalance(m.get_current_optimal_hs(s_model, por.hs))
# 
# for i in range(n):
#     #Save hs and time
#     for por, hs_m in zip(ports, hs_matrix):
#         hs_m[...,i] = por.hs 
#     
#     s_model.evolve_s_b()
#     
#     for por in ports:
#         por.update_pf_value()
#     
#     if i < n - 1:
#         for por, m in zip(ports,models):
#             por.rebalance(m.get_current_optimal_hs(s_model, por.hs)) 
# 
# pf_values = [por.pf_value for por in ports]
# 
# hedge_spots = s_model.spot
# option_values = option_por.get_portfolio_payoff(hedge_spots)
# Pnl
# Pnl = [np.array(pf_val) - np.array(option_values) for pf_val in pf_values]
# =============================================================================

#create hedge experiment engine
hedge_engine = HedgeEngineClass.HedgeEngineClass(n, s_model, models, option_por)
#run hedge experiment
hedge_engine.run_hedge_experiment(N_hedge_samples, init_pf, tc)

pf_values = hedge_engine.pf_values
hedge_spots = hedge_engine.hedge_spots
option_values = hedge_engine.option_values
Pnl = hedge_engine.Pnl
hs_matrix = hedge_engine.hs_matrix

if n_assets == 1:
    #Plot of hedge accuracy
    tmp_xs = np.linspace(0.5*S0,2*S0)[:,np.newaxis]
    tmp_option_values = option_por.get_portfolio_payoff(tmp_xs)
    
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
        
    current_hs = np.array(0.5)
    for m, name in zip(models[1:], model_names[1:]):
        plt.plot(np.arange(60,180)/100*s_model.S0,
                 [m.get_hs(time, current_hs, spot_tilde, rate) for spot_tilde in tmp_spots_tilde], 
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


