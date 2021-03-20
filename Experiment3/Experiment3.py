# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:58:51 2021

@author: mrgna
"""

# Experiment 3.0 Black Scholes with 1 asset and no transaction costs

import os
import sys

sys.path.insert(1, os.path.dirname(os.getcwd()))

import numpy as np
import gc
from tqdm import tqdm


import helper_functions
import nearPD
import HedgeEngineClass
import OptionClass
import StandardNNModel
import BlackScholesModel
import tensorflow as tf
import matplotlib.pyplot as plt

tf.config.set_visible_devices([], 'GPU')


n = 60  # 60
rate = 0.02
rate_change = 0
T = 3/12
tc = 0.0  # transaction cost

alpha = 0.95  # confidence level for CVaR

# Create stock model
S0 = 1

run = "BS"

exp_nr = 1
train_models = True


# Exp 3.x settings
if exp_nr == 0:
    tc = 0
elif exp_nr == 1:
    tc = 0.005

if run == "BS":
    n_assets = 4

    np.random.seed(69)

    mu = np.round(np.random.uniform(low=0, high=0.1, size=n_assets), 2)
    sigma = np.round(np.random.uniform(0.05, 0.4, n_assets), 2)

    corr = np.ones((n_assets, n_assets))
    for i in range(n_assets):
        for j in range(i, n_assets):
            if not i == j:
                # print(i,j)
                tmp_cor = np.random.uniform(0, 1)
                corr[i, j] = tmp_cor
                corr[j, i] = tmp_cor

    corr = nearPD.nearPD(corr, 1000)

    s_model = BlackScholesModel.BlackScholesModel(
        1, mu, sigma, corr, rate, T / n, n_assets=n_assets)

    # Setup call option with strike 1
    n_options = 10
    options = [OptionClass.Option(np.random.choice(["call", "put"]),
                                  [np.round((1+0.3*np.random.uniform(-1, 1))*S0, 2),
                                   T, np.round(S0*np.random.uniform(0.9, 0.94), 2)],
                                  np.random.randint(0, n_assets)) for _ in range(n_options)]
    units = list(np.round(np.random.uniform(
        low=-1, high=2, size=n_options), 2))
    option_por = OptionClass.OptionPortfolio(options, units)

# Option por
option_price = s_model.init_option(option_por)

# Create sample paths
N = 18  # 18
n_samples = 2**N

np.random.seed(69420)
x, y, banks = helper_functions.generate_dataset(
    s_model, n, n_samples, option_por)

#Samples with 0 correlation
corr0 = np.identity(n_assets)
s_model.corr = corr0

np.random.seed(69420)
x0, y0, banks0 = helper_functions.generate_dataset(
    s_model, n, n_samples, option_por)

#Samples with 0.5 correlation
corr05 = 0.5 * corr + 0.5 * corr0
s_model.corr = corr05

np.random.seed(69420)
x05, y05, banks05 = helper_functions.generate_dataset(
    s_model, n, n_samples, option_por)

#reset corr
s_model.corr = corr

############
# NN models
############

n_layers = 4
n_units = 5
tf.random.set_seed(69)


# Create NN model with rm
model1 = StandardNNModel.NN_simple_hedge(n_assets=n_assets, input_dim=1,
                                        n_layers=n_layers, n_units=n_units,
                                        activation='elu', final_activation=None,
                                        output2_dim=1)

model1.create_model(n, rate=0, dt=T / n, transaction_costs=tc, init_pf=0,
                   ignore_rates=True, ignore_minmax=True, ignore_info=True)

model1.create_rm_model(alpha=alpha)

# Create NN model with rm - 0 correlation
model0 = StandardNNModel.NN_simple_hedge(n_assets=n_assets, input_dim=1,
                                        n_layers=n_layers, n_units=n_units,
                                        activation='elu', final_activation=None,
                                        output2_dim=1)

model0.create_model(n, rate=0, dt=T / n, transaction_costs=tc, init_pf=0,
                   ignore_rates=True, ignore_minmax=True, ignore_info=True)

model0.create_rm_model(alpha=alpha)

# Create NN model with rm - 0.5 correlation
model05 = StandardNNModel.NN_simple_hedge(n_assets=n_assets, input_dim=1,
                                        n_layers=n_layers, n_units=n_units,
                                        activation='elu', final_activation=None,
                                        output2_dim=1)

model05.create_model(n, rate=0, dt=T / n, transaction_costs=tc, init_pf=0,
                   ignore_rates=True, ignore_minmax=True, ignore_info=True)

model05.create_rm_model(alpha=alpha)

# Training models
best_model_name1 = "best_model1_rm_3_{}.hdf5".format(exp_nr)
best_model_name0 = "best_model0_rm_3_{}.hdf5".format(exp_nr)
best_model_name05 = "best_model05_rm_3_{}.hdf5".format(exp_nr)


if train_models is True:
    # train CVaR
    for x, model, name in zip([x,x0,x05],[model1,model0, model05], 
                              [best_model_name1, best_model_name0, best_model_name05]):
        model.train_rm_model(x, epochs=100, batch_size=1024, patience=[5, 11], lr=0.01,
                             best_model_name = name)

model1.model_rm.load_weights(best_model_name1)
model0.model_rm.load_weights(best_model_name0)
model05.model_rm.load_weights(best_model_name05)

#del x0 and x05
del x0, y0, banks0
del x05, y05, banks05
gc.collect()


# Look at RM-cvar prediction and empirical cvar
test = model1.model_rm.predict(x)
print('model w:', test[0, 1])
print('model_rm cvar:', model1.get_J(x))
 
# Test CVAR network
test1 = - model1.model.predict(x)
print('emp cvar (in sample):', np.mean(
    test1[np.quantile(test1, alpha) <= test1]))


def cvar(w): return w + np.mean(np.maximum(test1 - w, 0) / (1 - alpha))


xs = np.linspace(0, option_price*2, 10000)
plt.plot(xs, [cvar(x) for x in xs])
plt.show()
print('emp cvar2 (in sample):', np.min([cvar(x) for x in xs]))
print('w:', xs[np.argmin([cvar(x) for x in xs])])

# Find zero cvar init_pf
init_pf_nn = model1.get_init_pf() + model1.get_J(x) * np.exp(-rate * T)

# Hedge simulations with fitted model
init_pf = option_price

N_hedge_samples = 50000

# create portfolios
models = [s_model, model1, model0, model05]
model_names = ["Analytical", "NN CVaR {}".format(alpha), "NN0 CVaR {}".format(alpha), "NN05 CVaR {}".format(alpha)]

#models = [s_model]
#model_names = [run]

# create hedge experiment engine 
np.random.seed(420)
hedge_engine = HedgeEngineClass.HedgeEngineClass(
    n, s_model, models, option_por)

# run hedge experiment
np.random.seed(69)
hedge_engine.run_hedge_experiment(N_hedge_samples, init_pf, tc)

pf_values = hedge_engine.pf_values
hedge_spots = hedge_engine.hedge_spots
option_values = hedge_engine.option_values
Pnl = hedge_engine.Pnl_disc
hs_matrix = hedge_engine.hs_matrix


gc.collect()

#################
# Plots
#################

dpi = 500

# Plot hs from nn vs optimal
for i in range(1):
    for j in range(n_assets):
        times = np.arange(n)/n
        for hs_m, name, ls in zip(hs_matrix, model_names, ["-", "--", "--","--"]):
            plt.plot(times, hs_m[i, j, :], ls, label=name, lw=2)
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("Units of $S$")
        plt.savefig("ex3_{}_hsfull_asset_{}_sample_{}.eps".format(
            exp_nr, j, i), bbox_inches='tight')
        plt.show()
        plt.close()

for i in range(1):
    for j in range(n_assets):
        times = np.arange(n)/n
        for hs_m, name, ls in zip(hs_matrix[:2], model_names[:2], ["-", "--"]):
            plt.plot(times, hs_m[i, j, :], ls, label=name, lw=2)
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("Units of $S$")
        plt.savefig("ex3_{}_hs_asset_{}_sample_{}.eps".format(
            exp_nr, j, i), bbox_inches='tight')
        plt.show()
        plt.close()

        

# Plot pnls on bar chart
for i in range(1, len(model_names)):
    plt.hist(Pnl[0], bins=100, label=model_names[0], alpha=0.8, density=True)
    plt.hist(Pnl[i], bins=100, label=model_names[i], alpha=0.5, density=True)
    #plt.title('Out of sample Pnl distribution')
    plt.legend()
    plt.xlabel("PnL")
    plt.savefig("ex3_{}_oos_pnldist_{}.png".format(
        exp_nr, model_names[i]), dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.close()

############
# Calculations
###########

# option price and p0
print("NN Risk p0:", model1.get_init_pf() + model1.get_J(x) * np.exp(-rate * T))
#print("NN0 Risk p0:", model0.get_init_pf() + model0.get_J(x) * np.exp(-rate * T))
#print("NN0 Risk p0:", model05.get_init_pf() + model05.get_J(x) * np.exp(-rate * T))
print("Option price:", option_price)

# Avg abs Pnl
for pnl, name in zip(Pnl, model_names):
    print("Avg abs PnL ({}):".format(name), np.round(np.mean(abs(pnl)), 5),
          '(', np.round(np.std(abs(pnl)), 5), ')',
          np.round(np.mean(abs(pnl)) / init_pf, 5))

# Avg squared Pnl
for pnl, name in zip(Pnl, model_names):
    print("Avg squared PnL ({}):".format(name), np.round(np.mean(pnl**2), 8))

# Avg Pbl
for pnl, name in zip(Pnl, model_names):
    print("Avg PnL ({}):".format(name), np.round(np.mean(pnl), 8))

# Calculate CVAR high
for pnl, name in zip(Pnl, model_names):
    tmp_loss = - pnl
    tmp_cvar = np.mean(tmp_loss[np.quantile(tmp_loss, alpha) <= tmp_loss])
    print('Out of sample CVAR{} ({}):'.format(alpha, name), tmp_cvar)

# Turnover
for por, name in zip(hedge_engine.ports, model_names):
    print('Avg. Turnover ({})'.format(name), np.mean(por.turnover, axis=0))

# Avg transaction costs
for por, name in zip(hedge_engine.ports, model_names):
    print('Avg. Transaction Costs ({})'.format(
        name), np.mean(np.sum(por.tc_hist, axis=1)))

print("Options")
print("Type:", [o.name for o in options])
print("Underlying:", [o.underlying for o in options])
print("Strike:", [o.params[0] for o in options])
print("Units:", units)

# Run second part of experiment with changed correlation matrix


def get_avg_pnl(Pnl): return np.array(
    [np.round(np.mean(pnl), 5) for pnl in Pnl])


def get_avg_abs_pnl(Pnl): return np.array(
    [np.round(np.mean(np.abs(pnl)), 5) for pnl in Pnl])


def get_avg_sq_pnl(Pnl): return np.array(
    [np.round(np.mean(pnl**2), 8) for pnl in Pnl])


def get_oos_cvar(Pnl): return np.array(
    [np.round(np.mean((-pnl)[np.quantile(-pnl, alpha) <= (-pnl)]), 5) for pnl in Pnl])


original_avg_pnl = get_avg_pnl(Pnl)
original_avg_abs_pnl = get_avg_abs_pnl(Pnl)
original_avg_sq_pnl = get_avg_sq_pnl(Pnl)
original_oos_cvar = get_oos_cvar(Pnl)

N_runs = 500
N_hedge_samples = 1000
shocks = [0,0.05,0.1,0.15]

avg_pnl_shocks = np.zeros((len(shocks), len(models)))
avg_abs_pnl_shocks = np.zeros_like(avg_pnl_shocks)
avg_sq_pnl_shocks = np.zeros_like(avg_pnl_shocks)
oos_cvar_shocks = np.zeros_like(avg_pnl_shocks)

avg_pnl_shocks_std = np.zeros_like(avg_pnl_shocks)
avg_abs_pnl_shocks_std = np.zeros_like(avg_pnl_shocks)
avg_sq_pnl_shocks_std = np.zeros_like(avg_pnl_shocks)
oos_cvar_shocks_std = np.zeros_like(avg_pnl_shocks)

np.random.seed(69*2)
for index, shock in enumerate(shocks):
    
    print(index,shock)
    
    tmp_avg_pnl = np.zeros((N_runs, len(models)))
    tmp_avg_abs_pnl = np.zeros_like(tmp_avg_pnl)
    tmp_avg_sq_pnl = np.zeros_like(tmp_avg_pnl)
    tmp_oos_cvar = np.zeros_like(tmp_avg_pnl)
    
    for k in tqdm(range(N_runs)):
        #print(k)
        # get new corr
        new_corr = np.array(corr)
        for i in range(n_assets):
            for j in range(i, n_assets):
                if not i == j:
                    tmp = np.random.normal() * shock
                    new_corr[i, j] += tmp
                    #new_corr[i, j] = np.random.uniform(0,1)
                    
                    new_corr[i, j] = np.minimum(new_corr[i, j], 1)
                    new_corr[j, i] = new_corr[i, j]
    
        new_corr = np.real(nearPD.nearPD(new_corr, 1000))
        
        #print(np.mean(np.abs(new_corr - corr)), np.mean(new_corr - corr))
        
        # change corr in hedge-engine
        s_model.corr = new_corr
    
        hedge_engine = HedgeEngineClass.HedgeEngineClass(
            n, s_model, models, option_por)
    
        # run hedge experiment
        hedge_engine.run_quick_hedge_exp(N_hedge_samples, init_pf, tc)
    
        # get tmp Pnl
        tmp_Pnl = hedge_engine.Pnl_disc
    
        # save tmp_Pnl
        tmp_avg_pnl[k, :] = get_avg_pnl(tmp_Pnl)
        tmp_avg_abs_pnl[k, :] = get_avg_abs_pnl(tmp_Pnl)
        tmp_avg_sq_pnl[k, :] = get_avg_sq_pnl(tmp_Pnl)
        tmp_oos_cvar[k, :] = get_oos_cvar(tmp_Pnl)
        
        
        gc.collect()
    
    avg_pnl_shocks[index,:] = np.mean(tmp_avg_pnl, axis = 0)
    avg_abs_pnl_shocks[index,:] = np.mean(tmp_avg_abs_pnl, axis = 0)
    avg_sq_pnl_shocks[index,:] = np.mean(tmp_avg_sq_pnl, axis = 0)
    oos_cvar_shocks[index,:] = np.mean(tmp_oos_cvar, axis = 0)
    
    avg_pnl_shocks_std[index,:] = np.std(tmp_avg_pnl, axis = 0)
    avg_abs_pnl_shocks_std[index,:] = np.std(tmp_avg_abs_pnl, axis = 0)
    avg_sq_pnl_shocks_std[index,:] = np.std(tmp_avg_sq_pnl, axis = 0)
    oos_cvar_shocks_std[index,:] = np.std(tmp_oos_cvar, axis = 0)
    
       
    
    # reset s_model
    s_model.corr = corr
    
oos_cvar_shocks_se = oos_cvar_shocks_std / np.sqrt(N_runs)

print("CVAR shocks:", oos_cvar_shocks)
print("CVAR shocks se", oos_cvar_shocks_se)