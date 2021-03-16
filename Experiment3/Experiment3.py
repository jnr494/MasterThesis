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
import helper_functions
import nearPD
import HedgeEngineClass
import OptionClass
import StandardNNModel
import BlackScholesModel
import tensorflow as tf
import matplotlib.pyplot as plt

tf.config.set_visible_devices([], 'GPU')


n = 10  # 60
rate = 0.02
rate_change = 0
T = 3/12
tc = 0.0  # transaction cost

alpha = 0.95  # confidence level for CVaR

# Create stock model
S0 = 1

run = "BS"

exp_nr = 0  # Set 0 for exp 2.0 and 1 for exp 2.1

# Exp 3.x settings
if exp_nr == 0:
    tc = 0

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
N = 16  # 18
n_samples = 2**N

x, y, banks = helper_functions.generate_dataset(
    s_model, n, n_samples, option_por)

############
# NN models
############

n_layers = 4
n_units = 5
tf.random.set_seed(69)


# Create NN model with rm
model = StandardNNModel.NN_simple_hedge(n_assets=n_assets, input_dim=1,
                                        n_layers=n_layers, n_units=n_units,
                                        activation='elu', final_activation=None,
                                        output2_dim=1)

model.create_model(n, rate=0, dt=T / n, transaction_costs=tc, init_pf=0,
                   ignore_rates=True, ignore_minmax=True, ignore_info=True)

model.create_rm_model(alpha=alpha)

# Training models
train_models = True

best_model_name = "best_model_rm_3_{}.hdf5".format(exp_nr)


if train_models is True:
    # train CVaR
    model.train_rm_model(x, epochs=100, batch_size=1024, patience=[5, 11], lr=0.01,
                         best_model_name=best_model_name)

model.model_rm.load_weights(best_model_name)


# Look at RM-cvar prediction and empirical cvar
test = model.model_rm.predict(x)
print('model w:', test[0, 1])
print('model_rm cvar:', model.get_J(x))

# Test CVAR network
test1 = - model.model.predict(x)
print('emp cvar (in sample):', np.mean(
    test1[np.quantile(test1, alpha) <= test1]))


def cvar(w): return w + np.mean(np.maximum(test1 - w, 0) / (1 - alpha))


xs = np.linspace(0, option_price*2, 10000)
plt.plot(xs, [cvar(x) for x in xs])
plt.show()
print('emp cvar2 (in sample):', np.min([cvar(x) for x in xs]))
print('w:', xs[np.argmin([cvar(x) for x in xs])])

# Find zero cvar init_pf
init_pf_nn = model.get_init_pf() + model.get_J(x) * np.exp(-rate * T)

# Hedge simulations with fitted model
init_pf = option_price

N_hedge_samples = 50000

# create portfolios
models = [s_model, model]
model_names = ["Analytical", "NN CVaR {}".format(alpha)]

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

#################
# Plots
#################

dpi = 500

# Plot hs from nn vs optimal
for i in range(1):
    for j in range(n_assets):
        times = np.arange(n)/n
        for hs_m, name, ls in zip(hs_matrix, model_names, ["-", "--", "--"]):
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
print("NN Risk p0:", model.get_init_pf() + model.get_J(x) * np.exp(-rate * T))
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

N_runs = 50
avg_pnl = np.zeros((N_runs, 2))
avg_abs_pnl = np.zeros_like(avg_pnl)
avg_sq_pnl = np.zeros_like(avg_pnl)
oos_cvar = np.zeros_like(avg_pnl)

N_hedge_samples = 10000

np.random.seed(69*2)
for shock in [0,0.02,0.04,0.06,0.08,0.1]:
    
    for k in range(N_runs):
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
        hedge_engine.run_hedge_experiment(N_hedge_samples, init_pf, tc)
    
        # get tmp Pnl
        tmp_Pnl = hedge_engine.Pnl_disc
    
        # save tmp_Pnl
        avg_pnl[k, :] = get_avg_pnl(tmp_Pnl)
        avg_abs_pnl[k, :] = get_avg_abs_pnl(tmp_Pnl)
        avg_sq_pnl[k, :] = get_avg_sq_pnl(tmp_Pnl)
        oos_cvar[k, :] = get_oos_cvar(tmp_Pnl)
    
    
    # reset s_model
    s_model.corr = corr
    print("Shock:",shock)
    for name, opnl, npnl in zip(["Avg pnl", "Avg abs pnl", "Avg squared pnl", "Oos CVAR"],
                                [original_avg_pnl, original_avg_abs_pnl,
                                    original_avg_sq_pnl, original_oos_cvar],
                                [avg_pnl, avg_abs_pnl, avg_sq_pnl, oos_cvar]):
        print(name, opnl, np.mean(npnl, axis=0), np.mean(npnl, axis=0) / opnl)
