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

n = 20
rate = 0.02
rate_change = 0
T = 1
tc = 0.0 #transaction cost


#Create binomial model
S0 = 1
#s_model = BinomialModel.Binom_model(S0, 0.1, 0.2, rate, 0.5, T/n, rate_change)
s_model = BlackScholesModel.BlackScholesModel(S0, 0.03, 0.2, rate, T / n)


#Create sample paths 
N = 14
n_samples = 2**N

s_model.reset_model(n_samples)
for j in range(n):
    s_model.evolve_s_b()

spots = s_model.spot_hist
banks = s_model.bank_hist
rates = s_model.rate_hist


spots_tilde = spots / banks

#Option
strike = 1

option = lambda spots: np.maximum(spots - strike,0)
#option = lambda spot: 0

option_price = s_model.init_option("call", [strike,T])

#Get option payoffs from samples
option_payoffs = option(spots[:,-1])[:,np.newaxis]

#Setup x and y
x = [spots_tilde[:,i:(i+1)]  for i in range(n+1)] \
    + [rates[:,i:(i+1)]  for i in range(n)] \
    + [banks[:,-1:],option_payoffs]
    
y = np.zeros(shape = (n_samples,1))

#Create NN model
model_mse = StandardNNModel.NN_simple_hedge(input_dim = 4, 
                                        base_output_dim = 5, base_n_layers = 2, base_n_units = 4, 
                                        n_layers = 2, n_units = 5, 
                                        activation = 'elu', final_activation = 'sigmoid')
model_mse.create_model(n, rate = 0, dt = T / n, transaction_costs = tc, init_pf = option_price, 
                   ignore_rates = True) #option_price

#Train model
for epochs, lr in zip([50,50,50,50],[1e-2, 1e-3, 1e-4, 1e-5]):
    print(epochs,lr)
    model_mse.train_model1(x, y, batch_size = 256, epochs = epochs, max_lr = [lr], min_lr = lr, step_size = epochs)

model_mse.model.load_weights("best_model.hdf5")
model_mse.model.trainable = False

alpha = 0.95
model_mse.create_rm_model(alpha = alpha)

for epochs, lr in zip([10,10,20],[1e-2, 1e-3, 1e-4]):
    print(epochs,lr)
    model_mse.train_rm_model(x, epochs, batch_size = 256, lr = lr)

#Create NN model with rm
model = StandardNNModel.NN_simple_hedge(input_dim = 4, 
                                        base_output_dim = 5, base_n_layers = 2, base_n_units = 4, 
                                        n_layers = 2, n_units = 5, 
                                        activation = 'elu', final_activation = 'sigmoid')
model.create_model(n, rate = 0, dt = T / n, transaction_costs = tc, init_pf = 0, 
                   ignore_rates = True) #option_price

model.create_rm_model(alpha = alpha)

model.model_rm.load_weights("best_model_rm.hdf5")
#Train model
for epochs, lr in zip([50,50],[1e-4, 1e-5]):
    print(epochs,lr)
    model.train_rm_model(x, epochs, batch_size = 256, lr = lr)

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
anal_price = option_price

init_pf = anal_price

def calculate_hs(time, hs, spot_tilde, rate):
    if type(time) == int or type(time) == float:
        time = np.ones_like(hs) * float(time)
    
    tmp_x = np.column_stack([spot_tilde, rate, hs,time])
    return np.squeeze(model.get_hs(tmp_x))

 
N_hedge_samples = 5000
hedge_spots = []
option_values = []

s_model.reset_model(N_hedge_samples)

#create portfolios
models = [s_model, model_mse, model]
model_names = ["BS", "NN MSE", "NN Risk"]
ports = []

#matrices to store investment in underlying for nn and optimal
hs_matrix = []

for m in models:
    ports.append(PortfolioClass.Portfolio(0, init_pf, s_model, transaction_cost = tc))
    hs_matrix.append(np.zeros((N_hedge_samples, n)))

#init rebalance
for por, m in zip(ports,models):
    por.rebalance(m.get_current_optimal_hs(s_model, por.hs))

for i in range(n):
    #Save hs and time
    for por, hs_m in zip(ports, hs_matrix):
        hs_m[:,i] = por.hs 
    
    s_model.evolve_s_b()
    
    for por in ports:
        por.update_pf_value()
    
    if i < n - 1:
        for por, m in zip(ports,models):
            por.rebalance(m.get_current_optimal_hs(s_model, por.hs)) 

pf_values = [por.pf_value for por in ports]

hedge_spots = s_model.spot
option_values = option(hedge_spots)

#Plot of hedge accuracy
tmp_xs = np.linspace(0.5*S0,2*S0)
tmp_option_values = option(tmp_xs)

for pf_vals, name in zip(pf_values, model_names):
    plt.plot(tmp_xs, tmp_option_values)
    plt.scatter(hedge_spots,pf_vals, color = 'black', s = 2)
    plt.title("Hedge Errors:" + name)
    plt.show()


#Pnl
Pnl = [np.array(pf_val) - np.array(option_values) for pf_val in pf_values]

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
for i in range(5):
    times = np.arange(n)/n
    for hs_m, name in zip(hs_matrix, model_names):
        plt.plot(times, hs_m[i,:], label = name)
    plt.legend()
    plt.show()

#Plot worst
for pnl, name in zip(Pnl, model_names):
    for hs_m, name2 in zip(hs_matrix, model_names): 
        plt.plot(times, hs_m[np.argmin(pnl),:], label = name2)
    plt.legend()
    plt.title('Worst {} Pnl'.format(name))
    plt.show()

#Plot hs at time T*0.8 over different spots
plt.plot(np.arange(60,120)/100,[s_model.get_optimal_hs(0.8*T,x/100) for x in np.arange(60,120)],
         label = "{} hs".format(model_names[0]))
for m, name in zip(models[1:], model_names[1:]):
    plt.plot(np.arange(60,120)/100,[m.get_hs(0.8*T,0.5,x/100, rate) for x in np.arange(60,120)], 
             label = "{} hs".format(name))

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
    

###########################################################################################
###########################################################################################
            ###########################################################################################
            ###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################

# =============================================================================
# N_hedge_samples = 5000
# pf_values = []
# pf_anal_values = []
# hedge_spots = []
# option_values = []
# 
# #matrices to store investment in underlying for nn and optimal
# hs_matrix = np.zeros((N_hedge_samples,n))
# hs_anal_matrix = np.zeros_like(hs_matrix)
# 
# port_nn = PortfolioClass.Portfolio(0, init_pf, s_model, transaction_cost = tc)
# port_nn.rebalance(model.get_current_optimal_hs(s_model, port_nn.hs))
# 
# port_anal = PortfolioClass.Portfolio(0, init_pf, s_model, transaction_cost = tc)
# port_anal.rebalance(s_model.get_current_optimal_hs(s_model, port_nn.hs))
# 
# for i in range(n):
#     #Save hs and time
#     hs_matrix[:,i] = port_nn.hs
#     hs_anal_matrix[:,i] = port_anal.hs
#     
#     #
#     s_model.evolve_s_b()
#     port_nn.update_pf_value()
#     port_anal.update_pf_value()
#     if i < n - 1:
#         port_nn.rebalance(model.get_current_optimal_hs(s_model, port_nn.hs)) 
#         port_anal.rebalance(s_model.get_current_optimal_hs(s_model, port_nn.hs))
# 
# pf_values = port_nn.pf_value
# pf_anal_values = port_anal.pf_value
# 
# hedge_spots = s_model.spot
# option_values = option(hedge_spots)
# 
# #Plot of hedge accuracy
# tmp_xs = np.linspace(0.5*S0,2*S0)
# tmp_option_values = option(tmp_xs)
# plt.plot(tmp_xs, tmp_option_values)
# plt.scatter(hedge_spots,pf_values, color = 'black', s = 2)
# plt.show()
# 
# plt.plot(tmp_xs, tmp_option_values)
# plt.scatter(hedge_spots,pf_anal_values, color = 'black', s = 2)
# plt.show()
# 
# #Pnl
# Pnl = np.array(pf_values) - np.array(option_values)
# 
# plt.scatter(hedge_spots,Pnl)
# plt.show()
# 
# Pnl_anal = np.array(pf_anal_values) - np.array(option_values)
# 
# plt.scatter(hedge_spots,Pnl_anal)
# plt.show()
# 
# print("Avg optimal abs PnL:", np.round(np.mean(abs(Pnl_anal)),5), 
#       '(',np.round(np.std(abs(Pnl_anal)),5),')',
#       np.round(np.mean(abs(Pnl_anal))  / init_pf,5))
# print("Avg NN abs PnL:", np.round(np.mean(abs(Pnl)),5), 
#       '(',np.round(np.std(abs(Pnl)),5),')',
#       np.round(np.mean(abs(Pnl)) / init_pf,5))
# 
# print("Avg squared optimal PnL:", np.round(np.mean(Pnl_anal**2),5))
# print("Avg squared NN PnL:", np.round(np.mean(Pnl**2),5))
# 
# 
# print("Avg optimal PnL:", np.round(np.mean(Pnl_anal),5))
# print("Avg NN PnL:", np.round(np.mean(Pnl),5))
# 
# #Plot hs from nn vs optimal
# for i in range(5):
#     times = np.arange(n)/n
#     plt.plot(times, hs_anal_matrix[i,:], label = 'Anal hedge')
#     plt.plot(times, hs_matrix[i,:],'--', label = 'NN hedge')
#     plt.legend()
#     plt.show()
# 
# #Plot worst nn
# plt.plot(times, hs_anal_matrix[np.argmin(Pnl),:], label = 'Anal hedge')
# plt.plot(times, hs_matrix[np.argmin(Pnl),:],'--', label = 'NN hedge')
# plt.legend()
# plt.title('Worst NN Pnl')
# plt.show()
# 
# #Plot worst anal
# plt.plot(times, hs_anal_matrix[np.argmin(Pnl_anal),:], label = 'Anal hedge')
# plt.plot(times, hs_matrix[np.argmin(Pnl_anal),:],'--', label = 'NN hedge')
# plt.legend()
# plt.title("Worst Anal Pnl")
# plt.show()
# 
# plt.plot(np.arange(60,120)/100,[model.get_hs(0.8*T,0.5,x/100, rate) for x in np.arange(60,120)], 
#          color = "orange", label = "NN hs")
# plt.plot(np.arange(60,120)/100,[s_model.get_optimal_hs(0.8*T,x/100) for x in np.arange(60,120)],
#          label = "Anal hs")
# plt.legend()
# plt.show()
# 
# plt.plot(rate * np.linspace(0.5,2,200),[model.get_hs(0.8*T,0.5,1, rate * x) for x in np.linspace(0.5,2,200)])
# plt.show()
# 
# #Plot pnls on bar chart
# plt.hist([Pnl,Pnl_anal], bins = 40, histtype='bar', label=['NN','Anal'])
# plt.title('Out of sample Pnl distribution')
# plt.legend()
# plt.show()
# 
# #Calculate CVAR
# nn_loss = - Pnl
# nn_cvar = np.mean(nn_loss[np.quantile(nn_loss, alpha) <= nn_loss])
# print('NN Out of sample CVAR:',nn_cvar)
# 
# anal_loss = - Pnl_anal
# anal_cvar = np.mean(anal_loss[np.quantile(anal_loss, alpha) <= anal_loss])
# print('Anal Out of sample CVAR:',anal_cvar)
# 
# 
# =============================================================================
