# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 13:25:54 2021

@author: Magnus Frandsen
"""

import numpy as np 
import matplotlib.pyplot as plt

import BinomialModel
import StandardNNModel

n = 10
rate = 0.05
rate_change = 0.05 #- rate * 2 / n
T = 1
tc = 0.00 #transaction cost


#Create binomial model
S0 = 1
bin_model = BinomialModel.Binom_model(S0, 0.1, 0.2, rate, 0.5, 1/n, rate_change)


#Create sample paths 
N = 12
n_samples = 2**N

spots = np.zeros((n_samples, n+1))
banks = np.zeros_like(spots)
rates = np.zeros_like(spots)

for i in range(n_samples):
    bin_model.reset_model()
    
    spots[i,0] = bin_model.spot
    banks[i,0] = bin_model.bank
    rates[i,0] = bin_model.rate
    
    for j in range(n):
        bin_model.evolve_s_b()
        spots[i,j+1] = bin_model.spot
        banks[i,j+1] = bin_model.bank
        rates[i,j+1] = bin_model.rate

spots_tilde = spots / banks

#Option
strike = 1

option = lambda spot: max([spot - strike,0])
#option = lambda spot: 0

opt_hedge = BinomialModel.Optimal_hedge(T, bin_model)
tmp_option_payoffs = [option(S0*opt_hedge.u**(opt_hedge.n-i) * opt_hedge.d**(i)) for i in range(opt_hedge.n+1)]
opt_hedge.calculate_hedge_price(tmp_option_payoffs)

option_price = opt_hedge.price[0,0]

#Get option payoffs from samples
option_payoffs = np.array([option(spot) for spot in spots[:,-1]])

#Setup x and y
x = [spots_tilde[:,i:(i+1)]  for i in range(n+1)] \
    + [rates[:,i:(i+1)]  for i in range(n)] \
    + [banks[:,-1:],option_payoffs[:,np.newaxis]]
y = np.zeros(shape = (n_samples,1))

#Create NN model
model = StandardNNModel.NN_simple_hedge(input_dim = 4,n_layers = 4, n_units = 5, 
                                        activation = 'elu', final_activation = 'sigmoid')
model.create_model(n, rate = 0, dt = T / n, mult_submodels = True, n_pr_submodel = 1, 
                   transaction_costs = tc, init_pf = 0, ignore_rates = True) #option_price

#Train model
model.compile_model()
# =============================================================================
# model.model.trainable = True
# model.train_model1(x, y, batch_size = 32, epochs = 100,
#                   max_lr = [0.01, 0.001, 1e-4, 1e-5], min_lr = 1e-5, step_size = 25)
# 
# model.model.load_weights("best_model.hdf5")
# model.model.trainable = False
# =============================================================================


#Play with RM model
import tensorflow as tf

alpha = 0.9

model.create_rm_model(alpha = alpha)
print(int(np.sum([tf.keras.backend.count_params(p) for p in set(model.model_rm.trainable_weights)])))


#model.compile_rm_w_target_loss()
for epochs, lr in zip([50,100,100,100],[1e-2, 1e-3, 1e-4, 1e-5]):
    print(epochs,lr)
    model.train_rm_model(x, epochs, batch_size = 256, lr = lr)

model.model_rm.load_weights("best_model.hdf5")

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


#Hedge simulations with fitted model

init_pf_nn = model.get_init_pf() + model.get_J(x) * np.exp(-rate * T)

pnl_0p0 = model.model.predict(x) 
pnl_from_p0 = lambda p0: pnl_0p0 + banks[:,-1] * p0

def cvar_from_p0(p0): 
    tmp_loss = - pnl_from_p0(p0)
    return np.mean(tmp_loss[np.quantile(tmp_loss,alpha) <= tmp_loss])

def binary_search(func,a,b, delta = 1e-6):
    a = a
    b = b
    tmp_delta = b - a

    while tmp_delta > delta:
        m = (b-a) / 2 + a
        f_m = func(m)
        
        if f_m < 0:
            b = m
        else:
            a = m
        
        tmp_delta= b - a
        
    return (b-a)/2 + a
    
init_pf_nn2 = binary_search(cvar_from_p0,init_pf_nn * 0, init_pf_nn * 2, delta = 1e-8)

print("init_pf1:",init_pf_nn)
print("init_pf2:",init_pf_nn2)

anal_price = option_price

init_pf = init_pf_nn2


def calculate_hs(time, hs, spot_tilde, rate):
    tmp_x = np.array([[spot_tilde, rate, hs,time]])
    #print(tmp_x)
    return model.get_hs(tmp_x)[0,0]

def calculate_hs_anal(time, spot):
    time_idx = int(np.round(time / T * n , 6))
    spot_place = np.argmax(abs(opt_hedge.St[:,time_idx] - spot) <= 1e-6)
    if np.count_nonzero(abs(opt_hedge.St[:,time_idx] - spot) <= 1e-6) != 1:
        print("FUCK", time, spot, time_idx)
    return opt_hedge.hs[spot_place,time_idx]

N_hedge_samples = 2000
pf_values = []
pf_anal_values = []
hedge_spots = []
option_values = []

#matrices to store investment in underlying for nn and optimal
hs_matrix = np.zeros((N_hedge_samples,n))
hs_anal_matrix = np.zeros_like(hs_matrix)

for h in range(N_hedge_samples):
    bin_model.reset_model()
    port_nn = BinomialModel.Portfolio(0, init_pf, bin_model, transaction_cost = tc)
    port_nn.rebalance(calculate_hs(bin_model.time, port_nn.hs, bin_model.spot / bin_model.bank, bin_model.rate))
    
    port_anal = BinomialModel.Portfolio(0, init_pf, bin_model, transaction_cost = tc)
    port_anal.rebalance(calculate_hs_anal(bin_model.time, bin_model.spot))
    
    for i in range(n):
        #Save hs and time
        hs_matrix[h,i] = port_nn.hs
        hs_anal_matrix[h,i] = port_anal.hs
        
        #
        bin_model.evolve_s_b()
        port_nn.update_pf_value()
        port_anal.update_pf_value()
        if i < n - 1:
            port_nn.rebalance(calculate_hs(bin_model.time, port_nn.hs, bin_model.spot / bin_model.bank, bin_model.rate))
            port_anal.rebalance(calculate_hs_anal(bin_model.time, bin_model.spot))
        
        #Save hs
        
    pf_values.append(port_nn.pf_value)
    pf_anal_values.append(port_anal.pf_value)
    
    hedge_spots.append(bin_model.spot)
    option_values.append(option(bin_model.spot))

#Plot of hedge accuracy
tmp_xs = np.linspace(0.5*S0,2*S0)
tmp_option_values = [option(i) for i in tmp_xs]
plt.plot(tmp_xs, tmp_option_values)
plt.scatter(hedge_spots,pf_values, color = 'black', s = 2)
plt.show()

plt.plot(tmp_xs, tmp_option_values)
plt.scatter(hedge_spots,pf_anal_values, color = 'black', s = 2)
plt.show()

#Pnl
Pnl = np.array(pf_values) - np.array(option_values)

plt.scatter(hedge_spots,Pnl)
plt.show()

Pnl_anal = np.array(pf_anal_values) - np.array(option_values)

plt.scatter(hedge_spots,Pnl_anal)
plt.show()

print("Avg optimal abs PnL:", np.round(np.mean(abs(Pnl_anal)),5), 
      '(',np.round(np.std(abs(Pnl_anal)),5),')',
      np.round(np.mean(abs(Pnl_anal))  / init_pf,5))
print("Avg NN abs PnL:", np.round(np.mean(abs(Pnl)),5), 
      '(',np.round(np.std(abs(Pnl)),5),')',
      np.round(np.mean(abs(Pnl)) / init_pf,5))

print("Avg optimal PnL:", np.round(np.mean(Pnl_anal),5))
print("Avg NN PnL:", np.round(np.mean(Pnl),5))

#Plot hs from nn vs optimal
for i in range(5):
    times = np.arange(n)/n
    plt.plot(times, hs_anal_matrix[i,:], label = 'Anal hedge')
    plt.plot(times, hs_matrix[i,:],'--', label = 'NN hedge')
    plt.legend()
    plt.show()

#Plot worst nn
plt.plot(times, hs_anal_matrix[np.argmin(Pnl),:], label = 'Anal hedge')
plt.plot(times, hs_matrix[np.argmin(Pnl),:],'--', label = 'NN hedge')
plt.legend()
plt.title('Worst NN Pnl')
plt.show()

#Plot worst anal
plt.plot(times, hs_anal_matrix[np.argmin(Pnl_anal),:], label = 'Anal hedge')
plt.plot(times, hs_matrix[np.argmin(Pnl_anal),:],'--', label = 'NN hedge')
plt.legend()
plt.title("Worst Anal Pnl")
plt.show()

plt.plot(np.arange(60,120)/100,[calculate_hs(0.8,0.5,x/100, rate) for x in np.arange(60,120)])
plt.show()

plt.plot(rate * np.linspace(0.5,2,200),[calculate_hs(0.8,0.5,1, rate * x) for x in np.linspace(0.5,2,200)])
plt.show()

#Plot pnls on bar chart
plt.hist([Pnl,Pnl_anal], bins = 10, histtype='bar', label=['NN','Anal'])
plt.title('Out of sample Pnl distribution')
plt.legend()
plt.show()

#Calculate CVAR
nn_loss = - Pnl
nn_cvar = np.mean(nn_loss[np.quantile(nn_loss, alpha) <= nn_loss])
print('NN Out of sample CVAR:',nn_cvar)

anal_loss = - Pnl_anal
anal_cvar = np.mean(anal_loss[np.quantile(anal_loss, alpha) <= anal_loss])
print('Anal Out of sample CVAR:',anal_cvar)