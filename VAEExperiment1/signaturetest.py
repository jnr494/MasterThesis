# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 18:19:37 2021

@author: mrgna
"""

import os
import sys
import time
sys.path.insert(1,os.path.dirname(os.getcwd()))

import numpy as np
import matplotlib.pyplot as plt

import iisignature
import BlackScholesModel


model = BlackScholesModel.BlackScholesModel(1,0.05, 0.3, np.array([[1]]), 0.01, dt = 0.01)

n = 20
for _ in range(n):
    model.evolve_s_b()
    
path = np.squeeze(model.spot_hist)[np.newaxis,:]

plt.plot(path[0])
plt.show()

def leadlag(path):
    repeat = np.repeat(np.squeeze(path),2)
    lead = repeat[1:]
    lag = repeat[:-1]
    
    leadlag = np.vstack((lead,lag)).T
    return leadlag

def path_from_lead_lag(leadlag_path,n):
    return leadlag_path.T[0][::2]
    

leadlag_path = leadlag(path)
test_path = path_from_lead_lag(leadlag_path,n)

for p in leadlag_path.T:
    plt.plot(p)
plt.show()

order = 10
iisignature.siglength(1, order)
iisignature.sig(path.T,order)

iisignature.siglength(2, order)
iisignature.sig(leadlag_path,order)

#logsig
order = 4
m = iisignature.logsiglength(2, order)
prep = iisignature.prepare(2, order)
t0 = time.time()
logsig = iisignature.logsig(leadlag_path,prep)
t1 = time.time()
print(np.round(t1-t0))


def MSE(x,y):
    return np.sqrt(np.mean((x-y)**2))

def MSE_y_deriv(x,y):
    return -2 / len(x) * (x - y) / MSE(x,y)

def absdif(x,y):
    return np.mean(np.abs((x-y) / x) / (1+np.arange(len(x))))

def absdif_y_deriv(x,y):
    x_y = (x-y) / x
    len_x = len(x)
    return - 1/x * np.sign(x_y) / (1+np.arange(len_x)) / len_x

def leadlag_deriv_to_path_deriv(leadlag_deriv,n):
    derivs = np.zeros((2*(n+1),2))
    derivs[:-1,1] = leadlag_deriv[:,1]
    derivs[1:,0] = leadlag_deriv[:,0]
    return np.sum(derivs.reshape((n+1,4)),axis = 1)

new_path = np.ones_like(path[0])

lr = 0.001
epochs = 100
losses = []



plt.plot(path[0])
plt.plot(new_path)
plt.show()

new_ll_path = leadlag(new_path)
new_logsig = iisignature.logsig(new_ll_path, prep)
new_loss = absdif(logsig, new_logsig)
for i in range(epochs):
    old_loss = new_loss
    new_logsig = iisignature.logsig(new_ll_path, prep)
    new_loss = absdif(logsig, new_logsig)
# =============================================================================
#     if new_loss > old_loss :
#         lr = max([lr / 10,0.01])
#         print(i,lr,new_loss,old_loss)
# =============================================================================
    
    derivs = absdif_y_deriv(logsig, new_logsig)
    new_ll_path_deriv = iisignature.logsigbackprop(derivs, new_ll_path, prep)
    new_path_deriv = leadlag_deriv_to_path_deriv(new_ll_path_deriv,n)
    
    new_path -= (lr / (abs(new_path_deriv)+1e-8))  * new_path_deriv #(lr / max(abs(new_path_deriv))) 
    new_ll_path = leadlag(new_path)
    print(i, new_loss)
    losses.append(new_loss)

new_path += - new_path[0] + path[0,0]
plt.plot(np.log(losses))
plt.show()

plt.plot(path[0])
plt.plot(new_path)
plt.show()

print(logsig)
print(new_logsig)
print((logsig - new_logsig)/logsig)

#decoder
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from sklearn.preprocessing import StandardScaler

def create_decoder(output_dim, latent_dim, layers_units, activation = "elu", final_activation = None):
    latent_inputs = keras.Input(shape=(latent_dim,))
    
    x = layers.Dense(layers_units[0], activation = activation)(latent_inputs)
    
    if len(layers_units) > 1:
        for units in layers_units[1:]:
            x = layers.Dense(units, activation = activation)(x)
            
    decoder_outputs = layers.Dense(output_dim, activation = final_activation)(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    
    return decoder

decoder = create_decoder(n+1,m, [10]*10)

#create training data
n_samples = 100000
model.reset_model(n_samples)
for _ in range(n):
    model.evolve_s_b()

paths = np.squeeze(model.spot_hist)

logsigs = np.empty((n_samples,m))
for i in range(n_samples):
    ll_path = leadlag(paths[i])
    logsigs[i] = iisignature.logsig(ll_path,prep)

scaler = StandardScaler()
scaler.fit(logsigs)
logsigs_norm = scaler.transform(logsigs)

decoder.compile(optimizer = "adam", loss = "mse")
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor= 0.1, patience=5, min_lr=1e-9, verbose = 2)
er = keras.callbacks.EarlyStopping(monitor = 'loss', patience = 11, verbose = 1)
decoder.fit(logsigs_norm,paths,epochs = 100, batch_size = 32,callbacks = [er,reduce_lr])

new_path = decoder.predict(scaler.transform(logsig[np.newaxis,:]))

plt.plot(path[0])
plt.plot(new_path[0])
plt.show()

#train again
new_paths = np.concatenate((paths,decoder.predict(logsigs)),axis = 0)
new_logsigs = np.empty((len(new_paths),m))

for i in range(len(new_paths)):
    ll_path = leadlag(new_paths[i])
    new_logsigs[i] = iisignature.logsig(ll_path,prep)

new_logsigs_norm = scaler.transform(new_logsigs)

decoder.compile(optimizer = "adam", loss = "mse")
decoder.fit(new_logsigs_norm,new_paths,epochs = 100, batch_size = 32,callbacks = [er,reduce_lr])

new_path = decoder.predict(scaler.transform(logsig[np.newaxis,:]))

plt.plot(path[0])
plt.plot(new_path[0])
plt.show()

new_ll_path = leadlag(new_path)
new_logsig = iisignature.logsig(new_ll_path, prep)

print(logsig)
print(new_logsig)


#global optimizer
model = BlackScholesModel.BlackScholesModel(1,0.05, 0.3, np.array([[1]]), 0.01, dt = 0.01)

n = 20
for _ in range(n):
    model.evolve_s_b()
    
path = np.squeeze(model.spot_hist)
leadlag_path = leadlag(path)

order = 4
m = iisignature.logsiglength(2, order)
prep = iisignature.prepare(2, order)
t0 = time.time()
logsig = iisignature.logsig(leadlag_path,prep)
t1 = time.time()
print(np.round(t1-t0))

def error(x,y):
    return np.mean(np.abs((x-y)/x) / (1+np.arange(len(logsig))))

def returns_2_logsig_error(returns, logsig, prep, n):
    new_path = np.ones((n+1))
    new_path[1:] = np.cumprod(returns + 1)
    
    new_ll_path = leadlag(new_path)
    new_logsig = iisignature.logsig(new_ll_path, prep)
    
    error_ = error(new_logsig, logsig)
    return error_

def return_2_path(returns):
    new_path = np.ones((n+1))
    new_path[1:] = np.cumprod(returns + 1)
    return new_path

def path_2_logsig_error(new_path, logsig, prep, n):
    new_ll_path = leadlag(new_path)
    new_logsig = iisignature.logsig(new_ll_path, prep)
    
    error_ = error(new_logsig, logsig)
    return error_

def path_from_points(points , n):
    path = np.ones((n+1))
    
    m = len(points)
    inc = n // (m-1)
    
    for i in range(m-2):
        path[i*inc:(i+1)*inc+1] = np.linspace(points[i],points[i+1],inc+1)
    
    inc2 = int(np.round(n / (m-1),0))
    path[-inc2:] = np.linspace(points[-2],points[-1], inc2)
    return path
    
    
    

#f = lambda returns: returns_2_logsig_error(returns, logsig, prep, n)
#f = lambda path: path_2_logsig_error(path, logsig, prep, n)

f = lambda points: path_2_logsig_error(path_from_points(points,n), logsig, prep, n)


from scipy.optimize import differential_evolution
n_points = 2
bounds = [(0.8,1.2) for _ in range(n_points)]
result = differential_evolution(f,bounds, maxiter = 1000)

#new_path = return_2_path(result.x)
new_path = path_from_points(result.x,n)
new_path += - new_path[0] + path[0]
plt.plot(path)
plt.plot(new_path)
plt.show()

new_ll_path = leadlag(new_path)
new_logsig = iisignature.logsig(new_ll_path, prep)

print(logsig)
print(new_logsig)
print((logsig - new_logsig)/logsig)
