# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 18:42:39 2021

@author: Magnus Frandsen
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Lambda, Input, BatchNormalization, Activation
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import gc
import copy
import matplotlib.pyplot as plt

gc.collect()
keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

def custom_mse(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true / 100, y_pred / 100)

def CLR(epoch: int, max_lrs: list,min_lr: float = 1e-6, step_size = 40) -> float:
    low_lr = min_lr
    
    super_state = int(np.floor(epoch / (2 * step_size)))
    if super_state <= len(max_lrs) - 1:
        max_lr = max_lrs[super_state]
    else:
        max_lr = max_lrs[-1]
    
    
    cycle = np.floor(1+epoch/(2*step_size))
    x = np.abs(epoch / step_size - 2*cycle + 1)
    lr = low_lr + (max_lr - low_lr) * np.max([0,1 - x])
    
    return lr 

def constmodel(input_dim, output_dim):
    inp = Input(shape = (input_dim,))
    out = Dense(output_dim, activation = None, use_bias = False)(inp[:,0:1] * 0 + 1)
    return Model(inputs = inp, outputs = out)

def cvar_loss_func(x, alpha):
    return tf.maximum(x,0) / (1 - alpha)

def rm_loss(y_true, y_pred):
    return y_pred[0,1] + tf.reduce_mean(y_pred[:,0])

def rm_target_loss(y_true, y_pred):
    return tf.abs(rm_loss(y_true, y_pred) - y_true[0,0])

def create_input_layers(n, n_assets):
    #Inputs
    spots = []
    for i in range(n+1):
        spots.append(Input(shape = (n_assets,),name ='Spot{}'.format(i)))
        
    rates = []
    for i in range(n):
        rates.append(Input(shape = (1,),name ='rate{}'.format(i)))
    
    bank_T = Input(shape = (1,),name ='Bank_T')
    
    option_payoff = Input(shape = (1,),name ='OptionPayoff')
    
    inputs = spots + rates + [bank_T, option_payoff]
    
    return spots, rates, bank_T, option_payoff, inputs
        

class NN_simple_hedge():
    def __init__(self, n_assets, input_dim, 
                 base_output_dim, base_n_layers, base_n_units, 
                 n_layers, n_units, 
                 activation = 'elu', final_activation = None):
        
        self.n_assets = n_assets
        self.input_dim = input_dim  + 2 * n_assets
        self.output_dim = n_assets
        
        
        self.base_output_dim = base_output_dim
        self.base_n_layers = base_n_layers
        self.base_n_units = base_n_units
        
        self.n_layers = n_layers
        self.n_units = n_units 
        self.activation = activation
        self.final_activation = final_activation
        
        #normalization constants
        self.m_x = 0
        self.s_x = 1

        self.create_submodel_calls = 0

    def create_submodel(self, input_dim, output_dim,
                        base_output_dim, base_n_layers, base_n_units, 
                        n_layers, n_units, 
                        activation, final_activation = None):
        
        if self.create_submodel_calls == 0:
            self.base_submodel = Sequential()
            self.base_submodel.add(Dense(n_units, activation = activation, input_dim = input_dim))
            for _ in range(base_n_layers-1):
                self.base_submodel.add(Dense(base_n_units, activation = activation)) 
            self.base_submodel.add(Dense(base_output_dim, activation = activation))
            self.base_submodel.summary()
                    
        inputs = Input(shape = (base_output_dim,))
        x = Dense(n_units, activation = activation)(inputs)
        for _ in range(n_layers - 1):
            x = Dense(n_units, activation = activation)(x)
        
        x = Dense(output_dim, activation = final_activation)(x)
        
        submodel = Model(inputs = inputs, outputs = x)
        submodel.summary()
        
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        submodel.compile(optimizer = optimizer, loss = 'mean_squared_error')
        
        self.create_submodel_calls += 1
        
        return submodel
        
# =============================================================================
#         submodel = Sequential()
#         submodel.add(Input(shape = (input_dim,)))
#         
#         #submodel.add(BatchNormalization())
#         submodel.add(Dense(n_units, activation = activation)) #, input_dim = input_dim
#         #submodel.add(BatchNormalization())
#         for _ in range(n_layers-1):
#             submodel.add(Dense(n_units, activation = activation))
#        #     submodel.add(BatchNormalization())  
#         submodel.add(Dense(output_dim, activation = final_activation))
#         
#         inputs = Input(shape = (input_dim,))
#         
#         submodel.summary()
#         optimizer = keras.optimizers.Adam(learning_rate=0.01)
#         submodel.compile(optimizer = optimizer, loss = 'mean_squared_error')
#         
#         return submodel
# =============================================================================
    
    def create_model(self, n, rate, dt, init_pf = None, transaction_costs = 0, 
                     ignore_rates = False):
        self.n = n
        self.T = n * dt
        self.transaction_costs = transaction_costs
        
        #ignore rates handling
        if ignore_rates is True:
            self.ignore_rates = 0
        else:
            self.ignore_rates = 1
        
        #init hs
        init_hs_comp = constmodel(self.base_output_dim, self.output_dim)
        
        #create submodels
        self.submodel = [init_hs_comp]
        for i in range(n -1):
            tmp_model = self.create_submodel(self.input_dim, self.output_dim,
                                             self.base_output_dim, self.base_n_layers, self.base_n_units,
                                             self.n_layers, self.n_units, 
                                             self.activation, self.final_activation)
            self.submodel.append(tmp_model) 

        #Inputs        
        spots, rates, bank_T, option_payoff, inputs = create_input_layers(self.n, self.n_assets)
        
        #pre-Computation
        R = tf.exp(float(rate) * dt)
        
        #dummy variables
        dummy_zeros = tf.stop_gradient(tf.math.multiply(bank_T,0))
        dummy_ones = tf.stop_gradient(tf.identity(dummy_zeros) + 1)
        
        dummy_zeros_expanded = tf.stop_gradient(tf.math.multiply(spots[0],0))
        
        #Tracking variables
        time = 0
        time_idx = int(np.round(time / self.T * self.n, 6))

        #lists of portfolios and portfolio values
        hs = []
        pf = []
        tc = []
        
        #Init pf
        if init_pf is None:
            self.init_pf_model = constmodel(1,1)
            init_pf = self.init_pf_model(dummy_zeros)
        else:
            init_pf = init_pf * dummy_ones
                
        #Computations        
        I1_0 = tf.concat([tf.math.log(spots[0] + 1), #log("Spots" +1)
                         rates[0] * self.ignore_rates, #Current rates
                         dummy_zeros_expanded, #Current hs
                         tf.identity(dummy_ones) * (tf.math.log(self.T - time + 1))], 1) #Log("Time to maturity" + 1)
        
        print(I1_0.shape, spots[0].shape, dummy_zeros_expanded.shape)
        hs_0 = self.submodel[time_idx](self.base_submodel(I1_0))
        
        tc_0 = tf.math.reduce_sum(tf.math.abs(hs_0) * spots[0], axis = 1, keepdims = True) * self.transaction_costs
        
        print("tc",tc_0.shape)
        print("init_pf",init_pf.shape)
        print((hs_0 * (spots[1] - spots[0])).shape)
        print(tf.math.reduce_sum(hs_0 * (spots[1] - spots[0] * R), axis = 1, keepdims = True).shape)
        pf_1 = tf.math.reduce_sum(hs_0 * (spots[1] - spots[0] * R), axis = 1, keepdims = True) + R * (init_pf - tc_0)
        
        print(0,pf_1.shape)
        
        time += dt
        time_idx = int(np.round(time / self.T * self.n, 6))
        
        tc.append(tc_0)
        hs.append(hs_0)
        pf.append(pf_1)
        
        if n > 1:
            for i in range(1,n):
                print(i,spots[i].shape, hs[-1].shape)
                tmp_I1 = tf.concat([tf.log(spots[i] + 1), #Spots
                                    rates[i] * self.ignore_rates, #current rates
                                    hs[-1], #Current hs
                                    tf.identity(dummy_ones)  * (tf.log(self.T - time + 1))], 1) #Log("Time to maturity" + 1)
                
                tmp_hs = self.submodel[time_idx](self.base_submodel(tmp_I1))

                tmp_tc = tf.math.reduce_sum(tf.math.abs(tmp_hs - hs[-1]) * spots[i],axis = 1, keepdims = True) * self.transaction_costs
                tmp_pf = tf.math.reduce_sum(tmp_hs * (spots[i+1] - spots[i]), axis = 1, keepdims = True) + R * (pf[-1] - tmp_tc)
                print(i,tmp_hs.shape, tmp_tc.shape, pf[-1].shape, tmp_pf.shape)
                time += dt
                time_idx = int(np.round(time / self.T * self.n, 6))
                
                hs.append(tf.identity(tmp_hs))
                pf.append(tf.identity(tmp_pf))
                tc.append(tmp_tc)
        
        
        total_pf = - option_payoff + bank_T * pf[-1]
        print(total_pf.shape)
        #Full model
        self.model_full = Model(inputs = inputs, outputs = [hs, pf, tc, init_pf])
        
        
        #Model for training
        self.model = Model(inputs = inputs, outputs = total_pf)
        #self.model.summary()
        self.compile_model()
        
        return self.model
    
    def create_rm_model(self, alpha = 0.95):
        #Risk measure model for training 
    
        #inputs
        spots, rates, bank_T, option_payoff, inputs = create_input_layers(self.n, self.n_assets)
        
        w = constmodel(1,1)(tf.stop_gradient(spots[0]))
        J = cvar_loss_func(- self.model(inputs) - w, alpha)
        rm_output = tf.concat([J,w], 1)
        self.model_rm = Model(inputs = inputs, outputs = rm_output)
        self.model_rm.compile(optimizer = "adam", loss = rm_loss)
        
    
    def compile_rm_w_target_loss(self):
        self.model_rm.compile(optimizer = "adam", loss = rm_target_loss)
    
    def compile_model(self, loss = 'mean_squared_error'):
        self.model.compile(optimizer = "adam", loss = loss) #'mean_squared_error')
    
    def train_model1(self, x, y, epochs, batch_size = 32, max_lr = [0.1,0.01], min_lr = 0.001, step_size = 40, 
                     best_model_name = "best_model.hdf5"):
        temp_CLR = lambda epoch: CLR(epoch,max_lr,min_lr, step_size)
        lr_schd = LearningRateScheduler(temp_CLR)
        mcp_save = ModelCheckpoint(best_model_name, save_best_only=True, monitor='loss', mode='min')
        callbacks = [lr_schd, mcp_save]
        
        #train
        self.model.fit(x, y, epochs = epochs, batch_size = batch_size, callbacks = callbacks, verbose = 2)
        
    def train_model2(self, x, y, epochs, batch_size = 32, patience = 10, learning_rate = 0.01, best_model_name = "best_model.hdf5") :                
        #train
        
        
        tmp_lr_lambda = lambda epoch: learning_rate
        lr_schd = LearningRateScheduler(tmp_lr_lambda)
        mcp_save = ModelCheckpoint(best_model_name, save_best_only=True, monitor='loss', mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor= 0.1, patience=5, min_lr=1e-7, verbose = 2)
        er = EarlyStopping(monitor = 'loss', patience = 11)
        callbacks = [er, mcp_save, reduce_lr]
        
        #train
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)
        self.model.fit(x, y, epochs = epochs, batch_size = batch_size, callbacks = callbacks)
        
    def find_optimal_lr(self, x, y, batch_size = 32, iterations = 100, min_lr = 1e-6, max_lr = 0.1, sma = 10):
        b = np.exp(np.log(max_lr / min_lr) / (iterations - 1))
        a = min_lr / (b**0)
        lr_lambda = lambda it: a * b**it 

        n = len(y)
        
        losses = []
        for i in range(iterations):
            lr = lr_lambda(i)
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
            
            batch_idx = np.random.choice(range(n),size = batch_size, replace = False)
            new_x = [tmp_x[batch_idx,:] for tmp_x in x]
        
            tmp_loss = self.model.train_on_batch(new_x, y[batch_idx,:])
            losses.append(tmp_loss)
            
        
        plt.plot(np.arange(0,iterations), losses)
        plt.show()
        
# =============================================================================
#         sma = 5
#         derivs = []
#         for i in range(sma-1, iterations):
#             derivs.append((losses[i] - losses[i - sma + 1]) / sma)
#         
#         plt.plot(np.arange(sma - 1,iterations), derivs)
#         plt.show()
#         
#         return losses, derivs, lr_lambda
# =============================================================================
        
        return losses, lr_lambda
    
    def get_J(self, x):
        rm_outputs = self.model_rm.predict(x)
        
        return rm_outputs[0,1] + np.mean(rm_outputs[:,0])
    
    def train_rm_model(self, x, epochs = 1, batch_size = 32, lr = 0.01, best_model_name = "best_model_rm.hdf5", reduce_lr = False):  
        temp_CLR = lambda epoch:lr
        lr_schd = LearningRateScheduler(temp_CLR)
        mcp_save = ModelCheckpoint(best_model_name, save_best_only=True, monitor='loss', mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor= 0.1, patience=5, min_lr=1e-7, verbose = 2)
        er = EarlyStopping(monitor = 'loss', patience = 11)
        callbacks = [er, mcp_save, reduce_lr]
        
        if reduce_lr is True:
            callbacks = [reduce_lr] + callbacks
        
        #tf.keras.backend.set_value(self.model_rm.optimizer.learning_rate, lr)
        y = np.ones(shape = (len(x[0][:,0]), 2))
        tf.keras.backend.set_value(self.model_rm.optimizer.learning_rate, lr)
        self.model_rm.fit(x, y, epochs = epochs, verbose = 2, callbacks = callbacks)
        self.model_rm.load_weights("best_model_rm.hdf5")
        
    def get_hs(self, time, hs, spot_tilde, rate):
        time_idx = int(np.round(time / self.T * self.n, 6))
        
        if type(time) == int or type(time) == float:
            time = np.ones_like(rate) * float(time)
    
        #x = np.column_stack([spot_tilde, rate, hs,time])
        
        
        #new_x = copy.deepcopy(x)
        
        log_spot = np.log(spot_tilde + 1)
        log_timetomat = np.log(self.T - time + 1)
        
        #new_x[:,0] = log_spot
        #new_x[:,-1] = log_timetomat
        #new_x[:,1] = x[:,1] * self.ignore_rates
        
        new_x = np.column_stack([log_spot, rate * self.ignore_rates, hs,log_timetomat])
        
        return np.squeeze(self.submodel[time_idx].predict(self.base_submodel.predict(new_x)))
    
    def get_current_optimal_hs(self, model, current_hs):
        return self.get_hs(model.time, current_hs, model.spot / model.bank[:,np.newaxis], model.rate)
        
    
    def get_init_pf(self):
        return self.model_full.predict([np.zeros((1,self.n_assets)) for j in range(self.n + 1)] \
                                       + [np.zeros((1,1)) for j in range(self.n + 2)])[-1][0,0]
        
if __name__ == "__main__":
    NN = NN_simple_hedge(2,1,2,4)
    NN.create_model(2,0,1.)
    t_hs0 = NN.submodel.predict(np.array([[2,0,0]]))
    t_pf1 = t_hs0 * 3 + (1 - t_hs0 * 2) * np.exp(0.*1)
    t_hs1 = NN.submodel.predict(np.array([[3,t_hs0,1.]]))
    t_pf2 = t_hs1 * 4 + (t_pf1 - t_hs1 * 3) * np.exp(0.*1)
    
    test1 = NN.model.predict([np.array([[1]]),np.array([[2,3,4]])])
