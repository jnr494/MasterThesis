# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 18:42:39 2021

@author: Magnus Frandsen
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Lambda, Input, BatchNormalization, Activation
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.initializers import VarianceScaling

import gc
import copy
import matplotlib.pyplot as plt

#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

gc.collect()
keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

def generator(C, r, batch_size):
    samples_per_epoch = C.shape[0]
    number_of_batches = samples_per_epoch / batch_size
    counter = 0

    while 1:
        X_batch = np.array(C[batch_size * counter:batch_size * (counter + 1)])
        y_batch = np.array(r[batch_size * counter:batch_size * (counter + 1)])
        counter += 1
        yield X_batch, y_batch

        # restart counter to yeild data in the next epoch as well
        if counter >= number_of_batches:
            counter = 0

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

def constmodel(input_dim, output_dim, output2_dim = 0):
    inp = Input(shape = (input_dim,))
    out = Dense(output_dim, activation = None, use_bias = False)(inp[:,0:1] * 0 + 1)
    
    if output2_dim == 0: 
        return Model(inputs = inp, outputs = out)
    else:
        inp2 = Input(shape = (output2_dim,))
        out2 = Dense(output2_dim, activation = None, use_bias = False)(inp[:,0:1] * 0 + 1)
        return Model(inputs = [inp, inp2], outputs = [out, out2])

def cvar_loss_func(x, alpha):
    return tf.nn.relu(x) / (1 - alpha)

def rm_loss(y_true, y_pred):
    return y_pred[0,1] + tf.reduce_mean(y_pred[:,0])

def rm_target_loss(y_true, y_pred):
    return tf.abs(rm_loss(y_true, y_pred) - y_true[0,0])

def create_input_layers(n, n_assets):
    #spots + rates + mins_spots + spot_returns + Bank_T + option_payoff
    inputs = Input(shape = (n_assets*(n+1) + n + 2*n_assets*n + 1 + 1,))
    
    spots = [inputs[:,(n_assets*i):(n_assets*(1+i))] for i in range(n+1)]
    rates = [inputs[:,(n_assets * (n+1) + i):(n_assets * (n+1) + i + 1)] for i in range(n)]
    min_spots = [inputs[:,(n_assets*(n+1) + n + n_assets*i):(n_assets*(n+1) + n + n_assets*(i+1))] for i in range(n)]
    spot_returns = [inputs[:,(n_assets*(n+1) + n + n*n_assets + n_assets*i):\
                           (n_assets*(n+1) + n + n*n_assets + n_assets*(i+1))] for i in range(n)]
    bank_T = inputs[:,-2:-1]
    option_payoff = inputs[:,-1:]

    return spots, rates, min_spots, spot_returns, bank_T, option_payoff, inputs

def calculate_tc(hs, spot):
    return tf.math.reduce_sum(tf.math.abs(hs) * spot, axis = 1, keepdims = True)

class Linear(keras.layers.Layer):
    def __init__(self, input_dim):
        super(Linear, self).__init__()
        self.w = self.add_weight(shape=(input_dim,), initializer="ones", trainable=True)
        self.b = self.add_weight(shape=(input_dim,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return self.w * inputs + self.b

def create_submodel(input_dim, output_dim, output2_dim,
                        n_layers, n_units, 
                        activation, final_activation = None,
                        batch_norm = True):
        
        inputs1 = Input(shape = (input_dim,))
        inputs2 = Input(shape = (output2_dim,))
        
        inputs = tf.concat([inputs1, inputs2], 1)
        
        if batch_norm is True:
            x = BatchNormalization()(inputs)
        else:
            x = inputs
        
        for _ in range(n_layers):
            x = Dense(n_units, kernel_initializer=VarianceScaling())(x)
            if batch_norm is True:
                x = BatchNormalization()(x)
            x = Activation(activation = activation)(x)
        
        
        output = Dense(output_dim, kernel_initializer=VarianceScaling())(x)
        output = Activation(activation = final_activation)(x)
        print(output.shape,output[:,0].shape)
        outputs = []
        for i in range(output_dim):
            outputs.append(Dense(1,kernel_initializer=VarianceScaling())(output[:,i:(i+1)]))
        
        output = tf.concat(outputs, axis = -1)
             
        if batch_norm is True:
            output2 = Dense(output2_dim, activation = final_activation, kernel_initializer=VarianceScaling())(x)
            submodel = Model(inputs = [inputs1, inputs2], outputs = [output,output2])
        else:
            submodel = Model(inputs = [inputs1, inputs2], outputs = output)
        submodel.summary()
        
        return submodel
    
class NN_simple_hedge():
    def __init__(self, n_assets,  
                 n_layers, n_units, 
                 activation = 'elu', final_activation = None, output2_dim =1):
        
        self.n_assets = n_assets
        self.input_dim = 2  + 4 * n_assets #3 comes from spots, min_spots, spot_returns, hs
        self.output_dim = n_assets
        self.output2_dim = output2_dim
        
        self.tmp_info = [None]
        
        self.n_layers = n_layers
        self.n_units = n_units 
        self.activation = activation
        self.final_activation = final_activation
        
    def create_model(self, n, rate, dt, init_pf = None, transaction_costs = 0, 
                     ignore_rates = True, ignore_minmax = True, ignore_returns = True, ignore_info = True,
                     ignore_pf = True):
        self.n = n
        self.T = n * dt
        self.transaction_costs = transaction_costs
        
        
        #ignore rates handling
        if ignore_rates is True:
            self.ignore_rates = 0
        else:
            self.ignore_rates = 1
            
        #ignore ignore_minmiax
        if ignore_minmax is True:
            self.ignore_minmax = 0
        else:
            self.ignore_minmax = 1
            
        #ignore returns
        if ignore_returns is True:
            self.ignore_returns = 0
        else:
            self.ignore_returns = 1
            
        #ignore ignore_minmiax
        if ignore_info is True:
            self.ignore_info = 0
        else:
            self.ignore_info = 1
        
        #ignore previous trade based on transaction costs
        if abs(self.transaction_costs) < 1e-7:
            self.ignore_hs = 0
        else:
            self.ignore_hs = 1
        
        #ignore pf
        if ignore_pf is True:
            self.ignore_pf = 0
        else:
            self.ignore_pf = 1
        
        #init hs
        init_hs_comp = constmodel(self.input_dim, self.output_dim, self.output2_dim)
        
        #create submodels
        self.submodel = [init_hs_comp]
        for i in range(n -1):
            tmp_model = create_submodel(self.input_dim, self.output_dim, self.output2_dim,
                                        self.n_layers, self.n_units, 
                                        self.activation, self.final_activation)
            self.submodel.append(tmp_model) 
        
        #create infomodel
        self.info_model = create_submodel(input_dim = self.input_dim, output_dim = self.output2_dim, 
                                          output2_dim = self.output2_dim, 
                                          n_layers = self.n_layers, n_units = self.n_units, 
                                          activation = self.activation, final_activation = self.final_activation,
                                          batch_norm = False)

        #Inputs        
        spots, rates, min_spots, spot_returns, bank_T, option_payoff, inputs = create_input_layers(self.n, self.n_assets)
        
        #dummy variables   
        dummy_zeros = tf.zeros_like(bank_T)
        dummy_ones = tf.ones_like(bank_T)
        
        dummy_zeros_expanded = tf.zeros_like(spots[0])
        dummy_zeros_expanded2 = tf.concat([dummy_zeros] * self.output2_dim, 1)
        
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
                         min_spots[0] * self.ignore_minmax, #Min spots information
                         spot_returns[0] * self.ignore_returns, #spot return information
                         dummy_zeros_expanded * self.ignore_hs,#current hs
                         init_pf * self.ignore_pf], 1) #current pf
        
        #init_info = dummy_zeros
        hs_0, tmp_info = self.submodel[time_idx]([I1_0, dummy_zeros_expanded2])
        #print(I1_0.shape, hs_0.shape, tmp_info.shape, dummy_zeros_expanded2.shape)
        #tmp_info = self.info_model([I1_0, dummy_zeros_expanded2])
        
        tc_0 = tf.math.reduce_sum(tf.math.abs(hs_0) * spots[0], axis = 1, keepdims = True) * self.transaction_costs

        pf_1 = tf.math.reduce_sum(hs_0 * (spots[1] - spots[0]), axis = 1, keepdims = True) + (init_pf - tc_0)
                
        time += dt
        time_idx = int(np.round(time / self.T * self.n, 6))
        
        tc.append(tc_0)
        hs.append(hs_0)
        pf.append(pf_1)
        
        if n > 1:
            for i in range(1,n):
                #print(i)
                tmp_I1 = tf.concat([tf.math.log(spots[i] + 1), #Spots
                                    rates[i] * self.ignore_rates, #current rates
                                    min_spots[i] * self.ignore_minmax, #Min spots information
                                    spot_returns[i] * self.ignore_returns, #spot return information
                                    hs[-1] * self.ignore_hs, #Current hs
                                    pf[-1] * self.ignore_pf], 1) #Current pf
                
                #print(tmp_I1.shape, (spot_returns[i] * self.ignore_returns).shape)
                tmp_hs, tmp_info = self.submodel[time_idx]([tmp_I1, tmp_info * self.ignore_info])
                #tmp_info = self.info_model([tmp_I1, tmp_info * self.ignore_info])

                tmp_tc = tf.math.reduce_sum(tf.math.abs(tmp_hs - hs[-1]) * spots[i],axis = 1, keepdims = True) * self.transaction_costs
                tmp_pf = tf.math.reduce_sum(tmp_hs * (spots[i+1] - spots[i]), axis = 1, keepdims = True) + (pf[-1] - tmp_tc)

                time += dt
                time_idx = int(np.round(time / self.T * self.n, 6))
                
                hs.append(tf.identity(tmp_hs))
                pf.append(tf.identity(tmp_pf))
                tc.append(tmp_tc)
        
        
        total_pf = - option_payoff + bank_T * pf[-1]
        
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
        spots, rates, min_spots, spot_returns, bank_T, option_payoff, inputs = create_input_layers(self.n, self.n_assets)
        
        #w = constmodel(1,1)(tf.stop_gradient(bank_T))
        w = constmodel(1,1)(tf.zeros_like(bank_T))
        J = cvar_loss_func(- self.model(inputs) - w, alpha)
        rm_output = tf.concat([J,w], 1)
        self.model_rm = Model(inputs = inputs, outputs = rm_output)
        self.model_rm.compile(optimizer = "adam", loss = rm_loss, run_eagerly=False)
        
    
    def compile_rm_w_target_loss(self):
        self.model_rm.compile(optimizer = "adam", loss = rm_target_loss)
    
    def compile_model(self, loss = 'mean_squared_error'):
        self.model.compile(optimizer = "adam", loss = loss, run_eagerly=False) #'mean_squared_error')
    
    
    def train_model(self, x, y, epochs, batch_size = 32, patience = [5,11], learning_rate = 0.01, best_model_name = "best_model.hdf5") :                
        mcp_save = ModelCheckpoint(best_model_name, save_best_only=True, monitor='loss', mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor= 0.1, patience=patience[0], min_lr=1e-6, verbose = 2)
        er = EarlyStopping(monitor = 'loss', patience = patience[1], verbose = 1)
        callbacks = [er, mcp_save, reduce_lr]
        
        #train
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)
        self.model.fit(x, y, epochs = epochs, batch_size = batch_size, callbacks = callbacks,
                       use_multiprocessing= True, workers = 8)

        
    def get_J(self, x):
        len_x = len(x)
        batch_size = min([len_x , int(2**15)])
        rm_outputs = self.model_rm.predict(x, batch_size = batch_size)
        
        
        return rm_outputs[0,1] + np.mean(rm_outputs[:,0])
    
    def train_rm_model(self, x, epochs = 1, batch_size = 32, patience = [5,11], lr = 0.01, best_model_name = "best_model_rm.hdf5"):  

        mcp_save = ModelCheckpoint(best_model_name, save_best_only=True, monitor='loss', mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor= 0.1, patience=patience[0], min_lr=1e-6, verbose = 1)
        er = EarlyStopping(monitor = 'loss', patience = patience[1], verbose = 1)
        
        callbacks = [er, mcp_save, reduce_lr]
        
        y = np.zeros(shape = (len(x),2))
        tf.keras.backend.set_value(self.model_rm.optimizer.learning_rate, lr)
        
# =============================================================================
#         #tf data
#         dataset = tf.data.Dataset.from_tensor_slices((x,y))
#         dataset = dataset.shuffle(buffer_size = len(x))
#         dataset = dataset.batch(batch_size = batch_size)
# =============================================================================
        
        self.model_rm.fit(x, y, epochs = epochs, batch_size = batch_size, verbose = 1, callbacks = callbacks, 
                          use_multiprocessing= True, workers = 8)
        
    def get_hs(self, time, hs, pf_tilde, spot_tilde, rate, min_spot, spot_return):
        time_idx = int(np.round(time / self.T * self.n, 6))
        
        #if type(time) == int or type(time) == float:
        #    time = np.ones_like(rate) * float(time)
        
        log_spot = np.log(spot_tilde + 1)
        
        new_x = np.column_stack([log_spot, rate * self.ignore_rates, min_spot * self.ignore_minmax, 
                                 spot_return * self.ignore_returns, hs * self.ignore_hs, pf_tilde * self.ignore_pf])
        
        #print(type(self.tmp_info), len(self.tmp_info), time)
        #print(type(self.tmp_info) is list, len(self.tmp_info) != len(hs), time < 1e-6)
        
        if type(self.tmp_info) is list or len(self.tmp_info) != len(hs) or time < 1e-6:
            self.tmp_info = np.zeros((len(hs),self.output2_dim))
        
        tmp_hs, self.tmp_info = self.submodel[time_idx].predict([new_x, self.tmp_info * self.ignore_info])
        #self.tmp_info = self.info_model.predict([new_x, self.tmp_info * self.ignore_info])
        #print(self.tmp_info)
        return np.squeeze(tmp_hs).reshape(hs.shape)
    
    def get_current_optimal_hs(self, model, current_hs, current_pf):
        
        return self.get_hs(model.time, current_hs, current_pf / model.bank[:,np.newaxis], 
                           model.spot / model.bank[:,np.newaxis], model.rate, 
                           model.min_spot, model.spot_return)
    
    def get_init_pf(self):
        return self.model_full.predict(np.zeros((1,self.n_assets * (self.n+1) + self.n + 2 * self.n * self.n_assets + 2)))[-1][0,0]

        
if __name__ == "__main__":
    NN = NN_simple_hedge(2,1,2,4)
    NN.create_model(2,0,1.)
    t_hs0 = NN.submodel.predict(np.array([[2,0,0]]))
    t_pf1 = t_hs0 * 3 + (1 - t_hs0 * 2) * np.exp(0.*1)
    t_hs1 = NN.submodel.predict(np.array([[3,t_hs0,1.]]))
    t_pf2 = t_hs1 * 4 + (t_pf1 - t_hs1 * 3) * np.exp(0.*1)
    
    test1 = NN.model.predict([np.array([[1]]),np.array([[2,3,4]])])
