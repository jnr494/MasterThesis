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

import gc
import copy
import matplotlib.pyplot as plt

#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

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
    return tf.nn.relu(x) / (1 - alpha) #tf.maximum(x,0) / (1 - alpha)

def rm_loss(y_true, y_pred):
    return y_pred[0,1] + tf.reduce_mean(y_pred[:,0])

def rm_target_loss(y_true, y_pred):
    return tf.abs(rm_loss(y_true, y_pred) - y_true[0,0])

def create_input_layers(n, n_assets):
# =============================================================================
#      #Inputs
#     spots = []
#     for i in range(n+1):
#         spots.append(Input(shape = (n_assets,),name ='Spot{}'.format(i)))
#         
#     rates = []
#     for i in range(n):
#         rates.append(Input(shape = (1,),name ='rate{}'.format(i)))
#     
#     bank_T = Input(shape = (1,),name ='Bank_T')
#     
#     option_payoff = Input(shape = (1,),name ='OptionPayoff')
#     
#     inputs = spots + rates + [bank_T, option_payoff]
# =============================================================================
    
    inputs = Input(shape = (n_assets * (n+1) + n + 1 + 1,))
    
    spots = [inputs[:,(n_assets*i):(n_assets*(1+i))] for i in range(n+1)]
    rates = [inputs[:,(n_assets * (n+1) + i):(n_assets * (n+1) + i + 1)] for i in range(n)]
    bank_T = inputs[:,-2:-1]
    option_payoff = inputs[:,-1:]

    return spots, rates, bank_T, option_payoff, inputs

def calculate_tc(hs, spot):
    return tf.math.reduce_sum(tf.math.abs(hs) * spot, axis = 1, keepdims = True)

def create_submodel(input_dim, output_dim,
                        n_layers, n_units, 
                        activation, final_activation = None):
                
        submodel = Sequential()
        print(input_dim)
        submodel.add(Input(shape = (input_dim,)))
        
        submodel.add(BatchNormalization())
        submodel.add(Dense(n_units, activation = activation)) #, input_dim = input_dim))
        #submodel.add(BatchNormalization())
        for _ in range(n_layers-1):
            submodel.add(Dense(n_units, activation = activation))
       #     submodel.add(BatchNormalization())  
        submodel.add(Dense(output_dim, activation = final_activation))
        
        submodel.summary()
        
        return submodel

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
        init_hs_comp = constmodel(self.input_dim, self.output_dim)
        
        #create submodels
        self.submodel = [init_hs_comp]
        for i in range(n -1):
            tmp_model = create_submodel(self.input_dim, self.output_dim,
                                        self.n_layers, self.n_units, 
                                        self.activation, self.final_activation)
            self.submodel.append(tmp_model) 

        #Inputs        
        spots, rates, bank_T, option_payoff, inputs = create_input_layers(self.n, self.n_assets)
        
        #dummy variables   
        dummy_zeros = tf.zeros_like(bank_T)
        dummy_ones = tf.ones_like(bank_T)
        
        dummy_zeros_expanded = tf.zeros_like(spots[0])
        
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
                         dummy_zeros_expanded], 1) #Current hs
        
        hs_0 = self.submodel[time_idx](I1_0)
        
        tc_0 = tf.math.reduce_sum(tf.math.abs(hs_0) * spots[0], axis = 1, keepdims = True) * self.transaction_costs

        pf_1 = tf.math.reduce_sum(hs_0 * (spots[1] - spots[0]), axis = 1, keepdims = True) + (init_pf - tc_0)
        
        #print(0,pf_1.shape)
        
        time += dt
        time_idx = int(np.round(time / self.T * self.n, 6))
        
        tc.append(tc_0)
        hs.append(hs_0)
        pf.append(pf_1)
        
        if n > 1:
            for i in range(1,n):
                tmp_I1 = tf.concat([tf.math.log(spots[i] + 1), #Spots
                                    rates[i] * self.ignore_rates, #current rates
                                    hs[-1]], 1) #Current hs
                
                tmp_hs = self.submodel[time_idx](tmp_I1)

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

    def create_model2(self, n, rate, dt, init_pf = None, transaction_costs = 0, 
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
        init_hs_comp = constmodel(self.input_dim, self.output_dim)
        
        #create submodels
        self.submodel = [init_hs_comp]
        for i in range(n -1):
            tmp_model = create_submodel(self.input_dim, self.output_dim,
                                        self.n_layers, self.n_units, 
                                        self.activation, self.final_activation)
            self.submodel.append(tmp_model) 

        #Inputs        
        spots, rates, bank_T, option_payoff, inputs = create_input_layers(self.n, self.n_assets)
        
        #dummy variables   
        dummy_zeros = tf.zeros_like(bank_T)
        dummy_ones = tf.ones_like(bank_T)
        
        dummy_zeros_expanded = tf.zeros_like(spots[0])
        
        #Tracking variables
        time = 0
        time_idx = int(np.round(time / self.T * self.n, 6))

        #lists of portfolios and portfolio values
        hs = []
        #pf = []
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
                         dummy_zeros_expanded], 1) #Current hs
        
        hs_0 = self.submodel[time_idx](I1_0)
        
        time += dt
        time_idx = int(np.round(time / self.T * self.n, 6))
        
        hs.append(hs_0)

        
        if n > 1:
            for i in range(1,n):
                tmp_I1 = tf.concat([tf.math.log(spots[i] + 1), #Spots
                                    rates[i] * self.ignore_rates, #current rates
                                    hs[-1]], 1) #Current hs
                
                tmp_hs = self.submodel[time_idx](tmp_I1)
                
                time += dt
                time_idx = int(np.round(time / self.T * self.n, 6))
                
                hs.append(tf.identity(tmp_hs))
        
        spots_matrix = tf.stack(spots, axis = 2)
        hs_matrix = tf.stack(hs,axis = 2)
        
        pf = hs_matrix * (spots_matrix[..., 1:] - spots_matrix[..., :-1])
        tc = tf.abs(hs_matrix - tf.concat([tf.expand_dims(dummy_zeros_expanded,2), hs_matrix[...,:-1]], axis = -1)) \
            * spots_matrix[...,:-1] * self.transaction_costs
        
        pf_tc = tf.expand_dims(tf.reduce_sum(pf - tc, axis = [1,2]), axis = -1)
        
        total_pf = - option_payoff + bank_T * (init_pf + pf_tc)
        
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
        #tmp_lr_lambda = lambda epoch: learning_rate
        #lr_schd = LearningRateScheduler(tmp_lr_lambda)
        mcp_save = ModelCheckpoint(best_model_name, save_best_only=True, monitor='loss', mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor= 0.1, patience=5, min_lr=1e-7, verbose = 2)
        er = EarlyStopping(monitor = 'loss', patience = 11, verbose = 1)
        callbacks = [er, mcp_save, reduce_lr]
        
        #train
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)
        self.model.fit(x, y, epochs = epochs, batch_size = batch_size, callbacks = callbacks)
        
    
    def get_J(self, x):
        rm_outputs = self.model_rm.predict(x)
        
        return rm_outputs[0,1] + np.mean(rm_outputs[:,0])
    
    def train_rm_model(self, x, epochs = 1, batch_size = 32, lr = 0.01, best_model_name = "best_model_rm.hdf5"):  
        #temp_CLR = lambda epoch:lr
        #lr_schd = LearningRateScheduler(temp_CLR)
        
        
        
        mcp_save = ModelCheckpoint(best_model_name, save_best_only=True, monitor='loss', mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor= 0.1, patience=5, min_lr=1e-7, verbose = 1)
        er = EarlyStopping(monitor = 'loss', patience = 11, verbose = 1)
        
        callbacks = [er, mcp_save, reduce_lr]
        
        #y = np.ones(shape = (len(x[0][:,0]), 2))
        y = np.zeros(shape = (len(x),2))
        tf.keras.backend.set_value(self.model_rm.optimizer.learning_rate, lr)
        
        #tf data
        dataset = tf.data.Dataset.from_tensor_slices((x,y))
        dataset = dataset.shuffle(buffer_size = len(x))
        dataset = dataset.batch(batch_size = batch_size)
        
        self.model_rm.fit(x, y, epochs = epochs, batch_size = batch_size, verbose = 1, callbacks = callbacks)
        #self.model_rm.fit(dataset, epochs = epochs, verbose = 1, callbacks = callbacks, use_multiprocessing=True, workers = 4)
        self.model_rm.load_weights("best_model_rm.hdf5")
        
    def get_hs(self, time, hs, spot_tilde, rate):
        time_idx = int(np.round(time / self.T * self.n, 6))
        
        if type(time) == int or type(time) == float:
            time = np.ones_like(rate) * float(time)
        
        log_spot = np.log(spot_tilde + 1)
        
        new_x = np.column_stack([log_spot, rate * self.ignore_rates, hs])
        
        return np.squeeze(self.submodel[time_idx].predict(new_x)).reshape(hs.shape)
        #np.squeeze(self.submodel[time_idx].predict(self.base_submodel.predict(new_x)))
    
    def get_current_optimal_hs(self, model, current_hs):
        return self.get_hs(model.time, current_hs, model.spot / model.bank[:,np.newaxis], model.rate)
        
    
    def get_init_pf(self):
        return self.model_full.predict(np.zeros((1,self.n_assets * (self.n+1) + self.n + 2)))[-1][0,0]
# =============================================================================
#         return self.model_full.predict([np.zeros((1,self.n_assets)) for j in range(self.n + 1)] \
#                                        + [np.zeros((1,1)) for j in range(self.n + 2)])[-1][0,0]
# =============================================================================

class CustomModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim, n, n_layers, n_units, activation):
		# call the parent constructor
        super(CustomModel, self).__init__()
        
        self.hs = []
        self.hs.append(constmodel(input_dim, output_dim))
        for i in range(n-1):
            self.hf.append(create_submodel(self, input_dim, output_dim, n_layers, n_units, activation))
        
        
        
        
    def call(self, inputs):
        
        return inputs
        
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

        
if __name__ == "__main__":
    NN = NN_simple_hedge(2,1,2,4)
    NN.create_model(2,0,1.)
    t_hs0 = NN.submodel.predict(np.array([[2,0,0]]))
    t_pf1 = t_hs0 * 3 + (1 - t_hs0 * 2) * np.exp(0.*1)
    t_hs1 = NN.submodel.predict(np.array([[3,t_hs0,1.]]))
    t_pf2 = t_hs1 * 4 + (t_pf1 - t_hs1 * 3) * np.exp(0.*1)
    
    test1 = NN.model.predict([np.array([[1]]),np.array([[2,3,4]])])
