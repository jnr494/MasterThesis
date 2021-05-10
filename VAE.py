# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 16:11:45 2021

@author: mrgna
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import lognorm
tf.config.set_visible_devices([], 'GPU')

import MarketGeneratorHelpFunctions
import BlackScholesModel

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class SampleNormalsLike(layers.Layer):
    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        dim = tf.shape(inputs)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return epsilon

#Momemt model
def create_moment_model(input_dim,output_dim, layers_units = [3,3,3,3], activation = "elu"):     
    inputs = keras.Input(shape=(input_dim,))
    
    x = layers.Dense(layers_units[0], activation = activation)(inputs)
    
    if len(layers_units) > 1:
        for units in layers_units[1:]:
            x = layers.Dense(units, activation = activation)(x)
    
    outputs = layers.Dense(output_dim, name="output")(x)
    mm_model = keras.Model(inputs, outputs)
    mm_model.summary()
    
    return mm_model

#Encoder
def create_encoder(input_dim, layers_units = [10,5], latent_dim = 2, activation = "elu", cond_dim = 0):
    
    encoder_inputs = keras.Input(shape=(input_dim + cond_dim,))
    
    x = layers.Dense(layers_units[0], activation = activation)(encoder_inputs)
    
    if len(layers_units) > 1:
        for units in layers_units[1:]:
            x = layers.Dense(units, activation = activation)(x)
    
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    
    return encoder

#Decoder
def create_decoder(output_dim, latent_dim, layers_units, activation = "elu", final_activation = None, cond_dim = 0):
    
    latent_cond_inputs = keras.Input(shape=(latent_dim + cond_dim,))

    x = layers.Dense(layers_units[0], activation = activation)(latent_cond_inputs)
    
    if len(layers_units) > 1:
        for units in layers_units[1:]:
            x = layers.Dense(units, activation = activation)(x)
            
    decoder_outputs = layers.Dense(output_dim, activation = final_activation)(x)
    decoder = keras.Model(latent_cond_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    
    return decoder

def create_real_decoder(decoder, c):
    latent_cond_inputs = keras.Input(shape=(decoder.input_shape[1],))
        
    decoder_outputs = decoder(latent_cond_inputs)
    
    batch, dim  = tf.shape(decoder_outputs)
    normals = tf.keras.backend.random_normal(shape=(batch, dim))
    
    real_decoder_outputs = decoder_outputs + tf.math.sqrt(c) * normals
    real_decoder = keras.Model(latent_cond_inputs, real_decoder_outputs, name="real_decoder")
    real_decoder.summary()
    
    return real_decoder

def calculate_cov_lag(x,lag):
    lead_x = x[:,lag:]
    lag_x = x[:,:-lag]
    #cov_x = tfp.stats.covariance(lead_x,lag_x, 0, event_axis = None)
    cov_x = tfp.stats.correlation(lead_x,lag_x, 0, event_axis = None)
    return cov_x

def calculate_cov_loss(x,y,lag):
    #covariance
    cov_x = calculate_cov_lag(x,lag)
    cov_y = calculate_cov_lag(y, lag)
    cov_loss = tf.reduce_mean(tf.math.abs(cov_x - cov_y))
    return cov_loss
    
#VAE Model
class VAE(keras.Model):
    def __init__(self, encoder, decoder,real_decoder, alpha = 0.9, beta = 0, gamma = 0, cond_type = 0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
        self.real_decoder = real_decoder
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.sample_normals_like = SampleNormalsLike()
        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.std_loss_tracker = keras.metrics.Mean(name="std_loss")
        self.cov_loss_tracker = keras.metrics.Mean(name="cov_loss")
        self.mean_loss_tracker = keras.metrics.Mean(name="mean_loss")
        self.mm_loss_tracker = keras.metrics.Mean(name="mm_loss")
        self.cmm_loss_tracker = keras.metrics.Mean(name="cmm_loss")
        
        self.decoder_output_dim = self.decoder.output_shape[1]
        self.encoder_input_dim = self.encoder.input_shape[1]
        
        self.encoder_decoder_trainable_w = self.encoder.trainable_weights + self.decoder.trainable_weights
        
        self.cond_n = self.encoder_input_dim - self.decoder_output_dim
        self.cond_type = cond_type
        self.cond_bool = self.cond_n > 0
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.std_loss_tracker,
            self.cov_loss_tracker,
            self.mean_loss_tracker,
            self.mm_loss_tracker,
            self.cmm_loss_tracker
        ]

    def train_step(self, data):
        data_cond, data_y = data
        
        if self.cond_bool is True:
            data_x = tf.concat([data_y,data_cond], axis = 1)
        else:
            data_x = data_y
                     
        with tf.GradientTape() as tape:
            
            #repeats = tf.constant([10,1], tf.int32)
            #data_x = tf.tile(data_x, repeats)
            #data_cond = tf.tile(data_cond, repeats)
            #data_y = tf.tile(data_y, repeats)
            
            z_mean, z_log_var, z = self.encoder(data_x)
            #reconstruction1 and 2
            if self.cond_bool is True:
                z1 = tf.concat([z,data_cond],axis = 1)
                z2 = tf.concat([self.sample_normals_like(z_mean),data_cond],axis = 1)
            else:
                z1 = z
                z2 = self.sample_normals_like(z_mean)
               
            reconstruction = self.decoder(z1)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(data_y - reconstruction), 1))
            
            #kl
            kl_loss = tf.reduce_mean(tf.reduce_sum(tf.square(z_mean) + tf.exp(z_log_var) - 1. - z_log_var, 1))
            
            mm_loss = 0
            mean_loss = 0
            std_loss = 0 
            cov_loss = 0
            cmm_loss = 0
            
            #std and mean loss
            reconstruction2 = self.real_decoder(z2) #!!!!!!!
            tmp_reconstruction = reconstruction2
             
            std_diff = tf.math.reduce_std(tmp_reconstruction,0) - tf.math.reduce_std(data_y,0)
            std_loss = tf.reduce_mean(tf.math.abs(std_diff))
            
            mean_diff = tf.math.reduce_mean(tmp_reconstruction,0) - tf.math.reduce_mean(data_y,0)
            mean_loss = tf.reduce_mean(tf.math.abs(mean_diff))
         
            #covariance
            if self.cond_bool is False:
                lags = self.decoder_output_dim - 1
                cov_loss = 0
                for lag in range(1,lags+1): 
                    cov_loss += (calculate_cov_loss(tf.abs(data_y), tf.abs(tmp_reconstruction), lag) + \
                                calculate_cov_loss(data_y, tmp_reconstruction, lag)) / lag
            else:
                if self.cond_type == 0:
                    lags = min([self.cond_n + self.decoder_output_dim - 1,20])
                    cov_loss = 0
                    for lag in range(1,lags+1): 
                        tmp_cond = data_cond[:,-lag:]
                        cond_recon = tf.concat([tmp_cond,reconstruction2], axis = 1)
                        cond_y = tf.concat([tmp_cond,data_y], axis = 1)
                        cov_loss += (calculate_cov_loss(tf.abs(cond_y), tf.abs(cond_recon), lag) + \
                                    calculate_cov_loss(cond_y, cond_recon, lag)) / lag
                elif self.cond_type == 1:
                    reconstruction1 = self.real_decoder(z1)
                    reconstruction_loss0 = tf.reduce_mean(tf.reduce_sum(tf.math.square(data_y - reconstruction1), 1))
                    reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.math.square(tf.square(data_y) - tf.square(reconstruction1)), 1))
                    cmm_loss = reconstruction_loss1 + reconstruction_loss0
                    
                    lags = self.decoder_output_dim - 1
                    cov_loss = 0
                    for lag in range(1,lags+1): 
                        cov_loss += (calculate_cov_loss(tf.abs(data_y), tf.abs(tmp_reconstruction), lag) + \
                                    calculate_cov_loss(data_y, tmp_reconstruction, lag)) / lag
                                                         
            
            mm_loss = mean_loss + std_loss + cov_loss 
            
            #total loss
            total_loss = self.alpha * reconstruction_loss + (1 - self.alpha) * kl_loss
            total_loss += self.beta * mm_loss
            total_loss += self.gamma * cmm_loss
            
        #grads = tape.gradient(total_loss, self.trainable_weights)
        #self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        grads = tape.gradient(total_loss, self.encoder_decoder_trainable_w)
        self.optimizer.apply_gradients(zip(grads, self.encoder_decoder_trainable_w))
        
        #Updated trackers
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.std_loss_tracker.update_state(std_loss)
        self.cov_loss_tracker.update_state(cov_loss)
        self.mean_loss_tracker.update_state(mean_loss)
        self.mm_loss_tracker.update_state(mm_loss)
        self.cmm_loss_tracker.update_state(cmm_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "std_loss": self.std_loss_tracker.result(),
            "cov_loss": self.cov_loss_tracker.result(),
            "mean_loss": self.mean_loss_tracker.result(),
            "mm_loss": self.mm_loss_tracker.result(),
            "cmm_loss": self.cmm_loss_tracker.result()
        }

def compile_vae(vae):
    vae.compile(optimizer=keras.optimizers.Adam())

def train_vae(vae, data_y, data_x = None, epochs = 2500, batch_size = 32, learning_rate = 0.01, patience = [10,100], 
              verbose = 2 , best_model_name = "best_vae_model.hdf5"):
    mcp_save = keras.callbacks.ModelCheckpoint(best_model_name, save_best_only=True, monitor='loss', mode='min')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor= 0.1, patience=patience[0], min_lr=1e-9, verbose = 2)
    er = keras.callbacks.EarlyStopping(monitor = 'loss', patience = patience[1], verbose = 1)
    
    callbacks = [er, mcp_save, reduce_lr]
    
    if best_model_name is None:
        callbacks.pop(1)
    
    #train
    if data_x is None:
        data_x = data_y
    
    tf.keras.backend.set_value(vae.optimizer.learning_rate, learning_rate)
    history = vae.fit(x = data_x, y = data_y, epochs = epochs, batch_size = batch_size, callbacks = callbacks,
                      use_multiprocessing= True, workers = 8, verbose = verbose)
    return history

def compile_and_train_mm_model(mm_model, x, y, epochs = 500, batch_size = 16, lr = 0.01, verbose = 2):
    mm_model.compile(optimizer=keras.optimizers.Adam(lr), loss = "mse")
    mm_model.fit(x, y, epochs = epochs, batch_size = batch_size, verbose = 2)

#Plot latent space
def plot_label_clusters(encoder, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

if __name__ == '__main__':
    n = 20
    N = 100
    T = 1/12
    s0, mu, sigma, rate, dt = (1, 0.06, 0.2, 0.01, T/n)
    bs_model = BlackScholesModel.BlackScholesModel(s0,mu, sigma, np.ones((1,1)), rate, dt)
    
    seed = 69+1
    log_returns = MarketGeneratorHelpFunctions.generate_data_for_MG(bs_model, n, N, overlap = False, seed = seed)
    log_returns_norm, scaler = MarketGeneratorHelpFunctions.transform_data(log_returns, minmax = False)
      
    actual_paths = MarketGeneratorHelpFunctions.convert_log_returns_to_paths(s0, log_returns)
    plt.plot(actual_paths[:100,:].T)
    plt.show()
    
    #Calculate logsignatures
    order = 4
    get_logsig = MarketGeneratorHelpFunctions.LogSignature(order)
    logsig_size = get_logsig.logsig_size
    logsigs = get_logsig(actual_paths)
    logsigs_norm, scaler_logsig = MarketGeneratorHelpFunctions.transform_data(logsigs, minmax = False)
    
    #Toggl logsigs on and off
    logsig = False
    if logsig is True:
        x_dim = logsig_size
        data_x = logsigs_norm
        latent_dim = n #x_dim
    else:
        x_dim = n
        data_x = log_returns_norm
        latent_dim = 20
    
    #VAE
    alpha = 0.2
    layers_units = [60]    
    encoder = create_encoder(x_dim, layers_units, latent_dim = latent_dim)
    decoder = create_decoder(n, latent_dim = latent_dim, layers_units = layers_units[::-1], final_activation = None)
    vae = VAE(encoder, decoder, alpha = alpha)
    
    compile_vae(vae)
    data_y = log_returns_norm
    train_vae(vae, data_y, data_x, epochs = 3000, batch_size = 128, learning_rate = 0.01, patience = [2000,10000])
    train_vae(vae, data_y, data_x, epochs = 3000, batch_size = 128, learning_rate = 0.001, patience = [2000,10000])
    train_vae(vae, data_y, data_x, epochs = 3000, batch_size = 128, learning_rate = 0.0001, patience = [2000,10000])
    
    n_samples = 100000
    sample_vae = MarketGeneratorHelpFunctions.sample_from_vae(decoder, n_samples, std = 1)
    log_return_vae = scaler.inverse_transform(sample_vae)
    sampled_paths = MarketGeneratorHelpFunctions.convert_log_returns_to_paths(s0, log_return_vae, )
    
    plt.plot(sampled_paths[:100,:].T)
    plt.show() 
    

# =============================================================================
#     #### Moment comparrison
#     paths = [actual_paths, sampled_paths]
#     print("Mean",[np.mean(p[:,-1]) for p in paths])
#     print("Std", [np.std(p[:,-1]) for p in paths])
# 
#     plot_label_clusters(encoder, data_x, actual_paths[:,-1])
#     plot_label_clusters(encoder, data_x, actual_paths[:,-1])
#     
#     
#     #QQ plot
#     
#     percentiles = np.linspace(0.01,0.99,100)
#     i = n
#     actual_quantiles = np.quantile(actual_paths[:,i], percentiles)
#     sampled_quantiles = np.quantile(sampled_paths[:,i], percentiles)
#     
#     #qq plot with theoretical quantiles of simulated bs paths    
#     plt.scatter(actual_quantiles, sampled_quantiles, s = 5)
#     plt.plot(actual_quantiles[[0,-1]],actual_quantiles[[0,-1]],c='k')
#     plt.show()
#     
#     #more precise actual quantiles
#     percentiles2 = np.linspace(0.001,0.999,999)
#     sampled_quantiles2 = np.quantile(sampled_paths[:,i], percentiles2)
#        
#     #test
#     s = sigma * np.sqrt(T)
#     loc = (mu - sigma**2/2)*T
#     scale = 1/s0
#     real_quantiles2 = lognorm.ppf(percentiles2, s = s, loc = loc, scale = scale)
#         
#     plt.scatter(real_quantiles2, sampled_quantiles2, s = 5)
#     plt.plot(real_quantiles2[[0,-1]],real_quantiles2[[0,-1]],c='k')
#     plt.show()
#     
#     #test of simulated paths
#     real_quantiles = lognorm.ppf(percentiles, s = s, loc = loc, scale = scale)
#     plt.scatter(real_quantiles, actual_quantiles, s = 5)
#     plt.plot(real_quantiles[[0,-1]],real_quantiles[[0,-1]],c='k')
#     plt.show()
#     
# ###################### Look at std and covariance
#     i = -1
#     print("Log returns")
#     print("Std:",np.std(log_returns[:,i]),np.std(log_return_vae[:,i]))
#     print("Cov:",np.cov(log_returns[:,i-1],log_returns[:,i])[0,1],
#           np.cov(log_return_vae[:,i-1],log_return_vae[:,i])[0,1])
#     print("Abs Cov:",np.cov(np.abs(actual_paths[:,i-1]),np.abs(actual_paths[:,i]))[0,1],
#           np.cov(np.abs(sampled_paths[:,i-1]),np.abs(sampled_paths[:,i]))[0,1])
#     
#     a = calculate_cov_lag(log_returns,1).numpy()
# 
# =============================================================================
