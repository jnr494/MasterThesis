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

#Encoder
def create_encoder(input_dim, layers_units = [10,5], latent_dim = 2, activation = "elu"):
    encoder_inputs = keras.Input(shape=(input_dim,))
    
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
def create_decoder(output_dim, latent_dim, layers_units, activation = "elu", final_activation = "sigmoid"):
    latent_inputs = keras.Input(shape=(latent_dim,))
    
    x = layers.Dense(layers_units[0], activation = activation)(latent_inputs)
    
    if len(layers_units) > 1:
        for units in layers_units[1:]:
            x = layers.Dense(units, activation = activation)(x)
            
    decoder_outputs = layers.Dense(output_dim, activation = final_activation)(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    
    return decoder

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
    def __init__(self, encoder, decoder, alpha = 0.2, beta = 0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.alpha = alpha
        self.beta = beta
        
        self.sample_normals_like = SampleNormalsLike()
        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.std_loss_tracker = keras.metrics.Mean(name="std_loss")
        self.cov_loss_tracker = keras.metrics.Mean(name="cov_loss")
        self.mean_loss_tracker = keras.metrics.Mean(name="mean_loss")
        
        self.decoder_output_dim = self.decoder.output_shape[1]
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.std_loss_tracker,
            self.cov_loss_tracker,
            self.mean_loss_tracker
        ]

    def train_step(self, data):
        data_x, data_y = data
        with tf.GradientTape() as tape:
            
            z_mean, z_log_var, z = self.encoder(data_x)
            #reconstruction
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.math.squared_difference(data_y, reconstruction), 1))
            #kl
            kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1. + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), 1))
            
            #std and mean loss
            reconstruction2 = self.decoder(self.sample_normals_like(z_mean))
            tmp_reconstruction = reconstruction2
            
            std_diff = tf.math.reduce_std(tmp_reconstruction,0) - tf.math.reduce_std(data_y,0)
            std_loss = tf.reduce_mean(tf.math.abs(std_diff))
            
            mean_diff = tf.math.reduce_mean(tmp_reconstruction,0) - tf.math.reduce_mean(data_y,0)
            mean_loss = tf.reduce_mean(tf.math.abs(mean_diff))
         
            #covariance
            lags = self.decoder_output_dim - 1
            cov_loss = 0
            for lag in range(1,lags+1): 
                cov_loss += (calculate_cov_loss(tf.abs(data_y), tf.abs(tmp_reconstruction), lag) + \
                            calculate_cov_loss(data_y, tmp_reconstruction, lag)) / lag
            
            print(reconstruction_loss.shape, kl_loss.shape, std_loss.shape)
            #total loss
            total_loss = (1 - self.alpha) * reconstruction_loss + self.alpha * kl_loss
            total_loss += self.beta * (mean_loss + std_loss + cov_loss)
            #total_loss = mean_loss + std_loss + cov_loss
            print("Total loss shape",total_loss.shape)
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.std_loss_tracker.update_state(std_loss)
        self.cov_loss_tracker.update_state(cov_loss)
        self.mean_loss_tracker.update_state(mean_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "std_loss": self.std_loss_tracker.result(),
            "cov_loss": self.cov_loss_tracker.result(),
            "mean_loss": self.mean_loss_tracker.result()
        }

def compile_vae(vae):
    vae.compile(optimizer=keras.optimizers.Adam())

def train_vae(vae, data_y, data_x = None, epochs = 2500, batch_size = 32, learning_rate = 0.01, patience = [10,100], 
              verbose = 2 , best_model_name = "best_vae_model.hdf5"):
    mcp_save = keras.callbacks.ModelCheckpoint(best_model_name, save_best_only=True, monitor='loss', mode='min')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor= 0.1, patience=patience[0], min_lr=1e-9, verbose = 2)
    er = keras.callbacks.EarlyStopping(monitor = 'loss', patience = patience[1], verbose = 1)
    
    callbacks = [er, mcp_save, reduce_lr]
   #callbacks = [] 
    
    #train
    if data_x is None:
        data_x = data_y
    
    tf.keras.backend.set_value(vae.optimizer.learning_rate, learning_rate)
    history = vae.fit(x = data_x, y = data_y, epochs = epochs, batch_size = batch_size, callbacks = callbacks,
                      use_multiprocessing= True, workers = 8, verbose = verbose)
    return history

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
