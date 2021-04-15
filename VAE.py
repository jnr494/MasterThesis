# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 16:11:45 2021

@author: mrgna
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
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
    
#VAE Model
class VAE(keras.Model):
    def __init__(self, encoder, decoder, alpha = 0.5, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.alpha = alpha
        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            #print(reconstruction.shape)
            #calculate losses
            reconstruction_loss = tf.reduce_sum(tf.math.squared_difference(data, reconstruction), 1)
            #reconstruction_loss = keras.losses.binary_crossentropy(data, reconstruction)
            
            kl_loss = -0.5 * tf.reduce_sum(1. + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), 1)
            
            #kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = tf.reduce_mean((1 - self.alpha) * reconstruction_loss + self.alpha * kl_loss)
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def compile_vae(vae):
    vae.compile(optimizer=keras.optimizers.Adam())

def train_vae(vae, data, epochs, batch_size, learning_rate, patience = [10,100], 
              verbose =2 , best_model_name = "best_vae_model.hdf5"):
    mcp_save = keras.callbacks.ModelCheckpoint(best_model_name, save_best_only=True, monitor='loss', mode='min')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor= 0.1, patience=patience[0], min_lr=1e-9, verbose = 2)
    er = keras.callbacks.EarlyStopping(monitor = 'loss', patience = patience[1], verbose = 1)
    
    callbacks = [er, mcp_save, reduce_lr]
   #callbacks = [] 
    
    #train
    tf.keras.backend.set_value(vae.optimizer.learning_rate, learning_rate)
    history = vae.fit(data, epochs = epochs, batch_size = batch_size, callbacks = callbacks,
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
    
    log_returns = MarketGeneratorHelpFunctions.generate_data_for_MG(bs_model, n, N)
    log_returns_norm, scaler = MarketGeneratorHelpFunctions.transform_data(log_returns, minmax = False)
    
# =============================================================================
#     log_returns_norm = log_returns
#     scaler = None
# =============================================================================
    
    actual_paths = MarketGeneratorHelpFunctions.convert_log_returns_to_paths(s0, log_returns)
    plt.plot(actual_paths[:100,:].T)
    plt.show()
    
    #VAE
    latent_dim = 10
    alpha = 0.2
    layers_units = [20,10]    
    encoder = create_encoder(n, layers_units, latent_dim = latent_dim)
    decoder = create_decoder(n, latent_dim = latent_dim, layers_units = layers_units[::-1], final_activation = None)
    vae = VAE(encoder, decoder, alpha = alpha)
    
    compile_vae(vae)
    train_vae(vae, log_returns_norm, epochs = 3000, batch_size = 128, learning_rate = 0.01, patience = [2000,10000])
    train_vae(vae, log_returns_norm, epochs = 3000, batch_size = 128, learning_rate = 0.001, patience = [2000,10000])
    train_vae(vae, log_returns_norm, epochs = 3000, batch_size = 128, learning_rate = 0.0001, patience = [2000,10000])
    
    n_samples = 100000
    sample_vae = MarketGeneratorHelpFunctions.sample_from_vae(decoder, n_samples)
    log_return_vae = scaler.inverse_transform(sample_vae)
    
    sampled_paths = MarketGeneratorHelpFunctions.convert_log_returns_to_paths(s0, log_return_vae, )
    
    plt.plot(sampled_paths[:100,:].T)
    plt.show() 

    #### Moment comparrison
    paths = [actual_paths, sampled_paths]
    print("Mean",[np.mean(p[:,-1]) for p in paths])
    print("Std", [np.std(p[:,-1]) for p in paths])

    plot_label_clusters(encoder, log_returns_norm, actual_paths[:,-1])
    plot_label_clusters(encoder, log_returns_norm, actual_paths[:,1])
    
    
    #QQ plot
    
    percentiles = np.linspace(0.01,0.99,100)
    i = n
    actual_quantiles = np.quantile(actual_paths[:,i], percentiles)
    sampled_quantiles = np.quantile(sampled_paths[:,i], percentiles)
    
    #qq plot with theoretical quantiles of simulated bs paths    
    plt.scatter(actual_quantiles, sampled_quantiles, s = 5)
    plt.plot(actual_quantiles[[0,-1]],actual_quantiles[[0,-1]],c='k')
    plt.show()
    
    #more precise actual quantiles
    percentiles2 = np.linspace(0.001,0.999,999)
    sampled_quantiles2 = np.quantile(sampled_paths[:,i], percentiles2)
    
# =============================================================================
#     log_returns2 = MarketGeneratorHelpFunctions.generate_data_for_MG(bs_model, n, 1000)
#     actual_paths2 = MarketGeneratorHelpFunctions.convert_log_returns_to_paths(s0, log_returns2)
#     
#     actual_quantiles2 = np.quantile(actual_paths2[:,i], percentiles2)
#     sampled_quantiles2 = np.quantile(sampled_paths[:,i], percentiles2)
#     
#     plt.scatter(actual_quantiles2, sampled_quantiles2, s = 5)
#     plt.plot(actual_quantiles2[[0,-1]],actual_quantiles2[[0,-1]],c='k')
#     plt.show()
# =============================================================================
    
    #test
    s = sigma * np.sqrt(T)
    loc = (mu - sigma**2/2)*T
    scale = 1/s0
    real_quantiles2 = lognorm.ppf(percentiles2, s = s, loc = loc, scale = scale)
        
    plt.scatter(real_quantiles2, sampled_quantiles2, s = 5)
    plt.plot(real_quantiles2[[0,-1]],real_quantiles2[[0,-1]],c='k')
    plt.show()
    
    #test of simulated paths
    real_quantiles = lognorm.ppf(percentiles, s = s, loc = loc, scale = scale)
    plt.scatter(real_quantiles, actual_quantiles, s = 5)
    plt.plot(real_quantiles[[0,-1]],real_quantiles[[0,-1]],c='k')
    plt.show()
    
# =============================================================================
#     plt.scatter(real_quantiles, actual_quantiles2, s = 5)
#     plt.plot(real_quantiles[[0,-1]],real_quantiles[[0,-1]],c='k')
#     plt.show()
# =============================================================================
