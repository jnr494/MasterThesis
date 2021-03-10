# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:36:49 2021

@author: mrgna
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

#Black scholes functions

def BScallprice(s0, sigma, r, T, K, greek = None, tensor = False):
    s0 = tf.dtypes.cast(s0, tf.float32)
    sigma = tf.dtypes.cast(sigma, tf.float32)
    r = tf.dtypes.cast(r, tf.float32)
    T = tf.dtypes.cast(T, tf.float32)
    K = tf.dtypes.cast(K, tf.float32)

    if greek is None:
        d1 = (tf.math.log(s0 / K) + (r + 0.5 * sigma **2) * T) / (sigma * tf.math.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        dist = tfd.Normal(loc=0., scale=1)
        Nd1 = dist.cdf(d1)
        Nd2 = dist.cdf(d2)
        
        price = s0 * Nd1 - tf.math.exp(-r * T) * K * Nd2
        
        if tensor is False:
            return price.numpy()
        else:
            return price
        return 
    elif greek == "delta":
        s0tf = tf.constant(s0)
        with tf.GradientTape() as g:
            g.watch(s0tf)
            price = BScallprice(s0tf, sigma, r, T, K, tensor = True)

            return g.gradient(price, s0tf).numpy()

def BSputprice(s0, sigma, r, T, K, greek = None, tensor = False):
    s0 = tf.dtypes.cast(s0, tf.float32)
    sigma = tf.dtypes.cast(sigma, tf.float32)
    r = tf.dtypes.cast(r, tf.float32)
    T = tf.dtypes.cast(T, tf.float32)
    K = tf.dtypes.cast(K, tf.float32)
    
    call = BScallprice(s0, sigma, r, T, K, greek, tensor)
    
    if greek is None:
        price = K * tf.math.exp(-r * T) + call - s0
        if tensor is False:
            return price.numpy()
        else:
            return price
    elif greek == "delta":
        return call - 1

def BSdigitalcall(s0, sigma, r, T, K, greek = None, tensor = False):
    s0 = tf.dtypes.cast(s0, tf.float32)
    sigma = tf.dtypes.cast(sigma, tf.float32)
    r = tf.dtypes.cast(r, tf.float32)
    T = tf.dtypes.cast(T, tf.float32)
    K = tf.dtypes.cast(K, tf.float32)

    if greek is None:
        d1 = (tf.math.log(s0 / K) + (r + 0.5 * sigma **2) * T) / (sigma * tf.math.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        dist = tfd.Normal(loc=0., scale=1)
        Nd2 = dist.cdf(d2)
        
        price = tf.math.exp(-r * T) * Nd2
        
        if tensor is False:
            return price.numpy()
        else:
            return price
        return 
    elif greek == "delta":
        s0tf = tf.constant(s0)
        with tf.GradientTape() as g:
            g.watch(s0tf)
            price = BSdigitalcall(s0tf, sigma, r, T, K, tensor = True)

            return g.gradient(price, s0tf).numpy()

############################################
#Down and out prices #######################
############################################

def BSdozcb(s0, sigma, r, T, L, greek = None, tensor = False):
    s0 = tf.dtypes.cast(s0, tf.float32)
    sigma = tf.dtypes.cast(sigma, tf.float32)
    r = tf.dtypes.cast(r, tf.float32)
    T = tf.dtypes.cast(T, tf.float32)
    L = tf.dtypes.cast(L, tf.float32)
    
    if greek is None:
        term1 = BSdigitalcall(s0, sigma, r, T, L, tensor = True)
        rtilde = r - sigma**2 / 2
        term2 = (L / s0)**(2*rtilde / sigma**2)
        term3 = BSdigitalcall(L**2 / s0, sigma, r, T, L, tensor = True)
        
        price = term1 - term2 * term3
        if tensor is False:    
            return price.numpy()
        else:
            return price
    elif greek == "delta":
        s0tf = tf.constant(s0)
        with tf.GradientTape() as g:
            g.watch(s0tf)
            price = BSdozcb(s0tf, sigma, r, T, L, tensor = True)

            return g.gradient(price, s0tf).numpy()

def BSdostock(s0, sigma, r, T, L, greek = None, tensor = False):
    s0 = tf.dtypes.cast(s0, tf.float32)
    sigma = tf.dtypes.cast(sigma, tf.float32)
    r = tf.dtypes.cast(r, tf.float32)
    T = tf.dtypes.cast(T, tf.float32)
    L = tf.dtypes.cast(L, tf.float32)
    
    if greek is None:
        term1 = BSdigitalcall(s0, sigma, r, T, L, tensor = True)
        rtilde = r - sigma**2 / 2
        term2 = (L / s0)**(2*rtilde / sigma**2)
        term3 = BSdigitalcall(L**2 / s0, sigma, r, T, L, tensor = True)
        term4 = BScallprice(s0, sigma, r, T, L, tensor = True)
        term5 = BScallprice(L**2 / s0, sigma, r, T, L, tensor = True)
        
        price = L * term1 - L * term2 * term3 + term4 - term2 * term5
        if tensor is False:    
            return price.numpy()
        else:
            return price
    elif greek == "delta":
        s0tf = tf.constant(s0)
        with tf.GradientTape() as g:
            g.watch(s0tf)
            price = BSdostock(s0tf, sigma, r, T, L, tensor = True)

            return g.gradient(price, s0tf).numpy()

def BSdocallprice(s0, sigma, r, T, K, L, greek = None, tensor = False):
    s0 = tf.dtypes.cast(s0, tf.float32)
    sigma = tf.dtypes.cast(sigma, tf.float32)
    r = tf.dtypes.cast(r, tf.float32)
    T = tf.dtypes.cast(T, tf.float32)
    K = tf.dtypes.cast(K, tf.float32)
    L = tf.dtypes.cast(L, tf.float32)
    
    if greek is None:
        
        term1 = BScallprice(s0, sigma, r, T, K, tensor = True)
    
        rtilde = r - sigma**2 / 2
        term2 = (L / s0)**(2 * rtilde / sigma**2)
        term3 = BScallprice(L**2 / s0, sigma, r, T, K, tensor = True)
        
        indicator = tf.dtypes.cast((L > K), tf.float32)
        term4 = BSdigitalcall(s0, sigma, r, T, L, tensor = True)
        term5 = BSdigitalcall(L**2 / s0, sigma, r, T, L, tensor = True)
        
        #print(term2.numpy(),term4.numpy(), term5.numpy())
        #print(indicator.numpy(), (indicator * (L-K) * (term4 - term2 * term5)).numpy())
        price = term1 - term2 * term3 + indicator * (L-K) * (term4 - term2 * term5)
        if tensor is False:    
            return price.numpy()
        else:
            return price
    elif greek == "delta":
        s0tf = tf.constant(s0)
        with tf.GradientTape() as g:
            g.watch(s0tf)
            price = BSdocallprice(s0tf, sigma, r, T, K, L, tensor = True)

            return g.gradient(price, s0tf).numpy()

def BSdoputprice(s0, sigma, r, T, K, L, greek = None, tensor = False):
    s0 = tf.dtypes.cast(s0, tf.float32)
    sigma = tf.dtypes.cast(sigma, tf.float32)
    r = tf.dtypes.cast(r, tf.float32)
    T = tf.dtypes.cast(T, tf.float32)
    K = tf.dtypes.cast(K, tf.float32)
    L = tf.dtypes.cast(L, tf.float32)
    
    if greek is None:
        term1 = BSdozcb(s0, sigma, r, T, L, tensor = True)
        term2 = BSdostock(s0, sigma, r, T, L, tensor = True)
        term3 = BSdocallprice(s0, sigma, r, T, K, L, tensor = True)
        
        price = K * term1 - term2 + term3
        if tensor is False:    
            return price.numpy()
        else:
            return price
    elif greek == "delta":
        s0tf = tf.constant(s0)
        with tf.GradientTape() as g:
            g.watch(s0tf)
            price = BSdoputprice(s0tf, sigma, r, T, K, L, tensor = True)

            return g.gradient(price, s0tf).numpy()    
# =============================================================================
# 
# s0, sigma, r, T, K, L = (np.array([100,100]), 0.2, 0.01, 1, np.array([100,90]),95)
# BSdocallprice(s0, sigma, r, T, K, L)
# #BSdocallprice(s0, sigma, r, T, K, L, greek = "delta")
# 6.261521 - 6.27407
# 
# BSdigitalcall(s0, sigma, r, T, L)
# L**2 / 100
# =============================================================================
