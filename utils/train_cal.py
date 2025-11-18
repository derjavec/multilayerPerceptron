import pandas as pd
import numpy as np


def activation(z, activation_type):
    if activation_type == 'relu':
        a = np.maximum(0, z)
        da_dz = (z > 0).astype(float)
    elif activation_type == 'softmax':
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        da_dz = None

    else:
        raise ValueError("Unknown activation type")
    return a, da_dz


def gradient_descent(X, intercept, coef, error, da_dz, alpha):

    if da_dz is None:
        delta = error
    else:
        delta = error * da_dz
    
    d_intercept = np.mean(delta, axis=0)
    d_coef = (delta.T @ X) / X.shape[0]
    intercept -= alpha * d_intercept
    coef -= alpha * d_coef
    
    return intercept, coef


def linear_regression(x, intercepts, coefs):
    coefs = np.array(coefs)
    intercepts = np.array(intercepts) 
    z = x @ coefs.T + intercepts 
    return z


def initialize_weights(config, input_dim):
    layer_sizes = config["layer"]
    
    intercepts = []
    coefs = []
    
    prev_dim = input_dim
    
    for neurons in layer_sizes:
        layer_coefs = np.random.randn(neurons, prev_dim) * np.sqrt(2 / prev_dim)
        layer_intercepts = np.zeros(neurons)  # intercepts típicamente inicializados en 0

        
        intercepts.append(layer_intercepts)
        coefs.append(layer_coefs)
        
        prev_dim = neurons
    
    return intercepts, coefs