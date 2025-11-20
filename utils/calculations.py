import numpy as np
import pandas as pd


def activation(z: np.ndarray, activation_type: str):
    """
    Apply activation function to the input array.
    """
    if activation_type == 'relu':
        a = np.maximum(0, z)
        da_dz = (z > 0).astype(float)
    elif activation_type.lower() == "sigmoid":
        a = 1 / (1 + np.exp(-z))
        da_dz = a * (1 - a)
    elif activation_type == 'softmax':
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        da_dz = None
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")

    return a, da_dz


def gradient_descent(X: np.ndarray, intercept: np.ndarray, coef: np.ndarray,
                     error: np.ndarray, da_dz: np.ndarray | None,
                     alpha: float):
    """
    Perform a single gradient descent step.
    """
    if da_dz is None:
        delta = error
    else:
        delta = error * da_dz

    d_intercept = np.mean(delta, axis=0)
    d_coef = (delta.T @ X) / X.shape[0]

    intercept -= alpha * d_intercept
    coef -= alpha * d_coef

    return intercept, coef


def linear_regression(x: np.ndarray, intercepts: np.ndarray, coefs: np.ndarray):
    """
    Compute linear combination (z = XW^T + b) for a layer.
    """
    coefs = np.array(coefs)
    intercepts = np.array(intercepts)
    z = x @ coefs.T + intercepts
    return z


def initialize_weights(config: dict, input_dim: int):
    """
    Initialize weights and biases for a multi-layer perceptron.
    """
    layer_sizes = config["layer"]

    intercepts = []
    coefs = []
    activations = config['activations']
    first_act = activations[0].strip().lower()

    if first_act == 'relu':
        scale = 2
    else:
        scale = 1
    
    prev_dim = input_dim

    for neurons in layer_sizes:
        layer_coefs = np.random.randn(neurons, prev_dim) * np.sqrt(scale / prev_dim)
        layer_intercepts = np.zeros(neurons)

        intercepts.append(layer_intercepts)
        coefs.append(layer_coefs)

        prev_dim = neurons

    return intercepts, coefs
