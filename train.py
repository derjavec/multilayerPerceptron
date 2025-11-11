import sys
from utils.get_config import get_config, read_config_file
from split import split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def scale(X: np.ndarray):
    """
    Scale an array to the [0, 1] range.
    """
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    scaled = (X - x_min) / (x_max - x_min)
    return scaled, x_min, x_max

def predict_values(intercept: float,
                   coef: np.ndarray,
                   X: np.ndarray) -> np.ndarray:
    """
    Predict output for input matrix X using linear model parameters.
    """
    try:
        intercept = float(intercept)
        return intercept + X @ coef
    except Exception as err:
        raise ValueError("Invalid intercept or coefficient") from err


def calculate_error(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Compute prediction error: predicted - true.
    """
    if true.shape != pred.shape:
        raise ValueError("Data length mismatch\
                          between true and predicted arrays")
    return pred - true


def gradient_descent(
    X: np.ndarray,
    alpha: float,
    error: np.ndarray,
    intercept: float,
    coef: np.ndarray,
):
    """
    Perform one step of gradient descent update.
    """
    d_intercept = np.mean(error)
    d_coef = np.mean(error[:, None] * X, axis=0)

    intercept -= alpha * d_intercept
    coef -= alpha * d_coef
    return intercept, coef


def train_neuron(X: np.ndarray, Y: np.ndarray, alpha: float, iterations: int):
    """
    Train a linear model using gradient descent
    until convergence or max iteration.
    """
    X_array = np.array(X, dtype=float)
    Y_array = np.array(Y, dtype=float)

    if X_array.shape[0] != Y_array.shape[0]:
        raise ValueError("Data length mismatch between X and Y")

    coef = np.random.randn(X_array.shape[1]) * 0.01
    intercept = np.random.randn() * 0.01

    epsilon = 1e-6
    mse_old = float("inf")

    scaled_X, x_min, x_max = scale(X_array)
    scaled_Y, y_min, y_max = scale(Y_array)

    for i in range(iterations):
        y = predict_values(intercept, coef, scaled_X)
        error = calculate_error(scaled_Y, y)
        mse = np.mean(error ** 2) / 2

        if abs(mse_old - mse) < epsilon:
            break

        mse_old = mse
        intercept, coef = gradient_descent(scaled_X,
                                           alpha, error,
                                           intercept, coef)

    coef_original = coef * (y_max - y_min) / (x_max - x_min)
    intercept_original = (
        y_min + intercept * (y_max - y_min) - np.sum(coef_original * x_min)
    )

    return intercept_original, coef_original

def activation(a_type, X, intercept, coef):

    z = intercept + np.dot(X, coef)

    if a_type == 'relu':       
        a = np.maximum(0, z)
        return a
    elif a_type == 'softmax':
        return z
    else:
        raise ValueError('Unknown actuvation type')


def train_layer(config, X, y, layer):

    neurons = config['layer'][layer]
    alpha = config['learning_rate']
    iterations = config['epochs']
    activation_type = config['activations'][layer].lower().strip()

    layer_weights = []
    layer_output = []
    for i in range(neurons):
        intercept, coef = train_neuron(X, y, alpha, iterations)
        layer_weights.append((intercept, coef))
        a = activation(activation_type, X, intercept, coef)
        layer_output.append(a)
    if activation_type == 'softmax':
        max_np = np.max(layer_output, axis=1, keepdims=True)
        exp = np.exp(layer_output - max_np)
        layer_output = exp / np.sum(exp, axis=1, keepdims=True)
    layer_output = np.array(layer_output).T
    return layer_weights, layer_output


def train_network(config, X, y):
    network_weights = []
    network_outputs = []

    input_X = X
    for layer in range(len(config['layer'])):  
        weights, output = train_layer(config, input_X, y, layer)
        network_weights.append(weights)
        network_outputs.append(output)
        input_X = np.array(output)
    return network_weights, network_outputs


def forward_pass(config, X, weights):

    input_X = X
    outputs = []
    for i, layer in enumerate(weights):
        intercepts = np.array([n[0] for n in layer])
        coefs = np.array([n[1] for n in layer])
        z = np.dot(input_X, coefs.T) + intercepts
        a_type = config['activations'][i].lower().strip()
        if a_type == 'relu':
            a = np.maximum(0, z)
        elif a_type == 'softmax':
            max_np = np.max(z, axis=1, keepdims=True)
            exp = np.exp(z - max_np)
            a = exp / np.sum(exp, axis=1, keepdims=True)
        else:
            raise ValueError('Unknown activation type')
        outputs.append(a)
        input_X = a
    return outputs
        

def prepare_df(df):

    fp_train, fp_val, fp_res = split(df)
    df_train = pd.read_csv(fp_train)
    df_val = pd.read_csv(fp_val)
    df_res = pd.read_csv(fp_res)
    X = df_train.iloc[:, 2:].to_numpy()
    y = df_train['Diagnosis'].map({'B': 0, 'M': 1}).to_numpy()
    X_val = df_val.iloc[:, 1:].to_numpy()
    y_val = df_res.to_numpy()
    return X, y, X_val, y_val



def main():
   
    config = get_config()
    df = pd.read_csv('./data/data.csv')
    X, y, X_val, y_res = prepare_df(df)
    weights, outputs = train_network(config, X, y)

    outputs = forward_pass(config, X_val, weights)
    prob = outputs[-1]
    print(prob)


if __name__ == '__main__':
    main()