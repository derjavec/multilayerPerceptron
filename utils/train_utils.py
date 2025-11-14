import pandas as pd
import numpy as np

from split import split, get_column_names


def activation(z, activation_type):
    if activation_type == 'relu':
        a = np.maximum(0, z)
        da_dz = (z > 0).astype(float)
    elif activation_type == 'softmax':
        exp_z = np.exp(z - np.max(z))
        a = exp_z / np.sum(exp_z)
        da_dz = 1
    else:
        raise ValueError("Unknown activation type")
    return a, da_dz

def gradient_descent(X, intercept, coef, error, da_dz, alpha):
    d_intercept = np.mean(error * da_dz)
    d_coef = np.mean((error * da_dz)[:, None] * X, axis=0)
    
    intercept -= alpha * d_intercept
    coef -= alpha * d_coef
    return intercept, coef


def scale(X: np.ndarray):
    """
    Scale an array to the [0, 1] range.
    """
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    scaled = (X - x_min) / (x_max - x_min)
    return scaled, x_min, x_max


def get_batches(X, y, batch_size):
    n_samples = X.shape[0]
    for i in range(0, n_samples, batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        yield X_batch, y_batch


def initialize_coef(x_train, neurons):
    layer_weights = []
    for _ in range(neurons):
        coef = np.random.randn(x_train.shape[1]) * 0.01
        intercept = np.random.randn() * 0.01
        layer_weights.append((intercept, coef))
    return layer_weights


def prepare_df(df):

    fp_train, fp_val, fp_res = split(df)
    df_train = pd.read_csv(fp_train)
    df_val = pd.read_csv(fp_val)
    df_res = pd.read_csv(fp_res)
    X = df_train.drop(columns=['ID', 'Diagnosis']).to_numpy()
    y = df_train['Diagnosis'].map({'B': 0, 'M': 1}).to_numpy()
    X_val = df_val.iloc[:, 1:].to_numpy()
    y_val = df_res.to_numpy()
    return X, y, X_val, y_val

def clean_df(df):
    df.columns = get_column_names()
    if 'Diagnosis' not in df.columns:
        raise ValueError('Class column missing')
    c_column = df['Diagnosis']
    df = df.drop(columns='Diagnosis')
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.apply(lambda col: col.fillna(col.mean()), axis=0)
    df['Diagnosis'] = c_column
    return df