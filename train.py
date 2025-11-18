import numpy as np
import pandas as pd
from utils.get_config import get_config, check_config
from utils.train_utils import prepare_df, get_batches, scale
from utils.train_cal import activation, gradient_descent, initialize_weights, linear_regression


def forward(x, config , intercepts, coefs):
    z_list = []
    a_list = []
    da_dz_list = []
    a_type_list = config['activations']
    input_x = x
    for layer_idx, _ in enumerate(config['layer']):
        z = linear_regression(input_x, intercepts[layer_idx], coefs[layer_idx])
        a, da_dz = activation(z, a_type_list[layer_idx])
        input_x = a
        z_list.append(z)
        a_list.append(a)
        da_dz_list.append(da_dz)
        
    return z_list, a_list, da_dz_list


def back_propagation(batch_x, batch_y, z_list, a_list, da_dz_list, intercepts, coefs, config):
    
    error = a_list[-1] - batch_y
    error_prop = error

    for layer in reversed(range(len(config['layer']))):
        # print('error_prop', error)
        if layer == 0:
            input_x = batch_x
        else:
            input_x = a_list[layer - 1]
        intercept, coef = gradient_descent(input_x, intercepts[layer], coefs[layer], error_prop, da_dz_list[layer], config['learning_rate'])
        if layer > 0:
            error_prop = (error_prop @ np.array(coefs[layer])) *  da_dz_list[layer - 1]
        intercepts[layer] = intercept
        coefs[layer] = coef
    return intercepts, coefs
    
def loss_cal(x, y, intercepts, coefs, config):
    _, a_list, _ = forward(x, config, intercepts, coefs) 
    y_pred = a_list[-1]
    loss = -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))
    return loss
    
    
def train_network(x_train, y_train, x_val, y_val, config):
    """Train a full multi-layer perceptron."""
    print("X_train shape:", x_train.shape)
    print("X_val shape:", x_val.shape)

    intercepts, coefs = initialize_weights(config, x_train.shape[1]) #1:Inicializar pesos una sola vez (antes de los epochs).
    # print(f'interceptos iniciales: {intercepts}, coef iniciales {coefs}')

    x_train_scaled, x_train_min, x_train_max = scale(x_train)
    x_val_scaled = (x_val - x_train_min) / (x_train_max - x_train_min)

    for epoch in range(1, config['epochs'] + 1):
        for batch_x, batch_y in get_batches(x_train_scaled, y_train, config['batch_size']):
            # print('intercepts', intercepts)
            # print('coefs', coefs)
            z_list, a_list, da_dz_list = forward(batch_x, config, intercepts, coefs)
            intercepts, coefs = back_propagation(batch_x, batch_y, z_list, a_list, da_dz_list, intercepts, coefs, config)
        
        loss = loss_cal(x_train_scaled, y_train, intercepts, coefs, config)
        val_loss = loss_cal(x_val_scaled, y_val, intercepts, coefs, config)
        print(f"Epoch {epoch}/{config['epochs']} - loss: {loss:.4f} - val_loss: {val_loss:.4f}")

    return intercepts, coefs


def main():
    """Main function to train network."""
    config = get_config()
    df = pd.read_csv("./data/data.csv")

    x_train, y_train, x_val, y_val = prepare_df(df)
    check_config(config, y_train)
    intercepts, coefs = train_network(x_train, y_train, x_val, y_val, config)


if __name__ == "__main__":
    main()
