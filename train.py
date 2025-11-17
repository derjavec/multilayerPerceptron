import numpy as np
import pandas as pd
from utils.get_config import get_config, get_config_items
from utils.train_utils import prepare_df, get_batches, activation, gradient_descent, initialize_weights


# def forward_layer(x, layer_weights, a_type):
#     """Compute actuvations for a layer."""
#     coefs = np.array([coef for intercept, coef in layer_weights])
#     intercepts = np.array([intercept for intercept, coef in layer_weights])

#     z = x @ coefs.T + intercepts

#     if a_type == "relu":
#         a = np.maximum(0, z)
#     elif a_type == "softmax":
#         max_z = np.max(z, axis=1, keepdims=True)
#         exp_z = np.exp(z - max_z)
#         a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
#     else:
#         raise ValueError(f"Unknown activation type: {a_type}")

#     return a


def backprop_neuron(x_batch, y_batch, intercept, coef, activation_type, alpha):
    """Compute gradient and update weights for a single neuron."""
    z = intercept + np.dot(x_batch, coef)
    a, da_dz = activation(z, activation_type)
    error = a - y_batch
    intercept, coef = gradient_descent(x_batch, intercept, coef, error, da_dz, alpha)
    return intercept, coef



# def train_layer(x, y, layer_idx, config):
#     """Train one layer of neurons."""
#     layers = config['layer']
#     neurons = layers[layer_idx]
#     a_type_list = config['activations']
#     a_type = a_type_list[layer_idx]
#     alpha = config['learning_rate']
    
   
#     layer_weights = []
#     for n in range(neurons):
#         intercept, coef = backprop_neuron(x, y, init_intercepts[n],
#                                             init_coefs[n], a_type, alpha)
#         layer_weights.append((intercept, coef))
#     # Compute layer output for next layer
#     layer_output = forward_layer(x, layer_weights, a_type)
#     return layer_weights, layer_output


# def get_class(y):
#     max_idx = []
#     for p in y:
#         max_idx.append(np.argmax(p))
#     return(max_idx)


def linear_regression(x, intercepts, coefs):
    coefs = np.array(coefs)            # (n_neurons, n_inputs)
    intercepts = np.array(intercepts) 
    z = x @ coefs.T + intercepts 
    return z

def forward(x, config , intercepts, coefs): #4 Para cada batch: forward completo por todas las capas (no entrenar capa a capa en el forward). Guardar z y a de cada capa.
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


def backprop(batch_x, batch_y, z_list, a_list, da_dz_list, intercepts, coefs, config):
    
    error = a_list[-1] - batch_y
    error_prop = error

    for layer in reversed(range(len(config['layer']))):
        if layer == 0:
            input_x = batch_x
        else:
            input_x = a_list[layer - 1]
        intercept, coef = gradient_descent(input_x, intercepts[layer], coefs[layer], error_prop, da_dz_list[layer], config['learning_rate'])
        if layer > 0:
            error_prop = (error_prop @ np.array(coefs[layer])) *  da_dz_list[layer - 1]

    return intercepts, coefs
    


def train_network(x_train, y_train, x_val, y_val, config):
    """Train a full multi-layer perceptron."""
    print("X_train shape:", x_train.shape)
    print("X_val shape:", x_val.shape)

    intercepts, coefs = initialize_weights(config, x_train.shape[1]) #1:Inicializar pesos una sola vez (antes de los epochs).
    final_intercepts = []
    final_coefs = []
    for epoch in range(1, config['epochs'] + 1): #Bucle de epochs.
        for batch_x, batch_y in get_batches(x_train, y_train, config['batch_size']): #Dentro: bucle por batches.
            z_list, a_list, da_dz_list = forward(batch_x, config, intercepts, coefs) 
            intercepts, coefs = backprop(batch_x, batch_y, z_list, a_list, da_dz_list, intercepts, coefs, config)
        # print(f'intercept {intercepts}')
        # print(f'coefs {coefs}')
        _, a_list, _ = forward(x_train, config, intercepts, coefs) 
        y_pred = a_list[-1]
        # print(y_pred)
        loss = -np.mean(np.sum(y_train * np.log(y_pred + 1e-8), axis=1))
        _, a_list, _ = forward(x_val, config, intercepts, coefs) 
        y_val_pred = a_list[-1]
        # print(y_val_pred)
        val_loss = -np.mean(np.sum(y_val * np.log(y_val_pred + 1e-8), axis=1))
        print(f"Epoch {epoch}/{config['epochs']} - loss: {loss:.4f} - val_loss: {val_loss:.4f}")

    return intercepts, coefs


# def forward_pass(x_input, network_weights, config):
#     """Compute network output for given input."""
#     input_x = x_input
#     for layer_idx, layer_weights in enumerate(network_weights):
#         activation_type = config["activations"][layer_idx].lower()
#         input_x = forward_layer(input_x, layer_weights, activation_type)
#     return input_x


def main():
    """Main function to train network."""
    config = get_config()
    df = pd.read_csv("./data/data.csv")

    x_train, y_train, x_val, y_val = prepare_df(df)
    intercepts, coefs = train_network(x_train, y_train, x_val, y_val, config)
    # print("Network weights:", network_weights)


if __name__ == "__main__":
    main()
