import numpy as np
import pandas as pd
from utils.get_config import get_config, get_config_items
from utils.train_utils import prepare_df, get_batches, activation, gradient_descent, initialize_coef


def forward_layer(x, layer_weights, a_type):
    """Compute actuvations for a layer."""
    coefs = np.array([coef for intercept, coef in layer_weights])
    intercepts = np.array([intercept for intercept, coef in layer_weights])

    z = x @ coefs.T + intercepts

    if a_type == "relu":
        a = np.maximum(0, z)
    elif a_type == "softmax":
        max_z = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - max_z)
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown activation type: {a_type}")

    return a


def backprop_neuron(x_batch, y_batch, intercept, coef, activation_type, alpha):
    """Compute gradient and update weights for a single neuron."""
    z = intercept + np.dot(x_batch, coef)
    a, da_dz = activation(z, activation_type)
    error = a - y_batch
    intercept, coef = gradient_descent(x_batch, intercept, coef, error, da_dz, alpha)
    return intercept, coef



def train_layer(x, y, layer_idx, config):
    """Train one layer of neurons."""
    layers = config['layer']
    neurons = layers[layer_idx]
    a_type_list = config['activations']
    a_type = a_type_list[layer_idx]
    alpha = config['learning_rate']
    
    init_intercepts, init_coefs = initialize_coef(x, neurons)
    layer_weights = []
    for n in range(neurons):
        intercept, coef = backprop_neuron(x, y, init_intercepts[n],
                                            init_coefs[n], a_type, alpha)
        layer_weights.append((intercept, coef))
    # Compute layer output for next layer
    layer_output = forward_layer(x, layer_weights, a_type)
    return layer_weights, layer_output


def get_class(y):
    max_idx = []
    for p in y:
        max_idx.append(np.argmax(p))
    return(max_idx)

def train_network(x_train, y_train, x_val, y_val, config):
    """Train a full multi-layer perceptron."""
    print("X_train shape:", x_train.shape)
    print("X_val shape:", x_val.shape)

    network_weights = []
    for epoch in range(1, config['epochs'] + 1):
        
        for batch_x, batch_y in get_batches(x_train, y_train, config['batch_size']):
            input_x = batch_x
            epoch_layer_weights = []
            for layer_idx, _ in enumerate(config['layer']):
                weights, output = train_layer(input_x, batch_y, layer_idx, config)
                epoch_layer_weights.append(weights)
                input_x = output
            network_weights = epoch_layer_weights
        
        val = x_train
        for layer_idx, layer_weights in enumerate(network_weights):
            val = forward_layer(val, layer_weights, config['activations'][layer_idx])
        y_pred = val
        print(y_pred)
        loss = np.mean((y_pred - y_train) ** 2) / 2

        # validación
        val = x_val
        for layer_idx, layer_weights in enumerate(network_weights):
            val = forward_layer(val, layer_weights, config['activations'][layer_idx])
        y_val_pred= val
        val_loss = np.mean((y_val_pred - y_val) ** 2) / 2
        print(f"Epoch {epoch}/{config['epochs']} - loss: {loss:.4f} - val_loss: {val_loss:.4f}")

    return network_weights


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
    network_weights = train_network(x_train, y_train, x_val, y_val, config)
    # print("Network weights:", network_weights)


if __name__ == "__main__":
    main()
