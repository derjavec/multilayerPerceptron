import numpy as np
import pandas as pd
from utils.get_config import get_config, get_config_items
from utils.train_utils import prepare_df, get_batches, activation, gradient_descent, initialize_coef


def forward_layer(x_input, layer_weights, activation_type):
    """Compute activations for a layer."""
    coefs = np.array([coef for intercept, coef in layer_weights])  # (n_neurons, n_features)
    intercepts = np.array([intercept for intercept, coef in layer_weights])  # (n_neurons,)

    z = x_input @ coefs.T + intercepts

    if activation_type == "relu":
        a = np.maximum(0, z)
    elif activation_type == "softmax":
        max_z = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - max_z)
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")

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
    neurons = config['layer'][layer_idx]
    a_type = config['activation_type'][layer_idx]
    alpha = config['alpha']
    
    init_weights = initialize_coef(x, neurons)

    new_weights = []
    for n in neurons:
        intercept, coef = backprop_neuron(x, y, intercept,
                                            coef, a_type, alpha)
        new_weights.append((intercept, coef))
    layer_weights = new_weights

        # # Compute losses at the end of epoch
        # y_pred = forward_layer(x_train, layer_weights, activation_type)
        # loss = np.mean((y_pred - y_train) ** 2) / 2

        # y_val_pred = forward_layer(x_val, layer_weights, activation_type)
        # val_loss = np.mean((y_val_pred - y_val) ** 2) / 2
        # print(f"Epoch {epoch}/{epochs} - loss: {loss:.4f} - val_loss: {val_loss:.4f}")


    # Compute layer output for next layer
    layer_output = forward_layer(x_train, layer_weights, activation_type)
    return layer_weights, layer_output


def train_network(x_train, y_train, x_val, y_val, config):
    """Train a full multi-layer perceptron."""
    print("X_train shape:", x_train.shape)
    print("X_val shape:", x_val.shape)

    for epoch in range(1, epochs + 1):
        for batch_x, batch_y in get_batches(x_train, y_train, batch_size):
            for layer_idx in eumerate(config['layer'])
                input_x = batch_x
                weights, output = train_layer(input_x, batch_y, layer_idx, config)
                input_x = output
    # input_x = x_train
    # network_weights = []

    # for layer_idx, neurons in enumerate(config["layer"]):
    #     weights, output = train_layer(input_x, y_train, x_val, y_val,
    #                                   neurons, layer_idx, config)
    #     network_weights.append(weights)
    #     input_x = output

    # return network_weights


def forward_pass(x_input, network_weights, config):
    """Compute network output for given input."""
    input_x = x_input
    for layer_idx, layer_weights in enumerate(network_weights):
        activation_type = config["activations"][layer_idx].lower()
        input_x = forward_layer(input_x, layer_weights, activation_type)
    return input_x


def main():
    """Main function to train network."""
    config = get_config()
    df = pd.read_csv("./data/test.csv")

    x_train, y_train, x_val, y_val = prepare_df(df)
    network_weights = train_network(x_train, y_train, x_val, y_val, config)
    print("Network weights:", network_weights)


if __name__ == "__main__":
    main()
