import numpy as np
import pandas as pd
import pickle
from utils.config import get_config, check_config
from utils.mlp_utils import prepare_for_training, get_batches, scale
from utils.calculations import (
    activation,
    gradient_descent,
    initialize_weights,
    linear_regression,
)
from utils.plots import loss_plot, acc_plot
from bonus.adam import init_adam_state, back_propagation_bonus
from bonus.history import build_history


def forward(x, config, intercepts, coefs):
    """
    Perform the forward pass of the MLP.
    """
    z_list = []
    a_list = []
    da_dz_list = []

    input_x = x
    for layer_idx, _ in enumerate(config["layer"]):
        z = linear_regression(
            input_x, intercepts[layer_idx], coefs[layer_idx]
        )
        a, da_dz = activation(z, config["activations"][layer_idx])

        z_list.append(z)
        a_list.append(a)
        da_dz_list.append(da_dz)

        input_x = a

    return z_list, a_list, da_dz_list


def back_propagation(batch_x, batch_y, z_list, a_list,
                     da_dz_list, intercepts, coefs, config):
    """
    Perform the backward pass and update weights.
    """
    error = a_list[-1] - batch_y
    error_prop = error

    for layer in reversed(range(len(config["layer"]))):
        if layer == 0:
            input_x = batch_x
        else:
            input_x = a_list[layer - 1]

        intercept, coef = gradient_descent(
            input_x,
            intercepts[layer],
            coefs[layer],
            error_prop,
            da_dz_list[layer],
            config["learning_rate"],
        )

        if layer > 0:
            error_prop = (
                error_prop @ np.array(coefs[layer])
            ) * da_dz_list[layer - 1]

        intercepts[layer] = intercept
        coefs[layer] = coef

    return intercepts, coefs


def multilayer_perceptron(df, config):
    """
    Train a multi-layer perceptron using backpropagation.
    """
    x_train, y_train, x_val, y_val = prepare_for_training(df)
    check_config(config, y_train)

    print("X_train shape:", x_train.shape)
    print("X_val shape:", x_val.shape)

    intercepts, coefs = initialize_weights(
        config, x_train.shape[1]
    )

    x_train_scaled, x_min, x_max = scale(x_train)
    x_val_scaled = (x_val - x_min) / (x_max - x_min)

    if config['optimizer'] == 'adam':
        adam_state = init_adam_state(intercepts, coefs)
    history = None
    for epoch in range(1, config["epochs"] + 1):
        for batch_x, batch_y in get_batches(
            x_train_scaled, y_train, config["batch_size"]
        ):
            z_list, a_list, da_dz_list = forward(
                batch_x, config, intercepts, coefs
            )
            if config['bonus']:
                intercepts, coefs = back_propagation_bonus(
                    batch_x,
                    batch_y,
                    z_list,
                    a_list,
                    da_dz_list,
                    intercepts,
                    coefs,
                    config,
                    adam_state,
                )
            else:
                intercepts, coefs = back_propagation(
                    batch_x,
                    batch_y,
                    z_list,
                    a_list,
                    da_dz_list,
                    intercepts,
                    coefs,
                    config,
                )
        _, a_list, _ = forward(x_train_scaled, config, intercepts, coefs)
        y_pred = a_list[-1]

        _, a_val_list, _ = forward(x_val_scaled, config, intercepts, coefs)
        y_val_pred = a_val_list[-1]
        history = build_history(history, epoch, y_train, y_pred, y_val, y_val_pred, config)

    return intercepts, coefs


def main():
    """
    Main entry point to train the multilayer perceptron.
    """
    config = get_config()
    df = pd.read_csv(config['dataset'])
    intercepts, coefs = multilayer_perceptron(df, config)
    
    model = {
        "coefs": coefs,
        "intercepts": intercepts,
        "config": config
    }
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved as 'model.pkl'.")


if __name__ == "__main__":
    main()
