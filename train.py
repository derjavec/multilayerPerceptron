import numpy as np
import pandas as pd
import pickle
from utils.get_config import get_config, check_config
from utils.train_utils import prepare_df, get_batches, scale
from utils.train_cal import (
    activation,
    gradient_descent,
    initialize_weights,
    linear_regression,
)
from utils.train_plot import loss_plot, acc_plot


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


def loss_and_accuracy(x, y, intercepts, coefs, config):
    """
    Compute cross-entropy loss and accuracy.
    """
    _, a_list, _ = forward(x, config, intercepts, coefs)
    y_pred = a_list[-1]

    loss = -np.mean(
        np.sum(y * np.log(y_pred + 1e-8), axis=1)
    )

    pred_class = np.argmax(y_pred, axis=1)
    true_class = np.argmax(y, axis=1)
    acc = np.mean(pred_class == true_class)

    return loss, acc


def multilayer_perceptron(df, config):
    """
    Train a multi-layer perceptron using backpropagation.
    """
    x_train, y_train, x_val, y_val = prepare_df(df)
    check_config(config, y_train)

    print("X_train shape:", x_train.shape)
    print("X_val shape:", x_val.shape)

    intercepts, coefs = initialize_weights(
        config, x_train.shape[1]
    )

    x_train_scaled, x_min, x_max = scale(x_train)
    x_val_scaled = (x_val - x_min) / (x_max - x_min)

    loss_list = []
    val_loss_list = []
    acc_list = []
    val_acc_list = []

    for epoch in range(1, config["epochs"] + 1):
        for batch_x, batch_y in get_batches(
            x_train_scaled, y_train, config["batch_size"]
        ):
            z_list, a_list, da_dz_list = forward(
                batch_x, config, intercepts, coefs
            )
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

        loss, acc = loss_and_accuracy(
            x_train_scaled, y_train, intercepts, coefs, config
        )
        val_loss, val_acc = loss_and_accuracy(
            x_val_scaled, y_val, intercepts, coefs, config
        )

        loss_list.append(loss)
        val_loss_list.append(val_loss)
        acc_list.append(acc)
        val_acc_list.append(val_acc)

        print(
            f"Epoch {epoch}/{config['epochs']} - "
            f"loss: {loss:.4f} - val_loss: {val_loss:.4f}"
        )

    loss_plot(loss_list, val_loss_list)
    acc_plot(acc_list, val_acc_list)

    return intercepts, coefs


def main():
    """
    Main entry point to train the multilayer perceptron.
    """
    config = get_config()
    df = pd.read_csv(config['config_file'])
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
