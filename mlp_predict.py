import sys
import pickle
import numpy as np
import pandas as pd

from mlp_train import forward
from utils.config import get_model_and_dataset
from utils.mlp_utils import prepare_for_prediction, scale


def predict(df, intercepts, coefs, config):
    """
    Perform predictions on a dataset using the trained MLP.

    Args:
        df (pd.DataFrame): Dataset to predict.
        intercepts (list of np.ndarray): List of intercept vectors per layer.
        coefs (list of np.ndarray): List of weight matrices per layer.
        config (dict): Network configuration.

    Returns:
        np.ndarray: Array of predicted probabilities.
    """
    x, y_val = prepare_for_prediction(df)
    x_scaled, _, _ = scale(x)

    # Forward pass
    _, a_list, _ = forward(x_scaled, config, intercepts, coefs)
    y_pred = a_list[-1]

    # Predicted class (0 = B, 1 = M)
    pred_class = np.argmax(y_pred, axis=1)
    labels = np.array(["B", "M"])
    predicted_labels = labels[pred_class]

    print("Predicted classes for the dataset:")
    print(predicted_labels)

    # If ground truth exists, compute loss and accuracy
    if y_val is not None:
        true_class = np.argmax(y_val, axis=1)
        # Binary Cross-Entropy using probability of class M
        prob_M = y_pred[:, 1]
        eps = 1e-8
        loss = -np.mean(
            true_class * np.log(prob_M + eps) + (1 - true_class) * np.log(1 - prob_M + eps)
        )
        acc = np.mean(pred_class == true_class)
        print(f"\nBinary Cross-Entropy Loss: {loss:.4f}")
        print(f"Accuracy: {acc * 100:.2f}%")

    return y_pred



def main():
    """Main entry point: load model and run predictions on the dataset."""
    model_path, dataset = get_model_and_dataset()
    df = pd.read_csv(dataset)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    coefs = model["coefs"]
    intercepts = model["intercepts"]
    config = model["config"]

    predict(df, intercepts, coefs, config)


if __name__ == "__main__":
    main()
