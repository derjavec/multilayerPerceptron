import sys
import pickle
import numpy as np
import pandas as pd
import os

from mlp_train import forward
from utils.config import get_model_and_dataset
from utils.mlp_utils import prepare_for_prediction, scale


def predict(df, intercepts, coefs, config):
    """
    Perform predictions on a dataset using the trained MLP.
    """
    x, y_val = prepare_for_prediction(df)
    x_scaled, _, _ = scale(x)

    _, a_list, _ = forward(x_scaled, config, intercepts, coefs)
    y_pred = a_list[-1]

    loss, acc = None, None

    if y_val is not None:
        true_class = np.argmax(y_val, axis=1)
        pred_class = np.argmax(y_pred, axis=1)

        prob_M = y_pred[:, 1]
        eps = 1e-8
        loss = -np.mean(
            true_class * np.log(prob_M + eps) +
            (1 - true_class) * np.log(1 - prob_M + eps)
        )
        acc = np.mean(pred_class == true_class)

        print(f"Binary Cross-Entropy Loss: {loss:.4f}")
        print(f"Accuracy: {acc * 100:.2f}%")

    return y_pred, loss, acc


def load_models(models_paths):
    """
    Load all models from a list of paths.
    """
    models = []

    for model_path in models_paths:
        model_name = os.path.splitext(os.path.basename(model_path))[0]

        print(f"\n📦 Loading model: {model_name}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        models.append((model_name, model))

    return models


def run_prediction_for_model(df, model_name, model):
    """
    Runs prediction for one model and returns predictions + stats.
    """
    coefs = model["coefs"]
    intercepts = model["intercepts"]
    config = model["config"]

    print(f"🔮 Predicting with model: {model_name}")

    y_pred, loss, acc = predict(df, intercepts, coefs, config)

    y_pred_class = np.argmax(y_pred, axis=1)

    labels = np.array(["B", "M"])
    predicted_labels = labels[y_pred_class]

    stats = {
        "loss": float(loss) if loss is not None else None,
        "acc": float(acc) if acc is not None else None
    }

    return predicted_labels, stats


def build_predictions_df(predictions_dict):
    """Create dataframe from predictions."""
    return pd.DataFrame(predictions_dict)


def build_stats_df(stats_dict):
    """Create dataframe from stats."""
    return pd.DataFrame(stats_dict, index=["loss", "acc"])


def save_predictions_csv(df_pred, df_stats, output_folder="generated_files"):
    """
    Merge predictions + stats and save into a single CSV.
    """
    df_final = pd.concat([df_pred, df_stats])

    df_final.insert(0, "id", df_final.index)

    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, "predictions.csv")
    df_final.to_csv(output_path, index=False)

    print(f"\n✅ Predictions saved to: {output_path}")


def main():
    models_paths, dataset_path = get_model_and_dataset()

    if not models_paths:
        print("❌ No models found.")
        return

    df = pd.read_csv(dataset_path)

    models = load_models(models_paths)

    predictions = {}
    stats = {}

    for model_name, model in models:
        preds, model_stats = run_prediction_for_model(df, model_name, model)
        predictions[model_name] = preds
        stats[model_name] = model_stats

    df_pred = build_predictions_df(predictions)
    df_stats = build_stats_df(stats)

    save_predictions_csv(df_pred, df_stats)


if __name__ == "__main__":
    main()
