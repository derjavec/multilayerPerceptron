import argparse
import json
import os
import ast
import numpy as np

DEFAULT_CONFIG = {
    "dataset" : './data/data.csv',
    "layer": [32, 16, 16, 2],
    "activations": ["relu", "relu", "relu", "softmax"],
    "epochs": 50,
    "batch_size": 8,
    "learning_rate": 0.01,
    "bonus" : 0
}


def read_config_file(file_path: str) -> dict:
    """
    Read a configuration file in JSON or TXT format.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".json":
        with open(file_path) as f:
            return json.load(f)

    if ext == ".txt":
        config = {}
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, value = line.split("=")
                key = key.strip()
                value = value.strip()
                try:
                    config[key] = ast.literal_eval(value)
                except Exception:
                    config[key] = value
        return config

    raise ValueError("Unknown file format. Please use .json or .txt")


def replace_dict_values(base: dict, new: dict) -> dict:
    """
    Replace values in a base dictionary with values from a new dictionary.
    """
    for k, v in new.items():
        if v is not None:
            base[k] = v
    return base


def check_config(config: dict, y: np.ndarray):
    """
    Validate network configuration.
    """
    layers = config["layer"]
    activations = config["activations"]

    if len(layers) != len(activations):
        raise ValueError(
            "Configuration Error: 'layer' and 'activations' must have the same length."
        )

    for i, val in enumerate(layers):
        try:
            float(val)
        except Exception:
            raise ValueError(
                f"Configuration Error: value in 'layer' at index {i} ('{val}') must be numeric."
            )

    for key in ["epochs", "batch_size", "learning_rate"]:
        try:
            float(config[key])
        except Exception:
            raise ValueError(
                f"Configuration Error: '{key}' must be numeric. Found: {config[key]}"
            )

    if activations[-1].strip().lower() != 'softmax':
        raise ValueError(
                f"Configuration Error: last activation should be 'softmax', and it's {activations[-1].strip().lower()}"
            )
    n_classes = len(np.unique(y))
    if n_classes != int(layers[-1]):
        raise ValueError(
            f"Configuration Error: last layer must contain {n_classes} neurons, "
            f"but found {layers[-1]}"
        )


def get_config() -> dict:
    """
    Load and parse the configuration from defaults, command-line arguments,
    or a file.
    """
    config = DEFAULT_CONFIG.copy()

    parser = argparse.ArgumentParser(description="Train a neural network")
    parser.add_argument(
        "config_file",
        nargs="?", 
        type=str,
        help="Path to config file (optional)",
    )
    parser.add_argument(
        "--dataset",
        nargs="?", 
        type=str,
        help="Path to dataset (optional)",
    )
    parser.add_argument(
        "--bonus",
        action="store_true",
        help="Excecute bonus (optional)",
    )
    parser.add_argument("--layer", nargs="+", type=int, help="Number of neurons per layer")
    parser.add_argument("--activations", nargs="+", type=str, help="Activation functions per layer")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")

    args = parser.parse_args()
    args_dict = vars(args)

    if args_dict.get("config_file"):
        config_dict = read_config_file(args_dict.pop("config_file"))
        config = replace_dict_values(config, config_dict)

    config = replace_dict_values(config, args_dict)
    return config


def get_model_and_dataset():
    """
    Load and parse the configuration from defaults, command-line arguments,
    or a file.
    """
    model = 'model.pkl'
    dataset = './data/data.csv'

    parser = argparse.ArgumentParser(description="Train a neural network")
    parser.add_argument(
        "--dataset",
        nargs="?", 
        type=str,
        help="Path to dataset file (optional)",
    )
    parser.add_argument("--model", nargs="?", type=str, help="mlp model")

    args = parser.parse_args()
    if args.dataset:
        dataset = args.dataset
    if args.model:
        model = args.model
    return model, dataset
