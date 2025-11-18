import argparse
import json
import os
import ast
import numpy as np


DEFAULT_CONFIG = {
    'layer': [16, 16, 2],
    'activations': ['relu', 'relu', 'softmax'],
    'epochs': 50,
    'batch_size': 16,
    'learning_rate': 0.01
}


def read_config_file(file_path):
    ext = os.path.splitext(file_path)[1]
    
    if ext == '.json':
        with open(file_path) as f:
            return json.load(f)
    
    elif ext == '.txt':
        config = {}
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                key, value = line.split('=')
                key = key.strip()
                value = value.strip()
                
                try:
                    config[key] = ast.literal_eval(value)
                except:
                    config[key] = value
        return config
    
    else:
        raise ValueError('Unknown file format, please use .json or .txt')

def replace_dict_values(base, new):
    for k, v in new.items():
        if v is not None:
            base[k] = v
    return base


def check_config(config, y):
    layer = config['layer']
    activations = config['activations']

    if len(layer) != len(activations):
        raise ValueError("Configuration Error: 'layer' and 'activations' must be the same length.")

    for i, val in enumerate(layer):
        try:
            float(val)
        except:
            raise ValueError(f"Configuration Error: value in 'layer' at index {i} ('{val}') must be numeric.")
    try:
        float(config["epochs"])
    except:
        raise ValueError(f"Configuration Error: 'epochs' must be numeric. Found: {config['epochs']}")

    try:
        float(config["batch_size"])
    except:
        raise ValueError(f"Configuration Error: 'batch_size' must be numeric. Found: {config['batch_size']}")

    try:
        float(config["learning_rate"])
    except:
        raise ValueError(f"Configuration Error: 'learning_rate' must be numeric. Found: {config['learning_rate']}")

    n_classes = len(np.unique(y))
    if n_classes != int(layer[-1]):
        raise ValueError(
            f"Configuration Error: last layer must contain {n_classes} neurons, "
            f"but found {layer[-1]}"
        )



def get_config():

    config = DEFAULT_CONFIG.copy()
    
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument(
        'config_file',
        nargs='?',
        type=str,
        help='Path to config JSON file (optional)'
    )
    parser.add_argument("--layer", nargs="+", type=int, help="Number of neurons per layer")
    parser.add_argument("--activations", nargs="+", type=str, help="Activation functions per layer")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")

    args = parser.parse_args()
    args_dict = vars(args)
    if args_dict.get('config_file'):
        config_dict = read_config_file(args_dict['config_file'])
        config = replace_dict_values(config, config_dict)
        args_dict.pop('config_file', None)
    config = replace_dict_values(config, args_dict)
    return config


# def get_activation_type(config, layer_idx):
#     activation_type = config["activations"][layer_idx].lower()
#     return activation_type


# def get_config_items(config):

#     alpha = config["learning_rate"]
#     epochs = config["epochs"]
#     batch_size = config["batch_size"]
#     return alpha, epochs, batch_size
