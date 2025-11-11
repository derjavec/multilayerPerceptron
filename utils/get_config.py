import argparse
import json
import os


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
        f = open(file_path)
        return json.load(f)
    elif ext == '.txt':
        config = {}
        f = open(file_path)
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, value = line.split('=')
            key = key.strip()
            value = value.strip()
            config[key] = value
        return config
    else:
        raise ValueError('Unknown file format, please use .json or .txt')

def replace_dict_values(base, new):
    for k, v in new.items():
        if v is not None:
            base[k] = v
    return base

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
