import pandas as pd
import numpy as np
import sys
import pickle
from train import forward
from utils.get_config import get_model_and_dataset
from utils.train_utils import prepare_for_prediction, scale


def predict(df, intercepts, coefs, config):

    x = prepare_for_prediction(df)
    x_scaled, _, _ = scale(x)
    _, a_list, _ = forward(x_scaled, config, intercepts, coefs)
    y_pred = a_list[-1]
    y_class = np.argmax(y_pred, axis=1)
    labels = np.array(["B", "M"])
    predicted_labels = labels[y_class]
    print(predicted_labels)

def main():
    """Main function to train network."""
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