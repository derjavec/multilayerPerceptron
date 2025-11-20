import numpy as np
import pandas as pd
from typing import Tuple

from split import split, get_column_names


def scale(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale an array to the [0, 1] range.
    """
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    scaled = (X - x_min) / (x_max - x_min)
    return scaled, x_min, x_max


def get_batches(
    X: np.ndarray, y: np.ndarray, batch_size: int
):
    """
    Yield mini-batches of data.
    """
    n_samples = X.shape[0]
    for idx in range(0, n_samples, batch_size):
        yield X[idx:idx + batch_size], y[idx:idx + batch_size]


def classes_to_one_hot(
    y: np.ndarray, num_classes: int
) -> np.ndarray:
    """
    Convert a vector of class indices to one-hot encoding.
    """
    y = np.asarray(y, dtype=int)

    if y.min() < 0 or y.max() >= num_classes:
        raise ValueError("Class values out of valid range.")

    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


def prepare_for_training(
    df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare the dataset: clean, split, one-hot encode, and return matrices.
    """
    df_clean = clean_df(df)
    fp_train, fp_val, fp_res = split(df_clean)

    df_train = pd.read_csv(fp_train)
    df_val = pd.read_csv(fp_val)
    df_res = pd.read_csv(fp_res)

    train_features = df_train.drop(columns=["ID", "Diagnosis"])
    val_features = df_val.iloc[:, 1:]

    X_train = train_features.to_numpy()
    X_val = val_features.to_numpy()

    y_train_raw = df_train["Diagnosis"].map({"B": 0, "M": 1}).to_numpy()
    y_val_raw = df_res["Diagnosis"].map({"B": 0, "M": 1}).to_numpy()

    y_train = classes_to_one_hot(y_train_raw, num_classes=2)
    y_val = classes_to_one_hot(y_val_raw, num_classes=2)

    return X_train, y_train, X_val, y_val


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the input DataFrame:
    - Apply correct column names
    - Extract and restore the Diagnosis column
    - Convert numerical columns to numeric
    - Fill NaN values with column means

    """
    df = df.copy()
    df.columns = get_column_names()

    if "Diagnosis" not in df.columns:
        raise ValueError("Missing 'Diagnosis' column.")

    diagnosis_col = df["Diagnosis"]
    df = df.drop(columns="Diagnosis")

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.apply(lambda col: col.fillna(col.mean()), axis=0)

    df["Diagnosis"] = diagnosis_col
    return df


def prepare_for_prediction(df: pd.DataFrame):
    """
    Prepare the dataset for prediction:
    - Clean df
    - Select only numeric columns
    - Drop ID if present
    - Return X
    """
    df = clean_df(df)
    y = None
    if 'Diagnosis' in df.columns:
        y = df['Diagnosis'].map({"B": 0, "M": 1}).to_numpy()
        y_val = classes_to_one_hot(y, num_classes=2)
    numeric_df = df.select_dtypes(include=[np.number])

    if "ID" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["ID"])

    x = numeric_df.to_numpy()

    return x, y_val
