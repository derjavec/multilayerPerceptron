import pandas as pd
import numpy as np

from split import split, get_column_names

def prepare_df(df):
    df = clean_df(df)
    fp_train, fp_val, fp_res = split(df)
    df_train = pd.read_csv(fp_train)
    df_val = pd.read_csv(fp_val)
    df_res = pd.read_csv(fp_res)
    X = df_train.drop(columns=['ID', 'Diagnosis']).to_numpy()
    y = df_train['Diagnosis'].str.strip().str.upper().map({'B': 0, 'M': 1}).to_numpy()
    X_val = df_val.iloc[:, 1:].to_numpy()
    y_val = df_res.to_numpy()
    return X, y, X_val, y_val

def clean_df(df):
    df.columns = get_column_names()
    if 'Diagnosis' not in df.columns:
        raise ValueError('Class column missing')
    c_column = df['Diagnosis']
    df = df.drop(columns='Diagnosis')
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.apply(lambda col: col.fillna(col.mean()), axis=0)
    df['Diagnosis'] = c_column
    return df