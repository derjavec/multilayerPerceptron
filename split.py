import pandas as pd
import os
import sys

def get_column_names():
    column_names = [
        'ID', 'Diagnosis', 
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    return column_names


def split(df: pd.DataFrame, train_frac: float = 0.8):
    """
    Split a DataFrame into training and test sets, then save them as CSV files.

    """

    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be a float between 0 and 1.")

    # df.columns = get_column_names()
    df_shuffled = df.sample(frac=1, random_state=42)
    train_size = int(len(df_shuffled) * train_frac)

    df_train = df_shuffled.iloc[:train_size]
    df_val = df_shuffled.iloc[train_size:]
    output_folder = "generated_files"
    os.makedirs(output_folder, exist_ok=True)
    fp_train = os.path.join(output_folder, "train.csv")
    fp_val = os.path.join(output_folder, "validate.csv")
    fp_res = os.path.join(output_folder, "val_results.csv")
    df_train.to_csv(fp_train, index=False)
    val_results = df_val['Diagnosis']
    val_results.to_csv(fp_res, index=False)
    df_val = df_val.drop(columns=['Diagnosis'])
    df_val.to_csv(fp_val, index=False)
    return fp_train, fp_val, fp_res

def main():
    if len(sys.argv) != 2:
        raise ValueError('Usage: please run the progrm with the file you want to split')
    
    df = pd.read_csv(sys.argv[1])
    if df.empty:
        raise ValueError('The file provided is empty')
    split(df)


if __name__ == '__main__':
    main()