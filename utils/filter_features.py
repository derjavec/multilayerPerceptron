from utils.describe import describe
import pandas as pd

def get_description_per_class(df):
    """Compute descriptive statistics per class."""
    if 'Diagnosis' not in df.columns:
        raise ValueError('Class column missing')
    
    stats = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    des = {}
    for c in df['Diagnosis'].unique():
        df_c = df[df['Diagnosis'] == c]
        df_des = describe(df_c, stats)
        des[c] = df_des
    return des


def mean_dispersion(des_per_class, df):
    """Normalized difference of feature means between two classes using full dataset range."""
    c1, c2 = list(des_per_class.keys())
    mean1 = des_per_class[c1].loc["Mean"]
    mean2 = des_per_class[c2].loc["Mean"]
    diff = abs(mean1 - mean2)
    feature_range = df.select_dtypes(include="number").max() - df.select_dtypes(include="number").min()
    norm_diff = diff / feature_range.replace(0, 1e-9)
    return norm_diff


def median_dispersion(des_per_class, df):
    """Normalized difference of medians between two classes using full dataset range."""
    c1, c2 = list(des_per_class.keys())
    med1 = des_per_class[c1].loc["50%"]
    med2 = des_per_class[c2].loc["50%"]

    diff = abs(med1 - med2)
    feature_range = df.select_dtypes(include="number").max() - df.select_dtypes(include="number").min()
    norm_diff = diff / feature_range.replace(0, 1e-9)
    return norm_diff


def std_noise(des_per_class, df):
    """Normalized maximum standard deviation across two classes using full dataset range."""
    c1, c2 = list(des_per_class.keys())
    std1 = des_per_class[c1].loc["Std"]
    std2 = des_per_class[c2].loc["Std"]

    max_std = pd.concat([std1, std2], axis=1).max(axis=1)
    feature_range = df.select_dtypes(include="number").max() - df.select_dtypes(include="number").min()
    norm_std = max_std / feature_range.replace(0, 1e-9)
    return norm_std


def q_outliers(des_per_class, df):
    """Normalized maximum IQR across two classes using full dataset range."""
    c1, c2 = list(des_per_class.keys())
    q25_1 = des_per_class[c1].loc["25%"]
    q25_2 = des_per_class[c2].loc["25%"]
    q75_1 = des_per_class[c1].loc["75%"]
    q75_2 = des_per_class[c2].loc["75%"]

    iqr_max = pd.concat([q75_1, q75_2], axis=1).max(axis=1) - pd.concat([q25_1, q25_2], axis=1).min(axis=1)
    feature_range = df.select_dtypes(include="number").max() - df.select_dtypes(include="number").min()
    norm_iqr = iqr_max / feature_range.replace(0, 1e-9)
    return norm_iqr


def filter_features_by_description(des_per_class, df):
    """Select features that discriminate between classes with low noise and low outliers."""
    mean_data = mean_dispersion(des_per_class, df)
    median_data = median_dispersion(des_per_class, df)
    std_data = std_noise(des_per_class, df)
    iqr_data = q_outliers(des_per_class, df)
    # thresholds
    keep_mean = mean_data[mean_data >= 0]
    keep_median = median_data[median_data >= 0]
    keep_std = std_data[std_data <= 1]
    keep_iqr = iqr_data[iqr_data <= 1]

    features = list(
        keep_mean.index
        .intersection(keep_median.index)
        .intersection(keep_std.index)
        .intersection(keep_iqr.index)
    )
    return features


def filter_features(df):
    """Compute descriptive stats and select relevant features."""
    if 'ID' in df.columns:
        df = df.drop(columns='ID')
    des_per_class = get_description_per_class(df)
    features = filter_features_by_description(des_per_class, df)
    features.extend(['ID', 'Diagnosis'])
    return features
