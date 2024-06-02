import os
from typing import Tuple

import pandas as pd

BASE_PATH_DEFAULT = "../datasets_files/swat"
NAME_TRAIN_DEFAULT = "SWaT_data_train.csv"
NAME_VAL_DEFAULT = "SWaT_data_val.csv"


def load_swat_df_train(
    name: str = NAME_TRAIN_DEFAULT, path_to_dataset: str = BASE_PATH_DEFAULT
) -> Tuple[pd.DataFrame]:
    """
    Loads the training dataset from the given path and returns a pandas DataFrame.
    Args:
        path_to_dataset: Path to the dataset files.
    Returns:
        A pandas DataFrame containing the training dataset.
    """
    file = os.path.join(path_to_dataset, name)
    df_train = pd.read_csv(file)
    df_train_labels = (df_train["Normal/Attack"] == "Attack").astype(int)
    df_train = df_train.drop(columns=["Normal/Attack"])
    return df_train, df_train_labels


def load_swat_df_val(
    name: str = NAME_VAL_DEFAULT, path_to_dataset: str = BASE_PATH_DEFAULT
) -> Tuple[pd.DataFrame]:
    """
    Loads the validation dataset from the given path and returns a pandas DataFrame.
    Args:
        path_to_dataset: Path to the dataset files.
    Returns:
        A pandas DataFrame containing the validation dataset.
    """
    file = os.path.join(path_to_dataset, name)
    df_val = pd.read_csv(file)
    df_val_labels = (df_val["Normal/Attack"] == "Attack").astype(int)
    df_val = df_val.drop(columns=["Normal/Attack"])
    return df_val, df_val_labels


def split_val_df(
    df_val: pd.DataFrame, df_val_labels: pd.Series, val_size: float = 0.6
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split the validation dataset into two parts: validation and test.
    Args:
        df_val: The DataFrame with the validation data.
        df_val_labels: The labels of the validation data.
        val_size: The size of the validation set.
    Returns:
        The validation and test datasets.
    """
    val_size = int(val_size * len(df_val))
    df_test = df_val.iloc[val_size:]
    df_test_labels = df_val_labels.iloc[val_size:]
    df_val = df_val.iloc[:val_size]
    df_val_labels = df_val_labels.iloc[:val_size]
    return df_val, df_val_labels, df_test, df_test_labels


def load_swat_df(
    path_to_dataset: str = BASE_PATH_DEFAULT, val_size: int = 0.6
) -> Tuple[pd.DataFrame]:
    """
    Loads the dataset from the given path and returns a pandas DataFrame.
    Args:
        names: List of names of the files to load.
        path_to_dataset: Path to the dataset files.
    Returns:
        A pandas DataFrame containing the dataset.
    """
    df_train, df_train_labels = load_swat_df_train(path_to_dataset=path_to_dataset)
    df_val, df_val_labels = load_swat_df_val(path_to_dataset=path_to_dataset)
    df_val, df_val_labels, df_test, df_test_labels = split_val_df(
        df_val, df_val_labels, val_size=val_size
    )

    return df_train, df_train_labels, df_val, df_val_labels, df_test, df_test_labels
