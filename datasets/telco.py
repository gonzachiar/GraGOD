import os
from typing import Optional, Tuple

import pandas as pd
import torch

from datasets.data_processing import InterPolationMethods, preprocess_df

BASE_PATH_DEFAULT = "datasets_files/telco"


def load_telco_df(
    base_path: str | os.PathLike = BASE_PATH_DEFAULT,
) -> Tuple[pd.DataFrame, ...]:
    """
    Load the TELCO datasets as pandas DataFrames from the given path.
    Args:
        base_path: The path where the datasets are stored.
    Returns:
        Tuple of DataFrames for train, validation, and test datasets.
    """
    df_train = pd.read_csv(os.path.join(base_path, "TELCO_data_train.csv"))
    df_train_labels = pd.read_csv(os.path.join(base_path, "TELCO_labels_train.csv"))
    df_val = pd.read_csv(os.path.join(base_path, "TELCO_data_val.csv"))
    df_val_labels = pd.read_csv(os.path.join(base_path, "TELCO_labels_val.csv"))
    df_test = pd.read_csv(os.path.join(base_path, "TELCO_data_test.csv"))
    df_test_labels = pd.read_csv(os.path.join(base_path, "TELCO_labels_test.csv"))

    return df_train, df_train_labels, df_val, df_val_labels, df_test, df_test_labels


def load_telco_tp(base_path: str | os.PathLike = BASE_PATH_DEFAULT):
    """
    Load the TELCO datasets as Temporian EventSets from the given path.
    Args:
        base_path: The path where the datasets are stored.
    Returns:
        Tuple of EventSets for train, validation, and test datasets.
    """
    import temporian as tp

    es_train = tp.from_csv(
        os.path.join(base_path, "TELCO_data_train.csv"), timestamps="time"
    )
    es_label_train = tp.from_csv(
        os.path.join(base_path, "TELCO_labels_train.csv"), timestamps="time"
    )
    es_val = tp.from_csv(
        os.path.join(base_path, "TELCO_data_val.csv"), timestamps="time"
    )
    es_label_val = tp.from_csv(
        os.path.join(base_path, "TELCO_labels_val.csv"), timestamps="time"
    )
    es_test = tp.from_csv(
        os.path.join(base_path, "TELCO_data_test.csv"), timestamps="time"
    )
    es_label_test = tp.from_csv(
        os.path.join(base_path, "TELCO_labels_test.csv"), timestamps="time"
    )

    return es_train, es_label_train, es_val, es_label_val, es_test, es_label_test


def load_telco_training_data(
    base_path: str | os.PathLike = BASE_PATH_DEFAULT,
    normalize: bool = False,
    clean: bool = False,
    scaler=None,
    interpolate_method: Optional[InterPolationMethods] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Load the data for the telco dataset, splitted into train, val and test.
    Args:
        base_path: The path where the datasets are stored.
        normalize: Whether to normalize the data. Default is False.
        clean: Whether to clean the data. Default is False.
    Returns:
        Tuple of training data, training labels, validation data, validation labels,
        and test data.
    """
    (
        df_train,
        df_train_labels,
        df_val,
        df_val_labels,
        df_test,
        df_test_labels,
    ) = load_telco_df(base_path=base_path)

    X_train, X_train_labels = preprocess_df(
        data_df=df_train,
        labels_df=df_train_labels,
        normalize=normalize,
        clean=clean,
        scaler=scaler,
        interpolate_method=interpolate_method,
    )
    X_val, X_val_labels = preprocess_df(
        data_df=df_val,
        labels_df=df_val_labels,
        normalize=normalize,
        clean=clean,
        scaler=scaler,
        interpolate_method=interpolate_method,
    )
    X_test, X_test_labels = preprocess_df(
        data_df=df_test,
        labels_df=df_test_labels,
        normalize=normalize,
        clean=clean,
        scaler=scaler,
        interpolate_method=interpolate_method,
    )

    # Ignorin typing since is will never be None
    return (
        X_train,
        X_val,
        X_test,
        X_train_labels,
        X_val_labels,
        X_test_labels,
    )  # type: ignore
