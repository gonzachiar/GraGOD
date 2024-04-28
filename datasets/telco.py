import os
from typing import Literal, Tuple

import numpy as np
import pandas as pd
import torch

# from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler


def load_df(base_path: str | os.PathLike) -> Tuple[pd.DataFrame, ...]:
    df_train = pd.read_csv(os.path.join(base_path, "TELCO_data_train.csv"))
    df_train_labels = pd.read_csv(os.path.join(base_path, "TELCO_labels_train.csv"))
    df_val = pd.read_csv(os.path.join(base_path, "TELCO_data_val.csv"))
    df_val_labels = pd.read_csv(os.path.join(base_path, "TELCO_labels_val.csv"))
    df_test = pd.read_csv(os.path.join(base_path, "TELCO_data_test.csv"))
    df_test_labels = pd.read_csv(os.path.join(base_path, "TELCO_labels_test.csv"))

    return df_train, df_train_labels, df_val, df_val_labels, df_test, df_test_labels


def load_tp(base_path: str | os.PathLike):
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


def convert_df_to_tensor(df: pd.DataFrame) -> np.ndarray:
    X = np.array(df.values[:, 1:])
    X = np.vstack(X).astype(float)  # type:ignore

    return X


def load_data(base_path: str | os.PathLike) -> Tuple[np.ndarray, ...]:
    df_train, df_train_labels, df_val, df_val_labels, df_test, df_test_labels = load_df(
        base_path
    )
    X_train = convert_df_to_tensor(df_train)
    X_val = convert_df_to_tensor(df_val)
    X_test = convert_df_to_tensor(df_test)
    X_train_labels = convert_df_to_tensor(df_train_labels)
    X_val_labels = convert_df_to_tensor(df_val_labels)
    X_labels_test = convert_df_to_tensor(df_test_labels)

    return X_train, X_val, X_test, X_train_labels, X_val_labels, X_labels_test


def interpolate_data(
    data: np.ndarray, method: Literal["linear", "spline", "time"] = "spline"
) -> np.ndarray:
    df = pd.DataFrame(data)

    df.interpolate(method=method, inplace=True, order=3)
    interpolated_data = df.to_numpy()

    return interpolated_data


def normalize_data(data, scaler=None) -> Tuple[np.ndarray, MinMaxScaler]:
    data = np.asarray(data, dtype=np.float32)
    if np.any(sum(np.isnan(data))):
        data = np.nan_to_num(data)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    data = scaler.transform(data)
    print("Data normalized")

    return data, scaler


def load_training_data(
    base_path: str, normalize: bool = False, clean: bool = False
) -> Tuple[torch.Tensor, ...]:
    X_train, X_val, X_test, X_train_labels, X_val_labels, X_test_labels = load_data(
        base_path=base_path
    )

    if normalize:
        X_train, _ = normalize_data(X_train)
        X_val, _ = normalize_data(X_val)
        X_test, _ = normalize_data(X_test)

    if clean:
        mask = X_train_labels == 1.0
        X_train[mask] = np.nan
        # imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
        # imp_mean.fit(X_train)
        # X_train = imp_mean.transform(X_train)
        # X_val = imp_mean.transform(X_val)
        # X_test = imp_mean.transform(X_test)

    X_train = interpolate_data(X_train)
    X_val = interpolate_data(X_val)
    X_test = interpolate_data(X_test)

    print("Data cleaned!")

    X_train = torch.tensor(X_train).to(torch.float32)
    X_val = torch.tensor(X_val).to(torch.float32)
    X_test = torch.tensor(X_test).to(torch.float32)
    X_train_labels = torch.tensor(X_train_labels).to(torch.float32)
    X_val_labels = torch.tensor(X_val_labels).to(torch.float32)
    X_test_labels = torch.tensor(X_test_labels).to(torch.float32)

    return (
        X_train,
        X_val,
        X_test,
        X_train_labels,
        X_val_labels,
        X_test_labels,
    )
