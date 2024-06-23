import json
from typing import Literal, Optional

import torch
import yaml
from sklearn.model_selection import train_test_split

from gragod import DATASETS, INTERPOLATION_METHODS

PARAM_FILE_TYPE = Literal["yaml", "json"]


def _split_train_val_test(
    X: torch.Tensor,
    Y: Optional[torch.Tensor],
    test_size: float = 0.1,
    val_size: float = 0.1,
    shuffle: bool = True,
    random_state: int = 42,
):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state,
        stratify=Y,
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train,
        Y_train,
        test_size=val_size,
        shuffle=shuffle,
        random_state=random_state,
        stratify=Y_train,
    )

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def load_params(base_path: str, type: PARAM_FILE_TYPE) -> dict:
    if type == "yaml":
        with open(base_path, "r") as yaml_file:
            params = yaml.safe_load(yaml_file)
    elif type == "json":
        with open(base_path, "r") as json_file:
            params = json.load(json_file)
    else:
        raise ValueError(f"Type must be one of {PARAM_FILE_TYPE}")

    return params


def load_training_data(
    dataset: DATASETS,
    test_size: float = 0.1,
    val_size: float = 0.1,
    shuffle: bool = True,
    random_state: int = 42,
    normalize: bool = False,
    clean: bool = False,
    interpolate_method: Optional[INTERPOLATION_METHODS] = None,
):
    if dataset == "mihaela":
        from datasets.mihaela import load_mihaela_service_training_data

        # TODO: Load all services
        X, Y = load_mihaela_service_training_data(
            service_name="Clash_of_Clans",
            normalize=normalize,
            clean=clean,
            interpolate_method=interpolate_method,
        )

        return _split_train_val_test(
            X=X,
            Y=Y,
            test_size=test_size,
            val_size=val_size,
            shuffle=shuffle,
            random_state=random_state,
        )

    elif dataset == "telco":
        from datasets.telco import load_telco_training_data

        print(
            "Telco data is already splited, ignoring arguments: 'test_size'"
            ", 'val_size', 'shuffle' and 'random_state'"
        )

        return load_telco_training_data(
            normalize=normalize, clean=clean, interpolate_method=interpolate_method
        )

    else:
        raise ValueError(f"{dataset} is an unkown dataset")
