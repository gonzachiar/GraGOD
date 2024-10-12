import json

import pytorch_lightning as pl
import torch
import yaml
from numpy.typing import ArrayLike

from datasets import load_swat_training_data, load_telco_training_data
from gragod import Datasets, InterPolationMethods, ParamFileTypes
from gragod.utils import get_logger

logger = get_logger()


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)


def _split_train_val_test(
    X: torch.Tensor,
    Y: torch.Tensor | None,
    groups: ArrayLike | None,
    test_size: float = 0.1,
    val_size: float = 0.1,
    shuffle: bool = True,
    random_state: int = 42,
):
    # https://rasbt.github.io/mlxtend/user_guide/evaluate/GroupTimeSeriesSplit/#example-1-multiple-training-groups-with-train-size-specified
    # fmt: off
    # import ipdb; from pprint import pprint as pp; ipdb.set_trace(context=10);
    # fmt: on

    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     X,
    #     Y,
    #     test_size=test_size,
    #     shuffle=shuffle,
    #     random_state=random_state,
    #     stratify=Y,
    # )
    # X_train, X_val, Y_train, Y_val = train_test_split(
    #     X_train,
    #     Y_train,
    #     test_size=val_size,
    #     shuffle=shuffle,
    #     random_state=random_state,
    #     stratify=Y_train,
    # )

    # return X_train, X_val, X_test, Y_train, Y_val, Y_test
    pass


def load_params(base_path: str, file_type: ParamFileTypes) -> dict:
    """
    Load the parameters from the given file.
    Args:
        base_path: The path to the parameters file.
        type: The enum with the type of the parameters file.
    Returns:
        The parameters as a dictionary.
    """
    if file_type == ParamFileTypes.YAML:
        with open(base_path, "r") as yaml_file:
            params = yaml.safe_load(yaml_file)
    elif file_type == ParamFileTypes.JSON:
        with open(base_path, "r") as json_file:
            params = json.load(json_file)
    else:
        raise ValueError(f"Type must be one of {ParamFileTypes.__members__.keys()}")

    return params


def load_training_data(
    dataset: Datasets,
    test_size: float = 0.1,
    val_size: float = 0.1,
    shuffle: bool = True,
    random_state: int = 42,
    normalize: bool = False,
    clean: bool = False,
    interpolate_method: InterPolationMethods | None = None,
):
    # TODO:
    # - Fix datasets loading path
    # - Load all services in Miahela

    # if dataset == "mihaela":
    #     from datasets.mihaela import load_mihaela_service_training_data

    #     X, Y = load_mihaela_service_training_data(
    #         service_name="Clash_of_Clans",
    #         normalize=normalize,
    #         clean=clean,
    #         interpolate_method=interpolate_method,
    #     )

    #     return _split_train_val_test(
    #         X=X,
    #         Y=Y,
    #         test_size=test_size,
    #         val_size=val_size,
    #         shuffle=shuffle,
    #         random_state=random_state,
    #     )
    if dataset == Datasets.SWAT:
        return load_swat_training_data(
            normalize=normalize, clean=clean, interpolate_method=interpolate_method
        )
    elif dataset == Datasets.TELCO:
        logger.warning(
            "Telco data is already splited, ignoring arguments: 'test_size'"
            ", 'val_size', 'shuffle' and 'random_state'"
        )

        return load_telco_training_data(
            normalize=normalize, clean=clean, interpolate_method=interpolate_method
        )

    else:
        raise ValueError(f"{dataset} is an unkown dataset")
