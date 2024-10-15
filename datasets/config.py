from enum import Enum
from typing import Dict, Type

from pydantic import BaseModel

from gragod import Datasets


class SWATPaths(Enum):
    BASE_PATH = "datasets_files/swat"
    NAME_TRAIN = "SWaT_data_train.csv"
    NAME_VAL = "SWaT_data_val.csv"


class TELCOPaths(Enum):
    BASE_PATH = "datasets_files/telco"


class DatasetConfig(BaseModel):
    normalize: bool
    paths: Type[Enum]


class SWATConfig(DatasetConfig):
    normalize: bool = False
    paths: Type[Enum] = SWATPaths


class TELCOConfig(DatasetConfig):
    normalize: bool = False
    paths: Type[Enum] = TELCOPaths


def get_dataset_config(dataset: Datasets) -> DatasetConfig:
    DATASET_CONFIGS: Dict[Datasets, DatasetConfig] = {
        Datasets.SWAT: SWATConfig(),
        Datasets.TELCO: TELCOConfig(),
    }
    return DATASET_CONFIGS[dataset]
