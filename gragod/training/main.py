import json
from typing import Literal

import yaml

PARAM_FILE_TYPE = Literal["yaml", "json"]


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
