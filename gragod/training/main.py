import json
from enum import Enum

import yaml


class ParamFileTypes(Enum):
    YAML = "yaml"
    JSON = "json"


def load_params(base_path: str, type: ParamFileTypes) -> dict:
    """
    Load the parameters from the given file.
    Args:
        base_path: The path to the parameters file.
        type: The enum with the type of the parameters file.
    Returns:
        The parameters as a dictionary.
    """
    if type == "yaml":
        with open(base_path, "r") as yaml_file:
            params = yaml.safe_load(yaml_file)
    elif type == "json":
        with open(base_path, "r") as json_file:
            params = json.load(json_file)
    else:
        raise ValueError(f"Type must be one of {ParamFileTypes.__members__.keys()}")

    return params
