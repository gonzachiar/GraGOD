import os

import pandas as pd


def load_dataset(names: list[str], path_to_dataset: str = "../datasets_files/swat"):
    """
    Loads the dataset from the given path and returns a pandas DataFrame.
    Args:
        names: List of names of the files to load.
        path_to_dataset: Path to the dataset files.
    Returns:
        A pandas DataFrame containing the dataset.
    """
    dataframes = []
    for name in names:
        file = os.path.join(path_to_dataset, name)
        dataframes.append(pd.read_csv(file, sep=","))

    return pd.concat(dataframes)
