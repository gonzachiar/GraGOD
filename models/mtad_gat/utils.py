import numpy as np
import pandas as pd
from pandas._typing import ArrayLike

# NOTE:
#   - cumsum may have some problems


def adjust_anomaly_scores(
    scores: ArrayLike, dataset: str, is_train: bool, lookback: int
):
    """
    Method for MSL and SMAP where channels have been concatenated as part of
    the preprocessing

    Args:
        scores: anomaly_scores
        dataset: name of dataset
        is_train: if scores is from train set
        lookback: lookback (window size) used in model
    """

    # Remove errors for time steps when transition to new channel
    # (as this will be impossible for model to predict)
    if dataset.upper() not in ["SMAP", "MSL"]:
        return scores

    adjusted_scores = scores.copy()
    if is_train:
        md = pd.read_csv(f"./datasets/data/{dataset.lower()}_train_md.csv")
    else:
        md = pd.read_csv("./datasets/data/labeled_anomalies.csv")
        md = md[md["spacecraft"] == dataset.upper()]

    md = md[md["chan_id"] != "P-2"]

    # Sort values by channel
    md = md.sort_values(by=["chan_id"])

    # Getting the cumulative start index for each channel
    sep_cuma = np.cumsum(md["num_values"].values) - lookback
    sep_cuma = sep_cuma[:-1]
    buffer = np.arange(1, 20)
    i_remov = np.sort(
        np.concatenate(
            (
                sep_cuma,
                np.array([i + buffer for i in sep_cuma]).flatten(),
                np.array([i - buffer for i in sep_cuma]).flatten(),
            )
        )
    )
    i_remov = i_remov[(i_remov < len(adjusted_scores)) & (i_remov >= 0)]
    i_remov = np.sort(np.unique(i_remov))
    if len(i_remov) != 0:
        adjusted_scores[i_remov] = 0

    # Normalize each concatenated part individually
    sep_cuma = np.cumsum(md["num_values"].values) - lookback
    s = [0] + sep_cuma.tolist()
    for c_start, c_end in [(s[i], s[i + 1]) for i in range(len(s) - 1)]:
        e_s = adjusted_scores[c_start : c_end + 1]

        e_s = (e_s - np.min(e_s)) / (np.max(e_s) - np.min(e_s))
        adjusted_scores[c_start : c_end + 1] = e_s

    return adjusted_scores
