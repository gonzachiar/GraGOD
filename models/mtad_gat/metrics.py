import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from gragod.training import load_training_data

DATA_PATH = "data"
PARAMS_FILE = "models/mtad_gat/params.yaml"
RANDOM_SEED = 42


def main(params, feature: Optional[int] = None):
    window_size = params["train_params"]["window_size"]
    (
        X_train,
        X_val,
        X_test,
        X_labels_train,
        X_labels_val,
        X_labels_test,
    ) = load_training_data(
        dataset=params["dataset"],
        test_size=params["train_params"]["test_size"],
        val_size=params["train_params"]["val_size"],
        shuffle=params["train_params"]["shuffle"],
        normalize=params["train_params"]["normalize_data"],
        interpolate_method=params["train_params"]["interpolate_method"],
        random_state=RANDOM_SEED,
        clean=False,
    )

    df = pd.read_pickle(
        os.path.join(
            params["predictor_params"]["save_path"].format(feature=feature),
            "train_output.pkl",
        )
    )

    target_dim = feature
    columns = range(X_train.shape[1])
    n_features = columns if target_dim is None else [target_dim]

    pred_columns = [f"A_Pred_{i}" for i in n_features]
    # recon_pred_columns = [f"Recon_{i}" for i in n_features]
    score_columns = [f"A_Score_{i}" for i in n_features]
    scores = df[score_columns]
    prediction = df[pred_columns]
    # y_pred = df[recon_pred_columns].to_numpy()
    thresholds = df[[f"Thresh_{i}" for i in n_features]].iloc[0].to_numpy()

    S = scores.to_numpy()
    mask = (X_labels_train == 1.0)[window_size:]

    mean_anomalies = []
    mean_normal = []
    if len(n_features) == 1:
        mask = mask[:, target_dim]
        print(f"Mean score of anomalies for feature {target_dim}: {S[mask[:]].mean()}")
        print(f"Mean score of normal for feature {target_dim}: {S[~mask].mean()}")
        mean_anomalies.append(S[mask].mean())
        mean_normal.append(S[~mask].mean())

    else:
        for i in n_features:
            # print(
            #     f"Mean score of anomalies for feature {i}:"
            # "{S[:, i][mask[:,i]].mean()}"
            # )
            # print(f"Mean score of normal for feature {i}:"
            # "{S[:, i][~mask[:,i]].mean()}")
            mean_anomalies.append(S[:, i][mask[:, i]].mean())
            mean_normal.append(S[:, i][~mask[:, i]].mean())
    mean_anomalies = np.array(mean_anomalies)
    mean_normal = np.array(mean_normal)

    # Optimize thresholds
    from scipy.optimize import minimize_scalar
    from sklearn.metrics import f1_score

    print("Optimizing thresholds")
    thresholds = []
    for i in n_features:
        real_value = X_labels_train[window_size:, i]

        def objective_function(threshold):
            prediction = S[:, i] > threshold
            return -f1_score(real_value, prediction)

        result = minimize_scalar(
            objective_function,
            bounds=(S.min(axis=0)[i], S.max(axis=0)[i]),
            method="bounded",
        )
        thresholds.append(result.x)

    thresholds = (mean_anomalies + mean_normal) / 1.5
    prediction = S > thresholds

    if len(n_features) == 1:
        i = target_dim
        real_value = X_labels_train[window_size:, i]
        prediction = S > thresholds[0]
        precision = prediction[real_value == 1.0].sum() / prediction.sum()
        recall = prediction[real_value == 1.0].sum() / (real_value == 1.0).sum()
        print(f"Precision for feature {i}: {precision}")
        print(f"Recall for feature {i}: {recall}")
        print(f"F1 for feature {i}: {2 * (precision * recall) / (precision + recall)}")
        print(f"Predictions per feature: {prediction.sum(axis=0)}")
        print(f"Real values per feature: {real_value.sum(axis=0)}")
    else:
        recalls = {}
        precisions = {}
        f1_scores = {}
        for i in n_features:
            # print(f"Threshold for feature {i}: {thresholds[i]}")
            real_value = X_labels_train[window_size:, i]
            prediction = S[:, i] > thresholds[i]
            precision = prediction[real_value == 1.0].sum() / prediction.sum()
            recall = prediction[real_value == 1.0].sum() / (real_value == 1.0).sum()
            # print(f"Precision for feature {i}: {precision}")
            # print(f"Recall for feature {i}: {recall}")
            # print(
            #     "F1 for feature {i}:"
            #     f"{2 * (precision * recall) / (precision + recall)}"
            # )
            recalls[i] = recall
            precisions[i] = precision
            f1_scores[i] = float(2 * (precision * recall) / (precision + recall))

        real_values = X_labels_train[window_size:, n_features]
        predictions = S > thresholds
        # print(
        #     f"Normal value acc:"
        #     f"{(~predictions[real_value == 0.0]).sum() / (real_value == 0.0).sum()}"
        # )
        # print(
        #     "Recall:"
        # .   "f{predictions[real_value == 1.0].sum() / (real_value == 1.0).sum()}"
        # )
        # print("Precision:"
        #       f"{predictions[real_value == 1.0].sum() / predictions.sum()}")

    for key, value in recalls.items():
        print(
            f"Feature {key}"
            f"| Precision {precisions[key]:.3}"
            f"| Recall: {value:.3}"
            f"| F1: {f1_scores[key]:.3}"
            f"| Mean Score Anomalies: {mean_anomalies[key]:.3}"
            f"| Mean Score Normal: {mean_normal[key]:.3}"
            f"| Number of predictions: {predictions[:, key].sum()}"
            f"| Number of anomalies: {int(real_values[:, key].sum())}"
        )


if __name__ == "__main__":
    with open(PARAMS_FILE, "r") as yaml_file:
        params = yaml.safe_load(yaml_file)

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--feature",
        type=int,
        default=None,
    )
    args = argparser.parse_args()

    main(params, args.feature)
