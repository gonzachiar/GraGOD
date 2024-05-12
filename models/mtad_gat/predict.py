import argparse
import os
from typing import Optional

import torch
import yaml

from gragod.training import load_training_data
from models.mtad_gat.model import MTAD_GAT
from models.mtad_gat.predictor import Predictor

DATA_PATH = "data"
PARAMS_FILE = "models/mtad_gat/params.yaml"
RANDOM_SEED = 42


def main(params, feature: Optional[int] = None):
    X_train, _, X_test, *_ = load_training_data(
        dataset=params["dataset"],
        test_size=params["test_size"],
        val_size=params["val_size"],
        shuffle=params["shuffle"],
        random_state=RANDOM_SEED,
        normalize=params["normalize_data"],
        clean=False,
        interpolate_method=params["interpolate_method"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(
        os.path.join(
            params["predictor_params"]["save_path"].format(feature=feature), "model.pt"
        ),
        map_location=torch.device(device),
    )

    window_size = params["train_params"]["window_size"]
    n_features = X_train.shape[1]
    out_dim = X_train.shape[1]

    model = MTAD_GAT(
        n_features,
        window_size,
        out_dim,
        dropout=params["train_params"]["dropout"],
        **params["model_params"],
    )
    model.load_state_dict(state_dict)
    model.eval()
    if device == "cuda":
        model = model.to(device)

    predictor_params = params["predictor_params"]
    predictor_params["target_dims"] = feature
    predictor = Predictor(
        model,
        window_size,
        n_features,
        predictor_params,
    )
    predictor.predict_anomalies(
        torch.tensor(X_train), torch.tensor(X_test), None, save_output=True
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
