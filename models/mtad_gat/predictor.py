import json
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from gragod import PathType
from models.mtad_gat.dataset import SlidingWindowDataset
from models.mtad_gat.eval_methods import (
    adjust_predicts,
    bf_search,
    epsilon_eval,
    find_epsilon,
    pot_eval,
)
from models.mtad_gat.model import MTAD_GAT
from models.mtad_gat.utils import adjust_anomaly_scores

# TODO:
#   - Check if Tensor is the correct type


class Predictor:
    """
    MTAD-GAT predictor class.

    Args:
        model: MTAD-GAT model (pre-trained) used to forecast and reconstruct
        window_size: Length of the input sequence
        n_features: Number of input features
        pred_args: params for thresholding and predicting anomalies

    """

    def __init__(
        self,
        model: MTAD_GAT,
        window_size: int,
        n_features: int,
        pred_args: dict,
        summary_file_name: PathType = "summary.txt",
    ):
        self.model = model
        self.window_size = window_size
        self.n_features = n_features
        self.dataset = pred_args["dataset"]
        self.target_dims = pred_args["target_dims"]
        self.scale_scores = pred_args["scale_scores"]
        self.q = pred_args["q"]
        self.level = pred_args["level"]
        self.dynamic_pot = pred_args["dynamic_pot"]
        self.use_mov_av = pred_args["use_mov_av"]
        self.gamma = pred_args["gamma"]
        self.reg_level = pred_args["reg_level"]
        self.save_path = pred_args["save_path"].format(feature=self.target_dims)
        self.batch_size = 256
        self.use_cuda = True
        self.pred_args = pred_args
        self.summary_file_name = summary_file_name

    def get_score(self, values: torch.Tensor):
        """
        Method that calculates anomaly score using given model and data

        Args:
            values: 2D array of multivariate time series data, shape (N, k)
        Returns:
            np array of anomaly scores + dataframe with prediction for each channel
            and global anomalies
        """

        print("Predicting and calculating anomaly scores..")
        data = SlidingWindowDataset(values, self.window_size, self.target_dims)
        loader = DataLoader(data, batch_size=self.batch_size, shuffle=False)
        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"

        self.model.eval()
        preds = []
        recons = []
        with torch.no_grad():
            for x, y in tqdm(loader):
                x = x.to(device)
                y = y.to(device)
                y_hat, _ = self.model(x)

                # Shifting input to include the observed value (y)
                # when doing the reconstruction
                recon_x = torch.cat((x[:, 1:, :], y), dim=1)
                _, window_recon = self.model(recon_x)

                preds.append(y_hat.detach().cpu().numpy())
                # Extract last reconstruction only
                recons.append(window_recon[:, -1, :].detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        recons = np.concatenate(recons, axis=0)
        actual = values.detach().cpu().numpy()[self.window_size :]

        anomaly_scores = np.zeros_like(actual)
        df_dict = {}
        for i in (
            [self.target_dims]
            if self.target_dims is not None
            else range(preds.shape[1])
        ):
            df_dict[f"Forecast_{i}"] = preds[:, i]
            df_dict[f"Recon_{i}"] = recons[:, i]
            df_dict[f"True_{i}"] = actual[:, i]
            a_score = np.sqrt((preds[:, i] - actual[:, i]) ** 2) + self.gamma * np.sqrt(
                (recons[:, i] - actual[:, i]) ** 2
            )

            if self.scale_scores:
                q75, q25 = np.percentile(a_score, [75, 25])
                iqr = q75 - q25
                median = np.median(a_score)
                a_score = (a_score - median) / (1 + iqr)

            anomaly_scores[:, i] = a_score
            df_dict[f"A_Score_{i}"] = a_score

        df = pd.DataFrame(df_dict)
        anomaly_scores = np.mean(anomaly_scores, 1)
        df["A_Score_Global"] = anomaly_scores

        return df

    def predict_anomalies(
        self,
        train: torch.Tensor,
        test: torch.Tensor,
        true_anomalies: Optional[torch.Tensor],
        load_scores: bool = False,
        save_output: bool = True,
        scale_scores: bool = False,
    ):
        """
        Predicts anomalies

        Args:
            train: 2D array of train multivariate time series data
            test: 2D array of test multivariate time series data
            true_anomalies: true anomalies of test set, None if not available
            save_scores: Whether to save anomaly scores of train and test
            load_scores: Whether to load anomaly scores instead of calculating them
            save_output: Whether to save output dataframe
            scale_scores: Whether to feature-wise scale anomaly scores
        """

        if load_scores:
            print("Loading anomaly scores")

            train_pred_df = pd.read_pickle(f"{self.save_path}/train_output.pkl")
            test_pred_df = pd.read_pickle(f"{self.save_path}/test_output.pkl")

            train_anomaly_scores = train_pred_df["A_Score_Global"].values
            test_anomaly_scores = test_pred_df["A_Score_Global"].values

        else:
            train_pred_df = self.get_score(train)
            test_pred_df = self.get_score(test)

            train_anomaly_scores = train_pred_df["A_Score_Global"].values
            test_anomaly_scores = test_pred_df["A_Score_Global"].values

            train_anomaly_scores = adjust_anomaly_scores(
                train_anomaly_scores, self.dataset, True, self.window_size
            )
            test_anomaly_scores = adjust_anomaly_scores(
                test_anomaly_scores, self.dataset, False, self.window_size
            )

            # Update df
            train_pred_df["A_Score_Global"] = train_anomaly_scores
            test_pred_df["A_Score_Global"] = test_anomaly_scores

        if self.use_mov_av:
            smoothing_window = int(self.batch_size * self.window_size * 0.05)
            train_anomaly_scores = (
                pd.DataFrame(train_anomaly_scores)
                .ewm(span=smoothing_window)
                .mean()
                .values.flatten()
            )
            test_anomaly_scores = (
                pd.DataFrame(test_anomaly_scores)
                .ewm(span=smoothing_window)
                .mean()
                .values.flatten()
            )

        # Find threshold and predict anomalies at feature-level
        # (for plotting and diagnosis purposes)
        out_dim = self.n_features

        all_preds = np.zeros((len(test_pred_df), out_dim))

        for i in [self.target_dims] if self.target_dims is not None else range(out_dim):
            train_feature_anom_scores = train_pred_df[f"A_Score_{i}"].values
            test_feature_anom_scores = test_pred_df[f"A_Score_{i}"].values
            epsilon = find_epsilon(train_feature_anom_scores, reg_level=2)

            train_feature_anom_preds = (train_feature_anom_scores >= epsilon).astype(
                int
            )
            test_feature_anom_preds = (test_feature_anom_scores >= epsilon).astype(int)

            train_pred_df[f"A_Pred_{i}"] = train_feature_anom_preds
            test_pred_df[f"A_Pred_{i}"] = test_feature_anom_preds

            train_pred_df[f"Thresh_{i}"] = epsilon
            test_pred_df[f"Thresh_{i}"] = epsilon

            all_preds[:, i] = test_feature_anom_preds

        # Global anomalies (entity-level) are predicted using aggregation of anomaly
        # scores across all features
        # These predictions are used to evaluate performance, as true anomalies are
        # labeled at entity-level
        # Evaluate using different threshold methods: brute-force, epsilon and
        # peaks-over-treshold
        e_eval = epsilon_eval(
            train_anomaly_scores,
            test_anomaly_scores,
            true_anomalies,
            reg_level=self.reg_level,
        )
        p_eval = pot_eval(
            train_anomaly_scores,
            test_anomaly_scores,
            true_anomalies,
            q=self.q,
            level=self.level,
            dynamic=self.dynamic_pot,
        )
        if true_anomalies is not None:
            bf_eval = bf_search(
                test_anomaly_scores,
                true_anomalies,
                start=0.01,
                end=2,
                step_num=100,
                verbose=False,
            )
        else:
            bf_eval = {}

        print(f"Results using epsilon method:\n {e_eval}")
        print(f"Results using peak-over-threshold method:\n {p_eval}")
        print(f"Results using best f1 score search:\n {bf_eval}")

        for k, v in e_eval.items():
            if not isinstance(e_eval[k], list):
                e_eval[k] = float(v)
        for k, v in p_eval.items():
            if not isinstance(p_eval[k], list):
                p_eval[k] = float(v)
        for k, v in bf_eval.items():
            bf_eval[k] = float(v)

        # Save
        summary = {"epsilon_result": e_eval, "pot_result": p_eval, "bf_result": bf_eval}
        with open(f"{self.save_path}/{self.summary_file_name}", "w") as f:
            json.dump(summary, f, indent=2)

        # Save anomaly predictions made using epsilon method (could be changed to pot
        # or bf-method)
        if save_output:
            global_epsilon = e_eval["threshold"]
            test_pred_df["A_True_Global"] = true_anomalies
            train_pred_df["Thresh_Global"] = global_epsilon
            test_pred_df["Thresh_Global"] = global_epsilon
            train_pred_df["A_Pred_Global"] = (
                train_anomaly_scores >= global_epsilon
            ).astype(int)
            test_preds_global = (test_anomaly_scores >= global_epsilon).astype(int)
            # Adjust predictions according to evaluation strategy
            if true_anomalies is not None:
                test_preds_global = adjust_predicts(
                    None, true_anomalies, global_epsilon, pred=test_preds_global
                )
            test_pred_df["A_Pred_Global"] = test_preds_global

            print(f"Saving output to {self.save_path}/<train/test>_output.pkl")
            train_pred_df.to_pickle(f"{self.save_path}/train_output.pkl")
            test_pred_df.to_pickle(f"{self.save_path}/test_output.pkl")

        print("-- Done.")