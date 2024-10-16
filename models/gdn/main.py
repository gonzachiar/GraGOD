import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from gragod import ParamFileTypes
from gragod.training import load_params, set_seeds
from models.gdn.dataset import TimeDataset
from models.gdn.evaluate import (
    get_best_performance_data,
    get_full_err_scores,
    get_val_performance_data,
)
from models.gdn.model import GDN
from models.gdn.preprocess import build_loc_net
from models.gdn.test import test
from models.gdn.train import train


class Main:
    """
    Main class for running the GDN (Graph Deviation Network) model.

    This class handles the initialization, training, and evaluation of the GDN model.
    It manages data loading, model creation, and result reporting.

    Attributes:
        train_config (dict): Configuration for training.
        env_config (dict): Environment configuration.
        datestr (str): Date string for saving results.
        device (str): Device to run the model on (CPU/GPU).
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        test_dataloader (DataLoader): DataLoader for test data.
        model (GDN): The GDN model instance.
    """

    def __init__(self, params):
        """
        Initialize the Main class.

        Args:
            train_config (dict): Configuration for training.
            model_config (dict): Configuration for the model.
            env_config (dict): Environment configuration.
            dataset (str): Name of the dataset to use.
        """
        self.train_config = params["train"]
        self.env_config = params["env"]
        self.datestr = None

        dataset = params["dataset"]
        df_train = pd.read_csv(f"./datasets_files/{dataset}/train.csv", index_col=0)
        df_test = pd.read_csv(f"./datasets_files/{dataset}/test.csv", index_col=0)

        column_names_list = df_train.columns.tolist()

        if "attack" in df_train.columns:
            df_train = df_train.drop(columns=["attack"])

        # Create a fully connected graph
        base_graph_structure = {
            ft: [other_ft for other_ft in column_names_list if other_ft != ft]
            for ft in column_names_list
        }

        self.device = params["train"]["device"]

        fc_edge_index = build_loc_net(base_graph_structure, list(column_names_list))
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)

        cfg = {
            "slide_win": params["train"]["slide_win"],
            "slide_stride": params["train"]["slide_stride"],
        }

        train_dataset = TimeDataset(df_train, fc_edge_index, mode="train", config=cfg)
        test_dataset = TimeDataset(df_test, fc_edge_index, mode="test", config=cfg)

        self.train_dataloader, self.val_dataloader, self.test_dataloader = (
            self.get_loaders(
                train_dataset,
                test_dataset,
                params["train"]["batch"],
                val_ratio=params["train"]["val_ratio"],
            )
        )

        self.model = GDN(
            [fc_edge_index],
            len(column_names_list),
            dim=params["model"]["dim"],
            input_dim=params["train"]["slide_win"],
            out_layer_num=params["model"]["out_layer_num"],
            out_layer_inter_dim=params["model"]["out_layer_inter_dim"],
            topk=params["model"]["topk"],
        ).to(self.device)

    def run(self):
        """
        Run the main process of training, testing, and evaluating the model.

        This method handles the entire pipeline from loading/training the model
        to getting the final scores.
        """
        if len(self.env_config["load_model_path"]) > 0:
            model_save_path = self.env_config["load_model_path"]
        else:
            model_save_path = self.get_save_path()[0]

            self.train_log = train(
                self.model,
                model_save_path,
                config=self.train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
            )

        # test
        self.model.load_state_dict(torch.load(model_save_path, weights_only=True))
        best_model = self.model.to(self.device)

        _, self.test_result = test(best_model, self.test_dataloader, self.device)
        _, self.val_result = test(best_model, self.val_dataloader, self.device)

        self.get_score(self.test_result, self.val_result)

    def get_loaders(self, train_dataset, test_dataset, batch, val_ratio=0.1):
        """
        Create DataLoader objects for train, validation, and test datasets.

        Args:
            train_dataset (Dataset): The training dataset.
            test_dataset (Dataset): The test dataset.
            batch (int): Batch size for DataLoaders.
            val_ratio (float): Ratio of training data to use for validation.

        Returns:
            tuple: Contains train_dataloader, val_dataloader, and test_dataloader.
        """
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat(
            [indices[:val_start_index], indices[val_start_index + val_use_len :]]
        )
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index : val_start_index + val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)

        train_dataloader = DataLoader(train_subset, batch_size=batch, shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch, shuffle=False)

        test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

        return train_dataloader, val_dataloader, test_dataloader

    def get_score(self, test_result, val_result):
        """
        Calculate and print the model's performance scores.

        Args:
            test_result (list): Results from testing the model.
            val_result (list): Results from validating the model.
        """
        np_test_result = np.array(test_result)

        test_labels = np_test_result[2, :, 0].tolist()

        test_scores, normal_scores = get_full_err_scores(test_result, val_result)

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)
        top1_val_info = get_val_performance_data(
            test_scores, normal_scores, test_labels, topk=1
        )

        print("\n=========================** Result **============================\n")

        info = None
        if self.env_config["report"] == "best":
            info = top1_best_info
        elif self.env_config["report"] == "val":
            info = top1_val_info

        print(f"F1 score: {info[0]}")
        print(f"precision: {info[1]}")
        print(f"recall: {info[2]}\n")

    def get_save_path(self):
        """
        Generate paths for saving the model and results.

        Returns:
            list: Contains paths for saving the model and results.
        """
        dir_path = self.env_config["save_path"]

        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime("%m|%d-%H:%M:%S")
        datestr = self.datestr

        paths = [
            f"./saved_models/{dir_path}/best_{datestr}.pt",
            f"./results/{dir_path}/{datestr}.csv",
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--params_file", type=str, default="models/gdn/params.yaml")

    params = load_params("models/gdn/params.yaml", file_type=ParamFileTypes.YAML)

    # Set random seeds
    set_seeds(params["env"]["random_seed"])

    main = Main(params)
    main.run()
