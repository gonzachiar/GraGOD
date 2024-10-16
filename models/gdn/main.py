import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from models.gdn.datasets import TimeDataset
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
    def __init__(self, train_config, model_config, env_config, dataset):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        df_train = pd.read_csv(f"./data/{dataset}/train.csv", index_col=0)
        df_test = pd.read_csv(f"./data/{dataset}/test.csv", index_col=0)

        column_names_list = df_train.columns.tolist()

        if "attack" in df_train.columns:
            df_train = df_train.drop(columns=["attack"])

        # Create a fully connected graph
        base_graph_structure = {
            ft: [other_ft for other_ft in column_names_list if other_ft != ft]
            for ft in column_names_list
        }

        self.device = train_config["device"]

        fc_edge_index = build_loc_net(base_graph_structure, list(column_names_list))
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)

        cfg = {
            "slide_win": train_config["slide_win"],
            "slide_stride": train_config["slide_stride"],
        }

        train_dataset = TimeDataset(df_train, fc_edge_index, mode="train", config=cfg)
        test_dataset = TimeDataset(df_test, fc_edge_index, mode="test", config=cfg)

        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.get_loaders(
            train_dataset,
            test_dataset,
            train_config["batch"],
            val_ratio=train_config["val_ratio"],
        )

        self.model = GDN(
            [fc_edge_index],
            len(column_names_list),
            dim=model_config["dim"],
            input_dim=train_config["slide_win"],
            out_layer_num=model_config["out_layer_num"],
            out_layer_inter_dim=model_config["out_layer_inter_dim"],
            topk=model_config["topk"],
        ).to(self.device)

    def run(self):
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

        np_test_result = np.array(test_result)

        test_labels = np_test_result[2, :, 0].tolist()

        test_scores, normal_scores = get_full_err_scores(test_result, val_result)

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)

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

        dir_path = self.env_config["save_path"]

        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime("%m|%d-%H:%M:%S")
        datestr = self.datestr

        paths = [
            f"./pretrained/{dir_path}/best_{datestr}.pt",
            f"./results/{dir_path}/{datestr}.csv",
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    # Load configuration from YAML file
    with open("models/gdn/params.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    train_config = config["train"]
    model_config = config["model"]
    env_config = config["env"]
    dataset = config["dataset"]

    # Set random seeds
    set_seeds(env_config["random_seed"])

    os.environ["PYTHONHASHSEED"] = str(env_config["random_seed"])

    main = Main(train_config, model_config, env_config, dataset)
    main.run()
