import argparse

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets.config import get_dataset_config
from gragod import InterPolationMethods, ParamFileTypes
from gragod.training import (
    get_column_names_list,
    load_params,
    load_training_data,
    set_seeds,
)
from gragod.types import cast_dataset
from models.gdn.dataset import TimeDataset
from models.gdn.evaluate import (
    get_best_performance_data,
    get_full_err_scores,
    get_val_performance_data,
)
from models.gdn.model import GDN
from models.gdn.preprocess import build_loc_net
from models.gdn.test import test

RANDOM_SEED = 42


set_seeds(RANDOM_SEED)


def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction="mean")

    return loss


def _get_attack_or_not_attack(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor.sum(dim=1) > 0).float()


def get_score(test_result, val_result, report: str):
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
    if report == "best":
        info = top1_best_info
    elif report == "val":
        info = top1_val_info

    print(f"F1 score: {info[0]}")
    print(f"precision: {info[1]}")
    print(f"recall: {info[2]}\n")


def get_dataloader(
    X,
    y,
    edge_index,
    batch_size: int,
    n_workers: int,
    config: dict,
    is_train: bool = False,
):
    dataset = TimeDataset(X, y, edge_index, is_train=is_train, config=config)
    return DataLoader(dataset, batch_size=batch_size, num_workers=n_workers)


def main(
    dataset_name: str,
    model_params: dict,
    test_size: float = 0.1,
    val_size: float = 0.1,
    clean: bool = True,
    interpolate_method: InterPolationMethods | None = None,
    batch_size: int = 264,
    n_workers: int = 0,
    init_lr: float = 0.001,
    weight_decay: float = 0.0,
    n_epochs: int = 30,
    device: str = "mps",
    params: dict = {},
):
    dataset = cast_dataset(dataset_name)
    dataset_config = get_dataset_config(dataset=dataset)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_training_data(
        dataset=dataset,
        test_size=test_size,
        val_size=val_size,
        normalize=dataset_config.normalize,
        clean=clean,
        interpolate_method=interpolate_method,
    )
    y_train = _get_attack_or_not_attack(y_train)
    y_val = _get_attack_or_not_attack(y_val)
    y_test = _get_attack_or_not_attack(y_test)
    column_names_list = get_column_names_list(dataset)

    # Create a fully connected graph
    base_graph_structure = {
        ft: [other_ft for other_ft in column_names_list if other_ft != ft]
        for ft in column_names_list
    }

    fc_edge_index = build_loc_net(base_graph_structure, list(column_names_list))

    cfg = {
        "slide_win": params["model_params"]["window_size"],
        "slide_stride": params["model_params"]["stride"],
    }

    train_loader = get_dataloader(
        X_train, y_train, fc_edge_index, batch_size, n_workers, cfg, is_train=True
    )
    val_loader = get_dataloader(
        X_val, y_val, fc_edge_index, batch_size, n_workers, cfg, is_train=False
    )
    test_loader = get_dataloader(
        X_test, y_test, fc_edge_index, batch_size, n_workers, cfg, is_train=False
    )

    model = GDN(
        [fc_edge_index],
        len(column_names_list),
        dim=model_params["dim"],
        input_dim=model_params["window_size"],
        out_layer_num=model_params["out_layer_num"],
        out_layer_inter_dim=model_params["out_layer_inter_dim"],
        topk=model_params["topk"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=init_lr,
        weight_decay=weight_decay,
    )

    train_loss_list = []

    acu_loss = 0
    min_loss = 1e8

    i = 0
    early_stop_win = 15

    model.train()

    stop_improve_count = 0

    dataloader = train_loader

    for i_epoch in range(n_epochs):

        acu_loss = 0
        model.train()

        for x, labels, attack_labels, edge_index in dataloader:

            x, labels, edge_index = [
                item.float().to(device) for item in [x, labels, edge_index]
            ]

            optimizer.zero_grad()
            out = model(x).float().to(device)
            loss = loss_func(out, labels)

            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())
            acu_loss += loss.item()

            i += 1

        # each epoch
        print(
            "epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})".format(
                i_epoch, n_epochs, acu_loss / len(dataloader), acu_loss
            ),
            flush=True,
        )

        # use val dataset to judge
        if val_loader is not None:

            val_loss, val_result = test(model, val_loader, device)

            if val_loss < min_loss:
                # torch.save(model.state_dict(), save_path)

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1

            if stop_improve_count >= early_stop_win:
                break

        else:
            if acu_loss < min_loss:
                # torch.save(model.state_dict(), save_path)
                min_loss = acu_loss

    _, test_result = test(model, test_loader, device)

    get_score(test_result, val_result, params["env_params"]["report"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_file", type=str, default="models/gdn/params.yaml")
    args = parser.parse_args()
    params = load_params(args.params_file, file_type=ParamFileTypes.YAML)

    main(
        dataset_name=params["dataset"],
        **params["train_params"],
        model_params=params["model_params"],
        params=params,
    )
