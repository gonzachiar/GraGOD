import torch
import torch.nn.functional as F

from models.gdn.test import test


def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction="mean")

    return loss


def train(
    model=None,
    save_path="models/gdn/model.pth",
    config={},
    train_dataloader=None,
    val_dataloader=None,
):

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["decay"])

    train_loss_list = []

    device = config["device"]

    acu_loss = 0
    min_loss = 1e8

    i = 0
    epoch = config["epoch"]
    early_stop_win = 15

    model.train()

    stop_improve_count = 0

    dataloader = train_dataloader

    for i_epoch in range(epoch):

        acu_loss = 0
        model.train()

        for x, labels, attack_labels, edge_index in dataloader:

            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

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
                i_epoch, epoch, acu_loss / len(dataloader), acu_loss
            ),
            flush=True,
        )

        # use val dataset to judge
        if val_dataloader is not None:

            val_loss, val_result = test(model, val_dataloader, device)

            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1

            if stop_improve_count >= early_stop_win:
                break

        else:
            if acu_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = acu_loss

    return train_loss_list
