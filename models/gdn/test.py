import torch
import torch.nn as nn


def test(model, dataloader, device):
    loss_func = nn.MSELoss(reduction="mean")

    test_loss_list = []

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    model.eval()

    i = 0
    acu_loss = 0
    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]

        with torch.no_grad():
            predicted = model(x).float().to(device)

            loss = loss_func(predicted, y)

            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)

        test_loss_list.append(loss.item())
        acu_loss += loss.item()

        i += 1

    test_predicted_list = t_test_predicted_list.tolist()
    test_ground_list = t_test_ground_list.tolist()
    test_labels_list = t_test_labels_list.tolist()

    avg_loss = sum(test_loss_list) / len(test_loss_list)

    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]