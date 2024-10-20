import numpy as np
from scipy.stats import iqr, rankdata
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def eval_scores(
    scores: list[float],
    true_scores: list[float],
    th_steps: int,
    return_thresold: bool = False,
) -> tuple[list[float], list[float]]:
    """
    Calculate F1 scores for different thresholds.

    Args:
        scores (list): Anomaly scores.
        true_scores (list): Ground truth labels.
        th_steps (int): Number of threshold steps.
        return_thresold (bool, optional): Whether to return thresholds. Defaults to False.

    Returns:
        list: F1 scores for each threshold.
        list: Thresholds (if return_thresold is True).
    """
    padding_list = [0] * (len(true_scores) - len(scores))

    if len(padding_list) > 0:
        scores = padding_list + scores

    scores_sorted = rankdata(scores, method="ordinal")
    th_steps = th_steps
    th_vals = np.array(range(th_steps)) * 1.0 / th_steps
    fmeas = [None] * th_steps
    thresholds = [None] * th_steps
    for i in range(th_steps):
        cur_pred = scores_sorted > th_vals[i] * len(scores)

        fmeas[i] = f1_score(true_scores, cur_pred)

        score_index = scores_sorted.tolist().index(int(th_vals[i] * len(scores) + 1))
        thresholds[i] = scores[score_index]

    if return_thresold:
        return fmeas, thresholds
    return fmeas


def get_err_median_and_iqr(
    predicted: list[float], groundtruth: list[float]
) -> tuple[float, float]:
    """
    Calculate the median and interquartile range of the absolute error.

    Args:
        predicted (list): Predicted values.
        groundtruth (list): Ground truth values.

    Returns:
        tuple: Median and IQR of the absolute error.
    """
    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    err_iqr = iqr(np_arr)

    return err_median, err_iqr


def get_full_err_scores(
    test_result: list[float], val_result: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate error scores for all features.

    Args:
        test_result (list): Test results.
        val_result (list): Validation results.

    Returns:
        tuple: All scores and normal distribution scores.
    """
    np_test_result = np.array(test_result)
    np_val_result = np.array(val_result)

    all_scores = None
    all_normals = None
    feature_num = np_test_result.shape[-1]

    for i in range(feature_num):
        test_re_list = np_test_result[:2, :, i]
        val_re_list = np_val_result[:2, :, i]

        scores = get_err_scores(test_re_list, val_re_list)
        normal_dist = get_err_scores(val_re_list, val_re_list)

        if all_scores is None:
            all_scores = scores
            all_normals = normal_dist
        else:
            all_scores = np.vstack((all_scores, scores))
            all_normals = np.vstack((all_normals, normal_dist))

    return all_scores, all_normals


def get_final_err_scores(
    test_result: list[float], val_result: list[float]
) -> np.ndarray:
    """
    Get final error scores.

    Args:
        test_result (list): Test results.
        val_result (list): Validation results.

    Returns:
        numpy.ndarray: Final error scores.
    """
    full_scores, all_normals = get_full_err_scores(
        test_result, val_result, return_normal_scores=True
    )

    all_scores = np.max(full_scores, axis=0)

    return all_scores


def get_err_scores(
    test_res: tuple[list[float], list[float]], val_res: tuple[list[float], list[float]]
) -> np.ndarray:
    """
    Calculate error scores.

    Args:
        test_res (tuple): Test results (predictions, ground truth).
        val_res (tuple): Validation results (predictions, ground truth).

    Returns:
        numpy.ndarray: Smoothed error scores.
    """
    test_predict, test_gt = test_res
    val_predict, val_gt = val_res

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)

    test_delta = np.abs(
        np.subtract(
            np.array(test_predict).astype(np.float64),
            np.array(test_gt).astype(np.float64),
        )
    )
    epsilon = 1e-2

    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)

    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i - before_num : i + 1])

    return smoothed_err_scores


def get_loss(predict: list[float], gt: list[float]) -> float:
    """
    Calculate mean squared error loss.

    Args:
        predict (list): Predicted values.
        gt (list): Ground truth values.

    Returns:
        float: Mean squared error.
    """
    ground_truth_list = np.array(gt)
    predicted_list = np.array(predict)

    loss = mean_squared_error(predicted_list, ground_truth_list)

    return loss


def get_f1_scores(
    total_err_scores: np.ndarray, gt_labels: list[float], topk: int = 1
) -> list[float]:
    """
    Calculate F1 scores for top-k error scores.

    Args:
        total_err_scores (numpy.ndarray): Total error scores.
        gt_labels (list): Ground truth labels.
        topk (int, optional): Number of top scores to consider. Defaults to 1.

    Returns:
        list: F1 scores.
    """
    print("total_err_scores", total_err_scores.shape)
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(
        total_err_scores, range(total_features - topk - 1, total_features), axis=0
    )[-topk:]

    topk_indices = np.transpose(topk_indices)

    total_topk_err_scores = []

    for i, indexs in enumerate(topk_indices):

        sum_score = sum(
            score
            for k, score in enumerate(
                sorted([total_err_scores[index, i] for j, index in enumerate(indexs)])
            )
        )

        total_topk_err_scores.append(sum_score)

    final_topk_fmeas = eval_scores(total_topk_err_scores, gt_labels, 400)

    return final_topk_fmeas


def get_val_performance_data(
    total_err_scores: np.ndarray,
    normal_scores: np.ndarray,
    gt_labels: list[float],
    topk: int = 1,
) -> tuple[float, float, float, float, float]:
    """
    Get performance data for validation set.

    Args:
        total_err_scores (numpy.ndarray): Total error scores.
        normal_scores (numpy.ndarray): Normal scores.
        gt_labels (list): Ground truth labels.
        topk (int, optional): Number of top scores to consider. Defaults to 1.

    Returns:
        tuple: F1 score, precision, recall, AUC score, and threshold.
    """
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(
        total_err_scores, range(total_features - topk - 1, total_features), axis=0
    )[-topk:]

    total_topk_err_scores = np.sum(
        np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0
    )

    thresold = np.max(normal_scores)

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    f1 = f1_score(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return f1, pre, rec, auc_score, thresold


def get_best_performance_data(
    total_err_scores: np.ndarray,
    gt_labels: list[float],
    topk: int = 1,
) -> tuple[float, float, float, float, float]:
    """
    Get best performance data.

    Args:
        total_err_scores (numpy.ndarray): Total error scores.
        gt_labels (list): Ground truth labels.
        topk (int, optional): Number of top scores to consider. Defaults to 1.

    Returns:
        tuple: Best F1 score, precision, recall, AUC score, and threshold.
    """
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(
        total_err_scores, range(total_features - topk - 1, total_features), axis=0
    )[-topk:]

    total_topk_err_scores = np.sum(
        np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0
    )

    final_topk_fmeas, thresolds = eval_scores(
        total_topk_err_scores, gt_labels, 400, return_thresold=True
    )

    th_i = final_topk_fmeas.index(max(final_topk_fmeas))
    thresold = thresolds[th_i]

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return max(final_topk_fmeas), pre, rec, auc_score, thresold


def print_score(
    test_result: list[float],
    val_result: list[float],
    report: str,
) -> None:
    """
    Calculate and print the model's performance scores.

    Args:
        test_result (list): Results from testing the model.
        val_result (list): Results from validating the model.
        report (str): Type of report to generate ('best' or 'val').
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
