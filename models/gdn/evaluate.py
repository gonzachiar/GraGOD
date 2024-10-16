import numpy as np
from scipy.stats import iqr, rankdata
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


# calculate F1 scores
def eval_scores(scores, true_scores, th_steps, return_thresold=False):
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


def get_err_median_and_iqr(predicted, groundtruth):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    err_iqr = iqr(np_arr)

    return err_median, err_iqr


def get_full_err_scores(test_result, val_result):
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


def get_final_err_scores(test_result, val_result):
    full_scores, all_normals = get_full_err_scores(
        test_result, val_result, return_normal_scores=True
    )

    all_scores = np.max(full_scores, axis=0)

    return all_scores


def get_err_scores(test_res, val_res):
    test_predict, test_gt = test_res
    val_predict, val_gt = val_res

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)

    test_delta = np.abs(
        np.subtract(np.array(test_predict).astype(np.float64), np.array(test_gt).astype(np.float64))
    )
    epsilon = 1e-2

    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)

    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i - before_num : i + 1])

    return smoothed_err_scores


def get_loss(predict, gt):

    ground_truth_list = np.array(gt)
    predicted_list = np.array(predict)

    loss = mean_squared_error(predicted_list, ground_truth_list)

    return loss


def get_f1_scores(total_err_scores, gt_labels, topk=1):
    print("total_err_scores", total_err_scores.shape)
    # remove the highest and lowest score at each timestep
    total_features = total_err_scores.shape[0]

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(
        total_err_scores, range(total_features - topk - 1, total_features), axis=0
    )[-topk:]

    topk_indices = np.transpose(topk_indices)

    total_topk_err_scores = []
    # topk_anomaly_sensors = []

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


def get_val_performance_data(total_err_scores, normal_scores, gt_labels, topk=1):
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(
        total_err_scores, range(total_features - topk - 1, total_features), axis=0
    )[-topk:]

    total_topk_err_scores = []

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


def get_best_performance_data(total_err_scores, gt_labels, topk=1):

    total_features = total_err_scores.shape[0]

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(
        total_err_scores, range(total_features - topk - 1, total_features), axis=0
    )[-topk:]

    total_topk_err_scores = []

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
