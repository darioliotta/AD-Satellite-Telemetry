###############################
# Imports # Imports # Imports #
###############################

# Note: Code in this file is modified from
# https://github.com/NetManAIOps/OmniAnomaly/blob/master/omni_anomaly/eval_methods.py

import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Union
from TranADPlus.src.spot import SPOT

def calc_point2point(predict, actual) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Calculate f1 score and roc_auc by predict and actual.

    `predict`: the predicted anomalies in a vector (n,) array (np.ndarray)  
    `actual`: the actual anomalies in a vector format (n,) array (np.ndarray)

    Returns: `tuple[float, float, float, float, float, float, float, float]`

    Tested (`test_omnianomaly.test_calcpoint2point`)
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    epsilon = 0. if TP + FP != 0 else 0.00001
    precision = TP / (TP + FP + epsilon)
    epsilon = 0. if TP + FN != 0 else 0.00001
    recall = TP / (TP + FN + epsilon)
    epsilon = 0. if precision + recall != 0 else 0.00001
    f1 = 2 * precision * recall / (precision + recall + epsilon)

    # Check if actual and predict are each a single value, apply bandaid
    # of setting first value of actual to a different one
    actual_vals = np.unique(actual, return_counts = True)
    if actual_vals[0].shape[0] == 1:
        actual[0] = 0 if actual[0] == 1 else 1
    
    roc_auc = roc_auc_score(actual, predict)
    return f1, precision, recall, TP, TN, FP, FN, roc_auc

def adjust_predicts(score, label,
                    threshold,
                    pred=None,
                    calc_latency=False,
                    return_original_pred=False) -> Union[np.ndarray, tuple[np.ndarray, float], tuple[np.ndarray, float, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    (Basically just marks entire regions of anomaly labels as correctly identified if a single region is correctly identified)

    `score`: The anomaly score (n,) (np.ndarray)  
    `label`: The ground-truth label (n,) (np.ndarray)  
    `threshold`: The threshold of anomaly score. A point is labeled as "anomaly" if its score is *higher* than the threshold. (float)  
    `pred`: if not None, adjust `pred` and ignore `score` and `threshold` (np.ndarray or None)  
    `calc_latency`: True or false, if true returns latency (bool)  
    `return_original_pred`: True or false, if true return unadjusted pred value (bool) 

    Returns: `Union[np.ndarray, tuple[np.ndarray, float], tuple[np.ndarray, float, np.ndarray], tuple[np.ndarray, np.ndarray]]`  

    Tested (`test_omnianomaly.test_adjust_predicts`)
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict_og = score > threshold
        predict = predict_og.copy()
    else:
        predict_og = pred
        predict = predict_og.copy()
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True

    # Convert from bool to float
    predict_og = predict_og.astype(float)
    predict = predict.astype(float)

    if calc_latency:
        if return_original_pred:
            return predict, latency / (anomaly_count + 1e-4), predict_og
        else:
            return predict, latency / (anomaly_count + 1e-4)
    else:
        if return_original_pred:
            return predict, predict_og
        else:
            return predict

def calc_seq(score, label, threshold) -> tuple[float, float, float, float, float, float, float, float, float]:
    """
    Find the f1 score by from provided threshold.
    Method from MTAD-GAT (https://github.com/ML4ITS/mtad-gat-pytorch)

    `score`: The anomaly score (n,) (np.ndarray)  
    `label`: The ground-truth label (n,) (np.ndarray)  
    `threshold`: The threshold of anomaly score. A point is labeled as "anomaly" if its score is *higher* than the threshold. (float)  

    Returns: `tuple[float, float, float, float, float, float, float, float, float]`

    Tested (`test_omnianomaly.test_calc_seq)
    """
    predict, latency = adjust_predicts(score, label, threshold, calc_latency=True)
    return calc_point2point(predict, label), latency

def bf_search(score, label, start, end, step_num, display_freq=1, verbose=True) -> dict:
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Method from MTAD-GAT (https://github.com/ML4ITS/mtad-gat-pytorch)

    `score`: The anomaly score (n,) (np.ndarray)    
    `label`: The ground-truth label (n,) (np.ndarray)    
    `start`: Starting threshold (floaot)  
    `end`:  Ending threshold (float)  
    `step_num`: Number of thresholds to test (int)  
    `display_freq`: How frequently to print BF search info (int)  
    `verbose`: Enable some print statements (bool)  

    Returns: `dict()`

    Tested (`test_omnianomaly.test_bf_search`)
    """

    if verbose:
        print(f"Finding best f1-score by searching for threshold..")
        print(f"Search range [{start}, {end}], {step_num} steps")
    thresholds = np.linspace(start, end, step_num)
    m = (-1.0, -1.0, -1.0)
    m_t = 0.0
    m_l = 0
    for i, threshold in enumerate(thresholds):
        target, latency = calc_seq(score, label, threshold)
        if target[0] > m[0]:
            m_t = threshold
            m = target
            m_l = latency
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)

    return {
        "bf_f1": m[0],
        "bf_precision": m[1],
        "bf_recall": m[2],
        "bf_TP": m[3],
        "bf_TN": m[4],
        "bf_FP": m[5],
        "bf_FN": m[6],
        'bf_ROC/AUC': m[7],
        "bf_threshold": m_t,
        "bf_latency": m_l,
    }


def pot_eval(init_score, score, label, q=1e-5, level=0.98):
    """
    Run POT method on given score. Returns result dictionary

    `init_score`: The data to get init threshold. (np.ndarray)  
        For `OmniAnomaly`, it should be the anomaly score of train set.  
    `score`: The data to run POT method. For `OmniAnomaly`, it should be the anomaly score of test set. (np.ndarray)  
    `label`: Labeled anomalous regions (np.ndarray)  
    `q`: Detection level (risk) (float)  
    `level` (float): Probability associated with the initial threshold t

    Returns: `dict`
    """
    while True:
        try:
            s = SPOT(q)  # SPOT object
            s.fit(init_score, score)  # data import
            s.initialize(level=level, min_extrema=False, verbose=False)  # initialization step
            break
        except:
            level = 0.999*level
    ret = s.run(dynamic=False)  # run
    pot_th = np.mean(ret['thresholds'])
    pred, p_latency, pred_og = adjust_predicts(score, label, pot_th, calc_latency=True, return_original_pred=True)
    p_t = calc_point2point(pred, label)
    return {
        'pot_f1': p_t[0],
        'pot_precision': p_t[1],
        'pot_recall': p_t[2],
        'pot_TP': p_t[3],
        'pot_TN': p_t[4],
        'pot_FP': p_t[5],
        'pot_FN': p_t[6],
        'pot_ROC/AUC': p_t[7],
        'pot_threshold': pot_th
    }, pred_og, pred
