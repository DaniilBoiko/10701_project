import numpy as np
from sklearn.metrics import ndcg_score
from tqdm.autonotebook import tqdm

def ndcg_10(y_true, y_pred):
    return ndcg_score(y_true, y_pred, k=10)

def ndcg_5(y_true, y_pred):
    return ndcg_score(y_true, y_pred, k=5)

def evaluate_groupwise(y_true, y_pred, group_ids, metric):
    unique_group_ids = np.unique(group_ids)
    scores = np.zeros_like(unique_group_ids, dtype=np.float64)
    for i, group_id in enumerate(tqdm(unique_group_ids)):
        group_mask = group_ids == group_id
        scores[i] = metric([y_true[group_mask]], [y_pred[group_mask]])
    return np.mean(scores)

def compute_groupwise_metrics(y_true, y_pred, group_ids, metrics):
    return {metric.__name__: evaluate_groupwise(y_true, y_pred, group_ids, metric) for metric in metrics}