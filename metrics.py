"""
Comprehensive Metrics for Centrality Prediction Evaluation

This module provides various metrics to evaluate the quality of centrality predictions:
- Ranking-based metrics (Kendall's Tau, Spearman's Rho)
- Top-K metrics (Precision@K, Recall@K, F1@K)
- Information Retrieval metrics (NDCG@K, MAP)
- Regression metrics (MSE, MAE, R²)
"""

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def kendall_tau(y_pred, y_true):
    """
    Kendall's Tau correlation coefficient.
    Measures ordinal association between predicted and true rankings.
    
    Returns: float in [-1, 1], where 1 = perfect agreement
    """
    tau, _ = kendalltau(y_pred, y_true)
    return tau


def spearman_rho(y_pred, y_true):
    """
    Spearman's rank correlation coefficient.
    
    Returns: float in [-1, 1]
    """
    rho, _ = spearmanr(y_pred, y_true)
    return rho


def precision_at_k(y_pred, y_true, k):
    """
    Precision@K: What fraction of top-K predicted nodes are actually in top-K true nodes?
    
    Args:
        y_pred: predicted scores
        y_true: true scores
        k: number of top items to consider
        
    Returns: float in [0, 1]
    """
    top_k_pred = set(np.argsort(y_pred)[-k:])
    top_k_true = set(np.argsort(y_true)[-k:])
    
    return len(top_k_pred & top_k_true) / k


def recall_at_k(y_pred, y_true, k):
    """
    Recall@K: What fraction of top-K true nodes are captured in top-K predictions?
    
    Returns: float in [0, 1]
    """
    top_k_pred = set(np.argsort(y_pred)[-k:])
    top_k_true = set(np.argsort(y_true)[-k:])
    
    return len(top_k_pred & top_k_true) / k


def f1_at_k(y_pred, y_true, k):
    """
    F1@K: Harmonic mean of Precision@K and Recall@K
    
    Returns: float in [0, 1]
    """
    prec = precision_at_k(y_pred, y_true, k)
    rec = recall_at_k(y_pred, y_true, k)
    
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def ndcg_at_k(y_pred, y_true, k):
    """
    Normalized Discounted Cumulative Gain@K
    Measures ranking quality with position-based discounting.
    
    Returns: float in [0, 1], where 1 = perfect ranking
    """
    # Get top-k indices by prediction
    top_k_indices = np.argsort(y_pred)[-k:][::-1]
    
    # DCG: sum of (relevance / log2(position + 1))
    dcg = 0.0
    for i, idx in enumerate(top_k_indices):
        relevance = y_true[idx]
        dcg += relevance / np.log2(i + 2)  # +2 because positions start at 1
    
    # IDCG: DCG of perfect ranking
    top_k_true_values = np.sort(y_true)[-k:][::-1]
    idcg = 0.0
    for i, relevance in enumerate(top_k_true_values):
        idcg += relevance / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def mean_average_precision(y_pred, y_true, k_values=[5, 10, 20]):
    """
    Mean Average Precision across multiple K values
    
    Returns: float in [0, 1]
    """
    aps = []
    for k in k_values:
        aps.append(precision_at_k(y_pred, y_true, k))
    return np.mean(aps)


def top_k_overlap(y_pred, y_true, k):
    """
    Jaccard similarity of top-K sets
    
    Returns: float in [0, 1]
    """
    top_k_pred = set(np.argsort(y_pred)[-k:])
    top_k_true = set(np.argsort(y_true)[-k:])
    
    intersection = len(top_k_pred & top_k_true)
    union = len(top_k_pred | top_k_true)
    
    return intersection / union if union > 0 else 0.0


def evaluate_all_metrics(y_pred, y_true, k_values=[5, 10, 20, 50]):
    """
    Compute all metrics for a single prediction.
    
    Args:
        y_pred: predicted centrality scores (numpy array)
        y_true: true centrality scores (numpy array)
        k_values: list of K values for top-K metrics
        
    Returns:
        dict: all computed metrics
    """
    results = {
        # Ranking correlations
        'kendall_tau': kendall_tau(y_pred, y_true),
        'spearman_rho': spearman_rho(y_pred, y_true),
        
        # Regression metrics
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2_score': r2_score(y_true, y_pred),
    }
    
    # Top-K metrics for each K
    for k in k_values:
        if k <= len(y_pred):
            results[f'precision@{k}'] = precision_at_k(y_pred, y_true, k)
            results[f'recall@{k}'] = recall_at_k(y_pred, y_true, k)
            results[f'f1@{k}'] = f1_at_k(y_pred, y_true, k)
            results[f'ndcg@{k}'] = ndcg_at_k(y_pred, y_true, k)
            results[f'overlap@{k}'] = top_k_overlap(y_pred, y_true, k)
    
    # MAP
    valid_k = [k for k in k_values if k <= len(y_pred)]
    if valid_k:
        results['map'] = mean_average_precision(y_pred, y_true, valid_k)
    
    return results


def aggregate_metrics(metrics_list):
    """
    Aggregate metrics across multiple graphs.
    
    Args:
        metrics_list: list of metric dicts
        
    Returns:
        dict: mean and std for each metric
    """
    if not metrics_list:
        return {}
    
    # Get all metric names
    metric_names = metrics_list[0].keys()
    
    aggregated = {}
    for name in metric_names:
        values = [m[name] for m in metrics_list if name in m and not np.isnan(m[name])]
        if values:
            aggregated[f'{name}_mean'] = np.mean(values)
            aggregated[f'{name}_std'] = np.std(values)
    
    return aggregated
