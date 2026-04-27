import numpy as np


def calculate_rmse(y_pred, y_true):
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def calculate_mae(y_pred, y_true):
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_pred - y_true)))


def calculate_mape(y_pred, y_true):
    """Mean Absolute Percentage Error (%)."""
    # Floor denominator to avoid extreme percentages when targets are near zero
    return float(np.mean(np.abs(y_pred - y_true) / np.clip(np.abs(y_true), 1.0, None)) * 100)


def calculate_r2(y_pred, y_true):
    """R² (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def calculate_hypervolume(pareto_objectives, ref_point=None):
    """
    Calculate hypervolume for a Pareto front.
    
    Args:
        pareto_objectives: (N, 2) array of [val_mse, complexity] values
        ref_point: (2,) reference point; default = (max_val_mse * 1.1, max_complexity * 1.1)
    
    Returns:
        Hypervolume value (larger is better after normalization)
    """
    if len(pareto_objectives) == 0:
        return 0.0
    
    objs = np.asarray(pareto_objectives)
    if ref_point is None:
        # Use a point dominated by all Pareto solutions
        ref_point = (np.max(objs[:, 0]) * 1.1, np.max(objs[:, 1]) * 1.1)
    
    ref_point = np.asarray(ref_point)
    hv = 0.0
    
    # Sort by first objective
    sorted_idx = np.argsort(objs[:, 0])
    sorted_objs = objs[sorted_idx]
    
    prev_width = ref_point[1]
    for i, obj in enumerate(sorted_objs):
        width = max(0, ref_point[1] - obj[1])
        height = max(0, ref_point[0] - obj[0])
        if i > 0:
            height = max(0, ref_point[0] - sorted_objs[i, 0]) - max(0, ref_point[0] - sorted_objs[i - 1, 0])
        hv += width * height
    
    return float(hv)


def calculate_igd(pareto_objectives, pareto_reference):
    """
    Inverted Generational Distance (IGD).
    
    Lower is better. Measures average distance from reference front to found Pareto front.
    
    Args:
        pareto_objectives: (N, 2) found Pareto front
        pareto_reference: (M, 2) reference Pareto front
    
    Returns:
        IGD value
    """
    if len(pareto_reference) == 0 or len(pareto_objectives) == 0:
        return float('inf')
    
    objs = np.asarray(pareto_objectives)
    ref = np.asarray(pareto_reference)
    
    # For each reference point, find minimum distance to found front
    distances = []
    for ref_point in ref:
        min_dist = np.min(np.linalg.norm(objs - ref_point, axis=1))
        distances.append(min_dist)
    
    return float(np.mean(distances))


def compute_test_metrics(y_pred, y_true, scaling_params=None):
    """
    Compute all single-objective test metrics.
    
    Args:
        y_pred: predictions (any scale)
        y_true: ground truth (any scale)
        scaling_params: optional dict with "mean" and "std" for original scale data
    
    Returns:
        dict with RMSE, MAE, MAPE, R²
    """
    # Normalize if scaling_params provided
    if scaling_params:
        mean = scaling_params["mean"]
        std = scaling_params["std"]
        y_pred_norm = (y_pred - mean) / std
        y_true_norm = (y_true - mean) / std
    else:
        y_pred_norm = y_pred
        y_true_norm = y_true
    
    return {
        "rmse": calculate_rmse(y_pred, y_true),           # original scale
        "mae": calculate_mae(y_pred, y_true),             # original scale
        "mape": calculate_mape(y_pred, y_true),           # percentage
        "r2": calculate_r2(y_pred_norm, y_true_norm),     # normalized
    }
