import numpy as np


def calculate_rmse(y_pred, y_true):
    """Root mean squared error (RMSE)."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def calculate_mae(y_pred, y_true):
    """Mean absolute error (MAE)."""
    return float(np.mean(np.abs(y_pred - y_true)))


def calculate_mape(y_pred, y_true):
    """Mean absolute percentage error (MAPE), in percent."""
    # Avoid tiny denominators that blow up the percentage.
    return float(np.mean(np.abs(y_pred - y_true) / np.clip(np.abs(y_true), 1.0, None)) * 100)


def calculate_r2(y_pred, y_true):
    """R^2 (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def calculate_hypervolume(pareto_objectives, ref_point=None):
    """
    Compute 2D hypervolume for a Pareto front.

    Args:
        pareto_objectives: (N, 2) array of [val_mse, complexity] values to minimize
        ref_point: (2,) reference point; defaults to a fixed global point

    Returns:
        Hypervolume value (larger is better)
    """
    if len(pareto_objectives) == 0:
        return 0.0
    
    objs = np.asarray(pareto_objectives)
    
    # Pick a reference point that dominates all solutions.
    if ref_point is None:
        # Fixed global reference point for fair comparison across runs.
        # 10.0 is large for normalized MSE; 2,000,000 is above typical model sizes.
        ref_point = (10.0, 2000000.0)
    
    ref_point = np.asarray(ref_point, dtype=float)
    
    # Sort by the first objective (ascending).
    sorted_idx = np.argsort(objs[:, 0])
    sorted_objs = objs[sorted_idx]
    
    hv = 0.0
    prev_x = ref_point[0]
    max_y = ref_point[1]
    
    # Sweep from right to left along the first objective.
    for i in range(len(sorted_objs) - 1, -1, -1):
        x_i = sorted_objs[i, 0]
        y_i = sorted_objs[i, 1]
        
        # Only accumulate if the reference point dominates this solution.
        if x_i < ref_point[0] and y_i < ref_point[1]:
            # Width is the x-distance to the next point.
            width = prev_x - x_i
            # Height is the capped y-distance.
            height = min(max_y, ref_point[1]) - y_i
            hv += width * height
            # Update max_y for the next iteration.
            max_y = min(max_y, y_i)
        
        prev_x = x_i
    
    return float(max(0.0, hv))


def calculate_igd(pareto_objectives, pareto_reference):
    """
    Inverted generational distance (IGD).

    Lower is better. Measures average distance from the reference front
    to the found Pareto front.

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
    
    # For each reference point, find the nearest point on the found front.
    distances = []
    for ref_point in ref:
        min_dist = np.min(np.linalg.norm(objs - ref_point, axis=1))
        distances.append(min_dist)
    
    return float(np.mean(distances))


def compute_test_metrics(y_pred, y_true, scaling_params=None):
    """
    Compute the standard single-objective test metrics.

    Args:
        y_pred: predictions (any scale)
        y_true: ground truth (any scale)
        scaling_params: optional dict with "mean" and "std" for original scale data

    Returns:
        dict with RMSE, MAE, MAPE, and R^2
    """
    # Normalize when scaling params are provided.
    if scaling_params:
        mean = scaling_params["mean"]
        std = scaling_params["std"]
        y_pred_norm = (y_pred - mean) / std
        y_true_norm = (y_true - mean) / std
    else:
        y_pred_norm = y_pred
        y_true_norm = y_true
    
    return {
        "rmse": calculate_rmse(y_pred, y_true),       # original scale metric
        "mae": calculate_mae(y_pred, y_true),         # original scale metric
        "mape": calculate_mape(y_pred, y_true),       # percent error
        "r2": calculate_r2(y_pred_norm, y_true_norm), # normalized scale
    }
