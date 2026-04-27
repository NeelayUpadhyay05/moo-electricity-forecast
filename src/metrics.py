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
    Calculate 2D hypervolume using the WFG algorithm.
    
    Args:
        pareto_objectives: (N, 2) array of [val_mse, complexity] values (minimize both)
        ref_point: (2,) reference point; default = (max_val_mse * 1.1, max_complexity * 1.1)
    
    Returns:
        Hypervolume value (larger is better)
    """
    if len(pareto_objectives) == 0:
        return 0.0
    
    objs = np.asarray(pareto_objectives)
    
    # Ensure reference point dominates all solutions
    if ref_point is None:
        ref_point = (np.max(objs[:, 0]) * 1.1, np.max(objs[:, 1]) * 1.1)
    
    ref_point = np.asarray(ref_point, dtype=float)
    
    # Sort by first objective (ascending)
    sorted_idx = np.argsort(objs[:, 0])
    sorted_objs = objs[sorted_idx]
    
    hv = 0.0
    prev_x = ref_point[0]
    max_y = ref_point[1]
    
    # Sweep from right to left along first objective
    for i in range(len(sorted_objs) - 1, -1, -1):
        x_i = sorted_objs[i, 0]
        y_i = sorted_objs[i, 1]
        
        # Only accumulate if point is dominated by reference
        if x_i < ref_point[0] and y_i < ref_point[1]:
            # Width is the x-distance to the next point
            width = prev_x - x_i
            # Height is minimum of current y and previous max_y
            height = min(max_y, ref_point[1]) - y_i
            hv += width * height
            # Update max_y for next iteration
            max_y = min(max_y, y_i)
        
        prev_x = x_i
    
    return float(max(0.0, hv))


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
