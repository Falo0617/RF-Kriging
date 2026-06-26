"""
封装 Kriging 的常用函数：OK 插值、LOO OK、对点或网格做 residual kriging。

注意：
- 请确保传入的 station_x/station_y 是在同一线性度量下（比如已投影为米），
  因为 variogram/距离以欧氏距离计算，使用经纬度（degrees）会导致不合理的变异函数。
- pykrige 的 grid 输出形状通常是 (len(grid_y), len(grid_x))，与 meshgrid 的 ordering 相关。
"""

import warnings
import numpy as np
from pykrige.ok import OrdinaryKriging


def perform_ok_interpolation(station_x, station_y, station_z, grid_res=50, variogram_model='spherical'):
    station_x = np.asarray(station_x)
    station_y = np.asarray(station_y)
    station_z = np.asarray(station_z)

    if len(station_x) == 0:
        raise ValueError("No station coordinates provided to perform_ok_interpolation.")

    x_min, x_max = station_x.min(), station_x.max()
    y_min, y_max = station_y.min(), station_y.max()
    pad_x = (x_max - x_min) * 0.1 if x_max > x_min else 1.0
    pad_y = (y_max - y_min) * 0.1 if y_max > y_min else 1.0

    grid_x = np.linspace(x_min - pad_x, x_max + pad_x, grid_res)
    grid_y = np.linspace(y_min - pad_y, y_max + pad_y, grid_res)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    try:
        ok = OrdinaryKriging(
            station_x, station_y, station_z,
            variogram_model=variogram_model,
            verbose=False,
            enable_plotting=False
        )
        # pykrige execute('grid', xs, ys) -> returns array shaped (len(ys), len(xs))
        ok_grid, ss = ok.execute('grid', grid_x, grid_y)
        ok_grid = np.asarray(ok_grid)
        ok_grid = np.nan_to_num(ok_grid, nan=np.nanmean(station_z))
    except Exception as e:
        warnings.warn(f"OK interpolation failed, returning constant grid. Error: {e}")
        ok_grid = np.full_like(grid_xx, np.nanmean(station_z))

    return grid_xx, grid_yy, ok_grid


def perform_ok_loo(station_x, station_y, station_z, variogram_model='spherical'):
    """
    Leave-one-out Ordinary Kriging predictions at the station points.
    Returns a numpy array of predictions (same length as inputs). If a point
    cannot be predicted, the corresponding value will be np.nan.
    For very small n (<=3) we fallback to returning np.full(n, np.nan).
    """
    station_x = np.asarray(station_x)
    station_y = np.asarray(station_y)
    station_z = np.asarray(station_z)

    n = len(station_x)
    if n == 0:
        return np.array([])
    if n <= 3:
        warnings.warn("perform_ok_loo: too few stations (<=3), returning NaNs for LOO predictions.")
        return np.full(n, np.nan)

    preds = np.full(n, np.nan)

    # loop LOO (pykrige requires rebuild per LOO)
    for i in range(n):
        try:
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            ok = OrdinaryKriging(
                station_x[mask], station_y[mask], station_z[mask],
                variogram_model=variogram_model,
                verbose=False,
                enable_plotting=False
            )
            p, ss = ok.execute('points', np.array([station_x[i]]), np.array([station_y[i]]))
            preds[i] = float(p[0])
        except Exception:
            preds[i] = np.nan

    return preds


def krige_residuals_to_points(train_x, train_y, train_resid, target_x, target_y, variogram_model='spherical'):
    """
    Krige training residuals to target points. On failure returns zeros array.
    """
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    train_resid = np.asarray(train_resid)
    target_x = np.asarray(target_x)
    target_y = np.asarray(target_y)

    if len(train_x) == 0:
        warnings.warn("krige_residuals_to_points: no training points, returning zeros.")
        return np.zeros(len(target_x))

    try:
        rk = OrdinaryKriging(train_x, train_y, train_resid,
                             variogram_model=variogram_model, verbose=False, enable_plotting=False)
        res_pred, ss = rk.execute('points', target_x, target_y)
        return np.asarray(res_pred).ravel()
    except Exception as e:
        warnings.warn(f"krige_residuals_to_points failed, returning zeros. Error: {e}")
        return np.zeros(len(target_x))


def krige_residuals_to_grid(train_x, train_y, train_resid, grid_x, grid_y, variogram_model='spherical'):
    """
    Krige training residuals onto a grid specified by grid_x (1d) and grid_y (1d).
    Returns a 2D array shaped (len(grid_y), len(grid_x)) consistent with pykrige output.
    On failure returns zeros with the same shape.
    """
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    train_resid = np.asarray(train_resid)
    grid_x = np.asarray(grid_x)
    grid_y = np.asarray(grid_y)

    if len(train_x) == 0:
        warnings.warn("krige_residuals_to_grid: no training points, returning zero grid.")
        gx, gy = np.meshgrid(grid_x, grid_y)
        return np.zeros_like(gx)

    try:
        rk = OrdinaryKriging(train_x, train_y, train_resid,
                             variogram_model=variogram_model, verbose=False, enable_plotting=False)
        res_grid, ss = rk.execute('grid', grid_x, grid_y)
        res_grid = np.asarray(res_grid)
        res_grid = np.nan_to_num(res_grid, nan=0.0)
        return res_grid
    except Exception as e:
        warnings.warn(f"krige_residuals_to_grid failed, returning zero grid. Error: {e}")
        gx, gy = np.meshgrid(grid_x, grid_y)
        return np.zeros_like(gx)