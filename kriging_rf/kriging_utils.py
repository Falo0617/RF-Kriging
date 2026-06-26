"""
封装 Kriging 的常用函数：OK 插值、LOO OK、对点或网格做 residual kriging。

扩展说明：
 - perform_ok_loo 新增参数:
     neighbors: int or None，LOO 时对每个待预测点仅使用最近 neighbors 个训练点来拟合变异函数（加速/稳定）
     use_cache: bool, cache_path: str, overwrite: bool 用于缓存 LOO 结果 (np.savez)
"""
import warnings
import numpy as np
from pykrige.ok import OrdinaryKriging
from scipy.spatial import cKDTree
import os


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
        ok_grid, ss = ok.execute('grid', grid_x, grid_y)
        ok_grid = np.asarray(ok_grid)
        ok_grid = np.nan_to_num(ok_grid, nan=np.nanmean(station_z))
    except Exception as e:
        warnings.warn(f"OK interpolation failed, returning constant grid. Error: {e}")
        ok_grid = np.full_like(grid_xx, np.nanmean(station_z))

    return grid_xx, grid_yy, ok_grid


def perform_ok_loo(station_x, station_y, station_z,
                   variogram_model='spherical',
                   neighbors=None,
                   use_cache=False, cache_path='ok_loo.npz', overwrite=False):
    """
    Leave-one-out Ordinary Kriging predictions at the station points.
    - neighbors: 如果为 int，则对每个 LOO 拟合仅使用该点的最近 neighbors 个训练点（若 neighbors >= n-1 则使用所有）。
    - use_cache: 如果 True，会尝试读取/写入 cache_path。cache 保存字段 'ok_loo' (numpy array)。
    返回 numpy array (len == n)，缺失处为 np.nan。
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

    # try cache
    if use_cache and os.path.exists(cache_path) and not overwrite:
        try:
            data = np.load(cache_path, allow_pickle=True)
            ok_loo = data.get('ok_loo', None)
            if ok_loo is not None and len(ok_loo) == n:
                return ok_loo
        except Exception:
            pass

    preds = np.full(n, np.nan)
    pts = np.column_stack([station_x, station_y])
    tree = cKDTree(pts)

    for i in range(n):
        try:
            # select neighbor indices (exclude i)
            if neighbors is None:
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                idxs = np.nonzero(mask)[0]
            else:
                # k nearest including self
                k = min(max(1, int(neighbors) + 1), n)  # +1 to include self
                dists, inds = tree.query(pts[i], k=k)
                inds = np.atleast_1d(inds)
                # remove self from inds
                idxs = [int(j) for j in inds if int(j) != i]
                if len(idxs) == 0:
                    preds[i] = np.nan
                    continue

            ok = OrdinaryKriging(
                station_x[idxs], station_y[idxs], station_z[idxs],
                variogram_model=variogram_model,
                verbose=False,
                enable_plotting=False
            )
            p, ss = ok.execute('points', np.array([station_x[i]]), np.array([station_y[i]]))
            preds[i] = float(p[0])
        except Exception:
            preds[i] = np.nan

    # write cache if requested
    if use_cache:
        try:
            np.savez_compressed(cache_path, ok_loo=preds)
        except Exception:
            pass

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
