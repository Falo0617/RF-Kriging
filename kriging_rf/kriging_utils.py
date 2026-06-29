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
                   use_cache=False,
                   cache_path='ok_loo.npz',
                   overwrite=False):
    """
    对每个站点执行留一法普通克里金（LOO OK）预测。
    若 neighbors 指定，则仅使用最近的 neighbors 个站点；否则使用全部站点。
    结果可缓存为 .npz 文件。
    """
    station_x = np.asarray(station_x)
    station_y = np.asarray(station_y)
    station_z = np.asarray(station_z)
    n = len(station_x)
    if n == 0:
        return np.array([])
    if n <= 3:
        warnings.warn("perform_ok_loo: too few stations (<=3), returning NaNs")
        return np.full(n, np.nan)

    # 尝试从缓存加载
    if use_cache and not overwrite and cache_path and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            cached = data.get('preds')
            if cached is not None and len(cached) == n:
                # 检查是否全部为 NaN（可能是之前失败的结果），若是则重新计算
                if not np.all(np.isnan(cached)):
                    print(f"[OK LOO] Loaded cached predictions from {cache_path}")
                    return cached
                else:
                    print(f"[OK LOO] Cached file contains all NaN, recomputing...")
            else:
                print(f"[OK LOO] Cached file length mismatch or missing, recomputing...")
        except Exception as e:
            print(f"[OK LOO] Failed to load cache: {e}, recomputing...")

    # ========== 坐标标准化 ==========
    x_mean, x_std = station_x.mean(), station_x.std()
    y_mean, y_std = station_y.mean(), station_y.std()
    if x_std < 1e-10:
        x_std = 1.0
    if y_std < 1e-10:
        y_std = 1.0
    station_x_norm = (station_x - x_mean) / x_std
    station_y_norm = (station_y - y_mean) / y_std
    # ==============================

    preds = np.full(n, np.nan)
    pts = np.column_stack([station_x_norm, station_y_norm])
    tree = cKDTree(pts)

    # 如果 neighbors 为 None 或大于 n-1，则使用全部其他点
    if neighbors is None or neighbors >= n:
        use_all = True
        k_eff = n - 1
    else:
        use_all = False
        k_eff = min(neighbors, n - 1)

    for i in range(n):
        try:
            if use_all:
                idxs = [j for j in range(n) if j != i]
            else:
                # 查询最近 k_eff 个邻居（排除自身）
                dists, idxs = tree.query(pts[i], k=k_eff + 1)
                # idxs[0] 是自身，去掉
                idxs = idxs[1:] if len(idxs) > 1 else []
                if len(idxs) < 2:
                    # 邻居太少，跳过
                    continue

            # 用标准化坐标做克里金
            ok = OrdinaryKriging(
                station_x_norm[idxs],
                station_y_norm[idxs],
                station_z[idxs],
                variogram_model=variogram_model,
                verbose=False,
                enable_plotting=False
            )
            p, ss = ok.execute('points',
                               np.array([station_x_norm[i]]),
                               np.array([station_y_norm[i]]))
            preds[i] = float(p[0])
        except Exception as e:
            # 打印调试信息（可选）
            # print(f"LOO failed for index {i}: {e}")
            preds[i] = np.nan

    # 缓存结果
    if use_cache and cache_path:
        try:
            np.savez_compressed(cache_path, preds=preds)
            print(f"[OK LOO] Saved predictions to {cache_path}")
        except Exception as e:
            print(f"[OK LOO] Failed to save cache: {e}")

    return preds


def krige_residuals_to_points(train_x, train_y, train_resid, target_x, target_y,
                              variogram_model='spherical'):
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    train_resid = np.asarray(train_resid)
    target_x = np.asarray(target_x)
    target_y = np.asarray(target_y)

    if len(train_x) == 0:
        return np.zeros(len(target_x))

    # ========== 新增：坐标标准化（基于训练集） ==========
    x_mean, x_std = train_x.mean(), train_x.std()
    y_mean, y_std = train_y.mean(), train_y.std()
    if x_std < 1e-10:
        x_std = 1.0
    if y_std < 1e-10:
        y_std = 1.0
    train_x_norm = (train_x - x_mean) / x_std
    train_y_norm = (train_y - y_mean) / y_std
    target_x_norm = (target_x - x_mean) / x_std
    target_y_norm = (target_y - y_mean) / y_std
    # =================================================

    try:
        rk = OrdinaryKriging(
            train_x_norm, train_y_norm, train_resid,  # 用标准化坐标
            variogram_model=variogram_model,
            verbose=False,
            enable_plotting=False
        )
        res_pred, ss = rk.execute('points', target_x_norm, target_y_norm)  # 用标准化坐标
        return np.asarray(res_pred).ravel()
    except Exception as e:
        warnings.warn(f"krige_residuals_to_points failed, returning zeros. Error: {e}")
        return np.zeros(len(target_x))


def krige_residuals_to_grid(train_x, train_y, train_resid,
                            grid_x, grid_y,
                            variogram_model='spherical'):
    """
    对残差进行克里金插值到网格（坐标标准化版）

    参数:
        train_x, train_y: 训练站点坐标（任意单位，建议为米）
        train_resid: 训练站点残差值
        grid_x, grid_y: 待插值网格坐标（一维数组，与训练坐标同单位）
        variogram_model: 半变异函数模型（默认'spherical'）

    返回:
        残差网格（二维，形状为 len(grid_y) x len(grid_x)）
    """
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    train_resid = np.asarray(train_resid)
    grid_x = np.asarray(grid_x)
    grid_y = np.asarray(grid_y)

    if len(train_x) == 0 or len(train_resid) == 0:
        return np.zeros((len(grid_y), len(grid_x)))

    # ---------- 坐标标准化（基于训练集） ----------
    x_mean, x_std = train_x.mean(), train_x.std()
    y_mean, y_std = train_y.mean(), train_y.std()
    # 防止 std=0 的情况（所有点同 x 或同 y）
    if x_std < 1e-10:
        x_std = 1.0
    if y_std < 1e-10:
        y_std = 1.0

    train_x_norm = (train_x - x_mean) / x_std
    train_y_norm = (train_y - y_mean) / y_std
    grid_x_norm = (grid_x - x_mean) / x_std
    grid_y_norm = (grid_y - y_mean) / y_std
    # -----------------------------------------

    try:
        rk = OrdinaryKriging(
            train_x_norm, train_y_norm, train_resid,
            variogram_model=variogram_model,
            verbose=False,
            enable_plotting=False
        )
        # 执行网格插值，传入标准化后的网格坐标
        z_grid, ss = rk.execute('grid', grid_x_norm, grid_y_norm)
        return z_grid
    except Exception as e:
        warnings.warn(f"krige_residuals_to_grid failed, returning zeros. Error: {e}")
        return np.zeros((len(grid_y), len(grid_x)))