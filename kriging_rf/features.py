"""
特征与坐标工具：经纬->平面坐标、DEM 插值、局部站点统计（density/mean/std/dist_to_nearest）
"""
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import warnings

warnings.filterwarnings('ignore')


def stations_to_xy(df_stations, lon_center=None, lat_center=None):
    df = df_stations.copy()
    if lon_center is None:
        lon_center = df['longitude'].mean()
    if lat_center is None:
        lat_center = df['latitude'].mean()
    df['x'] = (df['lon'] - lon_center) * 111.32 * 1000
    df['y'] = (df['lat'] - lat_center) * 111.32 * 1000
    return df, lon_center, lat_center


def interpolate_dem_to_grid(station_x, station_y, station_elev, grid_xx, grid_yy):
    points = np.column_stack([station_x, station_y])
    grid_points = np.column_stack([grid_xx.ravel(), grid_yy.ravel()])
    grid_dem = griddata(points, station_elev, grid_points, method='linear').reshape(grid_xx.shape)
    if np.any(np.isnan(grid_dem)):
        grid_dem_nn = griddata(points, station_elev, grid_points, method='nearest').reshape(grid_xx.shape)
        grid_dem = np.where(np.isnan(grid_dem), grid_dem_nn, grid_dem)
    return grid_dem


def local_spatial_stats(station_x, station_y, station_z, radius_m=50000):
    """
    计算每个站点的局部统计量（在给定半径内）：
      local_mean, local_std, n_neighbors, dist_to_nearest
    返回 dict of numpy arrays, 与输入站点顺序一致
    """
    pts = np.column_stack([station_x, station_y])
    tree = cKDTree(pts)
    n = len(station_x)
    local_mean = np.full(n, np.nan)
    local_std = np.full(n, np.nan)
    n_neighbors = np.zeros(n, dtype=int)
    dist_to_nearest = np.full(n, np.nan)

    for i in range(n):
        idxs = tree.query_ball_point(pts[i], r=radius_m)
        # remove self
        idxs = [j for j in idxs if j != i]
        n_neighbors[i] = len(idxs)
        if len(idxs) == 0:
            local_mean[i] = station_z[i]
            local_std[i] = 0.0
            dist_to_nearest[i] = np.nan
        else:
            vals = station_z[idxs]
            local_mean[i] = np.nanmean(vals)
            local_std[i] = np.nanstd(vals)
            dists, _ = tree.query(pts[i], k=2)  # first is 0 (self), second is nearest
            dist_to_nearest[i] = dists[1] if len(dists) > 1 else np.nan

    return {
        'local_mean': local_mean,
        'local_std': local_std,
        'n_neighbors': n_neighbors,
        'dist_to_nearest': dist_to_nearest
    }