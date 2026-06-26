"""
特征与坐标工具：经纬->平面坐标、DEM 插值、局部站点统计（density/mean/std/dist_to_nearest）
兼容并标准化多种常见站点字段名（latitude/lat, longitude/lon, elev1/elevation/elev, TEM/temperature）
返回的 DataFrame 至少包含列: ['station_id','lat','lon','elevation','temperature','x','y']
"""
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import warnings

warnings.filterwarnings('ignore')


def _pick_column(df, candidates, default=None):
    """从候选列名中选第一个存在于 df.columns 的（按顺序）"""
    for c in candidates:
        if c in df.columns:
            return c
    return default


def stations_to_xy(df_stations, lon_center=None, lat_center=None):
    """
    将站点表（可能含多种列名变体）标准化并生成 x,y (米) 投影坐标。
    输入 df_stations 常见列（任选）:
      - station id: 'station_id' or 'Station_Id_C' or 'id'
      - latitude: 'lat' or 'latitude'
      - longitude: 'lon' or 'longitude'
      - elevation: 'elev1' or 'elevation' or 'elev'
      - temperature: 'TEM' or 'temperature' or 'temp'
      - name: 'name' (optional)
    返回 (df_out, lon_center, lat_center)：
      df_out 包含至少这些列: ['station_id','lat','lon','elevation','temperature','name','x','y']
    """
    df = df_stations.copy()

    # Detect columns
    col_id = _pick_column(df, ['station_id', 'Station_Id_C', 'id'], default=None)
    col_lat = _pick_column(df, ['lat', 'latitude'], default=None)
    col_lon = _pick_column(df, ['lon', 'longitude'], default=None)
    col_elev = _pick_column(df, ['elev1', 'elevation', 'elev'], default=None)
    col_temp = _pick_column(df, ['TEM', 'temperature', 'temp'], default=None)
    col_name = _pick_column(df, ['name', 'station_name'], default=None)

    if col_lat is None or col_lon is None:
        raise KeyError("stations_to_xy: input dataframe must contain latitude and longitude (lat/lon or latitude/longitude).")

    # Build standardized DataFrame
    out = {}
    out['station_id'] = df[col_id] if col_id is not None else (df.index.astype(str))
    out['lat'] = df[col_lat].astype(float)
    out['lon'] = df[col_lon].astype(float)
    out['elevation'] = df[col_elev].astype(float) if col_elev is not None else np.zeros(len(df))
    out['temperature'] = df[col_temp].astype(float) if col_temp is not None else np.full(len(df), np.nan)
    out['name'] = df[col_name] if col_name is not None else out['station_id']

    out_df = out = __import__('pandas').DataFrame(out)

    # center lon/lat
    if lon_center is None:
        lon_center = float(out_df['lon'].mean())
    if lat_center is None:
        lat_center = float(out_df['lat'].mean())

    # convert degrees to meters approximately (local equirectangular)
    out_df['x'] = (out_df['lon'] - lon_center) * 111.32 * 1000.0
    out_df['y'] = (out_df['lat'] - lat_center) * 111.32 * 1000.0

    # ensure column order / names expected downstream
    # downstream expects: x,y,elevation,temperature,station_id, maybe name
    cols = ['station_id', 'name', 'lat', 'lon', 'x', 'y', 'elevation', 'temperature']
    for c in cols:
        if c not in out_df.columns:
            out_df[c] = np.nan
    return out_df[cols], lon_center, lat_center


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