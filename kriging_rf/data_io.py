"""
数据加载与缓存封装。
优先使用 data_preprocessor.py (如果存在)，否则退回到直接读取 CSV 的简易实现。
提供 load_and_select(...) 直接返回已筛选的站点 DataFrame。
支持温度（TEMP）、露点温度（DEWP）和风速（WDSP）三种目标变量。
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# try to import your existing preprocessor (optional)
try:
    from data_preprocessor import preprocess_gsod_folder, load_preprocessed, filter_stations_by_radius  # type: ignore
    HAS_PREPROCESSOR = True
except Exception:
    HAS_PREPROCESSOR = False


def _load_csv_fallback(folder_path: str, year: Optional[int] = None, target_col: str = 'temperature') -> pd.DataFrame:
    """
    当没有 preprocessor 时使用的轻量 CSV 读取（尽量只读取必需列）

    参数:
        folder_path: GSOD CSV 文件所在文件夹
        year: 可选，筛选特定年份
        target_col: 目标变量列名，'temperature', 'dew_point' 或 'wind_speed'

    返回:
        DataFrame，包含 station_id, latitude, longitude, elevation,
        temperature, dew_point, wind_speed, data_count
    """
    # 确定目标列名、缩放因子、合理范围
    if target_col == 'dew_point':
        value_col = 'DEWP'
        value_scale = 10.0
        min_val, max_val = -50, 40
        target_field = 'dew_point'
    elif target_col == 'wind_speed':
        value_col = 'WDSP'
        value_scale = 10.0  # 0.1 m/s -> m/s
        min_val, max_val = 0, 100
        target_field = 'wind_speed'
    else:  # temperature
        value_col = 'TEMP'
        value_scale = 10.0
        min_val, max_val = -60, 60
        target_field = 'temperature'

    records = []
    folder = Path(folder_path)

    for fp in folder.glob("*.csv"):
        try:
            df = pd.read_csv(fp, encoding='utf-8', low_memory=False)

            # 检查必要列
            if 'LATITUDE' not in df.columns or 'LONGITUDE' not in df.columns:
                continue
            if value_col not in df.columns:
                continue

            # 年份筛选
            if 'DATE' in df.columns and year is not None:
                df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
                df = df[df['DATE'].dt.year == int(year)]
                if df.empty:
                    continue

            # 读取目标值
            val_mean = df[value_col].mean()
            if abs(val_mean) > 100:
                val_mean = val_mean / value_scale
            if pd.isna(val_mean) or val_mean < min_val or val_mean > max_val:
                continue

            lat = df['LATITUDE'].iloc[0]
            lon = df['LONGITUDE'].iloc[0]
            elev = df['ELEVATION'].iloc[0] if 'ELEVATION' in df.columns else 0.0

            if pd.isna(lat) or pd.isna(lon):
                continue

            # 构建记录
            record = {
                'station_id': fp.stem,
                'latitude': float(lat),
                'longitude': float(lon),
                'elevation': float(elev) if not pd.isna(elev) else 0.0,
                'data_count': len(df),
                'source_file': str(fp)
            }

            # 存入主目标字段
            record[target_field] = float(val_mean)

            # 同时尝试读取其他辅助列（如果存在）
            if 'TEMP' in df.columns:
                temp_mean = df['TEMP'].mean() / 10.0
                if not pd.isna(temp_mean) and -60 < temp_mean < 60:
                    record['temperature'] = float(temp_mean)
            if 'DEWP' in df.columns:
                dewp_mean = df['DEWP'].mean() / 10.0
                if not pd.isna(dewp_mean) and -50 < dewp_mean < 40:
                    record['dew_point'] = float(dewp_mean)
            if 'WDSP' in df.columns:
                wdsp_mean = df['WDSP'].mean() / 10.0
                if not pd.isna(wdsp_mean) and 0 <= wdsp_mean <= 60:
                    record['wind_speed'] = float(wdsp_mean)

            records.append(record)
        except Exception:
            continue

    return pd.DataFrame(records)


def load_from_parquet(parquet_path: str,
                      aggregate: bool = True,
                      year: Optional[int] = None,
                      target_col: str = 'temperature',
                      verbose: bool = True) -> pd.DataFrame:
    """
    Load a parquet file and normalize column names for downstream pipeline.

    Behavior:
      - Accepts many common column names and maps them to canonical ones:
          station_id, lat/lon (and latitude/longitude), elevation, temperature,
          dew_point, wind_speed, name, Year
      - If aggregate=True: groups by station_id and returns one row per station with:
          station_id, latitude, longitude, lat, lon, elevation,
          temperature (mean of target), data_count, name, Year (if present)
      - If aggregate=False: returns the full table with standardized column names.
      - year: if provided and Year column found, filter to that year.
      - target_col: 'temperature', 'dew_point' or 'wind_speed'

    Returns:
      pandas.DataFrame (empty DataFrame if loading fails).
    """
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        if verbose:
            print(f"Failed to read parquet {parquet_path}: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        if verbose:
            print("Parquet loaded but empty.")
        return pd.DataFrame()

    # helper: build lowercase -> original name map
    cols_lower = {c.lower(): c for c in df.columns}

    def get_col_name(candidates):
        for cand in candidates:
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        return None

    # candidates
    station_col = get_col_name(['Station_Id_C', 'Station_Id', 'station_id', 'StationId', 'id'])
    lat_col = get_col_name(['lat', 'latitude', 'LAT', 'Latitude'])
    lon_col = get_col_name(['lon', 'longitude', 'LON', 'Longitude'])
    elev_col = get_col_name(['elev1', 'elev', 'elevation', 'ELEV', 'Elevation'])
    name_col = get_col_name(['name', 'station_name', 'station'])
    year_col = get_col_name(['Year', 'YEAR', 'year'])

    # 目标列
    temp_col = get_col_name(['TEM', 'TEMP', 'temp', 'temperature', 'Temperature'])
    dewp_col = get_col_name(['DEWP', 'dewp', 'dew_point', 'DewPoint'])
    wdsp_col = get_col_name(['WDSP', 'wdsp', 'wind_speed', 'WindSpeed'])

    if station_col is None:
        raise ValueError("Cannot find station id column in parquet. Candidates: Station_Id_C, Station_Id, station_id.")
    if lat_col is None or lon_col is None:
        if verbose:
            print("Warning: latitude/longitude columns not found in parquet. Expected 'lat'/'lon' or 'latitude'/'longitude'.")

    df_work = df.copy()

    # rename existing columns to canonical lower-case names
    rename_map = {}
    if station_col:
        rename_map[station_col] = 'station_id'
    if lat_col:
        rename_map[lat_col] = 'lat'
    if lon_col:
        rename_map[lon_col] = 'lon'
    if elev_col:
        rename_map[elev_col] = 'elevation'
    if name_col:
        rename_map[name_col] = 'name'
    if year_col:
        rename_map[year_col] = 'Year'

    # 根据 target_col 确定主列和辅助列
    if target_col == 'dew_point' and dewp_col:
        rename_map[dewp_col] = 'dew_point'
        if temp_col:
            rename_map[temp_col] = 'temperature_aux'
        if wdsp_col:
            rename_map[wdsp_col] = 'wind_speed_aux'
    elif target_col == 'wind_speed' and wdsp_col:
        rename_map[wdsp_col] = 'wind_speed'
        if temp_col:
            rename_map[temp_col] = 'temperature_aux'
        if dewp_col:
            rename_map[dewp_col] = 'dew_point_aux'
    else:
        # default temperature
        if temp_col:
            rename_map[temp_col] = 'temperature'
        if dewp_col:
            rename_map[dewp_col] = 'dew_point_aux'
        if wdsp_col:
            rename_map[wdsp_col] = 'wind_speed_aux'

    df_work = df_work.rename(columns=rename_map)

    # ensure numeric types
    if 'lat' in df_work.columns:
        df_work['lat'] = pd.to_numeric(df_work['lat'], errors='coerce')
        df_work['latitude'] = df_work['lat']
    elif 'latitude' in df_work.columns:
        df_work['latitude'] = pd.to_numeric(df_work['latitude'], errors='coerce')
        df_work['lat'] = df_work['latitude']

    if 'lon' in df_work.columns:
        df_work['lon'] = pd.to_numeric(df_work['lon'], errors='coerce')
        df_work['longitude'] = df_work['lon']
    elif 'longitude' in df_work.columns:
        df_work['longitude'] = pd.to_numeric(df_work['longitude'], errors='coerce')
        df_work['lon'] = df_work['longitude']

    if 'elevation' in df_work.columns:
        df_work['elevation'] = pd.to_numeric(df_work['elevation'], errors='coerce').fillna(0.0)
    else:
        if 'elev1' in df_work.columns:
            df_work['elevation'] = pd.to_numeric(df_work['elev1'], errors='coerce').fillna(0.0)
        else:
            df_work['elevation'] = 0.0

    # 处理主目标列
    if target_col == 'dew_point':
        if 'dew_point' in df_work.columns:
            df_work['dew_point'] = pd.to_numeric(df_work['dew_point'], errors='coerce')
        else:
            if verbose:
                print("Warning: DEWP column not found, falling back to temperature.")
            df_work['dew_point'] = df_work.get('temperature_aux', np.nan)
            if 'temperature_aux' in df_work.columns:
                df_work['dew_point'] = df_work['temperature_aux']
    elif target_col == 'wind_speed':
        if 'wind_speed' in df_work.columns:
            df_work['wind_speed'] = pd.to_numeric(df_work['wind_speed'], errors='coerce')
        else:
            if verbose:
                print("Warning: WDSP column not found, falling back to temperature.")
            df_work['wind_speed'] = df_work.get('temperature_aux', np.nan)
    else:
        # temperature
        if 'temperature' in df_work.columns:
            df_work['temperature'] = pd.to_numeric(df_work['temperature'], errors='coerce')
        else:
            if verbose:
                print("Warning: TEMP column not found, falling back to dew_point or wind_speed.")
            if 'dew_point_aux' in df_work.columns:
                df_work['temperature'] = df_work['dew_point_aux']
            elif 'wind_speed_aux' in df_work.columns:
                df_work['temperature'] = df_work['wind_speed_aux']
            else:
                df_work['temperature'] = np.nan

    # optional year filtering
    if year is not None and 'Year' in df_work.columns:
        try:
            df_work['Year'] = pd.to_numeric(df_work['Year'], errors='coerce').astype('Int64')
            df_work = df_work[df_work['Year'] == int(year)]
            if verbose:
                print(f"Filtered parquet to Year == {year}, remaining rows: {len(df_work)}")
        except Exception:
            if verbose:
                print("Year column present but failed to filter; continuing without year filter.")

    # aggregate per station if requested
    if aggregate:
        agg_dict = {}
        if 'latitude' in df_work.columns:
            agg_dict['latitude'] = ('latitude', lambda s: s.dropna().iloc[0] if s.dropna().shape[0] > 0 else np.nan)
        if 'longitude' in df_work.columns:
            agg_dict['longitude'] = ('longitude', lambda s: s.dropna().iloc[0] if s.dropna().shape[0] > 0 else np.nan)
        agg_dict['elevation'] = ('elevation', lambda s: float(s.dropna().iloc[0]) if s.dropna().shape[0] > 0 else 0.0)

        # 根据 target_col 选择主聚合列
        if target_col == 'dew_point' and 'dew_point' in df_work.columns:
            agg_dict['dew_point'] = ('dew_point', 'mean')
            # 辅助列
            if 'temperature_aux' in df_work.columns:
                agg_dict['temperature'] = ('temperature_aux', 'mean')
            if 'wind_speed_aux' in df_work.columns:
                agg_dict['wind_speed'] = ('wind_speed_aux', 'mean')
        elif target_col == 'wind_speed' and 'wind_speed' in df_work.columns:
            agg_dict['wind_speed'] = ('wind_speed', 'mean')
            if 'temperature_aux' in df_work.columns:
                agg_dict['temperature'] = ('temperature_aux', 'mean')
            if 'dew_point_aux' in df_work.columns:
                agg_dict['dew_point'] = ('dew_point_aux', 'mean')
        else:
            # temperature (default)
            if 'temperature' in df_work.columns:
                agg_dict['temperature'] = ('temperature', 'mean')
            if 'dew_point' in df_work.columns:
                agg_dict['dew_point'] = ('dew_point', 'mean')
            if 'wind_speed' in df_work.columns:
                agg_dict['wind_speed'] = ('wind_speed', 'mean')

        if 'name' in df_work.columns:
            agg_dict['name'] = ('name', lambda s: s.dropna().iloc[0] if s.dropna().shape[0] > 0 else '')

        df_work['_obs_count'] = 1
        agg_dict['data_count'] = ('_obs_count', 'sum')

        try:
            df_grp = df_work.groupby('station_id').agg(**{k: (v[0], v[1]) for k, v in agg_dict.items()}).reset_index()
        except Exception:
            # fallback manual aggregation
            groups = []
            for sid, g in df_work.groupby('station_id'):
                rec = {'station_id': sid}
                rec['latitude'] = float(g['latitude'].dropna().iloc[0]) if 'latitude' in g and g['latitude'].dropna().shape[0] > 0 else np.nan
                rec['longitude'] = float(g['longitude'].dropna().iloc[0]) if 'longitude' in g and g['longitude'].dropna().shape[0] > 0 else np.nan
                rec['elevation'] = float(g['elevation'].dropna().iloc[0]) if 'elevation' in g and g['elevation'].dropna().shape[0] > 0 else 0.0
                if target_col == 'dew_point':
                    rec['dew_point'] = float(g['dew_point'].mean()) if 'dew_point' in g else np.nan
                    rec['temperature'] = rec['dew_point']
                elif target_col == 'wind_speed':
                    rec['wind_speed'] = float(g['wind_speed'].mean()) if 'wind_speed' in g else np.nan
                    rec['temperature'] = rec['wind_speed']
                else:
                    rec['temperature'] = float(g['temperature'].mean()) if 'temperature' in g else np.nan
                rec['name'] = g['name'].dropna().iloc[0] if 'name' in g and g['name'].dropna().shape[0] > 0 else ''
                rec['data_count'] = int(g.shape[0])
                groups.append(rec)
            df_grp = pd.DataFrame(groups)

        # ensure lat/lon short names
        if 'latitude' in df_grp.columns:
            df_grp['lat'] = df_grp['latitude']
        if 'longitude' in df_grp.columns:
            df_grp['lon'] = df_grp['longitude']

        # ensure temperature column exists (for compatibility)
        if 'temperature' not in df_grp.columns:
            if target_col == 'dew_point' and 'dew_point' in df_grp.columns:
                df_grp['temperature'] = df_grp['dew_point']
            elif target_col == 'wind_speed' and 'wind_speed' in df_grp.columns:
                df_grp['temperature'] = df_grp['wind_speed']
            else:
                df_grp['temperature'] = np.nan

        # reorder
        cols_out = ['station_id', 'lat', 'lon', 'latitude', 'longitude', 'elevation', 'temperature', 'data_count', 'name']
        if 'dew_point' in df_grp.columns:
            cols_out.insert(6, 'dew_point')
        if 'wind_speed' in df_grp.columns:
            cols_out.insert(7, 'wind_speed')

        final_cols = [c for c in cols_out if c in df_grp.columns]
        df_out = df_grp[final_cols].copy().reset_index(drop=True)

        if verbose:
            print(f"Aggregated to {len(df_out)} stations from parquet.")
            if target_col == 'dew_point':
                print(f"  Using dew_point as target variable.")
            elif target_col == 'wind_speed':
                print(f"  Using wind_speed as target variable.")
            else:
                print(f"  Using temperature as target variable.")
        return df_out

    else:
        # not aggregating: ensure canonical columns exist
        if 'latitude' not in df_work.columns and 'lat' in df_work.columns:
            df_work['latitude'] = df_work['lat']
        if 'longitude' not in df_work.columns and 'lon' in df_work.columns:
            df_work['longitude'] = df_work['lon']
        # ensure temperature column exists
        if 'temperature' not in df_work.columns:
            if target_col == 'dew_point' and 'dew_point' in df_work.columns:
                df_work['temperature'] = df_work['dew_point']
            elif target_col == 'wind_speed' and 'wind_speed' in df_work.columns:
                df_work['temperature'] = df_work['wind_speed']
            else:
                df_work['temperature'] = np.nan
        return df_work


def load_and_select(folder_path: str,
                    target_lat: float,
                    target_lon: float,
                    radius_km: float = 200.0,
                    max_stations: int = 100,
                    year: Optional[int] = None,
                    target_col: str = 'temperature',
                    cache_path: str = "gsod_preprocessed.parquet",
                    manifest_path: str = "gsod_manifest.json",
                    force_preprocess: bool = False,
                    verbose: bool = True) -> pd.DataFrame:
    """
    加载站点数据并按距离筛选返回 top-N 最近站点。

    参数:
        folder_path: GSOD CSV 文件所在文件夹（或缓存文件目录）
        target_lat: 中心点纬度
        target_lon: 中心点经度
        radius_km: 搜索半径（公里）
        max_stations: 最大站点数
        year: 可选，筛选特定年份
        target_col: 目标变量列名，'temperature', 'dew_point' 或 'wind_speed'
        cache_path: 缓存文件路径
        manifest_path: manifest 文件路径
        force_preprocess: 是否强制重新预处理
        verbose: 是否打印详细信息

    返回:
        DataFrame 包含: station_id, latitude, longitude, elevation,
                         temperature (主目标), dew_point, wind_speed, data_count
    """
    if HAS_PREPROCESSOR:
        try:
            if force_preprocess or not os.path.exists(cache_path):
                if verbose:
                    print("Running data_preprocessor to build cache...")
                df_all, _ = preprocess_gsod_folder(
                    folder_path, out_path=cache_path, manifest_path=manifest_path,
                    year=year, force=force_preprocess, verbose=verbose
                )
            else:
                if verbose:
                    print(f"Loading cached preprocessed data from {cache_path}...")
                df_all = load_preprocessed(cache_path)

            if df_all is None or df_all.empty:
                if verbose:
                    print("Preprocessed cache empty — falling back to CSV scanning.")
                df_all = _load_csv_fallback(folder_path, year=year, target_col=target_col)
            else:
                # 如果 preprocessor 返回的 DataFrame 缺少目标列，补充
                if target_col == 'dew_point' and 'dew_point' not in df_all.columns:
                    if verbose:
                        print("DEWP not in preprocessed cache, attempting to add from CSV scan...")
                    df_dewp = _load_csv_fallback(folder_path, year=year, target_col='dew_point')
                    if not df_dewp.empty and 'dew_point' in df_dewp.columns:
                        df_all = df_all.merge(df_dewp[['station_id', 'dew_point']], on='station_id', how='left')
                elif target_col == 'wind_speed' and 'wind_speed' not in df_all.columns:
                    if verbose:
                        print("WDSP not in preprocessed cache, attempting to add from CSV scan...")
                    df_wind = _load_csv_fallback(folder_path, year=year, target_col='wind_speed')
                    if not df_wind.empty and 'wind_speed' in df_wind.columns:
                        df_all = df_all.merge(df_wind[['station_id', 'wind_speed']], on='station_id', how='left')

        except Exception as e:
            if verbose:
                print(f"Preprocessor failed ({e}), falling back to CSV scanning.")
            df_all = _load_csv_fallback(folder_path, year=year, target_col=target_col)

        # 筛选
        try:
            df_sel = filter_stations_by_radius(df_all, target_lat, target_lon, radius_km, max_stations=max_stations)
        except Exception:
            if verbose:
                print("filter_stations_by_radius failed, using manual distance calculation.")
            if 'latitude' not in df_all.columns or 'longitude' not in df_all.columns:
                raise ValueError("DataFrame missing latitude/longitude columns for filtering.")
            lat_diff = (df_all['latitude'] - target_lat) * 111.32
            lon_diff = (df_all['longitude'] - target_lon) * 111.32 * np.cos(np.radians(target_lat))
            df_all['distance'] = (lat_diff**2 + lon_diff**2) ** 0.5
            df_sel = df_all[df_all['distance'] <= radius_km].sort_values('distance').head(max_stations).reset_index(drop=True)
            if 'distance' in df_sel.columns:
                df_sel = df_sel.drop(columns=['distance'])

        # 确保 temperature 列存在
        if 'temperature' not in df_sel.columns:
            if target_col == 'dew_point' and 'dew_point' in df_sel.columns:
                df_sel['temperature'] = df_sel['dew_point']
            elif target_col == 'wind_speed' and 'wind_speed' in df_sel.columns:
                df_sel['temperature'] = df_sel['wind_speed']
            else:
                df_sel['temperature'] = np.nan

        return df_sel
    else:
        if verbose:
            print("data_preprocessor not found — scanning CSVs directly (slower).")
        df_all = _load_csv_fallback(folder_path, year=year, target_col=target_col)

        if df_all is None or df_all.empty:
            return pd.DataFrame()

        # compute approximate distance and select
        lat_diff = (df_all['latitude'] - target_lat) * 111.32
        lon_diff = (df_all['longitude'] - target_lon) * 111.32 * np.cos(np.radians(target_lat))
        df_all['distance'] = (lat_diff**2 + lon_diff**2) ** 0.5
        df_sel = df_all[df_all['distance'] <= radius_km].sort_values('distance').head(max_stations).reset_index(drop=True)
        if 'distance' in df_sel.columns:
            df_sel = df_sel.drop(columns=['distance'])

        # ensure temperature column exists
        if 'temperature' not in df_sel.columns:
            if target_col == 'dew_point' and 'dew_point' in df_sel.columns:
                df_sel['temperature'] = df_sel['dew_point']
            elif target_col == 'wind_speed' and 'wind_speed' in df_sel.columns:
                df_sel['temperature'] = df_sel['wind_speed']
            else:
                df_sel['temperature'] = np.nan

        return df_sel