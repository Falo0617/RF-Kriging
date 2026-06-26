"""
数据加载与缓存封装。
优先使用 data_preprocessor.py (如果存在)，否则退回到直接读取 CSV 的简易实现。
提供 load_and_select(...) 直接返回已筛选的站点 DataFrame。
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

# try to import your existing preprocessor (optional)
try:
    from data_preprocessor import preprocess_gsod_folder, load_preprocessed, filter_stations_by_radius  # type: ignore
    HAS_PREPROCESSOR = True
except Exception:
    HAS_PREPROCESSOR = False


def _load_csv_fallback(folder_path: str, year: int = None) -> pd.DataFrame:
    """当没有 preprocessor 时使用的轻量 CSV 读取（尽量只读取必需列）"""
    records = []
    folder = Path(folder_path)
    for fp in folder.glob("*.csv"):
        try:
            df = pd.read_csv(fp)
            # basic columns check
            if 'LATITUDE' not in df.columns or 'LONGITUDE' not in df.columns or 'TEMP' not in df.columns:
                continue
            if 'DATE' in df.columns and year is not None:
                df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
                df = df[df['DATE'].dt.year == int(year)]
                if df.empty:
                    continue
            temp_mean = df['TEMP'].mean()
            if abs(temp_mean) > 100:
                temp_mean = temp_mean / 10.0
            if pd.isna(temp_mean) or temp_mean < -60 or temp_mean > 60:
                continue
            lat = df['LATITUDE'].iloc[0]
            lon = df['LONGITUDE'].iloc[0]
            elev = df['ELEVATION'].iloc[0] if 'ELEVATION' in df.columns else 0.0
            if pd.isna(lat) or pd.isna(lon):
                continue
            records.append({
                'station_id': fp.stem,
                'latitude': float(lat),
                'longitude': float(lon),
                'elevation': float(elev) if not pd.isna(elev) else 0.0,
                'temperature': float(temp_mean),
                'data_count': len(df),
                'source_file': str(fp)
            })
        except Exception:
            continue
    return pd.DataFrame(records)

def load_from_parquet(parquet_path: str, aggregate: bool = True, year: int | None = None, verbose: bool = True) -> pd.DataFrame:
    """
    Load a parquet file and normalize column names for downstream pipeline.

    Behavior:
      - Accepts many common column names and maps them to canonical ones:
          station_id, lat/lon (and latitude/longitude), elevation, temperature, name, Year
      - If aggregate=True: groups by station_id and returns one row per station with:
          station_id, latitude, longitude, lat, lon, elevation, temperature (mean), data_count, name, Year (if present)
      - If aggregate=False: returns the full table with standardized column names (but no grouping)
      - year: if provided and Year column found, data will be filtered to that year before aggregating.

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
    temp_col = get_col_name(['TEM', 'TEMP', 'temp', 'temperature', 'Temperature'])
    lat_col = get_col_name(['lat', 'latitude', 'LAT', 'Latitude'])
    lon_col = get_col_name(['lon', 'longitude', 'LON', 'Longitude'])
    elev_col = get_col_name(['elev1', 'elev', 'elevation', 'ELEV', 'Elevation'])
    name_col = get_col_name(['name', 'station_name', 'station'])
    year_col = get_col_name(['Year', 'YEAR', 'year'])

    # require station id and some coordinates
    if station_col is None:
        raise ValueError("Cannot find station id column in parquet. Candidates: Station_Id_C, Station_Id, station_id.")
    if lat_col is None or lon_col is None:
        # we will not raise immediately: try to continue but warn
        if verbose:
            print("Warning: latitude/longitude columns not found in parquet. Expected 'lat'/'lon' or 'latitude'/'longitude'.")

    # create a working copy with canonical column names
    df_work = df.copy()

    # rename existing columns to canonical lower-case names in df_work
    rename_map = {}
    if station_col: rename_map[station_col] = 'station_id'
    if temp_col: rename_map[temp_col] = 'temperature'
    if lat_col: rename_map[lat_col] = 'lat'
    if lon_col: rename_map[lon_col] = 'lon'
    if elev_col: rename_map[elev_col] = 'elevation'
    if name_col: rename_map[name_col] = 'name'
    if year_col: rename_map[year_col] = 'Year'  # keep Year capitalized for clarity

    df_work = df_work.rename(columns=rename_map)

    # ensure numeric types where relevant
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
        # try elev1 fallback like some files
        if 'elev1' in df_work.columns:
            df_work['elevation'] = pd.to_numeric(df_work['elev1'], errors='coerce').fillna(0.0)
        else:
            df_work['elevation'] = 0.0

    if 'temperature' in df_work.columns:
        df_work['temperature'] = pd.to_numeric(df_work['temperature'], errors='coerce')
    else:
        # try TEM uppercase fallback if not captured (should have been renamed)
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
        # group by station_id
        group_cols = ['station_id']
        agg_dict = {}
        # for coords/elev/name take first non-null
        if 'latitude' in df_work.columns:
            agg_dict['latitude'] = ('latitude', lambda s: s.dropna().iloc[0] if s.dropna().shape[0] > 0 else np.nan)
        if 'longitude' in df_work.columns:
            agg_dict['longitude'] = ('longitude', lambda s: s.dropna().iloc[0] if s.dropna().shape[0] > 0 else np.nan)
        agg_dict['elevation'] = ('elevation', lambda s: float(s.dropna().iloc[0]) if s.dropna().shape[0] > 0 else 0.0)
        if 'temperature' in df_work.columns:
            agg_dict['temperature'] = ('temperature', 'mean')
        else:
            agg_dict['temperature'] = ('temperature', lambda s: np.nan)
        if 'name' in df_work.columns:
            agg_dict['name'] = ('name', lambda s: s.dropna().iloc[0] if s.dropna().shape[0] > 0 else '')
        # data_count
        df_work['_obs_count'] = 1
        agg_dict['data_count'] = ('_obs_count', 'sum')

        # use pandas groupby.agg with named aggregation for clarity (pandas >= 0.25)
        named_aggs = {}
        # build named_aggs careful for older/newer pandas:
        for out_name, (in_col, fn) in agg_dict.items():
            named_aggs[out_name] = pd.NamedAgg(column=in_col, aggfunc=fn) if hasattr(pd, 'NamedAgg') else (in_col, fn)

        try:
            df_grp = df_work.groupby('station_id').agg(**named_aggs).reset_index()
        except Exception:
            # fallback: manual aggregation
            groups = []
            for sid, g in df_work.groupby('station_id'):
                rec = {'station_id': sid}
                rec['latitude'] = float(g['latitude'].dropna().iloc[0]) if 'latitude' in g and g['latitude'].dropna().shape[0] > 0 else np.nan
                rec['longitude'] = float(g['longitude'].dropna().iloc[0]) if 'longitude' in g and g['longitude'].dropna().shape[0] > 0 else np.nan
                rec['elevation'] = float(g['elevation'].dropna().iloc[0]) if g['elevation'].dropna().shape[0] > 0 else 0.0
                rec['temperature'] = float(g['temperature'].mean()) if 'temperature' in g else np.nan
                rec['name'] = g['name'].dropna().iloc[0] if 'name' in g and g['name'].dropna().shape[0] > 0 else ''
                rec['data_count'] = int(g.shape[0])
                groups.append(rec)
            df_grp = pd.DataFrame(groups)

        # ensure lat/lon short names are present (for features.stations_to_xy)
        if 'latitude' in df_grp.columns:
            df_grp['lat'] = df_grp['latitude']
        if 'longitude' in df_grp.columns:
            df_grp['lon'] = df_grp['longitude']

        # reorder common columns
        cols_out = ['station_id', 'lat', 'lon', 'latitude', 'longitude', 'elevation', 'temperature', 'data_count', 'name']
        final_cols = [c for c in cols_out if c in df_grp.columns]
        df_out = df_grp[final_cols].copy().reset_index(drop=True)
        if verbose:
            print(f"Aggregated to {len(df_out)} stations from parquet.")
        return df_out

    else:
        # not aggregating: ensure canonical columns exist
        if 'latitude' not in df_work.columns and 'lat' in df_work.columns:
            df_work['latitude'] = df_work['lat']
        if 'longitude' not in df_work.columns and 'lon' in df_work.columns:
            df_work['longitude'] = df_work['lon']
        # try to keep common columns available
        return df_work

def load_and_select(folder_path: str,
                    target_lat: float,
                    target_lon: float,
                    radius_km: float = 200.0,
                    max_stations: int = 100,
                    year: int = None,
                    cache_path: str = "gsod_preprocessed.parquet",
                    manifest_path: str = "gsod_manifest.json",
                    force_preprocess: bool = False,
                    verbose: bool = True) -> pd.DataFrame:
    """
    加载站点数据并按距离筛选返回 top-N 最近站点。
    优先使用 data_preprocessor 的缓存/增量逻辑；否则使用 CSV fallback。
    返回 DataFrame 包含: station_id, latitude, longitude, elevation, temperature, data_count
    """
    if HAS_PREPROCESSOR:
        if force_preprocess or not os.path.exists(cache_path):
            if verbose:
                print("Running data_preprocessor to build cache...")
            df_all, _ = preprocess_gsod_folder(folder_path, out_path=cache_path, manifest_path=manifest_path, year=year, force=force_preprocess, verbose=verbose)  # type: ignore
        else:
            if verbose:
                print(f"Loading cached preprocessed data from {cache_path}...")
            df_all = load_preprocessed(cache_path)  # type: ignore
        if df_all is None or df_all.empty:
            if verbose:
                print("Preprocessed cache empty — falling back to CSV scanning.")
            df_all = _load_csv_fallback(folder_path, year=year)
        # do selection by radius
        df_sel = filter_stations_by_radius(df_all, target_lat, target_lon, radius_km, max_stations=max_stations)  # type: ignore
        return df_sel
    else:
        if verbose:
            print("data_preprocessor not found — scanning CSVs directly (slower).")
        df_all = _load_csv_fallback(folder_path, year=year)
        if df_all is None or df_all.empty:
            return pd.DataFrame()
        # compute approximate distance and select
        lat_diff = (df_all['latitude'] - target_lat) * 111.32
        lon_diff = (df_all['longitude'] - target_lon) * 111.32 * np.cos(np.radians(target_lat))
        df_all['distance'] = (lat_diff**2 + lon_diff**2) ** 0.5
        df_sel = df_all[df_all['distance'] <= radius_km].sort_values('distance').head(max_stations).reset_index(drop=True)
        if 'distance' in df_sel.columns:
            df_sel = df_sel.drop(columns=['distance'])
        return df_sel