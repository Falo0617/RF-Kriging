#!/usr/bin/env python3
"""
Prepare Guangdong station cache for kriging_rf pipeline.

- 读取输入 parquet/csv
- 尝试识别并重命名常见列（Station_Id_C -> station_id, TEM -> temperature, elev1/elev/ELEVATION -> elevation, lat/lon 等）
- 清洗明显的无效观测（例：TEM 大于 100 或等于常见哨兵值 999017/999999 等）
- 按站点聚合（station_id），输出每站一行：station_id, lat, lon, elevation, temperature (mean), data_count, name
- 保存为 parquet（默认 guangdong_merged_cache.parquet）
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np
import glob

SENTINEL_VALUES = {999017, 999999, 99999, 9999}

def auto_find_input(data_dir='Guangzhou_data'):
    """在 data_dir 下按优先级寻找 parquet 然后 csv"""
    if not os.path.isdir(data_dir):
        return None
    # 优先 parquet
    pats = ['*.parquet', '*.parq', '*.pq', '*.parquet.gz', '*.csv', '*.txt']
    for p in pats:
        full = glob.glob(os.path.join(data_dir, p))
        if full:
            return full[0]
    return None

def find_col(df, candidates):
    """在 df 中查找第一匹配的列名（大小写不敏感）。返回列名或 None"""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def sanitize_temp(x):
    try:
        if pd.isna(x):
            return np.nan
        xv = float(x)
        if int(xv) in SENTINEL_VALUES:
            return np.nan
        if xv < -60 or xv > 60:
            return np.nan
        return xv
    except Exception:
        return np.nan

def sanitize_elev(x):
    try:
        if pd.isna(x):
            return np.nan
        xv = float(x)
        if int(xv) in SENTINEL_VALUES:
            return np.nan
        if xv < -500 or xv > 10000:
            return np.nan
        return xv
    except Exception:
        return np.nan

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # 识别列
    col_station = find_col(df, ['Station_Id_C', 'station_id', 'StationId', 'station'])
    col_temp = find_col(df, ['TEM', 'TEMP', 'temperature', 'temp'])
    col_lat = find_col(df, ['lat', 'LAT', 'latitude', 'LATITUDE'])
    col_lon = find_col(df, ['lon', 'LON', 'longitude', 'LONGITUDE'])
    col_elev = find_col(df, ['elev1', 'elevation', 'ELEVATION', 'elev'])
    col_name = find_col(df, ['name', 'station_name', 'stn_name'])

    if col_station is None:
        df['_tmp_idx'] = df.index.astype(str)
        col_station = '_tmp_idx'
        print("Warning: station id not found; using row index as station_id (temporary).")

    if col_lat is None or col_lon is None:
        raise ValueError("Latitude/Longitude columns not found. Found columns: " + ", ".join(df.columns))

    df2 = pd.DataFrame()
    df2['station_id'] = df[col_station].astype(str)
    df2['lat'] = pd.to_numeric(df[col_lat], errors='coerce')
    df2['lon'] = pd.to_numeric(df[col_lon], errors='coerce')
    if col_elev is not None:
        df2['elevation'] = df[col_elev].apply(sanitize_elev)
    else:
        df2['elevation'] = np.nan
    if col_temp is not None:
        df2['temperature_raw'] = df[col_temp]
        df2['temperature'] = df[col_temp].apply(sanitize_temp)
    else:
        df2['temperature_raw'] = np.nan
        df2['temperature'] = np.nan

    if col_name is not None:
        df2['name'] = df[col_name].astype(str)
    else:
        df2['name'] = df2['station_id']

    df2 = df2.dropna(subset=['lat','lon']).reset_index(drop=True)

    agg_funcs = {
        'lat': 'median',
        'lon': 'median',
        'elevation': lambda x: np.nanmedian(x.values) if np.any(~np.isnan(x.values)) else np.nan,
        'temperature': lambda x: np.nanmean(x.values) if np.any(~np.isnan(x.values)) else np.nan,
        'temperature_raw': 'count',
        'name': lambda x: x.astype(str).mode().iloc[0] if len(x.astype(str).mode())>0 else x.astype(str).iloc[0]
    }
    grouped = df2.groupby('station_id').agg(agg_funcs).rename(columns={'temperature_raw': 'data_count'}).reset_index()

    grouped = grouped[['station_id','lat','lon','elevation','temperature','data_count','name']]

    grouped['latitude'] = grouped['lat']
    grouped['longitude'] = grouped['lon']

    return grouped

def main():
    # ✅ argparse 现在在 main() 函数内初始化
    p = argparse.ArgumentParser(description="Prepare Guangdong merged cache for kriging_rf")
    p.add_argument('--input', '-i', required=False, default=None, help='输入文件（.parquet 或 .csv）；若不指定，将在 Guangzhou_data 自动查找')
    p.add_argument('--out', '-o', default='guangdong_merged_cache.parquet', help='输出 parquet 文件路径')
    p.add_argument('--force', '-f', action='store_true', help='覆盖已存在的输出文件')
    args = p.parse_args()

    # ✅ 自动查找输入文件的逻辑也在这里
    if args.input is None:
        found = auto_find_input('Guangzhou_data')
        if found is None:
            print("No input specified and no files found in Guangzhou_data. 请用 --input 指定文件路径。")
            sys.exit(2)
        print("Auto-detected input:", found)
        args.input = found

    if not os.path.exists(args.input):
        print("Input file not found:", args.input)
        sys.exit(2)

    ext = os.path.splitext(args.input)[1].lower()
    if ext in ('.parquet', '.pq'):
        df = pd.read_parquet(args.input)
    elif ext in ('.csv', '.txt'):
        df = pd.read_csv(args.input)
    else:
        try:
            df = pd.read_parquet(args.input)
        except Exception:
            df = pd.read_csv(args.input)

    print("Loaded dataframe with columns:", list(df.columns))
    grouped = prepare(df)
    print(f"Aggregated to {len(grouped)} stations. Example rows:")
    print(grouped.head(10).to_string(index=False))

    if os.path.exists(args.out) and not args.force:
        print(f"Output file {args.out} exists. Use --force to overwrite.")
        sys.exit(0)

    try:
        grouped.to_parquet(args.out, index=False)
        print("Saved merged cache to:", args.out)
    except Exception as e:
        print("Failed to save parquet:", e)
        out_csv = os.path.splitext(args.out)[0] + ".csv"
        grouped.to_csv(out_csv, index=False)
        print("Saved fallback CSV to:", out_csv)

if __name__ == '__main__':
    main()