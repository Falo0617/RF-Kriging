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

SENTINEL_VALUES = {999017, 999999, 99999, 9999}

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
        # 如果温度绝对值极大，视为异常（单位通常是 °C，允许 -60..60）
        if xv < -60 or xv > 60:
            return np.nan
        # 某些数据是放大 10 倍的情况（极端少见），但这里不自动调整，留给人工检查
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
        if xv < -500 or xv > 10000:  # 海拔范围阈值
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
        # 如果没有 station id，尝试用经纬+name组合成 id
        df['_tmp_idx'] = df.index.astype(str)
        col_station = '_tmp_idx'
        print("Warning: station id not found; using row index as station_id (temporary).")

    if col_lat is None or col_lon is None:
        raise ValueError("Latitude/Longitude columns not found. Found columns: " + ", ".join(df.columns))

    # 先拷贝需要的列到标准名（降低后续模块出错概率）
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

    # 去掉经纬缺失的行
    df2 = df2.dropna(subset=['lat','lon']).reset_index(drop=True)

    # 聚合：按 station_id （同时保留 lat/lon 的中位/平均，如果同一 id 有不同经纬会用中位）
    agg_funcs = {
        'lat': 'median',
        'lon': 'median',
        'elevation': lambda x: np.nanmedian(x.values) if np.any(~np.isnan(x.values)) else np.nan,
        'temperature': lambda x: np.nanmean(x.values) if np.any(~np.isnan(x.values)) else np.nan,
        'temperature_raw': 'count',  # 用来计算 data_count 原始行数（含无效）
        'name': lambda x: x.astype(str).mode().iloc[0] if len(x.astype(str).mode())>0 else x.astype(str).iloc[0]
    }
    grouped = df2.groupby('station_id').agg(agg_funcs).rename(columns={'temperature_raw': 'data_count'}).reset_index()

    # 调整列名以满足 pipeline
    grouped = grouped[['station_id','lat','lon','elevation','temperature','data_count','name']]

    # 添加备用列名（pipeline 的不同模块可能查 'latitude'/'longitude'）
    grouped['latitude'] = grouped['lat']
    grouped['longitude'] = grouped['lon']

    return grouped

def main():
    p = argparse.ArgumentParser(description="Prepare Guangdong merged cache for kriging_rf")
    p.add_argument('--input', '-i', required=True, help='输入 parquet 或 CSV 文件路径')
    p.add_argument('--out', '-o', default='guangdong_merged_cache.parquet', help='输出 parquet 文件路径')
    p.add_argument('--force', '-f', action='store_true', help='覆盖已存在的输出文件')
    args = p.parse_args()

    if not os.path.exists(args.input):
        print("Input file not found:", args.input)
        sys.exit(2)

    # 读取输入（支持 parquet/csv）
    ext = os.path.splitext(args.input)[1].lower()
    if ext in ('.parquet', '.pq'):
        df = pd.read_parquet(args.input)
    elif ext in ('.csv', '.txt'):
        df = pd.read_csv(args.input)
    else:
        # 尝试自动判断 parquet first then csv
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

    # 保存 parquet（保持压缩）
    try:
        grouped.to_parquet(args.out, index=False)
        print("Saved merged cache to:", args.out)
    except Exception as e:
        print("Failed to save parquet:", e)
        # 退回保存 csv
        out_csv = os.path.splitext(args.out)[0] + ".csv"
        grouped.to_csv(out_csv, index=False)
        print("Saved fallback CSV to:", out_csv)

if __name__ == '__main__':
    main()