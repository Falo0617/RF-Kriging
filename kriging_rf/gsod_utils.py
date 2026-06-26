"""
GSOD (Global Summary of the Day) 数据处理工具

GSOD 是 NOAA/NCEI 提供的全球日尺度气象数据，包括温度、风速、降水等。
本模块用于：
1. 读取多个站点的 CSV 文件
2. 统一列名和数据格式
3. 按站点聚合
4. 清洗异常值
"""
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

SENTINEL_VALUES = {999.9, 9999.9, 999017, 999999, 99999, 9999}


def find_col(df, candidates):
    """在 df 中查找第一匹配的列名（大小写不敏感）。返回列名或 None"""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def sanitize_temp(x):
    """
    清洗温度数据
    - 检查哨兵值
    - 检查合理范围（-60 to 60°C）
    """
    try:
        if pd.isna(x):
            return np.nan
        xv = float(x)
        # 检查哨兵值（整数和浮点）
        if xv in SENTINEL_VALUES or int(xv) in SENTINEL_VALUES:
            return np.nan
        # GSOD 温度通常是 °F 或 °C
        if xv < -60 or xv > 60:
            return np.nan
        return xv
    except Exception:
        return np.nan


def sanitize_elev(x):
    """清洗海拔数据"""
    try:
        if pd.isna(x):
            return np.nan
        xv = float(x)
        if xv in SENTINEL_VALUES or int(xv) in SENTINEL_VALUES:
            return np.nan
        if xv < -500 or xv > 10000:
            return np.nan
        return xv
    except Exception:
        return np.nan


def read_gsod_csv(filepath):
    """
    读取单个 GSOD CSV 文件

    GSOD 标准列名（可能有变化）：
    - STN / USAF: 站点 ID
    - WBAN: WBAN ID
    - YEARMODA: 日期
    - TEMP: 平均温度 (°F)
    - LAT / LON: 经纬度
    - ELEV: 海拔
    - STNNAME: 站点名称

    返回: DataFrame 或 None（如果读取失败）
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Warning: Failed to read {filepath}: {e}")
        return None


def read_gsod_directory(directory, pattern="*.csv"):
    """
    读取目录下所有 GSOD CSV 文件并合并

    Args:
        directory: GSOD CSV 文件所在目录
        pattern: 文件匹配模式，默认 "*.csv"

    Returns:
        合并后的 DataFrame
    """
    csv_files = glob.glob(os.path.join(directory, pattern))

    if not csv_files:
        raise ValueError(f"No CSV files found in {directory} matching pattern {pattern}")

    print(f"Found {len(csv_files)} CSV files in {directory}")

    dfs = []
    for fpath in csv_files:
        df = read_gsod_csv(fpath)
        if df is not None:
            dfs.append(df)
            print(f"  ✓ Loaded {os.path.basename(fpath)} ({len(df)} rows)")

    if not dfs:
        raise ValueError("No GSOD files could be loaded")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(combined)} total records from {len(dfs)} files")
    print(f"Columns found: {list(combined.columns)}")

    return combined


def prepare_gsod(df: pd.DataFrame, temp_unit='F', drop_nan_temp=True, use_latest_date=True) -> pd.DataFrame:
    """
    处理 GSOD 数据：标准化列名、清洗数据、聚合按站点

    Args:
        df: 合并的 GSOD DataFrame
        temp_unit: 温度单位 'F'(华氏) 或 'C'(摄氏)，默认 'F'（GSOD 标准）
        drop_nan_temp: 是否删除温度为 NaN 的记录（默认 True，推荐）
        use_latest_date: 是否只保留最新日期的数据（默认 True）🔑 新增

    Returns:
        标准化后的 DataFrame（每行一个站点）
    """
    # 尝试识别各种列名变体
    col_station = find_col(df, ['STN', 'USAF', 'station_id', 'STATION', 'StationId'])
    col_wban = find_col(df, ['WBAN', 'wban', 'Wban'])
    col_temp = find_col(df, ['TEMP', 'temperature', 'TEMPERATURE', 'temp', 'MEAN', 'Mean'])
    col_lat = find_col(df, ['LAT', 'lat', 'latitude', 'LATITUDE'])
    col_lon = find_col(df, ['LON', 'lon', 'longitude', 'LONGITUDE'])
    col_elev = find_col(df, ['ELEV', 'elevation', 'ELEVATION', 'elev', 'altitude', 'ALT'])
    col_name = find_col(df, ['STNNAME', 'station_name', 'NAME', 'name', 'StationName', 'STNNAME'])
    col_date = find_col(df, ['YEARMODA', 'DATE', 'date', 'Year', 'YEAR'])  # 🔑 新增

    print(f"\n列名检测结果：")
    print(f"  Station ID: {col_station}")
    print(f"  Temperature: {col_temp}")
    print(f"  Latitude: {col_lat}")
    print(f"  Longitude: {col_lon}")
    print(f"  Elevation: {col_elev}")
    print(f"  Station Name: {col_name}")
    print(f"  Date: {col_date}")  # 🔑 新增

    # 验证必要列
    if col_lat is None or col_lon is None:
        raise ValueError(f"Latitude/Longitude columns not found. Found columns: {', '.join(df.columns)}")

    if col_station is None and col_wban is None:
        raise ValueError(f"Station ID columns (STN/USAF or WBAN) not found. Found columns: {', '.join(df.columns)}")

    if col_temp is None:
        raise ValueError(f"Temperature column not found. Found columns: {', '.join(df.columns)}")

    # 🔑 新增：关键步骤 - 按最新日期筛选
    if use_latest_date and col_date is not None:
        print(f"\n📅 日期列检测到，按最新日期筛选...")
        latest_date = df[col_date].max()
        print(f"   最新日期：{latest_date}")
        df_before = len(df)
        df = df[df[col_date] == latest_date].copy()
        df_after = len(df)
        print(f"   筛选前：{df_before} 条记录")
        print(f"   筛选后：{df_after} 条记录 (删除 {df_before - df_after} 条)")

    # 创建标准数据框
    df2 = pd.DataFrame()

    # 组合 station_id（优先用 STN，否则用 WBAN）
    if col_station is not None and col_wban is not None:
        df2['station_id'] = df[col_station].astype(str) + '_' + df[col_wban].astype(str)
    elif col_station is not None:
        df2['station_id'] = df[col_station].astype(str)
    else:
        df2['station_id'] = df[col_wban].astype(str)

    # 坐标和海拔
    df2['lat'] = pd.to_numeric(df[col_lat], errors='coerce')
    df2['lon'] = pd.to_numeric(df[col_lon], errors='coerce')

    if col_elev is not None:
        df2['elevation'] = df[col_elev].apply(sanitize_elev)
    else:
        df2['elevation'] = np.nan

    # 温度处理
    df2['temperature_raw'] = df[col_temp]
    df2['temperature'] = df[col_temp].apply(sanitize_temp)

    # 华氏转摄氏（如果需要）
    if temp_unit == 'F':
        valid = ~np.isnan(df2['temperature'])
        print(f"\n华氏 → 摄氏转换: {valid.sum()} 条记录")
        df2.loc[valid, 'temperature'] = (df2.loc[valid, 'temperature'] - 32) * 5 / 9

    # 站点名称
    if col_name is not None:
        df2['name'] = df[col_name].astype(str)
    else:
        df2['name'] = df2['station_id']

    print(f"\n数据清洗前:")
    print(f"  总记录数: {len(df2)}")
    print(f"  温度 NaN 数: {df2['temperature'].isna().sum()}")
    print(f"  经纬 NaN 数: {(df2['lat'].isna() | df2['lon'].isna()).sum()}")

    # 🔑 关键：删除温度为 NaN 的记录
    if drop_nan_temp:
        initial_count = len(df2)
        df2 = df2.dropna(subset=['temperature'])
        removed_count = initial_count - len(df2)
        if removed_count > 0:  # 🔑 新增条件判断
            print(f"\n删除温度 NaN 记录: {removed_count} 条 ({removed_count / initial_count * 100:.1f}%)")

    # 删除经纬缺失的行
    df2 = df2.dropna(subset=['lat', 'lon']).reset_index(drop=True)

    if len(df2) == 0:
        raise ValueError("No valid station coordinates after filtering")

    print(f"\n数据清洗后: {len(df2)} 条有效记录")

    # 按 station_id 聚合
    unique_stations = df2['station_id'].nunique()  # 🔑 新增
    if len(df2) > unique_stations:  # 🔑 新增条件判断
        print(f"\n🔄 按站点聚合（{len(df2)} 条记录 → {unique_stations} 个站点）...")  # 🔑 新增
        agg_funcs = {
            'lat': 'median',
            'lon': 'median',
            'elevation': lambda x: np.nanmedian(x.values) if np.any(~np.isnan(x.values)) else np.nan,
            'temperature': lambda x: np.nanmean(x.values) if np.any(~np.isnan(x.values)) else np.nan,
            'temperature_raw': 'count',
            'name': lambda x: x.astype(str).mode()[0] if len(x.astype(str).mode()) > 0 else x.astype(str).iloc[0]
        }

        grouped = df2.groupby('station_id', as_index=False).agg(agg_funcs)
        grouped = grouped.rename(columns={'temperature_raw': 'data_count'})
    else:  # 🔑 新增
        grouped = df2.copy()  # 🔑 新增
        grouped['data_count'] = 1  # 🔑 新增

    # 标准列顺序
    grouped = grouped[['station_id', 'lat', 'lon', 'elevation', 'temperature', 'data_count', 'name']]

    # 再次检查温度 NaN
    nan_temp_count = grouped['temperature'].isna().sum()
    if nan_temp_count > 0:
        print(f"\n⚠️  警告：聚合后仍有 {nan_temp_count} 个站点的温度为 NaN")
        print(f"   这些站点将在建模时被跳过")
        grouped = grouped.dropna(subset=['temperature'])

    # 添加备用列名（兼容其他模块）
    grouped['latitude'] = grouped['lat']
    grouped['longitude'] = grouped['lon']

    print(f"\n✅ 最终聚合结果：{len(grouped)} 个站点")  # 🔑 新增

    return grouped


def main():
    """命令行接口"""
    import argparse

    p = argparse.ArgumentParser(
        description="Prepare GSOD (Global Summary of the Day) station data for kriging_rf pipeline"
    )
    p.add_argument(
        '--input-dir', '-d',
        required=True,
        help='GSOD CSV 文件所在目录'
    )
    p.add_argument(
        '--pattern', '-p',
        default='*.csv',
        help='文件匹配模式（默认 *.csv）'
    )
    p.add_argument(
        '--temp-unit',
        choices=['F', 'C'],
        default='F',
        help='温度单位：F(华氏度，GSOD默认) 或 C(摄氏度)'
    )
    p.add_argument(
        '--out', '-o',
        default='gsod_merged_cache.parquet',
        help='输出 parquet 文件路径'
    )
    p.add_argument(
        '--force', '-f',
        action='store_true',
        help='覆盖已存在的输出文件'
    )

    args = p.parse_args()

    # 检查输入目录
    if not os.path.isdir(args.input_dir):
        print(f"Input directory not found: {args.input_dir}")
        return

    try:
        # 读取所有 GSOD CSV
        print(f"\nReading GSOD CSV files from: {args.input_dir}")
        df = read_gsod_directory(args.input_dir, pattern=args.pattern)

        print(f"\nPreparing data (temperature unit: {args.temp_unit})...")
        grouped = prepare_gsod(df, temp_unit=args.temp_unit, drop_nan_temp=True)

        print(f"\n✓ Aggregated to {len(grouped)} stations")
        print("\nExample rows:")
        print(grouped.head(10).to_string(index=False))

        # 检查输出文件是否存在
        if os.path.exists(args.out) and not args.force:
            print(f"\n✗ Output file {args.out} already exists. Use --force to overwrite.")
            return

        # 保存为 parquet
        try:
            grouped.to_parquet(args.out, index=False, compression='snappy')
            print(f"\n✓ Saved merged cache to: {args.out}")
            print(f"  File size: {os.path.getsize(args.out) / (1024**2):.2f} MB")
        except Exception as e:
            print(f"\n✗ Failed to save parquet: {e}")
            # 退回保存为 CSV
            out_csv = os.path.splitext(args.out)[0] + ".csv"
            grouped.to_csv(out_csv, index=False)
            print(f"✓ Saved fallback CSV to: {out_csv}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
