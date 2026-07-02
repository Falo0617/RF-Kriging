"""
GSOD 数据处理工具函数
支持读取多个站点 CSV 文件，聚合站点数据，支持温度（TEMP）和露点温度（DEWP）两种目标变量。
支持异常值过滤（temp_min / temp_max）。
"""
import os
import glob
import numpy as np
import pandas as pd


def read_gsod_directory(directory, pattern='*.csv'):
    """
    读取目录下所有 GSOD CSV 文件，返回一个包含所有站点原始数据的 DataFrame。

    从每个 CSV 中提取以下列：
        STATION, DATE, LATITUDE, LONGITUDE, ELEVATION, TEMP, DEWP, NAME

    参数:
        directory: 包含 CSV 文件的目录路径
        pattern: 文件匹配模式 (默认 '*.csv')

    返回:
        pandas.DataFrame: 合并后的原始数据
    """
    all_dfs = []
    for filepath in glob.glob(os.path.join(directory, pattern)):
        try:
            df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
            # 检查必要列
            required = ['STATION', 'DATE', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'TEMP']
            if not all(col in df.columns for col in required):
                continue
            # 提取需要的列
            cols = ['STATION', 'DATE', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'TEMP']
            if 'DEWP' in df.columns:
                cols.append('DEWP')
            else:
                df['DEWP'] = np.nan
            if 'NAME' in df.columns:
                cols.append('NAME')
            else:
                df['NAME'] = ''

            df_sub = df[cols].copy()
            df_sub['source_file'] = os.path.basename(filepath)
            all_dfs.append(df_sub)

        except Exception as e:
            print(f"Warning: failed to read {filepath}: {e}")
            continue

    if not all_dfs:
        raise ValueError("No valid GSOD CSV files found in directory: {}".format(directory))
    return pd.concat(all_dfs, ignore_index=True)


def prepare_gsod(df, temp_unit='F', drop_nan_temp=True, use_latest_date=True, target_col='temperature',
                 temp_min=-50.0, temp_max=60.0):
    """
    聚合站点数据，每个站点一行。

    参数:
        df: pandas.DataFrame，来自 read_gsod_directory 的原始数据
        temp_unit: 输入温度单位，'F' 或 'C'，GSOD 标准为华氏度（'F'）
        drop_nan_temp: 是否删除 TEMP 缺失的行
        use_latest_date: True 则每个站点只保留最新日期的记录；False 则保留所有记录的平均值
        target_col: 'temperature' 或 'dew_point'，指定主目标列
        temp_min: 温度最小阈值（摄氏度），低于此值视为异常，默认 -50°C
        temp_max: 温度最大阈值（摄氏度），高于此值视为异常，默认 60°C

    返回:
        pandas.DataFrame: 包含列 station_id, latitude, longitude, elevation,
                          temperature, dew_point, name
    """
    df = df.copy()

    # 温度处理
    # GSOD 中 TEMP 存储为 0.1°F (即华氏度 × 10)
    if 'TEMP' in df.columns:
        # 除以 10 得到华氏度
        df['TEMP_F'] = df['TEMP'] / 10.0
        # 转换为摄氏度
        if temp_unit.upper() == 'F':
            df['TEMP_C'] = (df['TEMP_F'] - 32) * 5 / 9
        else:
            df['TEMP_C'] = df['TEMP_F']  # 如果已经是摄氏度，直接使用
    else:
        df['TEMP_C'] = np.nan

    # 露点温度处理（DEWP 同样为 0.1°F）
    if 'DEWP' in df.columns:
        df['DEWP_F'] = df['DEWP'] / 10.0
        if temp_unit.upper() == 'F':
            df['DEWP_C'] = (df['DEWP_F'] - 32) * 5 / 9
        else:
            df['DEWP_C'] = df['DEWP_F']
    else:
        df['DEWP_C'] = np.nan

    # 日期处理
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    if use_latest_date:
        # 按站点分组，取最大日期对应的行
        idx = df.groupby('STATION')['DATE'].idxmax()
        df = df.loc[idx].reset_index(drop=True)
    else:
        # 按站点聚合平均值
        numeric_cols = ['LATITUDE', 'LONGITUDE', 'ELEVATION', 'TEMP_C', 'DEWP_C']
        agg_dict = {col: 'mean' for col in numeric_cols if col in df.columns}
        if 'NAME' in df.columns:
            agg_dict['NAME'] = lambda x: x.iloc[0] if len(x) > 0 else ''
        df = df.groupby('STATION').agg(agg_dict).reset_index()

    # 删除温度缺失的行（可选）
    if drop_nan_temp:
        df = df.dropna(subset=['TEMP_C'])

    # 重命名列
    rename_map = {
        'STATION': 'station_id',
        'LATITUDE': 'latitude',
        'LONGITUDE': 'longitude',
        'ELEVATION': 'elevation',
        'TEMP_C': 'temperature',
        'DEWP_C': 'dew_point',
        'NAME': 'name'
    }
    df = df.rename(columns=rename_map)

    # 确保 dew_point 列存在（若没有则补 NaN）
    if 'dew_point' not in df.columns:
        df['dew_point'] = np.nan

    # 根据 target_col 决定主列
    # 如果 target_col 是 dew_point，则将 dew_point 的值赋给 temperature（使下游兼容）
    if target_col == 'dew_point':
        if 'dew_point' in df.columns and not df['dew_point'].isna().all():
            df['temperature'] = df['dew_point']
        else:
            # 若 dew_point 全为 NaN，则使用 temperature 列（如果存在）
            pass

    # 确保必要的列存在
    required_cols = ['station_id', 'latitude', 'longitude', 'elevation', 'temperature']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # 处理 name 列
    if 'name' not in df.columns:
        df['name'] = ''

    # ========== 新增：异常值过滤 ==========
    before = len(df)

    # 过滤 temperature 列
    if 'temperature' in df.columns:
        df = df[(df['temperature'] >= temp_min) & (df['temperature'] <= temp_max)]

    # 过滤 dew_point 列（如果存在且非空）
    if 'dew_point' in df.columns and not df['dew_point'].isna().all():
        df = df[(df['dew_point'] >= temp_min) & (df['dew_point'] <= temp_max)]

    after = len(df)
    if after < before:
        print(f"  过滤掉 {before - after} 个异常站点（温度范围 {temp_min}~{temp_max}°C）")
    # =====================================

    # 选择最终输出的列（保持顺序）
    out_cols = ['station_id', 'latitude', 'longitude', 'elevation', 'temperature', 'dew_point', 'name']
    return df[out_cols].copy()