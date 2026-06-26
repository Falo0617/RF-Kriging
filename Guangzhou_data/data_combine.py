# preprocess_guangdong.py
import re
import pandas as pd
from pathlib import Path

def parse_station_line(line):
    # 去首尾空白
    s = line.strip()
    if not s:
        return None
    # 假设格式：省 + station_id(5位) + 中文站名 + lat_int lon_int elev1 elev2
    # 我们用正则提取第一个连续的数字串作为 station_id，然后在它之后找到站名直到遇到空白+数字（lat）
    m = re.search(r'(\d{4,6})', s)  # 找到站号
    if not m:
        return None
    stn_start = m.start()
    stn_end = m.end()
    station_id = s[stn_start:stn_end]
    prefix = s[:stn_start]  # 省 + 可能没有空格
    rest = s[stn_end:].strip()
    # rest 应该从站名开始，后面以空白分隔出数字字段
    parts = rest.split()
    if len(parts) < 3:
        # 有些行可能没有空格分隔，经纬紧跟，需要用另一种方式
        # 尝试把非数字中文站名与后面的数字分开：
        m2 = re.search(r'(\d{3,5})\s+(\d{4,6})', rest)
        if not m2:
            return None
        # fallback: naive split
        return None

    # parts[0] = 站名（如果站名里没有空格）；后面是 lat lon elev1 [elev2]
    name = parts[0]
    lat_str = parts[1]
    lon_str = parts[2]
    elev1 = float(parts[3]) if len(parts) >= 4 else None
    elev2 = float(parts[4]) if len(parts) >= 5 else None

    # 将整数形式的经纬转换成带小数形式：假设需要除以 100
    try:
        lat = float(lat_str) / 100.0
        lon = float(lon_str) / 100.0
    except ValueError:
        lat = None
        lon = None

    return {
        'station_id': int(station_id),
        'prov_prefix': prefix,
        'name': name,
        'lat_raw': lat_str,
        'lon_raw': lon_str,
        'lat': lat,
        'lon': lon,
        'elev1': elev1,
        'elev2': elev2
    }

def build_stations_df(station_file):
    rows = []
    with open(station_file, 'r', encoding='utf-8') as f:
        for line in f:
            parsed = parse_station_line(line)
            if parsed:
                rows.append(parsed)
    df = pd.DataFrame(rows)
    # 去重/排序等
    df = df.drop_duplicates(subset=['station_id']).set_index('station_id', drop=False)
    return df

def read_obs_file(obs_file):
    # S202606... 文件看起来以空白分隔，并有列头
    df = pd.read_table(obs_file, delim_whitespace=True, na_values=['999017','999999'], dtype={'Station_Id_C':int})
    return df

def merge_and_save(obs_file, station_file, out_parquet):
    stations = build_stations_df(station_file)
    print("stations:", stations.shape)
    obs = read_obs_file(obs_file)
    print("obs rows:", obs.shape)
    merged = obs.merge(stations[['lat','lon','elev1','name']], left_on='Station_Id_C', right_index=True, how='left')
    # 检查没有坐标的站点
    missing_coords = merged[merged['lat'].isna() | merged['lon'].isna()]['Station_Id_C'].unique()
    if len(missing_coords)>0:
        print("Warning: missing coords for station ids:", missing_coords)
    # 你可以在这里做更多预处理（重命名 TEM->temperature, 填充缺测等）
    merged.to_parquet(out_parquet, index=False)
    print("Saved parquet:", out_parquet)

if __name__ == '__main__':
    obs_file = 'S202606260910462856600.txt'     # 你给的时序文件路径
    station_file = '广东59071连南2473 11228 175.5 174.3.txt'  # 你的站点元数据
    out_parquet = 'guangdong_merged_cache.parquet'
    merge_and_save(obs_file, station_file, out_parquet)
