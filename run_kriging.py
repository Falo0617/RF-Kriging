#!/usr/bin/env python3
"""
Top-level runner: 读取 parquet 缓存或原始 GSOD CSV，运行 RF + Kriging pipeline。
支持温度（temperature）和露点温度（dew_point）两种目标变量。
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

from kriging_rf import data_io, features, modeling, kriging_utils, evaluate

# 如果 compare_ok_vs_rfok 在包中存在，则导入
try:
    from kriging_rf import compare_ok_vs_rfok
except Exception:
    compare_ok_vs_rfok = None


def parse_args():
    p = argparse.ArgumentParser(description="Run RF + Kriging residual pipeline using Guangdong parquet cache")
    p.add_argument('--cache', type=str, default='gsod_merged_cache.parquet',
                   help='预处理后的 parquet 文件（完整路径或相对于当前工作目录/脚本的路径）')
    p.add_argument('--cache-only', action='store_true', default=True,
                   help='仅从 parquet 缓存读取并退出（若找不到文件则退出）')
    p.add_argument('--lat', type=float, default=23.13)
    p.add_argument('--lon', type=float, default=113.26)
    p.add_argument('--radius', type=float, default=1000.0)
    p.add_argument('--max-stations', type=int, default=100)
    p.add_argument('--grid-res', type=int, default=10)
    p.add_argument('--n-splits', type=int, default=50)
    p.add_argument('--out-prefix', type=str, default='rfok')
    # neighbors 参数传给 kriging_utils.perform_ok_loo
    p.add_argument('--ok-neighbors', type=int, default=10, help='OK-LOO 时使用的邻域大小 k（最近 k 个邻居）')
    p.add_argument('--ok-cache', type=str, default='ok_loo.npz', help='OK-LOO 结果缓存路径')
    p.add_argument('--no-compare', dest='do_compare', action='store_false', help='禁用 OK vs RF-Kriging 比较')
    # 新增：目标变量选择
    p.add_argument('--target', type=str, default='dew_point', choices=['temperature', 'dew_point', 'wind_speed'],
                   help='目标变量: temperature (温度) 或 dew_point (露点温度)')
    p.set_defaults(do_compare=True)
    return p.parse_args()


def load_data(args):
    """
    根据参数加载数据：
      1. 如果指定了 target='dew_point'，尝试从 parquet 读取 DEWP 列
      2. 如果 parquet 中没有 DEWP 列，回退到从原始 GSOD CSV 扫描（使用 data_io.load_and_select）
      3. 如果 target='temperature'，直接使用 parquet 中的 temperature 列
    """
    cache_path = args.cache if os.path.isabs(args.cache) else os.path.join(os.getcwd(), args.cache)

    # 检查 parquet 是否存在
    parquet_exists = os.path.exists(cache_path)

    # 如果指定了 target='dew_point'，优先从 parquet 读取 DEWP 列
    if args.target == 'dew_point' and parquet_exists:
        try:
            df_test = pd.read_parquet(cache_path)
            if 'DEWP' in df_test.columns or 'dew_point' in df_test.columns:
                print(f"\n[INFO] 从 parquet 读取 DEWP 列: {cache_path}")
                df = data_io.load_from_parquet(cache_path, aggregate=True, year=None,
                                               target_col='dew_point', verbose=True)
                if df is not None and not df.empty and 'temperature' in df.columns:
                    print(f"[INFO] 成功从 parquet 加载露点温度数据，{len(df)} 个站点")
                    return df
            else:
                print(f"[INFO] parquet 中未找到 DEWP 列，回退到 CSV 扫描...")
        except Exception as e:
            print(f"[INFO] 从 parquet 读取 DEWP 失败 ({e})，回退到 CSV 扫描...")

    # 如果 parquet 不存在 或 目标列是 temperature 或 上述 DEWP 读取失败
    if parquet_exists and args.target == 'temperature':
        try:
            df = data_io.load_from_parquet(cache_path, aggregate=True, year=None,
                                           target_col='temperature', verbose=True)
            if df is not None and not df.empty and 'temperature' in df.columns:
                print(f"[INFO] 成功从 parquet 加载温度数据，{len(df)} 个站点")
                return df
        except Exception as e:
            print(f"[INFO] 从 parquet 读取温度失败 ({e})，回退到 CSV 扫描...")

    # 回退：使用 data_io.load_and_select 从原始 GSOD CSV 扫描
    print(f"\n[INFO] 从原始 GSOD CSV 扫描数据 (target={args.target})...")
    print(f"[INFO] 数据文件夹: {args.cache if os.path.isdir(args.cache) else os.path.dirname(cache_path)}")

    # 确定数据文件夹路径
    if os.path.isdir(args.cache):
        folder_path = args.cache
    else:
        # 如果 cache 是 parquet 文件路径，取所在目录
        folder_path = os.path.dirname(cache_path)
        # 如果目录不存在，尝试当前目录下的 2024_gsod_data
        if not os.path.exists(folder_path) or not folder_path:
            folder_path = os.path.join(os.getcwd(), "2024_gsod_data")

    if not os.path.exists(folder_path):
        print(f"[ERROR] 数据文件夹不存在: {folder_path}")
        print("请确保 GSOD CSV 文件位于该目录，或使用 --cache 指定正确的 parquet 路径。")
        return None

    df = data_io.load_and_select(
        folder_path=folder_path,
        target_lat=args.lat,
        target_lon=args.lon,
        radius_km=args.radius,
        max_stations=args.max_stations,
        year=2024,
        target_col=args.target,
        cache_path=cache_path if parquet_exists else None,
        force_preprocess=False,
        verbose=True
    )

    return df


def main():
    args = parse_args()
    print("Parameters:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # ========== 加载数据 ==========
    df = load_data(args)

    if df is None or df.empty:
        print("[ERROR] 数据加载失败，程序退出。")
        return

    # ========== 调试：检查数据加载 ==========
    print(f"\n[DEBUG] 加载到 {len(df)} 条记录")
    if len(df) > 0:
        print(f"[DEBUG] 列名: {df.columns.tolist()}")
        # 检查目标列是否存在
        if 'temperature' in df.columns:
            print(f"[DEBUG] 目标值范围: {df['temperature'].min():.2f} ~ {df['temperature'].max():.2f}")
        if 'latitude' in df.columns and 'longitude' in df.columns:
            print(f"[DEBUG] 经度范围: {df['longitude'].min():.4f} ~ {df['longitude'].max():.4f}")
            print(f"[DEBUG] 纬度范围: {df['latitude'].min():.4f} ~ {df['latitude'].max():.4f}")
        if 'elevation' in df.columns:
            print(f"[DEBUG] 海拔范围: {df['elevation'].min():.1f} ~ {df['elevation'].max():.1f}m")
    else:
        print("[DEBUG] ❌ 数据为空！")
        return
    # ======================================

    # ========== 按半径筛选站点（如果 data_io 未做筛选） ==========
    # data_io.load_and_select 已经做了筛选，这里作为补充保险
    if 'latitude' in df.columns and 'longitude' in df.columns:
        print(f"\n[DEBUG] 按半径 {args.radius}km 筛选站点...")
        lat_rad = np.radians(args.lat)
        lon_rad = np.radians(args.lon)
        lat_diff = np.radians(df['latitude'] - args.lat)
        lon_diff = np.radians(df['longitude'] - args.lon)
        a = np.sin(lat_diff / 2) ** 2 + np.cos(lat_rad) * np.cos(np.radians(df['latitude'])) * np.sin(lon_diff / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        df['_dist'] = 6371 * c
        df = df[df['_dist'] <= args.radius].drop(columns=['_dist']).head(args.max_stations)
        print(f"[DEBUG] 筛选后剩余 {len(df)} 个站点")
    # ===========================================================

    if df is None or df.empty:
        print("Loaded dataframe is empty. Exiting.")
        return

    # 下面将标准化字段并生成 x,y,elevation,temperature 列
    # 注意：features.stations_to_xy 已支持 target_col 参数
    df_xy, lon0, lat0 = features.stations_to_xy(df, lon_center=args.lon, lat_center=args.lat, target_col=args.target)

    # ========== 调试：检查坐标转换 ==========
    print(f"\n[DEBUG] stations_to_xy 转换后:")
    print(f"[DEBUG] x 范围: {df_xy['x'].min():.2f}m ~ {df_xy['x'].max():.2f}m")
    print(f"[DEBUG] y 范围: {df_xy['y'].min():.2f}m ~ {df_xy['y'].max():.2f}m")
    print(f"[DEBUG] 目标值范围: {df_xy['temperature'].min():.2f} ~ {df_xy['temperature'].max():.2f}")
    print(f"[DEBUG] 有效站点数: {len(df_xy)}")
    # ======================================

    # OK LOO：使用 kriging_utils 的新接口并支持缓存
    print("Computing or loading OK LOO at stations (neighbors=%d)..." % args.ok_neighbors)
    ok_loo = kriging_utils.perform_ok_loo(
        df_xy['x'].values, df_xy['y'].values, df_xy['temperature'].values,
        neighbors=args.ok_neighbors, cache_path=args.ok_cache, use_cache=True, overwrite=False
    )

    # ========== 调试：检查 OK LOO 结果 ==========
    if ok_loo is not None:
        valid_mask = ~np.isnan(ok_loo)
        print(f"[DEBUG] OK LOO 返回长度: {len(ok_loo)}, 有效值数: {valid_mask.sum()}")
        if valid_mask.sum() > 0:
            print(f"[DEBUG] OK LOO 有效值范围: {ok_loo[valid_mask].min():.4f} ~ {ok_loo[valid_mask].max():.4f}")
        else:
            print("[DEBUG] ⚠️ OK LOO 全部为 NaN！")
    else:
        print("[DEBUG] ⚠️ OK LOO 返回 None")
    # =========================================

    # OOF RF and corrected
    oof_rf, oof_corr, rf_full, scaler_full, monitor = modeling.generate_oof_corrected(df_xy, n_splits=args.n_splits)

    if monitor is not None:
        print("Monitor outputs written to:", getattr(monitor, 'out_dir', '(unknown)'))

    # final grid
    grid_xx, grid_yy, rf_grid, res_grid, corrected_grid = modeling.final_grid_prediction(df_xy, rf_full, scaler_full,
                                                                                         grid_res=args.grid_res)

    # evaluate & save
    summary, df_preds = evaluate.evaluate_and_save(df_xy, oof_rf, oof_corr, ok_loo, out_prefix=args.out_prefix)
    print(summary.to_string(index=False, float_format='{:0.4f}'.format))

    # plot (原有绘图函数)
    png = evaluate.plot_results(df_xy, grid_xx, grid_yy, ok_grid=None, rf_grid=rf_grid, res_grid=res_grid,
                                corrected_grid=corrected_grid, df_preds=df_preds, out_prefix=args.out_prefix)
    print("Saved figure:", png)

    # 如启用比较，则调用 compare_ok_vs_rfok（如果可用）
    if args.do_compare:
        if compare_ok_vs_rfok is None:
            print("compare_ok_vs_rfok module not found; skipping comparison.")
        else:
            print("Running OK vs RF-Kriging comparison...")
            # compare_and_plot wrapper 支持 (ok_loo, df_preds) 这种调用
            try:
                compare_ok_vs_rfok.compare_and_plot(ok_loo, df_preds, out_prefix=args.out_prefix)
            except Exception as e:
                print("compare_ok_vs_rfok failed:", e)


if __name__ == '__main__':
    main()