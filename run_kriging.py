#!/usr/bin/env python3
"""
Top-level runner: 只读取广东合并缓存 parquet（guangdong_merged_cache.parquet）并运行 RF + Kriging pipeline。
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
    p.add_argument('--lat', type=float, default=23.00)
    p.add_argument('--lon', type=float, default=113.00)
    p.add_argument('--radius', type=float, default=1000.0)
    p.add_argument('--max-stations', type=int, default=50)
    p.add_argument('--grid-res', type=int, default=10)
    p.add_argument('--n-splits', type=int, default=10)
    p.add_argument('--out-prefix', type=str, default='rfok')
    # neighbors 参数传给 kriging_utils.perform_ok_loo
    p.add_argument('--ok-neighbors', type=int, default=10, help='OK-LOO 时使用的邻域大小 k（最近 k 个邻居）')
    p.add_argument('--ok-cache', type=str, default='ok_loo.npz', help='OK-LOO 结果缓存路径')
    p.add_argument('--no-compare', dest='do_compare', action='store_false', help='禁用 OK vs RF-Kriging 比较')
    p.set_defaults(do_compare=True)
    return p.parse_args()


def main():
    args = parse_args()
    print("Parameters:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    cache_path = args.cache if os.path.isabs(args.cache) else os.path.join(os.getcwd(), args.cache)
    if not os.path.exists(cache_path):
        print(f"Parquet cache not found at: {cache_path}")
        print("Exiting. 请生成 guangdong_merged_cache.parquet 或指定正确路径。")
        return

    # 读取 parquet（假设格式与原 df 相同：包含站点经纬度/temperature 等所需字段）
    df = pd.read_parquet(cache_path)

    # ========== 调试：检查数据加载 ==========
    print(f"\n[DEBUG] 从 parquet 加载到 {len(df)} 条记录")
    if len(df) > 0:
        print(f"[DEBUG] 列名: {df.columns.tolist()}")
        print(f"[DEBUG] 温度范围: {df['temperature'].min():.2f} ~ {df['temperature'].max():.2f}°C")
        print(f"[DEBUG] 经度范围: {df['longitude'].min():.4f} ~ {df['longitude'].max():.4f}")
        print(f"[DEBUG] 纬度范围: {df['latitude'].min():.4f} ~ {df['latitude'].max():.4f}")
        print(f"[DEBUG] 海拔范围: {df['elevation'].min():.1f} ~ {df['elevation'].max():.1f}m")
    else:
        print("[DEBUG] ❌ 缓存文件为空！")
        return
    # ======================================

    # ========== 新增：按半径筛选站点 ==========
    print(f"\n[DEBUG] 按半径 {args.radius}km 筛选站点...")
    lat_rad = np.radians(args.lat)
    lon_rad = np.radians(args.lon)
    lat_diff = np.radians(df['latitude'] - args.lat)
    lon_diff = np.radians(df['longitude'] - args.lon)
    # 球面余弦公式近似计算距离（km）
    a = np.sin(lat_diff / 2) ** 2 + np.cos(lat_rad) * np.cos(np.radians(df['latitude'])) * np.sin(lon_diff / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    df['_dist'] = 6371 * c  # 地球半径 6371km
    df = df[df['_dist'] <= args.radius].drop(columns=['_dist']).head(args.max_stations)
    print(f"[DEBUG] 筛选后剩余 {len(df)} 个站点")
    # =========================================

    if df is None or df.empty:
        print("Loaded dataframe is empty. Exiting.")
        return

    # 下面将标准化字段并生成 x,y,elevation,temperature 列
    df_xy, lon0, lat0 = features.stations_to_xy(df, lon_center=args.lon, lat_center=args.lat)

    # ========== 调试：检查坐标转换 ==========
    print(f"\n[DEBUG] stations_to_xy 转换后:")
    print(f"[DEBUG] x 范围: {df_xy['x'].min():.2f}m ~ {df_xy['x'].max():.2f}m")
    print(f"[DEBUG] y 范围: {df_xy['y'].min():.2f}m ~ {df_xy['y'].max():.2f}m")
    print(f"[DEBUG] 温度范围: {df_xy['temperature'].min():.2f} ~ {df_xy['temperature'].max():.2f}°C")
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