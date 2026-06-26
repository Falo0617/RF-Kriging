#!/usr/bin/env python3
"""
Top-level runner: 只读取广东合并缓存 parquet（guangdong_merged_cache.parquet）并运行 RF + Kriging pipeline。
"""
import argparse
import os
import sys
import pandas as pd

from kriging_rf import data_io, features, modeling, kriging_utils, evaluate

# 如果 compare_ok_vs_rfok 在包中存在，则导入
try:
    from kriging_rf import compare_ok_vs_rfok
except Exception:
    compare_ok_vs_rfok = None

def parse_args():
    p = argparse.ArgumentParser(description="Run RF + Kriging residual pipeline using Guangdong parquet cache")
    p.add_argument('--cache', type=str, default='guangdong_merged_cache.parquet',
                   help='预处理后的 parquet 文件（完整路径或相对于当前工作目录/脚本的路径）')
    p.add_argument('--lat', type=float, default=23.00)
    p.add_argument('--lon', type=float, default=113.00)
    p.add_argument('--radius', type=float, default=1000.0)
    p.add_argument('--max-stations', type=int, default=100)
    p.add_argument('--grid-res', type=int, default=50)
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--out-prefix', type=str, default='rfok')
    p.add_argument('--year', type=int, default=None, help="如果提供则在聚合前按 Year 过滤")
    p.add_argument('--no-compare', dest='do_compare', action='store_false', help='禁用 OK vs RF-Kriging 比较')
    p.set_defaults(do_compare=True)
    return p.parse_args()


def main():
    args = parse_args()
    print("Parameters:")
    for k,v in vars(args).items():
        print(f"  {k}: {v}")

    cache_path = args.cache if os.path.isabs(args.cache) else os.path.join(os.getcwd(), args.cache)
    if not os.path.exists(cache_path):
        print(f"Parquet cache not found at: {cache_path}")
        print("Exiting. 请生成 guangdong_merged_cache.parquet 或指定正确路径。")
        return

    # 使用 data_io 提供的 helper 读取并标准化列；aggregate=True 返回每站一行
    df = data_io.load_from_parquet(cache_path, aggregate=True, year=args.year, verbose=True)
    if df is None or df.empty:
        print("Loaded dataframe is empty after load_from_parquet. Exiting.")
        return

    # df now should have at least 'lat' and 'lon' (and 'elevation','temperature')
    df_xy, lon0, lat0 = features.stations_to_xy(df, lon_center=args.lon, lat_center=args.lat)

    # OK LOO at stations
    print("Computing OK LOO at stations (can be slow)...")
    # use existing kriging_utils.perform_ok_loo signature (no changes assumed)
    ok_loo = kriging_utils.perform_ok_loo(df_xy['x'].values, df_xy['y'].values, df_xy['temperature'].values)

    # OOF RF and corrected
    oof_rf, oof_corr, rf_full, scaler_full, monitor = modeling.generate_oof_corrected(df_xy, n_splits=args.n_splits)

    if monitor is not None:
        print("Monitor outputs written to:", getattr(monitor, 'out_dir', '(unknown)'))

    # final grid
    grid_xx, grid_yy, rf_grid, res_grid, corrected_grid = modeling.final_grid_prediction(df_xy, rf_full, scaler_full, grid_res=args.grid_res)

    # evaluate & save
    summary, df_preds = evaluate.evaluate_and_save(df_xy, oof_rf, oof_corr, ok_loo, out_prefix=args.out_prefix)
    print(summary.to_string(index=False, float_format='{:0.4f}'.format))

    # plot
    png = evaluate.plot_results(df_xy, grid_xx, grid_yy, ok_grid=None, rf_grid=rf_grid, res_grid=res_grid, corrected_grid=corrected_grid, df_preds=df_preds, out_prefix=args.out_prefix)
    print("Saved figure:", png)

    # optional compare
    if args.do_compare and compare_ok_vs_rfok is not None:
        print("Running OK vs RF-Kriging comparison (compare_ok_vs_rfok)...")
        # note: compare_ok_vs_rfok.compare_ok_vs_rfok signature may differ; use the function you have
        try:
            compare_ok_vs_rfok.compare_ok_vs_rfok(df_xy, ok_loo, df_preds['oof_corr'].values if 'oof_corr' in df_preds else df_preds.get('oof_rf').values, out_prefix=args.out_prefix)
        except Exception as e:
            print("compare_ok_vs_rfok call failed; please adapt call to module interface. Error:", e)

if __name__ == '__main__':
    main()