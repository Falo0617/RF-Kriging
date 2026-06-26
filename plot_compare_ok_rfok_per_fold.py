#!/usr/bin/env python3
# plot_compare_ok_rfok_per_fold.py
# 读取 per_fold_metrics.csv，绘制每折 RMSE 折线 + 平滑曲线 + 改进百分比

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 配置
csv_path = Path("monitor/per_fold_metrics.csv")   # 如不在当前目录，改成绝对/相对路径
out_png = Path("rfok_vs_ok_per_fold.png")
smooth_window = 5   # 平滑窗口 (fold 数)。设为 0 或 1 表示不平滑

if not csv_path.exists():
    raise FileNotFoundError(f"{csv_path} not found. 请把 per_fold_metrics.csv 放到运行目录或调整 csv_path。")

df = pd.read_csv(csv_path)

# 确保按 fold 排序
if 'fold' in df.columns:
    df = df.sort_values('fold').reset_index(drop=True)
else:
    df['fold'] = np.arange(1, len(df) + 1)

folds = df['fold'].values
rmse_rf = df['rmse_rf'].values
rmse_corr = df['rmse_corr'].values

# 计算相对改进 (%)： (1 - corr / rf) * 100
with np.errstate(divide='ignore', invalid='ignore'):
    rel_pct = (1.0 - (rmse_corr / rmse_rf)) * 100.0

# 平滑（简单移动平均）
def smooth(arr, w):
    if w is None or w <= 1:
        return arr
    return np.convolve(arr, np.ones(w)/w, mode='same')

rmse_rf_s = smooth(rmse_rf, smooth_window)
rmse_corr_s = smooth(rmse_corr, smooth_window)
rel_pct_s = smooth(np.nan_to_num(rel_pct, nan=0.0), smooth_window)

# 绘图：两行（上：RMSE 折线；下：相对改进）
plt.figure(figsize=(12,8))
gs = plt.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.25)

ax0 = plt.subplot(gs[0])
ax0.plot(folds, rmse_rf, marker='o', linestyle='-', label='OK (LOO) RMSE', alpha=0.45)
ax0.plot(folds, rmse_corr, marker='o', linestyle='-', label='RF-OK (OOF) RMSE', alpha=0.45, color='tab:orange')
if smooth_window and smooth_window > 1:
    ax0.plot(folds, rmse_rf_s, linestyle='--', color='tab:blue', label=f'OK Smoothed (w={smooth_window})')
    ax0.plot(folds, rmse_corr_s, linestyle='--', color='tab:orange', label=f'RF-OK Smoothed (w={smooth_window})')

ax0.set_xlabel('Fold')
ax0.set_ylabel('RMSE')
ax0.set_title('Per-fold RMSE: OK (LOO) vs RF-Kriging (OOF)')
ax0.grid(alpha=0.3)
ax0.legend(loc='upper right')

ax1 = plt.subplot(gs[1], sharex=ax0)
ax1.bar(folds, rel_pct, color=['tab:green' if v>0 else 'tab:red' for v in rel_pct], alpha=0.6)
if smooth_window and smooth_window > 1:
    ax1.plot(folds, rel_pct_s, color='k', linestyle='-', label='Smoothed rel. improvement (%)')
ax1.axhline(0, color='k', linestyle='--', linewidth=0.8)
ax1.set_xlabel('Fold')
ax1.set_ylabel('Relative improvement (%)')
ax1.set_title('Per-fold relative improvement: (1 - RMSE_RF-OK / RMSE_OK) * 100')
ax1.grid(alpha=0.25)
if smooth_window and smooth_window > 1:
    ax1.legend()

plt.tight_layout()
plt.savefig(out_png, dpi=200)
print("Saved:", out_png.resolve())
plt.show()