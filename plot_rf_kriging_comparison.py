"""
绘制 RF-Kriging vs 普通 Kriging 的交叉验证误差对比图
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============ 配置 ============
# 如果你有 monitor 文件夹下的分折误差数据，可以在这里指定路径
# 否则使用下面的模拟数据演示（实际使用时替换为你的真实数据）
USE_REAL_DATA = False  # 改为 True 并指定路径后使用真实数据

MONITOR_DIR = Path("monitor")  # 你的 monitor 文件夹路径
OK_OOF_FILE = MONITOR_DIR / "ok_oof_errors.npy"   # 如果有保存的话
RF_OOF_FILE = MONITOR_DIR / "rf_oof_errors.npy"   # 如果有保存的话
# =============================


def generate_comparison_plot(ok_errors, rf_errors, save_path='rf_kriging_comparison.png'):
    """
    生成 RF-Kriging vs OK 的对比图
    参数:
        ok_errors: 列表，每个元素是一折的 OK 绝对误差数组
        rf_errors: 列表，每个元素是一折的 RF-Kriging 绝对误差数组
        save_path: 保存路径
    """
    # 准备箱线图数据
    n_folds = len(ok_errors)
    data = []
    labels = []
    colors = []

    for fold_idx in range(n_folds):
        # OK 误差
        for err in ok_errors[fold_idx]:
            data.append(err)
            labels.append(f'OK-F{fold_idx+1}')
            colors.append('skyblue')
        # RF-Kriging 误差
        for err in rf_errors[fold_idx]:
            data.append(err)
            labels.append(f'RF-F{fold_idx+1}')
            colors.append('lightcoral')

    df_plot = pd.DataFrame({
        '绝对误差 (°C)': data,
        '分组': labels,
        '方法': ['OK' if 'OK' in l else 'RF-Kriging' for l in labels]
    })

    # ========== 绘图 ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：分折箱线图
    ax1 = axes[0]
    # 按分组绘制箱线图
    order = []
    for f in range(n_folds):
        order.append(f'OK-F{f+1}')
    for f in range(n_folds):
        order.append(f'RF-F{f+1}')

    sns.boxplot(
        data=df_plot,
        x='分组',
        y='绝对误差 (°C)',
        hue='方法',
        palette={'OK': 'skyblue', 'RF-Kriging': 'lightcoral'},
        ax=ax1,
        order=order,
        dodge=False,
        fliersize=2,
        linewidth=1
    )

    ax1.set_title('各折交叉验证误差分布对比', fontsize=14, fontweight='bold')
    ax1.set_xlabel('交叉验证折')
    ax1.set_ylabel('绝对误差 (°C)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 在图上添加每折的均值标注
    for f in range(n_folds):
        ok_vals = ok_errors[f]
        rf_vals = rf_errors[f]
        ok_mean = np.mean(ok_vals)
        rf_mean = np.mean(rf_vals)
        # 在 OK 箱子上加横线
        ax1.text(f, ok_mean + 0.05, f'{ok_mean:.2f}', ha='center', va='bottom', fontsize=8, color='blue')
        # 在 RF 箱子上加横线
        ax1.text(f + n_folds, rf_mean + 0.05, f'{rf_mean:.2f}', ha='center', va='bottom', fontsize=8, color='red')

    # 右图：汇总箱线图
    ax2 = axes[1]
    all_ok = np.concatenate(ok_errors)
    all_rf = np.concatenate(rf_errors)
    data_agg = [all_ok, all_rf]
    bp = ax2.boxplot(data_agg, labels=['普通 Kriging', 'RF-Kriging'],
                     patch_artist=True, widths=0.5)

    # 设置颜色
    bp['boxes'][0].set_facecolor('skyblue')
    bp['boxes'][1].set_facecolor('lightcoral')

    ax2.set_title('所有折汇总误差对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('绝对误差 (°C)')
    ax2.grid(True, alpha=0.3)

    # 添加统计信息
    ok_rmse = np.sqrt(np.mean(np.concatenate(ok_errors)**2))
    rf_rmse = np.sqrt(np.mean(np.concatenate(rf_errors)**2))
    ok_mae = np.mean(np.concatenate(ok_errors))
    rf_mae = np.mean(np.concatenate(rf_errors))

    info = f'OK RMSE: {ok_rmse:.3f}°C\nRF-OK RMSE: {rf_rmse:.3f}°C\n'
    info += f'RMSE 降低: {(1 - rf_rmse/ok_rmse)*100:.1f}%'
    ax2.text(0.98, 0.95, info, transform=ax2.transAxes,
             ha='right', va='top', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ 对比图已保存为: {save_path}")


def load_monitor_data(monitor_dir):
    """
    从 monitor 目录加载各折误差数据
    如果找不到文件，则从最终的 df_preds 重建（需要用户提供）
    """
    ok_fold_errors = []
    rf_fold_errors = []

    # 方法1：从 monitor 目录读取（如果有保存）
    # 假设 monitor 目录下保存了每折的误差文件
    for f in range(1, 6):
        ok_file = monitor_dir / f'ok_fold_{f}_errors.npy'
        rf_file = monitor_dir / f'rf_fold_{f}_errors.npy'

        if ok_file.exists() and rf_file.exists():
            ok_fold_errors.append(np.load(ok_file))
            rf_fold_errors.append(np.load(rf_file))

    if len(ok_fold_errors) > 0:
        return ok_fold_errors, rf_fold_errors

    # 方法2：如果没有保存，使用模拟数据演示
    print("未找到 monitor 目录下的分折误差文件，使用模拟数据演示...")
    return generate_demo_data()


def generate_demo_data():
    """生成演示数据（模拟 RF-Kriging 优于 OK 的场景）"""
    np.random.seed(42)
    n_folds = 5
    n_samples_per_fold = 10

    ok_errors = []
    rf_errors = []

    for f in range(n_folds):
        # OK：误差较大，波动也大
        ok_err = np.random.gamma(2, 0.8, n_samples_per_fold) + 0.1
        # RF-Kriging：误差更小，更集中
        rf_err = np.random.gamma(1.5, 0.5, n_samples_per_fold) + 0.05

        ok_errors.append(ok_err)
        rf_errors.append(rf_err)

    return ok_errors, rf_errors


def load_from_run_output(df_preds):
    """
    从 run_kriging.py 输出的 df_preds 中提取误差
    在你的主程序中调用此函数
    """
    # 从 df_preds 中按折分组
    ok_errors = []
    rf_errors = []

    # 假设 df_preds 有 'fold', 'ok_error', 'rf_error' 列
    if 'fold' in df_preds.columns and 'ok_error' in df_preds.columns:
        for fold in df_preds['fold'].unique():
            fold_data = df_preds[df_preds['fold'] == fold]
            ok_errors.append(fold_data['ok_error'].values)
            rf_errors.append(fold_data['rf_error'].values)
        return ok_errors, rf_errors
    else:
        # 如果没有分折信息，将所有误差合并为一折
        ok_errors = [df_preds['ok_error'].values]
        rf_errors = [df_preds['rf_error'].values]
        return ok_errors, rf_errors


# ============ 主程序 ============
if __name__ == "__main__":
    if USE_REAL_DATA:
        # 尝试从 monitor 目录加载真实数据
        ok_errors, rf_errors = load_monitor_data(MONITOR_DIR)
    else:
        # 使用演示数据
        ok_errors, rf_errors = generate_demo_data()

    # 生成对比图
    generate_comparison_plot(ok_errors, rf_errors, 'rf_kriging_comparison.png')

    # 打印汇总统计
    all_ok = np.concatenate(ok_errors)
    all_rf = np.concatenate(rf_errors)
    print("\n" + "="*60)
    print("汇总统计")
    print("="*60)
    print(f"普通 Kriging  MAE: {np.mean(all_ok):.4f}°C, RMSE: {np.sqrt(np.mean(all_ok**2)):.4f}°C")
    print(f"RF-Kriging    MAE: {np.mean(all_rf):.4f}°C, RMSE: {np.sqrt(np.mean(all_rf**2)):.4f}°C")
    print(f"RMSE 降低: {(1 - np.sqrt(np.mean(all_rf**2))/np.sqrt(np.mean(all_ok**2)))*100:.2f}%")