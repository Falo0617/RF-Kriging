"""
评估、保存与绘图
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_metrics(obs, pred):
    mae = np.nanmean(np.abs(obs - pred))
    rmse = np.sqrt(np.nanmean((obs - pred) ** 2))
    return mae, rmse


def evaluate_and_save(df_xy, oof_rf, oof_corr, ok_loo=None, out_prefix='rfok'):
    z = df_xy['temperature'].values
    idx_rf = ~np.isnan(oof_rf)
    idx_corr = ~np.isnan(oof_corr)

    ok_mae = ok_rmse = np.nan
    if ok_loo is not None:
        ok_valid = ~np.isnan(ok_loo)
        ok_mae, ok_rmse = compute_metrics(z[ok_valid], ok_loo[ok_valid])

    rf_mae, rf_rmse = (compute_metrics(z[idx_rf], oof_rf[idx_rf]) if np.any(idx_rf) else (np.nan, np.nan))
    corr_mae, corr_rmse = (compute_metrics(z[idx_corr], oof_corr[idx_corr]) if np.any(idx_corr) else (np.nan, np.nan))

    records = [
        {'method': 'OK (LOO)', 'MAE': ok_mae, 'RMSE': ok_rmse},
        {'method': 'RF (OOF)', 'MAE': rf_mae, 'RMSE': rf_rmse},
        {'method': 'RF-OK (OOF corrected)', 'MAE': corr_mae, 'RMSE': corr_rmse},
    ]
    summary = pd.DataFrame(records)

    def rel_pct(base, new):
        if np.isnan(base) or np.isnan(new) or base == 0:
            return np.nan
        return (1 - new / base) * 100.0

    if not np.isnan(ok_rmse):
        summary['RMSE_vs_OK_%'] = summary['RMSE'].apply(lambda r: rel_pct(ok_rmse, r))
        summary['MAE_vs_OK_%'] = summary['MAE'].apply(lambda m: rel_pct(ok_mae, m))
    else:
        summary['RMSE_vs_OK_%'] = np.nan
        summary['MAE_vs_OK_%'] = np.nan

    summary.to_csv(f"{out_prefix}_model_comparison_summary.csv", index=False)
    df_out = df_xy.copy()
    df_out['oof_rf'] = oof_rf
    df_out['oof_corr'] = oof_corr
    if ok_loo is not None:
        df_out['ok_loo'] = ok_loo
    df_out.to_csv(f"{out_prefix}_per_station_predictions.csv", index=False)
    return summary, df_out


def plot_results(df_xy, grid_xx, grid_yy, ok_grid, rf_grid, res_grid, corrected_grid, df_preds, out_prefix='rfok'):
    x = df_xy['x'].values; y = df_xy['y'].values; z = df_xy['temperature'].values
    oof_rf = df_preds['oof_rf'].values if 'oof_rf' in df_preds else np.full(len(df_xy), np.nan)
    oof_corr = df_preds['oof_corr'].values if 'oof_corr' in df_preds else np.full(len(df_xy), np.nan)
    ok_loo = df_preds['ok_loo'].values if 'ok_loo' in df_preds else np.full(len(df_xy), np.nan)

    idx_ok = ~np.isnan(ok_loo)
    idx_rf = ~np.isnan(oof_rf)
    idx_corr = ~np.isnan(oof_corr)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    sc1 = ax1.scatter(x / 1000, y / 1000, c=z, s=50, cmap='RdYlBu_r', edgecolors='k')
    ax1.set_title('(a) Observed')
    plt.colorbar(sc1, ax=ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    if ok_grid is not None:
        im2 = ax2.contourf(grid_xx / 1000, grid_yy / 1000, ok_grid, 30, cmap='RdYlBu_r')
        ax2.set_title('(b) OK grid')
        plt.colorbar(im2, ax=ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.contourf(grid_xx / 1000, grid_yy / 1000, corrected_grid, 30, cmap='RdYlBu_r')
    ax3.set_title('(c) Corrected grid (RF + krige resid)')
    plt.colorbar(im3, ax=ax3)

    ax4 = fig.add_subplot(gs[1, 0])
    if idx_ok.any():
        ax4.scatter(z[idx_ok], ok_loo[idx_ok], alpha=0.6)
        ax4.plot([z.min(), z.max()], [z.min(), z.max()], 'k--')
    ax4.set_title('(d) Obs vs Pred OK')

    ax5 = fig.add_subplot(gs[1, 1])
    if idx_rf.any():
        ax5.scatter(z[idx_rf], oof_rf[idx_rf], alpha=0.6)
        ax5.plot([z.min(), z.max()], [z.min(), z.max()], 'k--')
    ax5.set_title('(e) Obs vs Pred RF')

    ax6 = fig.add_subplot(gs[1, 2])
    if idx_corr.any():
        ax6.scatter(z[idx_corr], oof_corr[idx_corr], alpha=0.6)
        ax6.plot([z.min(), z.max()], [z.min(), z.max()], 'k--')
    ax6.set_title('(f) Obs vs Pred RF-OK')

    res_ok = z - ok_loo
    res_rf = z - oof_rf
    res_corr = z - oof_corr

    ax7 = fig.add_subplot(gs[2, 0])
    sc7 = ax7.scatter(x / 1000, y / 1000, c=res_ok, cmap='coolwarm', s=50)
    ax7.set_title('(g) Residual OK')
    plt.colorbar(sc7, ax=ax7)

    ax8 = fig.add_subplot(gs[2, 1])
    sc8 = ax8.scatter(x / 1000, y / 1000, c=res_rf, cmap='coolwarm', s=50)
    ax8.set_title('(h) Residual RF')
    plt.colorbar(sc8, ax=ax8)

    ax9 = fig.add_subplot(gs[2, 2])
    sc9 = ax9.scatter(x / 1000, y / 1000, c=res_corr, cmap='coolwarm', s=50)
    ax9.set_title('(i) Residual RF-OK')
    plt.colorbar(sc9, ax=ax9)

    plt.tight_layout()
    out_png = f"{out_prefix}_comparison.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.show()
    return out_png