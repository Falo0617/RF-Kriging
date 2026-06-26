import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

def compare_ok_vs_rfok(df_xy, ok_preds, rfok_preds, out_prefix='compare_ok_rfok', save_plots=True):
    """
    比较传统 OK (LOO) 与 RF-Kriging (OOF RF-OK 或 OOF RF-OK+Residual)
    - df_xy: DataFrame with at least ['station_id','x','y','elevation','temperature']
    - ok_preds: array-like, same length as df_xy, LOO OK predictions
    - rfok_preds: array-like, same length as df_xy, OOF RF-OK (or corrected)
    - out_prefix:文件前缀
    - returns (summary_df, df_out)
    """
    obs = df_xy['temperature'].values
    ok = np.asarray(ok_preds, dtype=float)
    rfok = np.asarray(rfok_preds, dtype=float)

    # mask: 保证两者都存在（否则无法配对）
    mask = (~np.isnan(ok)) & (~np.isnan(rfok)) & (~np.isnan(obs))
    n_pairs = mask.sum()
    if n_pairs == 0:
        raise ValueError("No paired predictions available for comparison (all NaN).")

    obs_p = obs[mask]
    ok_p = ok[mask]
    rfok_p = rfok[mask]

    # metrics
    def mae(a,b): return np.nanmean(np.abs(a-b))
    def rmse(a,b): return np.sqrt(np.nanmean((a-b)**2))

    ok_mae, ok_rmse = mae(obs_p, ok_p), rmse(obs_p, ok_p)
    rfok_mae, rfok_rmse = mae(obs_p, rfok_p), rmse(obs_p, rfok_p)

    def rel_improve(base, new):
        if np.isnan(base) or base == 0 or np.isnan(new): return np.nan
        return (1 - new / base) * 100.0

    summary = pd.DataFrame([
        {'method': 'OK (LOO)', 'MAE': ok_mae, 'RMSE': ok_rmse},
        {'method': 'RF-OK (OOF)', 'MAE': rfok_mae, 'RMSE': rfok_rmse}
    ])
    summary['MAE_vs_OK_%'] = summary['MAE'].apply(lambda m: rel_improve(ok_mae, m))
    summary['RMSE_vs_OK_%'] = summary['RMSE'].apply(lambda r: rel_improve(ok_rmse, r))

    # paired test on absolute errors (OK vs RF-OK)
    ae_ok = np.abs(obs_p - ok_p)
    ae_rfok = np.abs(obs_p - rfok_p)
    try:
        stat, pval = wilcoxon(ae_ok, ae_rfok)
    except Exception:
        stat, pval = np.nan, np.nan

    # bootstrap RMSE diff (rfok - ok)
    def bootstrap_rmse_diff(a_obs, a_predA, a_predB, n_boot=2000, seed=42):
        rng = np.random.RandomState(seed)
        n = len(a_obs)
        diffs = []
        for _ in range(n_boot):
            idx = rng.randint(0, n, n)
            rm_a = np.sqrt(np.mean((a_obs[idx] - a_predA[idx])**2))
            rm_b = np.sqrt(np.mean((a_obs[idx] - a_predB[idx])**2))
            diffs.append(rm_b - rm_a)
        arr = np.array(diffs)
        lo, med, hi = np.percentile(arr, [2.5, 50, 97.5])
        return lo, med, hi

    lo, med, hi = bootstrap_rmse_diff(obs_p, ok_p, rfok_p, n_boot=2000)

    # per-station table
    df_out = df_xy.copy().reset_index(drop=True)
    df_out['ok_pred'] = ok
    df_out['rfok_pred'] = rfok
    df_out['err_ok'] = df_out['temperature'] - df_out['ok_pred']
    df_out['err_rfok'] = df_out['temperature'] - df_out['rfok_pred']
    df_out['abs_err_ok'] = np.abs(df_out['err_ok'])
    df_out['abs_err_rfok'] = np.abs(df_out['err_rfok'])
    df_out['abs_err_diff'] = df_out['abs_err_rfok'] - df_out['abs_err_ok']  # 正数表示 RF-OK 更差

    # save per-station csv
    try:
        df_out.to_csv(f"{out_prefix}_per_station_compare.csv", index=False)
    except Exception:
        pass

    # prints
    print("Comparison summary (OK vs RF-OK):")
    print(summary.to_string(index=False, float_format='{:0.4f}'.format))
    print(f"\nPaired Wilcoxon on absolute errors: stat={stat}, p-value={pval}")
    print(f"Bootstrap RMSE diff (RF-OK - OK) 95% CI: [{lo:.4f}, {hi:.4f}], median={med:.4f}  (negative => RF-OK better)")

    # plots
    if save_plots:
        # scatter obs vs preds (both)
        plt.figure(figsize=(6,6))
        plt.scatter(obs_p, ok_p, s=20, alpha=0.6, label='OK (LOO)')
        plt.scatter(obs_p, rfok_p, s=20, alpha=0.6, label='RF-OK (OOF)')
        mn = min(obs_p.min(), ok_p.min(), rfok_p.min())
        mx = max(obs_p.max(), ok_p.max(), rfok_p.max())
        plt.plot([mn,mx],[mn,mx],'k--')
        plt.xlabel('Observed'); plt.ylabel('Predicted'); plt.legend()
        plt.title('Observed vs Predicted (paired)')
        plt.savefig(f"{out_prefix}_obs_vs_pred.png", dpi=200)
        plt.close()

        # histogram of rmse differences per station (abs_err difference)
        plt.figure(figsize=(6,4))
        plt.hist(df_out.loc[mask,'abs_err_diff'], bins=40)
        plt.axvline(0, color='k', linestyle='--')
        plt.title('Absolute error difference per station (RF-OK - OK)')
        plt.xlabel('abs_err_diff (positive => RF-OK worse)')
        plt.savefig(f"{out_prefix}_abs_err_diff_hist.png", dpi=200)
        plt.close()

        # scatter map of which stations improved
        try:
            import numpy as _np
            coords = np.column_stack([df_out['x'], df_out['y']])
            improved = (df_out['abs_err_diff'] < 0)
            plt.figure(figsize=(6,5))
            plt.scatter(coords[~np.isnan(df_out['x']),0]/1000.0, coords[~np.isnan(df_out['x']),1]/1000.0,
                        c=np.where(improved,1,0), cmap='RdYlBu', s=40)
            plt.title('Stations improved (1) vs not (0) by RF-OK')
            plt.xlabel('x (km)'); plt.ylabel('y (km)')
            plt.savefig(f"{out_prefix}_spatial_improved_map.png", dpi=200)
            plt.close()
        except Exception:
            pass

    return summary, df_out, (stat, pval, (lo, med, hi))