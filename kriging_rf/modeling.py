"""
建模流程：RF OOF、在训练集上对残差做 Kriging 并对验证点修正；以及最后在全量上生成网格输出。
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from .kriging_utils import krige_residuals_to_points, krige_residuals_to_grid, perform_ok_interpolation


def generate_oof_corrected(df_xy, n_splits=5, rf_params=None, monitor_out_dir="monitor", monitor_enable=True):
    """
    生成 OOF 的 RF 基线与 RF+Kriging(residual) 修正，并在每个 fold 记录监控信息。

    输入:
      - df_xy: DataFrame，必须包含列 ['x','y','elevation','temperature']
      - n_splits: KFold 折数
      - rf_params: dict，传给 RandomForestRegressor 的超参（例如 {'n_estimators':200,'max_depth':14}）
      - monitor_out_dir: str，监控输出目录（若 None 则不启用监控图）
      - monitor_enable: bool，是否启用监控（控制是否记录并画图）

    返回:
      (oof_rf, oof_corr, rf_full, scaler_full, monitor)
      - monitor 为 ModelMonitor 实例（若 monitor_enable=False 仍返回 None）
    """
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold

    # local imports from package
    from .kriging_utils import krige_residuals_to_points
    # lazy import monitor to avoid hard dependency if module not present
    try:
        from .monitoring import ModelMonitor
    except Exception:
        ModelMonitor = None

    if rf_params is None:
        rf_params = {'n_estimators': 200, 'max_depth': 14}

    # prepare arrays
    n = len(df_xy)
    x = df_xy['x'].values
    y = df_xy['y'].values
    elev = df_xy['elevation'].values
    z = df_xy['temperature'].values

    oof_rf = np.full(n, np.nan)
    oof_corr = np.full(n, np.nan)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # setup monitor if available and requested
    monitor = None
    if monitor_enable and ModelMonitor is not None:
        monitor = ModelMonitor(out_dir=monitor_out_dir or "monitor", enable_plots=True)
    else:
        monitor = None

    # Fold loop
    for fold_i, (train_idx, val_idx) in enumerate(kf.split(np.arange(n)), start=1):
        print(f"[OOF] fold {fold_i}/{n_splits} train={len(train_idx)} val={len(val_idx)}")

        # prepare train/val features
        X_tr = np.column_stack([x[train_idx], y[train_idx], elev[train_idx]])
        X_val = np.column_stack([x[val_idx], y[val_idx], elev[val_idx]])
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        # train RF on train set
        rf = RandomForestRegressor(n_jobs=-1, random_state=42, **rf_params)
        rf.fit(X_tr_s, z[train_idx])

        # RF predictions on val
        rf_val = rf.predict(X_val_s)
        oof_rf[val_idx] = rf_val

        # compute train residuals and krige them to validation points
        rf_train_pred = rf.predict(X_tr_s)
        resid_train = z[train_idx] - rf_train_pred

        # use kriging util to predict residuals at validation coordinates
        try:
            res_pred = krige_residuals_to_points(x[train_idx], y[train_idx], resid_train, x[val_idx], y[val_idx])
        except Exception:
            # fallback: zeros
            res_pred = np.zeros(len(val_idx))

        oof_corr[val_idx] = rf_val + res_pred

        # logging to monitor if available
        if monitor is not None:
            try:
                y_train = z[train_idx]
                y_val = z[val_idx]
                coords_train = np.column_stack([x[train_idx], y[train_idx]])
                coords_val = np.column_stack([x[val_idx], y[val_idx]])
                monitor.log_fold(
                    fold=fold_i,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    y_train=y_train,
                    y_val=y_val,
                    rf_pred_train=rf_train_pred,
                    rf_pred_val=rf_val,
                    resid_train=resid_train,
                    krige_resid_val=res_pred,
                    coords_train=coords_train,
                    coords_val=coords_val,
                    extra={'rf_params': rf.get_params()}
                )
            except Exception:
                # do not interrupt main flow if monitoring fails
                pass

    # train final RF on full data for grid prediction
    X_full = np.column_stack([x, y, elev])
    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X_full)
    rf_full = RandomForestRegressor(n_jobs=-1, random_state=42, **rf_params)
    rf_full.fit(X_full_s, z)

    # after OOF complete, populate monitor with OOF arrays and feature importances
    if monitor is not None:
        try:
            coords_all = np.column_stack([x, y])
            monitor.set_oof(y_true=z, oof_rf=oof_rf, oof_corr=oof_corr, coords=coords_all)
            # feature names and importances (we used x,y,elevation)
            feature_names = ['x', 'y', 'elevation']
            try:
                fi = rf_full.feature_importances_
            except Exception:
                fi = None
            if fi is not None:
                monitor.set_feature_importance(feature_names, fi)
            monitor.finalize()
        except Exception:
            pass

    # return same outputs as before, plus monitor
    return oof_rf, oof_corr, rf_full, scaler_full, monitor


def final_grid_prediction(df_xy, rf_full, scaler_full, grid_res=50):
    """生成 RF 网格 + 全量残差 kriging 的最终 corrected_grid"""
    x = df_xy['x'].values; y = df_xy['y'].values; elev = df_xy['elevation'].values; z = df_xy['temperature'].values

    # build grid same as perform_ok_interpolation
    grid_xx, grid_yy, _ = perform_ok_interpolation(x, y, z, grid_res=grid_res)
    grid_dem = None
    # interpolate DEM from station elev
    from .features import interpolate_dem_to_grid
    grid_dem = interpolate_dem_to_grid(x, y, elev, grid_xx, grid_yy)

    grid_feat = np.column_stack([grid_xx.ravel(), grid_yy.ravel(), grid_dem.ravel()])
    grid_feat_s = scaler_full.transform(grid_feat)
    rf_grid = rf_full.predict(grid_feat_s).reshape(grid_xx.shape)

    # residuals on full data
    X_full = np.column_stack([x, y, elev])
    X_full_s = scaler_full.transform(X_full)
    rf_full_pred = rf_full.predict(X_full_s)
    resid_full = z - rf_full_pred

    grid_x = np.unique(grid_xx[0, :])
    grid_y = np.unique(grid_yy[:, 0])
    res_grid = krige_residuals_to_grid(x, y, resid_full, grid_x, grid_y)

    corrected_grid = rf_grid + res_grid
    return grid_xx, grid_yy, rf_grid, res_grid, corrected_grid