"""
kriging_rf/monitoring.py

ModelMonitor: 用于追踪 RF + Kriging residual pipeline 的训练/OOF 过程并生成可视化报告。
- 在每个 OOF fold 结束时调用 log_fold(...) 记录该折的指标与图像快照
- 在流程结束时调用 finalize(...) 生成总体报告图与 CSV
输出（默认到 out_dir）:
 - per_fold_metrics.csv
 - fold_XX_scatter.png, fold_XX_resid_hist.png, fold_XX_spatial_resid.png
 - overall_summary.png, overall_resid_hist.png, feature_importance.png (若传入 feature names / importances)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend('Agg')  # safe for headless servers

class ModelMonitor:
    def __init__(self, out_dir="monitor", enable_plots=True):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.enable_plots = enable_plots
        # storage
        self.fold_records = []  # list of dicts: fold, n_train, n_val, rmse_rf, mae_rf, rmse_corr, mae_corr
        # OOF accumulation
        self.oof_rf = None
        self.oof_corr = None
        self.y_true = None
        self.coords = None  # array [[x,y], ...] optional for spatial plots
        self.feature_names = None
        self.feature_importances = None

    def log_fold(self, fold, train_idx, val_idx,
                 y_train, y_val,
                 rf_pred_train, rf_pred_val,
                 resid_train=None, krige_resid_val=None,
                 coords_train=None, coords_val=None,
                 extra=None):
        """
        在每个 fold 结束时调用记录指标并保存折快照图。
        必需参数:
          - fold: int
          - train_idx, val_idx: indices in original dataset (optional, used for saving)
          - y_train, y_val: arrays of true values
          - rf_pred_train, rf_pred_val: RF predictions on train and val sets (train preds used for residuals)
        可选:
          - resid_train: residuals on train (y_train - rf_pred_train)
          - krige_resid_val: Kriging-predicted residuals for val (same length as y_val)
          - coords_train / coords_val: arrays shape (n,2) for spatial residual plots
          - extra: dict of any extra info to store
        """
        # compute rf metrics on val
        mae_rf = np.nanmean(np.abs(y_val - rf_pred_val))
        rmse_rf = np.sqrt(np.nanmean((y_val - rf_pred_val)**2))
        mae_corr = np.nan
        rmse_corr = np.nan
        if krige_resid_val is not None:
            pred_corr = rf_pred_val + krige_resid_val
            mae_corr = np.nanmean(np.abs(y_val - pred_corr))
            rmse_corr = np.sqrt(np.nanmean((y_val - pred_corr)**2))
        rec = {
            "fold": int(fold),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "mae_rf": float(mae_rf),
            "rmse_rf": float(rmse_rf),
            "mae_corr": (float(mae_corr) if not np.isnan(mae_corr) else None),
            "rmse_corr": (float(rmse_corr) if not np.isnan(rmse_corr) else None),
            "extra": extra or {}
        }
        self.fold_records.append(rec)

        # save scatter and resid hist for this fold
        if self.enable_plots:
            base = os.path.join(self.out_dir, f"fold_{fold:02d}")
            # obs vs pred scatter for RF and corrected (if available)
            try:
                plt.figure(figsize=(6,6))
                plt.scatter(y_val, rf_pred_val, s=25, alpha=0.6, label='RF')
                if krige_resid_val is not None:
                    plt.scatter(y_val, rf_pred_val + krige_resid_val, s=25, alpha=0.6, label='RF+KrigedResid')
                mn = np.nanmin(np.concatenate([y_val, rf_pred_val]))
                mx = np.nanmax(np.concatenate([y_val, rf_pred_val]))
                plt.plot([mn,mx], [mn,mx], 'k--', linewidth=0.8)
                plt.xlabel("Observed"); plt.ylabel("Predicted")
                plt.title(f"Fold {fold}: Obs vs Pred")
                plt.legend()
                plt.tight_layout()
                plt.savefig(base + "_scatter.png", dpi=150)
                plt.close()
            except Exception:
                pass

            # residual histogram
            try:
                plt.figure(figsize=(6,4))
                resid_rf = (y_val - rf_pred_val)
                plt.hist(resid_rf, bins=30, alpha=0.6, label='RF resid')
                if krige_resid_val is not None:
                    resid_corr = (y_val - (rf_pred_val + krige_resid_val))
                    plt.hist(resid_corr, bins=30, alpha=0.6, label='RF-OK resid')
                plt.legend()
                plt.title(f"Fold {fold} residuals")
                plt.xlabel("Residual (obs - pred)")
                plt.tight_layout()
                plt.savefig(base + "_resid_hist.png", dpi=150)
                plt.close()
            except Exception:
                pass

            # spatial residual plot if coordinates given
            try:
                if coords_val is not None and coords_val.shape[1] >= 2:
                    plt.figure(figsize=(6,5))
                    sc = plt.scatter(coords_val[:,0] / 1000.0, coords_val[:,1] / 1000.0, c=(y_val - rf_pred_val), cmap='coolwarm', s=40)
                    plt.colorbar(sc, label='RF residual')
                    plt.title(f"Fold {fold} RF residual (spatial km)")
                    plt.xlabel("x (km)"); plt.ylabel("y (km)")
                    plt.tight_layout()
                    plt.savefig(base + "_spatial_resid_rf.png", dpi=150)
                    plt.close()
                    if krige_resid_val is not None:
                        plt.figure(figsize=(6,5))
                        sc2 = plt.scatter(coords_val[:,0] / 1000.0, coords_val[:,1] / 1000.0, c=(y_val - (rf_pred_val + krige_resid_val)), cmap='coolwarm', s=40)
                        plt.colorbar(sc2, label='RF-OK residual')
                        plt.title(f"Fold {fold} RF-OK residual (spatial km)")
                        plt.xlabel("x (km)"); plt.ylabel("y (km)")
                        plt.tight_layout()
                        plt.savefig(base + "_spatial_resid_corr.png", dpi=150)
                        plt.close()
            except Exception:
                pass

        # flush per-fold JSON for reproducibility
        try:
            with open(os.path.join(self.out_dir, f"fold_{fold:02d}_meta.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def set_oof(self, y_true, oof_rf, oof_corr, coords=None):
        """在 OOF 全部fold完成后传入完整数组，便于总体绘图/保存"""
        self.y_true = np.asarray(y_true)
        self.oof_rf = np.asarray(oof_rf)
        self.oof_corr = np.asarray(oof_corr)
        if coords is not None:
            self.coords = np.asarray(coords)

    def set_feature_importance(self, feature_names, importances):
        """如果模型提供 feature importances，可传入以画图"""
        self.feature_names = list(feature_names)
        self.feature_importances = np.asarray(importances)

    def finalize(self):
        """保存 fold 表格与总体图（scatter, hist, fold-metrics）"""
        # save per-fold table
        try:
            df = pd.DataFrame(self.fold_records)
            df.to_csv(os.path.join(self.out_dir, "per_fold_metrics.csv"), index=False)
        except Exception:
            pass

        if not self.enable_plots:
            return

        # fold-level metrics plot
        try:
            folds = [r['fold'] for r in self.fold_records]
            rmse_rf = [r['rmse_rf'] for r in self.fold_records]
            rmse_corr = [r['rmse_corr'] if r['rmse_corr'] is not None else np.nan for r in self.fold_records]
            plt.figure(figsize=(8,4))
            plt.plot(folds, rmse_rf, '-o', label='RF RMSE')
            if not all(np.isnan(rmse_corr)):
                plt.plot(folds, rmse_corr, '-o', label='RF-OK RMSE')
            plt.xlabel("Fold"); plt.ylabel("RMSE"); plt.title("Per-fold RMSE")
            plt.grid(alpha=0.3); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, "per_fold_rmse.png"), dpi=150)
            plt.close()
        except Exception:
            pass

        # overall OOF scatter & hist
        try:
            if self.y_true is not None and self.oof_rf is not None:
                plt.figure(figsize=(6,6))
                mask = ~np.isnan(self.oof_rf)
                plt.scatter(self.y_true[mask], self.oof_rf[mask], s=20, alpha=0.6, label='RF OOF')
                if self.oof_corr is not None:
                    mask2 = ~np.isnan(self.oof_corr)
                    plt.scatter(self.y_true[mask2], self.oof_corr[mask2], s=20, alpha=0.6, label='RF-OK OOF')
                mn = np.nanmin(self.y_true); mx = np.nanmax(self.y_true)
                plt.plot([mn,mx], [mn,mx], 'k--')
                plt.xlabel("Observed"); plt.ylabel("Predicted"); plt.title("Overall OOF: Obs vs Pred")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, "oof_scatter.png"), dpi=150)
                plt.close()

                # residual hist
                plt.figure(figsize=(6,4))
                resid_rf = (self.y_true - self.oof_rf)
                plt.hist(resid_rf[~np.isnan(resid_rf)], bins=40, alpha=0.6, label='RF OOF')
                if self.oof_corr is not None:
                    resid_corr = (self.y_true - self.oof_corr)
                    plt.hist(resid_corr[~np.isnan(resid_corr)], bins=40, alpha=0.6, label='RF-OK OOF')
                plt.legend(); plt.title("OOF residuals"); plt.xlabel("Residual")
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, "oof_resid_hist.png"), dpi=150)
                plt.close()
        except Exception:
            pass

        # spatial residual map if coords available
        try:
            if self.coords is not None:
                import numpy as _np
                cx = self.coords[:,0] / 1000.0
                cy = self.coords[:,1] / 1000.0
                mask = ~np.isnan(self.oof_rf)
                plt.figure(figsize=(6,5))
                sc = plt.scatter(cx[mask], cy[mask], c=(self.y_true[mask] - self.oof_rf[mask]), cmap='coolwarm', s=40)
                plt.colorbar(sc, label='RF OOF resid')
                plt.title("Spatial RF OOF residuals (km)")
                plt.xlabel("x (km)"); plt.ylabel("y (km)")
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, "spatial_oof_resid_rf.png"), dpi=150)
                plt.close()
                if self.oof_corr is not None:
                    mask2 = ~np.isnan(self.oof_corr)
                    plt.figure(figsize=(6,5))
                    sc2 = plt.scatter(cx[mask2], cy[mask2], c=(self.y_true[mask2] - self.oof_corr[mask2]), cmap='coolwarm', s=40)
                    plt.colorbar(sc2, label='RF-OK OOF resid')
                    plt.title("Spatial RF-OK OOF residuals (km)")
                    plt.xlabel("x (km)"); plt.ylabel("y (km)")
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.out_dir, "spatial_oof_resid_corr.png"), dpi=150)
                    plt.close()
        except Exception:
            pass

        # feature importance
        try:
            if self.feature_names is not None and self.feature_importances is not None:
                inds = np.argsort(self.feature_importances)[::-1]
                names = [self.feature_names[i] for i in inds]
                imps = self.feature_importances[inds]
                plt.figure(figsize=(6,4))
                plt.barh(range(len(imps)), imps[::-1], align='center')
                plt.yticks(range(len(imps)), names[::-1])
                plt.title("Feature importances")
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, "feature_importances.png"), dpi=150)
                plt.close()
        except Exception:
            pass