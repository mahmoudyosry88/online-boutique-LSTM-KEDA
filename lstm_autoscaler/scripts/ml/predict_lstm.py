
import sys
import os
# Ensure the v2 root is in the Python path so we can import from config, scripts, etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

#!/usr/bin/env python3
"""
LSTM INFERENCE AND EVALUATION PIPELINE

Inputs:
    - best_lstm_final.pt: The trained PyTorch model weights and data scalers.
    - autoscaling_training_dataset.csv: The dataset containing the test split (last 15%).

Outputs:
    - predictions_tau_0.45.csv: A CSV file logging the actual vs. predicted target replicas.
    - prediction_plots/: A folder containing high-quality visual graphs showing the model's accuracy per microservice.

Process:
    1. Loads the pre-trained LSTM model and standard scalers.
    2. Isolates the Test dataset (which the model has never seen).
    3. Runs fast batch inference to predict raw floating-point replica counts.
    4. Applies an optimized rounding threshold (Tau = 0.45) to convert raw predictions into exact integer replica counts.
    5. Calculates evaluation metrics (Accuracy, MAE) and generates comparative visual plots.
"""

import argparse
import re
import math
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import StandardScaler
from config.config_loader import get_config

import warnings
warnings.filterwarnings("ignore")

cfg = get_config()
paths_cfg = cfg['paths']
infer_cfg = cfg['inference']
base_dir = paths_cfg['base_dir']

# -------------------------------------------------------------------
# Matplotlib config (Windows-safe font handling)
# -------------------------------------------------------------------
plt.rcParams["axes.unicode_minus"] = False

font_prop = None
candidate_fonts = [
    r"prediction_plots/font.ttf",
    r"fonts/font.ttf",
]

for fp in candidate_fonts:
    if Path(fp).is_file():
        try:
            font_prop = fm.FontProperties(fname=fp)
            break
        except Exception:
            pass

if font_prop is None:
    font_prop = fm.FontProperties()


from scripts.ml.model import GlobalLSTMRegressor


def load_checkpoint(path, device):
    """
    Purpose: Restores the trained PyTorch model architecture, weights, and the StandardScaler objects from the saved .pt file.
    """
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    params = ckpt["params"]

    model = GlobalLSTMRegressor(
        num_features=ckpt["num_features"],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x_scaler = StandardScaler()
    x_scaler.mean_ = ckpt["x_scaler_mean"]
    x_scaler.scale_ = ckpt["x_scaler_scale"]

    y_scaler = StandardScaler()
    y_scaler.mean_ = ckpt["y_scaler_mean"]
    y_scaler.scale_ = ckpt["y_scaler_scale"]

    print("Model loaded!")
    return model, ckpt, x_scaler, y_scaler

def get_test_df(csv_path, feature_cols, service_col, train_frac=0.70, val_frac=0.15):
    """
    Purpose: Safely isolates the chronological Test split (the final 15% of the data) so the model is evaluated on strictly unseen future data.
    """
    print(f"\nLoading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df = df.sort_values(["Timestamp", service_col]).reset_index(drop=True)

    df["ts_int"] = df["Timestamp"].astype("int64") // 10**9
    t_min, t_max = df["ts_int"].min(), df["ts_int"].max()

    # ---------------------------------------------------------
    # Academic Train/Val/Test Chronological Split (70-15-15)
    # ---------------------------------------------------------
    # 1. Calculate the point in time where Train (70%) and Val (15%) end.
    # We add both fractions (0.70 + 0.15 = 0.85) to find the 85% timestamp mark.
    t_val_end = t_min + (t_max - t_min) * (train_frac + val_frac)

    # 2. Isolate the implicitly remaining 15% for Testing.
    # By selecting all timestamps GREATER than the 85% mark, we extract 
    # the final 15% (unseen data) for testing generalization.
    test_df = df[df["ts_int"] > t_val_end].copy()
    
    if len(test_df) == 0:
        raise ValueError("Test dataframe is empty! Check train_frac and val_frac splits.")

    print(f"   Test rows : {len(test_df)} ({test_df['Timestamp'].min()} to {test_df['Timestamp'].max()})")
    return test_df

def apply_rounding_threshold(pred_raw, threshold):
    """
    Purpose: Converts continuous model predictions into discrete pod counts (integers) using a custom mathematical threshold instead of standard rounding.
    """
    fractional = pred_raw - math.floor(pred_raw)
    if fractional >= threshold:
        return max(1, int(math.ceil(pred_raw)))
    return max(1, int(math.floor(pred_raw)))

def predict_test_tau045(model, test_df, feature_cols, target_col,
                         service_col, lookback, x_scaler, y_scaler, device,
                         service_ids, constant_services, tau=0.45):
    """
    Purpose: Executes batched inference across all microservices, processes the sliding windows, scales inputs/outputs, and compiles the final error metrics.
    """
    all_rows = []
    constant_services = constant_services or set()

    with torch.no_grad():
        for service, g in test_df.groupby(service_col):
            g = g.sort_values("Timestamp").reset_index(drop=True)
            feat = g[feature_cols].values.astype(np.float32)
            target = g[target_col].values.astype(np.float32)
            times = g["Timestamp"].values
            users_vals = g["Users"].values if "Users" in g.columns else np.zeros(len(g))
            cpu_vals = g["CPU"].values if "CPU" in g.columns else np.zeros(len(g))
            mem_vals = g["Memory"].values if "Memory" in g.columns else np.zeros(len(g))
            lat_vals = g["Latency"].values if "Latency" in g.columns else np.zeros(len(g))
            rps_vals = g["RPS_frontend"].values if "RPS_frontend" in g.columns else np.zeros(len(g))



            svc_id = np.full((len(g), 1), service_ids[service], dtype=np.float32)
            is_constant = np.full((len(g), 1), 1.0 if service in constant_services else 0.0, dtype=np.float32)
            feat_with_meta = np.hstack([feat, svc_id, is_constant])
            feat_with_meta = x_scaler.transform(feat_with_meta).astype(np.float32)

            # Batch Inference Optimization
            all_tensors = []
            valid_times = []
            valid_actuals = []
            
            for i in range(lookback, len(g)):
                window = feat_with_meta[i - lookback:i]
                all_tensors.append(window)
                valid_times.append(times[i])
                valid_actuals.append(float(target[i]))
            
            if not all_tensors:
                continue
                
            tensor_batch = torch.from_numpy(np.array(all_tensors)).to(device)
            pred_scaled_batch = model(tensor_batch).cpu().numpy()
            
            pred_raw_batch = y_scaler.inverse_transform(pred_scaled_batch.reshape(-1, 1)).flatten()
            
            for i in range(len(pred_raw_batch)):
                pred_raw = float(pred_raw_batch[i])
                actual = valid_actuals[i]
                actual_round = round(actual)
                pred_rounded_opt = apply_rounding_threshold(pred_raw, tau)

                all_rows.append({
                    "Timestamp": pd.Timestamp(valid_times[i]),
                    "Service": service,
                    "Users": users_vals[i],
                    "CPU": cpu_vals[i],
                    "Memory": mem_vals[i],
                    "Latency": lat_vals[i],
                    "RPS_frontend": rps_vals[i],
                    "Actual_Replicas": actual_round,
                    "Predicted_Raw": round(pred_raw, 4),
                    "Raw_Abs_Error": round(abs(pred_raw - actual), 4),
                    "Predicted_Replicas_Optimized": pred_rounded_opt,
                    "Correct_Optimized": int(pred_rounded_opt == actual_round),
                    "Round_Abs_Error_Optimized": abs(pred_rounded_opt - actual_round),
                })

    return pd.DataFrame(all_rows)

def save_advanced_metrics_report(results_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "advanced_metrics_report.csv"
    
    rows = []
    
    for service, grp in results_df.groupby("Service"):
        y_true = grp["Actual_Replicas"].astype(int)
        y_pred = grp["Predicted_Replicas_Optimized"].astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        
        # Calculate macro averages for precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        # Type 1: Predicted > Actual
        type_1_count = (y_pred > y_true).sum()
        type_1_pct = (type_1_count / len(y_true)) * 100
        
        # Type 2: Predicted < Actual
        type_2_count = (y_pred < y_true).sum()
        type_2_pct = (type_2_count / len(y_true)) * 100
        
        rows.append({
            "Service": service,
            "Accuracy (%)": round(acc * 100, 2),
            "Precision (Macro)": round(precision, 4),
            "Recall (Macro)": round(recall, 4),
            "F1-Score (Macro)": round(f1, 4),
            "Type 1 Error (Over-provisioned) %": round(type_1_pct, 2),
            "Type 2 Error (Under-provisioned) %": round(type_2_pct, 2),
            "Total Samples": len(y_true)
        })
        
    df_report = pd.DataFrame(rows)
    df_report.to_csv(report_path, index=False)
    print(f"Saved Advanced Metrics Report CSV: {report_path}")

def print_report_tau045(results_df, tau=0.45):
    total = len(results_df)
    correct = int(results_df["Correct_Optimized"].sum())
    acc = correct / total * 100

    raw_mae = results_df["Raw_Abs_Error"].mean()
    round_mae = results_df["Round_Abs_Error_Optimized"].mean()

    print(f"\n{'='*100}")
    print(f"  LSTM AUTOSCALER - tau={tau:.2f} REPORT (BATCH OPTIMIZED)")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}")
    print(f"  Raw MAE                         : {raw_mae:.4f}")
    print(f"  tau={tau:.2f} Round MAE           : {round_mae:.4f}")
    print(f"  Accuracy (Optimized integer)   : {acc:.2f}% ({correct}/{total})\n")

def save_time_sorted_csv(results_df, output_path):
    df = results_df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(["Timestamp", "Service"]).reset_index(drop=True)
    df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    output_cols = [
        "Timestamp", "Service", "Users", "CPU", "Memory", "Latency", "RPS_frontend", "Actual_Replicas",
        "Predicted_Raw", "Raw_Abs_Error",
        "Predicted_Replicas_Optimized",
        "Correct_Optimized", "Round_Abs_Error_Optimized",
    ]
    df_out = df[output_cols]
    df_out.to_csv(output_path, index=False)
    print(f"Saved predictions CSV: {output_path}")

    # Save a rounded version (4 decimal places) for academic presentation
    rounded_path = output_path.replace('.csv', '_rounded.csv')
    df_rounded = df_out.copy()
    
    if "Memory" in df_rounded.columns:
        df_rounded["Memory"] = pd.to_numeric(df_rounded["Memory"], errors='coerce') / 1048576.0
        
    df_rounded.rename(columns={
        "Memory": "Memory_MiB", 
        "CPU": "CPU_Cores", 
        "Latency": "Latency_ms"
    }, inplace=True)

    cols_to_round = ["CPU_Cores", "Memory_MiB", "Latency_ms", "RPS_frontend", "Predicted_Raw", "Raw_Abs_Error"]
    for col in cols_to_round:
        if col in df_rounded.columns:
            df_rounded[col] = pd.to_numeric(df_rounded[col], errors='coerce').round(4)
    df_rounded.to_csv(rounded_path, index=False)
    print(f"Saved rounded predictions CSV: {rounded_path}")

def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")

def save_visual_reports(results_df, output_dir):
    output_dir = Path(output_dir)
    cm_dir = output_dir / "confusion_matrices"
    avp_dir = output_dir / "actual_vs_predicted"
    cm_dir.mkdir(parents=True, exist_ok=True)
    avp_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    try:
        for service, grp in results_df.groupby("Service"):
            grp = grp.sort_values("Timestamp")
            safe_service = _safe_filename(service)

            # Contiguous labels for confusion matrix
            min_label = int(min(grp["Actual_Replicas"].min(), grp["Predicted_Replicas_Optimized"].min()))
            max_label = int(max(grp["Actual_Replicas"].max(), grp["Predicted_Replicas_Optimized"].max()))
            labels = list(range(min_label, max_label + 1))

            cm = confusion_matrix(
                grp["Actual_Replicas"].astype(int),
                grp["Predicted_Replicas_Optimized"].astype(int),
                labels=labels
            )

            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111)
            im = ax.imshow(cm, cmap="Greens")
            ax.set_title(f"Confusion Matrix (τ=0.45) - {service}", fontproperties=font_prop)
            ax.set_xlabel("Predicted replicas", fontproperties=font_prop)
            ax.set_ylabel("Actual replicas", fontproperties=font_prop)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, fontproperties=font_prop)
            ax.set_yticklabels(labels, fontproperties=font_prop)

            max_value = cm.max() if cm.size else 0
            threshold_val = max_value / 2 if max_value else 0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > threshold_val else "black", fontproperties=font_prop)

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if cbar.ax is not None:
                for lbl in cbar.ax.get_yticklabels():
                    lbl.set_fontproperties(font_prop)
            
            fig.tight_layout()
            cm_path = cm_dir / f"cm_{safe_service}.png"
            fig.savefig(cm_path, dpi=150)
            plt.close(fig)
            saved_files.append(cm_path)

            fig = plt.figure(figsize=(12, 4))
            ax = fig.add_subplot(111)
            ax.plot(grp["Timestamp"], grp["Actual_Replicas"], label="Actual", linewidth=2, marker="o", markersize=3)
            ax.plot(grp["Timestamp"], grp["Predicted_Replicas_Optimized"], label="Predicted (τ=0.45)", linewidth=1.8, linestyle="--", marker="^", markersize=3)
            ax.set_title(f"Actual vs Predicted (τ=0.45) - {service}", fontproperties=font_prop)
            ax.set_xlabel("Timestamp", fontproperties=font_prop)
            ax.set_ylabel("Replicas", fontproperties=font_prop)
            ax.grid(True, alpha=0.3)
            ax.legend(prop=font_prop)
            
            # Explicitly format the x-axis to show clear time (HH:MM) without the day number
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            fig.autofmt_xdate()
            fig.tight_layout()
            
            avp_path = avp_dir / f"avp_{safe_service}.png"
            fig.savefig(avp_path, dpi=150)
            plt.close(fig)
            saved_files.append(avp_path)

    except Exception as e:
        print(f"Warning: Plotting failed: {e}")

    return saved_files

def main():
    parser = argparse.ArgumentParser(description="LSTM Inference")
    parser.add_argument("--checkpoint", default=os.path.join(base_dir, paths_cfg['checkpoint']))
    parser.add_argument("--csv", default=os.path.join(base_dir, paths_cfg['dataset_csv']))
    parser.add_argument("--output", default=os.path.join(base_dir, paths_cfg['predictions_csv']))
    parser.add_argument("--plots_dir", default=os.path.join(base_dir, paths_cfg['predictions_dir']))
    parser.add_argument("--tau", type=float, default=infer_cfg['tau'])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt, x_scaler, y_scaler = load_checkpoint(args.checkpoint, device)

    train_frac = ckpt["params"]["train_frac"]
    val_frac = ckpt["params"]["val_frac"]

    service_ids = ckpt["service_ids"]
    if 0 in service_ids or "0" in service_ids:
        service_ids = {v: int(k) for k, v in service_ids.items()}

    test_df = get_test_df(args.csv, ckpt["feature_cols"], "Service", train_frac, val_frac)

    results_df = predict_test_tau045(
        model, test_df, ckpt["feature_cols"], ckpt["target_col"],
        "Service", ckpt["lookback"], x_scaler, y_scaler, device,
        service_ids, ckpt.get("constant_services", set()), args.tau
    )

    print_report_tau045(results_df, args.tau)
    save_advanced_metrics_report(results_df, args.plots_dir)
    save_time_sorted_csv(results_df, args.output)
    saved_plots = save_visual_reports(results_df, args.plots_dir)
    print(f"Saved {len(saved_plots)} plot files under: {args.plots_dir}")

if __name__ == "__main__":
    main()