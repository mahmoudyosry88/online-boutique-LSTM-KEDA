import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

def create_presentation_plots():
    csv_path = os.path.join("data", "processed", "autoscaling_training_dataset.csv")
    out_dir = os.path.join("outputs", "presentation_plots")
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print("Dataset not found!")
        return

    df = pd.read_csv(csv_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    
    # 1. Feature Correlation Heatmap
    plt.figure(figsize=(10, 8))
    # Select only numeric columns for correlation
    numeric_cols = ["CPU", "Memory", "Latency", "RPS_frontend", "Users", "target_replicas"]
    corr = df[numeric_cols].corr()
    
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "1_Correlation_Heatmap.png"), dpi=300)
    plt.close()
    
    # 2. System Traffic Load (Frontend RPS vs Users)
    plt.figure(figsize=(12, 5))
    frontend_df = df[df["Service"] == "frontend"].copy()
    frontend_df = frontend_df.sort_values("Timestamp")
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    ax1.plot(frontend_df["Timestamp"], frontend_df["RPS_frontend"], color="blue", label="RPS (Requests/Sec)", linewidth=2)
    ax2.plot(frontend_df["Timestamp"], frontend_df["Users"], color="orange", label="Simulated Users", linewidth=2, linestyle="--")
    
    ax1.set_xlabel("Time (HH:MM)", fontsize=12)
    ax1.set_ylabel("RPS", color="blue", fontsize=12)
    ax2.set_ylabel("Active Users", color="orange", fontsize=12)
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.title("Locust Workload: Simulated Users vs Requests Per Second", fontsize=14, fontweight="bold")
    fig = plt.gcf()
    fig.autofmt_xdate()
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "2_Traffic_Load.png"), dpi=300)
    plt.close()

    # 3. Resource vs Scaling (Cartservice example)
    plt.figure(figsize=(12, 5))
    svc_df = df[df["Service"] == "cartservice"].copy()
    svc_df = svc_df.sort_values("Timestamp")
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    ax1.fill_between(svc_df["Timestamp"], svc_df["CPU"], color="red", alpha=0.3, label="CPU Usage (mCores)")
    ax1.plot(svc_df["Timestamp"], svc_df["CPU"], color="red", linewidth=1.5)
    
    ax2.step(svc_df["Timestamp"], svc_df["target_replicas"], color="green", label="Replicas (Pods)", linewidth=2.5, where='post')
    
    ax1.set_xlabel("Time (HH:MM)", fontsize=12)
    ax1.set_ylabel("CPU Usage", color="red", fontsize=12)
    ax2.set_ylabel("Replicas", color="green", fontsize=12)
    ax2.set_yticks(range(0, int(svc_df["target_replicas"].max()) + 2))
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.title("cartservice: CPU Spikes Triggering Scaling Events", fontsize=14, fontweight="bold")
    fig = plt.gcf()
    fig.autofmt_xdate()
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "3_CPU_vs_Scaling.png"), dpi=300)
    plt.close()

    # 4. Latency Stability
    plt.figure(figsize=(12, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    ax1.plot(frontend_df["Timestamp"], frontend_df["Latency"], color="purple", label="Response Latency (ms)", linewidth=2)
    ax2.step(frontend_df["Timestamp"], frontend_df["target_replicas"], color="teal", label="Frontend Replicas", linewidth=2.5, linestyle="--", where='post')
    
    ax1.set_xlabel("Time (HH:MM)", fontsize=12)
    ax1.set_ylabel("Latency (ms)", color="purple", fontsize=12)
    ax2.set_ylabel("Replicas", color="teal", fontsize=12)
    ax2.set_yticks(range(0, int(frontend_df["target_replicas"].max()) + 2))
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.title("Frontend: Maintaining Low Latency Under Pressure via Autoscaling", fontsize=14, fontweight="bold")
    fig = plt.gcf()
    fig.autofmt_xdate()
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "4_Latency_Stability.png"), dpi=300)
    plt.close()

    print(f"Successfully generated 4 presentation-ready plots in: {out_dir}")

if __name__ == "__main__":
    create_presentation_plots()
