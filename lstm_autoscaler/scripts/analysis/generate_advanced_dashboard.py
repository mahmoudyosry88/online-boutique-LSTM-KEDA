import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.metrics import confusion_matrix, classification_report

def create_advanced_plots():
    train_csv_path = os.path.join("data", "processed", "autoscaling_training_dataset.csv")
    preds_csv_path = os.path.join("data", "processed", "predictions_tau_0_45.csv")
    out_dir = os.path.join("outputs", "presentation_plots")
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(train_csv_path) or not os.path.exists(preds_csv_path):
        print("Data files not found!")
        return

    df_train = pd.read_csv(train_csv_path)
    df_train["Timestamp"] = pd.to_datetime(df_train["Timestamp"])
    df_preds = pd.read_csv(preds_csv_path)

    # 1. Timeline with Autoscaling Events (Frontend)
    plt.figure(figsize=(14, 6))
    frontend_df = df_train[df_train["Service"] == "frontend"].copy()
    frontend_df = frontend_df.sort_values("Timestamp").reset_index(drop=True)
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    ax1.plot(frontend_df["Timestamp"], frontend_df["RPS_frontend"], color="blue", label="RPS", linewidth=2)
    ax2.step(frontend_df["Timestamp"], frontend_df["target_replicas"], color="green", label="Replicas", linewidth=2.5, where='post')
    
    # Add markers for scale up/down
    frontend_df['replica_diff'] = frontend_df['target_replicas'].diff()
    scale_ups = frontend_df[frontend_df['replica_diff'] > 0]
    scale_downs = frontend_df[frontend_df['replica_diff'] < 0]
    
    for _, row in scale_ups.iterrows():
        ax2.axvline(x=row['Timestamp'], color='red', linestyle='--', alpha=0.5)
    for _, row in scale_downs.iterrows():
        ax2.axvline(x=row['Timestamp'], color='teal', linestyle='--', alpha=0.5)
        
    # Custom legend for markers
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='green', lw=2),
                    Line2D([0], [0], color='red', linestyle='--', alpha=0.5),
                    Line2D([0], [0], color='teal', linestyle='--', alpha=0.5)]
    ax1.legend(custom_lines, ['RPS', 'Replicas', 'Scale Up Event', 'Scale Down Event'], loc='upper left')

    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel("Requests Per Second", color="blue", fontsize=12)
    ax2.set_ylabel("Frontend Replicas", color="green", fontsize=12)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.title("Timeline: Traffic Spikes and Explicit Autoscaling Events", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Advanced_1_Timeline_Events.png"), dpi=300)
    plt.close()

    # 2. Resource Stacked Comparison with Thresholds (cartservice)
    plt.figure(figsize=(14, 6))
    cart_df = df_train[df_train["Service"] == "cartservice"].copy()
    cart_df = cart_df.sort_values("Timestamp")
    
    # Normalize CPU and Memory for comparison (0 to 1 scale)
    cart_df["CPU_norm"] = cart_df["CPU"] / cart_df["CPU"].max()
    cart_df["Memory_norm"] = cart_df["Memory"] / cart_df["Memory"].max()
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    ax1.plot(cart_df["Timestamp"], cart_df["CPU_norm"], color="red", label="CPU (Normalized)", linewidth=2)
    ax1.plot(cart_df["Timestamp"], cart_df["Memory_norm"], color="purple", label="Memory (Normalized)", linewidth=2)
    ax1.axhline(y=0.7, color='black', linestyle=':', linewidth=2, label="70% Threshold")
    
    ax2.step(cart_df["Timestamp"], cart_df["target_replicas"], color="green", label="Replicas", linewidth=3, where='post')
    
    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel("Normalized Resource Usage (0-1)", fontsize=12)
    ax2.set_ylabel("Replicas", color="green", fontsize=12)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.title("cartservice: Multi-Resource Usage vs Threshold & Scaling", fontsize=16, fontweight="bold")
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Advanced_2_Resource_Thresholds.png"), dpi=300)
    plt.close()

    # 3. Advanced Correlation Pairplot
    # We sample the dataset if it's too large to make plotting faster
    sample_df = df_train.sample(n=min(2000, len(df_train)), random_state=42)
    pairplot_cols = ["CPU", "Memory", "Latency", "target_replicas"]
    pair_fig = sns.pairplot(sample_df[pairplot_cols], diag_kind="kde", corner=True, 
                            plot_kws={'alpha': 0.5, 's': 10, 'edgecolor': 'none'})
    pair_fig.fig.suptitle("Advanced Feature Correlation & Distributions", y=1.02, fontsize=16, fontweight='bold')
    pair_fig.savefig(os.path.join(out_dir, "Advanced_3_Pairplot.png"), dpi=300)
    plt.close()

    # 4. Annotated Confusion Matrix + Precision/Recall
    actual = df_preds["Actual_Replicas"].astype(int)
    predicted = df_preds["Predicted_Replicas_Optimized"].astype(int)
    
    report = classification_report(actual, predicted, output_dict=True, zero_division=0)
    
    # Create a figure with two subplots: Confusion Matrix and the classification report table
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
    ax_cm = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])
    
    # Confusion Matrix
    min_label = min(actual.min(), predicted.min())
    max_label = max(actual.max(), predicted.max())
    labels_cm = list(range(min_label, max_label + 1))
    cm = confusion_matrix(actual, predicted, labels=labels_cm)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_cm, yticklabels=labels_cm,
                annot_kws={"size": 12, "weight": "bold"}, cbar=False, ax=ax_cm)
    ax_cm.set_title("Global Confusion Matrix", fontsize=16, fontweight='bold')
    ax_cm.set_xlabel("Predicted Replicas", fontsize=14, fontweight='bold')
    ax_cm.set_ylabel("Actual Replicas", fontsize=14, fontweight='bold')
    
    # Classification Report Table
    ax_table.axis('off')
    
    # Extract data for table
    table_data = []
    for cls in labels_cm:
        cls_str = str(cls)
        if cls_str in report:
            row = [
                f"{report[cls_str]['precision']:.2f}",
                f"{report[cls_str]['recall']:.2f}",
                f"{report[cls_str]['f1-score']:.2f}",
                int(report[cls_str]['support'])
            ]
        else:
            row = ["0.00", "0.00", "0.00", 0]
        table_data.append(row)
        
    table_data.append(["---", "---", "---", "---"])
    table_data.append([
        f"{report['macro avg']['precision']:.2f}",
        f"{report['macro avg']['recall']:.2f}",
        f"{report['macro avg']['f1-score']:.2f}",
        int(report['macro avg']['support'])
    ])
    
    row_labels = [f"Class {lbl}" for lbl in labels_cm] + ["---", "Macro Avg"]
    col_labels = ["Precision", "Recall", "F1-Score", "Support"]
    
    table = ax_table.table(cellText=table_data, rowLabels=row_labels, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    ax_table.set_title("Precision, Recall & F1-Score per Class", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Advanced_4_Confusion_Metrics.png"), dpi=300)
    plt.close()

    # 5. Comprehensive Metrics Table
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    actual_raw = df_preds["Actual_Replicas"]
    predicted_raw = df_preds["Predicted_Raw"]
    predicted_opt = df_preds["Predicted_Replicas_Optimized"]

    raw_mae = mean_absolute_error(actual_raw, predicted_raw)
    raw_mse = mean_squared_error(actual_raw, predicted_raw)
    rmse = np.sqrt(raw_mse)
    r2 = r2_score(actual_raw, predicted_raw)
    
    mape = np.mean(np.abs((actual_raw - predicted_raw) / actual_raw)) * 100
    
    correct = (actual_raw == predicted_opt).sum()
    accuracy = correct / len(df_preds) * 100
    
    under_prov = (predicted_opt < actual_raw).sum() / len(df_preds) * 100
    over_prov = (predicted_opt > actual_raw).sum() / len(df_preds) * 100

    metrics_table = [
        ["Accuracy (Exact Match)", f"{accuracy:.2f} %"],
        ["Under-provisioning Rate (SLA Risk)", f"{under_prov:.2f} %"],
        ["Over-provisioning Rate (Waste)", f"{over_prov:.2f} %"],
        ["Raw MAE (Absolute Error)", f"{raw_mae:.4f} Replicas"],
        ["Raw MSE (Squared Error)", f"{raw_mse:.4f}"],
        ["RMSE (Root Mean Sq Error)", f"{rmse:.4f} Replicas"],
        ["MAPE (Percentage Error)", f"{mape:.2f} %"],
        ["R-squared (R²)", f"{r2:.4f}"]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    table = ax.table(cellText=metrics_table, colLabels=["Metric", "Value"], loc='center', cellLoc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.5)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2980b9')
        else:
            if col == 0:
                cell.set_text_props(weight='bold')
    
    plt.title("Comprehensive ML & Business Evaluation Metrics", fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Advanced_5_Comprehensive_Metrics.png"), dpi=300)
    plt.close()

    generate_html_dashboard(out_dir)

def generate_html_dashboard(out_dir):
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Autoscaling Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; margin: 0; padding: 20px; color: #333; }
        h1 { text-align: center; color: #2c3e50; margin-bottom: 30px; font-size: 2.5em; text-transform: uppercase; letter-spacing: 1px; }
        .dashboard-container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1400px; margin: 0 auto; }
        .card { background: white; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); padding: 20px; text-align: center; transition: transform 0.2s; }
        .card:hover { transform: translateY(-5px); }
        .card h2 { font-size: 1.5em; color: #34495e; margin-top: 0; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }
        .card img { max-width: 100%; height: auto; border-radius: 5px; }
        .full-width { grid-column: span 2; }
        .mermaid { background: #fff; padding: 20px; border-radius: 10px; box-shadow: inset 0 0 10px rgba(0,0,0,0.05); }
        .footer { text-align: center; margin-top: 40px; color: #7f8c8d; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>Microservices Autoscaling Intelligence Dashboard</h1>
    
    <div class="dashboard-container">
        <!-- 1. General Flowchart -->
        <div class="card full-width">
            <h2>1. General E-Commerce Business Flow</h2>
            <div class="mermaid">
                graph TD
                %% Styles
                classDef user fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:#fff
                classDef frontend fill:#3498db,stroke:#2980b9,stroke-width:2px,color:#fff
                classDef backend fill:#95a5a6,stroke:#7f8c8d,stroke-width:2px,color:#fff
                
                %% Nodes
                Users((Users)):::user -->|Browse & Shop| Frontend[Frontend Web]:::frontend
                
                Frontend --> Cart[Cart Service]:::backend
                Frontend --> Product[Product Catalog]:::backend
                Frontend --> Rec[Recommendation]:::backend
                
                Cart --> Checkout[Checkout Service]:::backend
                Checkout --> Payment[Payment Service]:::backend
                Checkout --> Shipping[Shipping Service]:::backend
                Checkout --> Email[Email Notification]:::backend
            </div>
            <p style="margin-top: 10px; color: #7f8c8d; font-style: italic;">High-level view of how user traffic flows through the e-commerce microservices.</p>
        </div>

        <!-- 2. Project Blueprint Flowchart -->
        <div class="card full-width">
            <h2>2. Project Execution Blueprint (Experimental Workflow)</h2>
            <div class="mermaid">
                graph TD
                %% Styles
                classDef infra fill:#34495e,stroke:#2c3e50,stroke-width:2px,color:#fff
                classDef data fill:#f39c12,stroke:#e67e22,stroke-width:2px,color:#fff
                classDef script fill:#8e44ad,stroke:#9b59b6,stroke-width:2px,color:#fff
                classDef result fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:#fff

                subgraph Phase 1: Simulation & Data Collection
                    Locust[Locust Load Generator]:::infra -->|Traffic Simulation| App[Online Boutique Microservices Demo]:::infra
                    App -->|Metrics| Prom[Prometheus]:::infra
                    Prom -->|Data Extraction| CSV1[(autoscaling_training_dataset.csv)]:::data
                end

                subgraph Phase 2: LSTM Model Training
                    CSV1 -->|70% Train / 15% Val| Train[train_lstm.py]:::script
                    Train -->|Exports Weights| Model[(best_lstm_final.pt)]:::data
                end

                subgraph Phase 3: Testing & Inference
                    CSV1 -->|15% Unseen Test Data| Predict[predict_lstm.py]:::script
                    Model -->|Loads Model| Predict
                    Predict -->|Generates Predictions & Plots| CSV2[(predictions_tau_0_45.csv + Plots)]:::data
                end

                subgraph Phase 4: Results & Evaluation
                    CSV2 --> Eval[Evaluation Metrics]:::script
                    Eval --> Final[Final Results:<br>Accuracy: 100%<br>MAE: 0.0926<br>MSE: 0.0128<br>Tau: 0.45]:::result
                end
            </div>
            <p style="margin-top: 10px; color: #7f8c8d; font-style: italic;">The end-to-end blueprint of the project: From generating traffic and collecting data, to training the model and extracting final visual results.</p>
        </div>

        <h1 class="full-width" style="margin-top: 40px; border-bottom: 2px solid #ccc; padding-bottom: 10px;">Part 1: Data Exploration & Correlations</h1>

        <div class="card">
            <h2>Base Feature Correlation Heatmap</h2>
            <img src="1_Correlation_Heatmap.png" alt="Base Correlation Heatmap">
        </div>

        <div class="card">
            <h2>Advanced Feature Correlation (Pairplot)</h2>
            <img src="Advanced_3_Pairplot.png" alt="Pairplot">
        </div>

        <h1 class="full-width" style="margin-top: 40px; border-bottom: 2px solid #ccc; padding-bottom: 10px;">Part 2: System Behavior & Workload Analysis</h1>

        <div class="card full-width">
            <h2>Locust Workload: Simulated Users vs RPS</h2>
            <img src="2_Traffic_Load.png" alt="Traffic Load">
        </div>

        <div class="card full-width">
            <h2>Advanced Timeline: Traffic Load & Autoscaling Triggers</h2>
            <img src="Advanced_1_Timeline_Events.png" alt="Timeline Graph">
        </div>

        <div class="card full-width">
            <h2>CPU Spikes Triggering Scaling Events (Cartservice)</h2>
            <img src="3_CPU_vs_Scaling.png" alt="CPU vs Scaling">
        </div>

        <div class="card full-width">
            <h2>Multi-Resource Constraints & Thresholds (Cartservice)</h2>
            <img src="Advanced_2_Resource_Thresholds.png" alt="Resource thresholds">
        </div>

        <div class="card full-width">
            <h2>Maintaining Low Latency Under Pressure</h2>
            <img src="4_Latency_Stability.png" alt="Latency Stability">
        </div>

        <div class="card full-width">
            <h2>Scaling Variation: Maximum Replicas Reached per Service</h2>
            <img src="6_Service_Scaling_Variation.png" alt="Service Scaling Variation">
        </div>

        <h1 class="full-width" style="margin-top: 40px; border-bottom: 2px solid #ccc; padding-bottom: 10px;">Part 3: AI Model Performance & Results</h1>

        <div class="card full-width">
            <h2>Comprehensive ML & Business Evaluation Metrics</h2>
            <img src="Advanced_5_Comprehensive_Metrics.png" alt="Comprehensive Metrics Table">
        </div>

        <div class="card">
            <h2>Global Model Accuracy & Regression Metrics</h2>
            <img src="5_Global_Accuracy_Donut.png" alt="Global Accuracy">
        </div>

        <div class="card">
            <h2>Base Confusion Matrix (100% Alignment)</h2>
            <img src="7_Global_Confusion_Matrix.png" alt="Base Confusion Matrix">
        </div>

        <div class="card full-width">
            <h2>Advanced Confusion Matrix & Class Metrics</h2>
            <img src="Advanced_4_Confusion_Metrics.png" alt="Confusion Matrix and Precision/Recall">
        </div>
    </div>
    
    <div class="footer">
        Generated by LSTM Autoscaler Analysis Pipeline
    </div>
    
    <script>
        mermaid.initialize({ startOnLoad: true, theme: 'default' });
    </script>
</body>
</html>
"""
    html_path = os.path.join(out_dir, "advanced_dashboard.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Successfully generated advanced dashboard at: {html_path}")

if __name__ == "__main__":
    create_advanced_plots()
