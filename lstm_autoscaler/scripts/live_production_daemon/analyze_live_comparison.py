import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR    = r"c:\ex1\microservices-demo"
OUTPUT_BASE = os.path.join(BASE_DIR, "lstm_autoscaler", "outputs", "live_comparison_results")
HPA_CSV     = os.path.join(OUTPUT_BASE, "hpa_live_dataset.csv")
LSTM_CSV    = os.path.join(OUTPUT_BASE, "lstm_live_dataset.csv")
DECISION_LOG = os.path.join(OUTPUT_BASE, "lstm_daemon_decisions.csv")

# =============================================================================
# STYLING
# =============================================================================

HPA_COLOR  = "#E63946"   # Red  — Reactive (HPA)
LSTM_COLOR = "#2A9D8F"   # Teal — Proactive (LSTM)
BG_COLOR   = "#0D1117"   # Dark background
GRID_COLOR = "#21262D"
TEXT_COLOR = "#E6EDF3"

plt.rcParams.update({
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    BG_COLOR,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "text.color":        TEXT_COLOR,
    "grid.color":        GRID_COLOR,
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
})

# =============================================================================
# METRIC CALCULATION
# =============================================================================

def calculate_metrics(df, name):
    """
    Calculates key performance indicators from the dataset.
    """
    avg_replicas  = df['target_replicas'].mean() if 'target_replicas' in df.columns else 0
    p95_latency   = df['Latency'].quantile(0.95) if len(df) > 0 else 0
    sla_violations = len(df[df['Latency'] > 500]) if len(df) > 0 else 0
    slo_compliance = 100 - ((sla_violations / len(df)) * 100) if len(df) > 0 else 100
    avg_rps        = df['RPS_frontend'].mean() if 'RPS_frontend' in df.columns else 0
    error_rate     = 0.0

    print(f"\n{'='*30}")
    print(f"  {name} Metrics")
    print(f"{'='*30}")
    print(f"  Avg Replicas  : {avg_replicas:.2f}")
    print(f"  P95 Latency   : {p95_latency:.2f} ms")
    print(f"  SLO Compliance: {slo_compliance:.2f}%")
    print(f"  Avg RPS       : {avg_rps:.2f} req/s")
    print(f"  Error Rate    : {error_rate:.2f}%")

    return {
        "Replicas":  avg_replicas,
        "P95":       p95_latency,
        "SLO":       slo_compliance,
        "RPS":       avg_rps,
        "ErrorRate": error_rate,
    }

# =============================================================================
# CHART 1: SUMMARY BAR DASHBOARD (2x2)
# =============================================================================

def plot_summary_dashboard(hpa_m, lstm_m):
    """
    2×2 bar chart dashboard — one bar per metric comparing HPA vs LSTM.
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('HPA (Reactive) vs LSTM (Proactive)\nPerformance Summary',
                 fontsize=18, fontweight='bold', color=TEXT_COLOR, y=0.98)

    metrics_spec = [
        ('P95 Latency (ms)',        'P95',      axs[0, 0], 'Lower is Better ↓',  True),
        ('SLO Compliance (%)',       'SLO',      axs[0, 1], 'Higher is Better ↑', False),
        ('Avg Replicas (Resource Cost)', 'Replicas', axs[1, 0], 'Lower is Better ↓', True),
        ('Avg Throughput (RPS)',     'RPS',      axs[1, 1], 'Higher is Better ↑', False),
    ]

    for title, key, ax, ylabel, lower_is_better in metrics_spec:
        labels = ['HPA\n(Reactive)', 'LSTM\n(Proactive)']
        values = [hpa_m[key], lstm_m[key]]

        # Highlight the winner in green, loser in red
        if lower_is_better:
            colors = [HPA_COLOR if values[0] <= values[1] else "#888",
                      LSTM_COLOR if values[1] <= values[0] else "#888"]
        else:
            colors = [HPA_COLOR if values[0] >= values[1] else "#888",
                      LSTM_COLOR if values[1] >= values[0] else "#888"]

        bars = ax.bar(labels, values, color=colors, width=0.45,
                      edgecolor=GRID_COLOR, linewidth=1.2)
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis='y')
        ax.spines[['top', 'right']].set_visible(False)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval * 1.01,
                    f'{yval:.2f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold', color=TEXT_COLOR)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_BASE, "01_summary_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {path}")

# =============================================================================
# CHART 2: TIME-SERIES OVERLAY (Latency + Replicas + RPS)
# =============================================================================

def plot_timeseries(hpa_df, lstm_df):
    """
    3-panel time-series chart showing how each system behaved over time.
    X-axis is minutes elapsed since the experiment started.
    """
    fig = plt.figure(figsize=(16, 13))
    fig.suptitle('HPA vs LSTM — Behaviour Over Time',
                 fontsize=18, fontweight='bold', color=TEXT_COLOR)
    gs = gridspec.GridSpec(3, 1, hspace=0.45)

    panels = [
        ('Latency p95 (ms)',    'Latency',         'ms'),
        ('Replicas (all svcs avg)', 'target_replicas', 'Pods'),
        ('Frontend RPS',        'RPS_frontend',    'req/s'),
    ]

    def to_minutes(df):
        """Convert Timestamp column to elapsed minutes from the first row."""
        df = df.copy()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        # Aggregate across all services per timestamp for clarity
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        df = df.groupby('Timestamp')[numeric_cols].mean().reset_index()
        df['elapsed_min'] = (
            df['Timestamp'] - df['Timestamp'].iloc[0]
        ).dt.total_seconds() / 60
        return df

    hpa_ts  = to_minutes(hpa_df)
    lstm_ts = to_minutes(lstm_df)

    for i, (title, col, unit) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        if col in hpa_ts.columns:
            ax.plot(hpa_ts['elapsed_min'], hpa_ts[col],
                    color=HPA_COLOR,  linewidth=1.8, label='HPA (Reactive)', alpha=0.9)
        if col in lstm_ts.columns:
            ax.plot(lstm_ts['elapsed_min'], lstm_ts[col],
                    color=LSTM_COLOR, linewidth=1.8, label='LSTM (Proactive)', alpha=0.9)

        ax.set_title(title, fontsize=13, pad=6)
        ax.set_ylabel(unit, fontsize=10)
        ax.set_xlabel('Elapsed Time (minutes)', fontsize=10)
        ax.legend(loc='upper right', fontsize=9,
                  facecolor=BG_COLOR, edgecolor=GRID_COLOR)
        ax.grid(True)
        ax.spines[['top', 'right']].set_visible(False)

        # Mark the burst window (minutes 10-20 of a typical 30-min test)
        ax.axvspan(10, 20, alpha=0.08, color='yellow', label='Burst Window')
        ax.text(10.3, ax.get_ylim()[1] * 0.92, '⚡ Burst',
                fontsize=9, color='yellow', alpha=0.8)

    path = os.path.join(OUTPUT_BASE, "02_timeseries_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {path}")

# =============================================================================
# CHART 3: PER-SERVICE REPLICA HEATMAP
# =============================================================================

def plot_replica_heatmap(hpa_df, lstm_df):
    """
    Side-by-side heatmaps showing average replicas per service for HPA vs LSTM.
    """
    services = [
        'adservice', 'cartservice', 'checkoutservice', 'currencyservice',
        'emailservice', 'frontend', 'paymentservice', 'productcatalogservice',
        'recommendationservice', 'shippingservice'
    ]

    def avg_replicas_per_service(df):
        if 'Service' not in df.columns or 'target_replicas' not in df.columns:
            return pd.Series({s: 1.0 for s in services})
        return df.groupby('Service')['target_replicas'].mean().reindex(services, fill_value=1.0)

    hpa_avgs  = avg_replicas_per_service(hpa_df)
    lstm_avgs = avg_replicas_per_service(lstm_df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Average Replicas Per Service',
                 fontsize=16, fontweight='bold', color=TEXT_COLOR)

    for ax, avgs, title, cmap in [
        (axes[0], hpa_avgs,  'HPA (Reactive)',   'Reds'),
        (axes[1], lstm_avgs, 'LSTM (Proactive)', 'Greens'),
    ]:
        data = avgs.values.reshape(-1, 1)
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=1, vmax=10)
        ax.set_yticks(range(len(services)))
        ax.set_yticklabels(services, fontsize=10)
        ax.set_xticks([])
        ax.set_title(title, fontsize=13, pad=10)
        plt.colorbar(im, ax=ax, label='Avg Replicas')

        for idx, val in enumerate(avgs.values):
            ax.text(0, idx, f'{val:.1f}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white')

    fig.tight_layout()
    path = os.path.join(OUTPUT_BASE, "03_replica_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {path}")

# =============================================================================
# CHART 4: LSTM DECISION LOG TIMELINE
# =============================================================================

def plot_decision_log():
    """
    Scatter plot showing every scaling decision the LSTM daemon made,
    colour-coded by action type (SCALED UP, SCALED DOWN, SKIPPED, NO-OP).
    """
    if not os.path.exists(DECISION_LOG):
        print(f"⚠️  Decision log not found at {DECISION_LOG}, skipping chart 4.")
        return

    df = pd.read_csv(DECISION_LOG)
    if df.empty:
        print("⚠️  Decision log is empty, skipping chart 4.")
        return

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['elapsed_min'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds() / 60

    action_colors = {
        'SCALED':  LSTM_COLOR,
        'NO-OP':   '#888888',
        'SKIPPED': '#F4A261',
    }

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_title('LSTM Daemon — Scaling Decision Timeline',
                 fontsize=15, fontweight='bold', color=TEXT_COLOR)

    services = df['Service'].unique()
    svc_to_y = {s: i for i, s in enumerate(sorted(services))}

    for _, row in df.iterrows():
        action_key = 'SCALED' if 'SCALED' in str(row['Action_Taken']) else row['Action_Taken']
        color = action_colors.get(action_key, '#FFFFFF')
        y_pos = svc_to_y.get(row['Service'], 0)
        ax.scatter(row['elapsed_min'], y_pos, color=color, s=60, alpha=0.85, zorder=3)

    ax.set_yticks(list(svc_to_y.values()))
    ax.set_yticklabels(list(svc_to_y.keys()), fontsize=9)
    ax.set_xlabel('Elapsed Time (minutes)', fontsize=11)
    ax.grid(True, axis='x')
    ax.spines[['top', 'right']].set_visible(False)

    # Legend
    legend_handles = [
        mpatches.Patch(color=LSTM_COLOR, label='Scale Executed'),
        mpatches.Patch(color='#888888',  label='No-Op (Already at Target)'),
        mpatches.Patch(color='#F4A261',  label='Skipped (Cooldown)'),
    ]
    ax.legend(handles=legend_handles, loc='upper right',
              facecolor=BG_COLOR, edgecolor=GRID_COLOR, fontsize=10)

    path = os.path.join(OUTPUT_BASE, "04_lstm_decisions_timeline.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Orchestrates all 4 charts and prints the final metrics comparison table.
    """
    if not os.path.exists(HPA_CSV) or not os.path.exists(LSTM_CSV):
        print(f"❌ ERROR: Dataset CSVs not found.\n   Expected:\n   {HPA_CSV}\n   {LSTM_CSV}")
        return

    print("📊 Loading datasets...")
    hpa_df  = pd.read_csv(HPA_CSV)
    lstm_df = pd.read_csv(LSTM_CSV)

    hpa_m  = calculate_metrics(hpa_df,  "HPA Baseline")
    lstm_m = calculate_metrics(lstm_df, "LSTM Daemon")

    # Delta summary
    print("\n📈 Delta (LSTM vs HPA):")
    for key, label, lower_better in [
        ('P95',      'P95 Latency',    True),
        ('SLO',      'SLO Compliance', False),
        ('Replicas', 'Avg Replicas',   True),
        ('RPS',      'Avg RPS',        False),
    ]:
        delta = lstm_m[key] - hpa_m[key]
        pct   = (delta / hpa_m[key] * 100) if hpa_m[key] != 0 else 0
        win   = (delta < 0) if lower_better else (delta > 0)
        icon  = "✅" if win else "❌"
        print(f"  {icon} {label:20s}: {delta:+.2f}  ({pct:+.1f}%)")

    print("\n🎨 Generating charts...")
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    plot_summary_dashboard(hpa_m, lstm_m)
    plot_timeseries(hpa_df, lstm_df)
    plot_replica_heatmap(hpa_df, lstm_df)
    plot_decision_log()

    print(f"\n✅ All charts saved to: {OUTPUT_BASE}")

if __name__ == "__main__":
    main()
