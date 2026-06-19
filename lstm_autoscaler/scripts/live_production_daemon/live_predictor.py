#!/usr/bin/env python3
import sys
import os
import time
import csv
import requests
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import subprocess

# Ensure the project root is in the Python path to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Import the model loader and rounding logic from our ML pipeline
from scripts.ml.predict_lstm import load_checkpoint, apply_rounding_threshold

# =============================================================================
# CONFIGURATION
# =============================================================================

# Define the local Prometheus endpoint where metrics will be scraped from
PROMETHEUS_URL = "http://localhost:9090"
# Target Kubernetes namespace
NAMESPACE = "default"
# Absolute path to the trained LSTM model file
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'best_lstm_final.pt')
# The rounding threshold (Tau). A value of 0.45 makes the model slightly aggressive in scaling up
TAU = 0.45
SCALEDOBJECT_SUFFIX = "-prom-cpu"
# Interval in seconds to poll Prometheus and run inference (Must match the 30s bucket size from training)
POLL_INTERVAL = 30

# =============================================================================
# SAFETY GUARDRAILS
# =============================================================================

# Guardrail 1: COOLDOWN  Minimum seconds between scaling decisions PER SERVICE.
# Prevents rapid oscillation (e.g., scale up then immediately scale down).
# Kubernetes itself takes ~30-60s to spin up a pod, so acting faster is pointless.
SCALE_COOLDOWN_SECONDS = 120  # 2 minutes

# Guardrail 2: MAX DELTA  Maximum number of pods to add/remove in a single decision.
# Prevents the model from making catastrophic jumps (e.g., 1 -> 10 pods at once).
MAX_DELTA_PER_CYCLE = 3

# Guardrail 3: HARD LIMITS  Absolute bounds the model can never violate.
MIN_REPLICAS = 1
MAX_REPLICAS = 10

# Guardrail 4: PROMETHEUS TIMEOUT  If Prometheus doesn't respond within N seconds,
# skip the current cycle entirely (keep current replicas, do nothing).
PROMETHEUS_TIMEOUT_SECONDS = 5

# =============================================================================
# SERVICES & QUERIES
# =============================================================================

# List of all microservices in the boutique application
SERVICES = ["adservice", "cartservice", "checkoutservice", "currencyservice",
            "emailservice", "frontend", "paymentservice", "productcatalogservice",
            "recommendationservice", "shippingservice"]

# Define PromQL queries to fetch the live instantaneous metrics for the model
QUERIES = {
    # Calculate CPU usage rate over the last 1 minute
    "CPU": f'sum by (pod) (rate(container_cpu_usage_seconds_total{{namespace="{NAMESPACE}", container!="", container!="POD"}}[1m]))',
    # Fetch the current memory working set size
    "Memory": f'sum by (pod) (container_memory_working_set_bytes{{namespace="{NAMESPACE}", container!="", container!="POD"}})',
    # Calculate the 95th percentile latency over the last 1 minute
    "Latency": f'histogram_quantile(0.95, sum by (le, destination_service_name) (rate(istio_request_duration_milliseconds_bucket{{reporter="source", destination_service_namespace="{NAMESPACE}"}}[1m])))',
    # Calculate the total Requests Per Second (RPS) hitting the frontend
    "RPS_frontend": f'sum(rate(istio_requests_total{{reporter="source", destination_service_namespace="{NAMESPACE}", destination_service_name="frontend"}}[1m]))'
}

# =============================================================================
# STATE TRACKING
# =============================================================================

# Keep track of last scaled replicas to avoid redundant API calls to Kubernetes
last_scaled_replicas = {svc: 1 for svc in SERVICES}

# Guardrail 1 state: timestamp (epoch seconds) of the last scale action per service
last_scale_time = {svc: 0.0 for svc in SERVICES}

# =============================================================================
# DECISION LOG (CSV)
# =============================================================================

# Path for the structured decision log  records every prediction the daemon makes
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'live_comparison_results')
os.makedirs(LOG_DIR, exist_ok=True)
DECISION_LOG_PATH = os.path.join(LOG_DIR, 'lstm_daemon_decisions.csv')

def init_decision_log():
    """
    Creates the CSV decision log file and writes the header row.
    Called once at daemon startup.
    """
    with open(DECISION_LOG_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Timestamp', 'Service', 'Raw_Prediction', 'Rounded_Prediction',
            'Clamped_Prediction', 'Previous_Replicas', 'Action_Taken',
            'Skip_Reason'
        ])
    print(f" Decision log initialized: {DECISION_LOG_PATH}")

def log_decision(service, raw_pred, rounded_pred, clamped_pred,
                 previous_replicas, action_taken, skip_reason=""):
    """
    Appends a single scaling decision row to the CSV log.
    Records both executed and skipped decisions for full transparency.
    """
    with open(DECISION_LOG_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            service,
            f"{raw_pred:.4f}" if raw_pred is not None else "N/A",
            rounded_pred if rounded_pred is not None else "N/A",
            clamped_pred if clamped_pred is not None else "N/A",
            previous_replicas,
            action_taken,
            skip_reason
        ])

# =============================================================================
# PROMETHEUS HELPERS
# =============================================================================

def query_prometheus(query_str):
    """
    Executes an HTTP GET request to Prometheus with the given PromQL query.
    Returns None on timeout or error (Guardrail 4: Fallback).
    """
    try:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query_str},
            timeout=PROMETHEUS_TIMEOUT_SECONDS
        )
        if r.status_code == 200:
            return r.json().get('data', {}).get('result', [])
    except requests.exceptions.Timeout:
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Prometheus TIMEOUT  skipping cycle (Guardrail 4)")
        return None  # None signals a timeout/failure to the caller
    except Exception as e:
        print(f"  Prometheus query failed: {e}")
        return None
    return []

def get_service_name(pod_name):
    """
    Extracts the base service name from a dynamically generated Kubernetes Pod name.
    """
    for svc in SERVICES:
        if svc in pod_name:
            return svc
    return None

def fetch_current_metrics():
    """
    Fetches the current metrics for all services and aggregates them into a Pandas DataFrame.
    Returns None if any Prometheus query times out (triggering the fallback guardrail).
    """
    # Fetch CPU data
    cpu_data = query_prometheus(QUERIES["CPU"])
    if cpu_data is None:
        return None  # Propagate timeout signal up

    cpu_map = {svc: 0.0 for svc in SERVICES}
    for res in cpu_data:
        pod = res['metric'].get('pod', '')
        svc = get_service_name(pod)
        if svc:
            cpu_map[svc] += float(res['value'][1])

    # Fetch Memory data
    mem_data = query_prometheus(QUERIES["Memory"])
    if mem_data is None:
        return None

    mem_map = {svc: 0.0 for svc in SERVICES}
    for res in mem_data:
        pod = res['metric'].get('pod', '')
        svc = get_service_name(pod)
        if svc:
            mem_map[svc] += float(res['value'][1])

    # Fetch Latency data
    lat_data = query_prometheus(QUERIES["Latency"])
    if lat_data is None:
        return None

    lat_map = {svc: 0.0 for svc in SERVICES}
    for res in lat_data:
        svc_dest = res['metric'].get('destination_service_name', '')
        if svc_dest in lat_map:
            val = float(res['value'][1])
            if not np.isnan(val):
                lat_map[svc_dest] = val

    # Fetch global frontend RPS
    rps_data = query_prometheus(QUERIES["RPS_frontend"])
    if rps_data is None:
        return None

    rps_val = (
        float(rps_data[0]['value'][1])
        if rps_data and len(rps_data) > 0 and not np.isnan(float(rps_data[0]['value'][1]))
        else 0.0
    )

    # Build a list of metric dictionaries for each service
    metrics = []
    for svc in SERVICES:
        metrics.append({
            "Service": svc,
            "CPU": cpu_map[svc],
            "Memory": mem_map[svc],
            "Latency": lat_map[svc],
            "RPS_frontend": rps_val,
            "Users": 100  # Dummy value required by the model's expected feature shape
        })
    return pd.DataFrame(metrics)

# =============================================================================
# KUBERNETES ACTUATION
# =============================================================================

def actuate_kubernetes(service, target_replicas, raw_pred, rounded_pred):
    """
    Applies all safety guardrails, then executes 'kubectl scale' if appropriate.

    Guardrail 1  Cooldown: Skip if we scaled this service recently.
    Guardrail 2  Max Delta: Clamp the change to MAX_DELTA_PER_CYCLE pods.
    Guardrail 3  Hard Limits: Enforce MIN_REPLICAS / MAX_REPLICAS bounds.
    """
    global last_scaled_replicas, last_scale_time
    now = time.time()
    previous = last_scaled_replicas.get(service, 1)

    # --- Guardrail 1: Cooldown check ---
    seconds_since_last_scale = now - last_scale_time.get(service, 0.0)
    if seconds_since_last_scale < SCALE_COOLDOWN_SECONDS:
        remaining = int(SCALE_COOLDOWN_SECONDS - seconds_since_last_scale)
        log_decision(service, raw_pred, rounded_pred, None,
                     previous, "SKIPPED", f"Cooldown: {remaining}s remaining")
        return

    # --- Guardrail 2: Max Delta clamp ---
    delta = target_replicas - previous
    if abs(delta) > MAX_DELTA_PER_CYCLE:
        # Clamp: move by at most MAX_DELTA_PER_CYCLE in the correct direction
        clamped = previous + (MAX_DELTA_PER_CYCLE if delta > 0 else -MAX_DELTA_PER_CYCLE)
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {service}: Delta clamped "
              f"{previous}{target_replicas} limited to {previous}{clamped} (Guardrail 2)")
    else:
        clamped = target_replicas

    # --- Guardrail 3: Hard limits ---
    clamped = max(MIN_REPLICAS, min(clamped, MAX_REPLICAS))

    # NO-OP: Already at the desired scale
    if previous == clamped:
        log_decision(service, raw_pred, rounded_pred, clamped,
                     previous, "NO-OP", "Already at target replicas")
        return

    # All guardrails passed  execute the scale command
    direction = " UP" if clamped > previous else " DOWN"
    print(f"[{datetime.now().strftime('%H:%M:%S')}]  SCALE {direction}: "
          f"{service} {previous}  {clamped} pods  (raw={raw_pred:.3f})")

    scaledobject_name = f"{service}{SCALEDOBJECT_SUFFIX}"
    cmd = f'kubectl patch scaledobject {scaledobject_name} -n {NAMESPACE} --type=merge -p="{{\\"spec\\":{{\\"minReplicaCount\\":{clamped}}}}}"'
    subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Update state
    last_scaled_replicas[service] = clamped
    last_scale_time[service] = now

    log_decision(service, raw_pred, rounded_pred, clamped,
                 previous, f"SCALED ({previous}{clamped})", "")

# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    """
    The main continuous loop that drives the AI autoscaler daemon.
    """
    print("=" * 60)
    print("  LSTM Production Daemon  Starting")
    print(f"   Guardrail 1: Cooldown     = {SCALE_COOLDOWN_SECONDS}s per service")
    print(f"   Guardrail 2: Max Delta    = {MAX_DELTA_PER_CYCLE} pods per cycle")
    print(f"   Guardrail 3: Hard Limits  = [{MIN_REPLICAS}, {MAX_REPLICAS}] pods")
    print(f"   Guardrail 4: Prom Timeout = {PROMETHEUS_TIMEOUT_SECONDS}s")
    print("=" * 60)

    # Initialize the structured decision log
    init_decision_log()

    # Select GPU if available, otherwise fallback to CPU for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Device: {device}")

    # Load the trained model architecture, weights, and scalers
    model, ckpt, x_scaler, y_scaler = load_checkpoint(CHECKPOINT_PATH, device)

    # Extract metadata from the model checkpoint
    feature_cols = ckpt["feature_cols"]
    lookback = ckpt["lookback"]
    service_ids = ckpt["service_ids"]

    # Ensure service_ids keys are strings
    if 0 in service_ids or "0" in service_ids:
        service_ids = {v: int(k) for k, v in service_ids.items()}
    constant_services = ckpt.get("constant_services", set())

    # Create a history buffer to store the recent sequences needed for the LSTM lookback window
    history_buffer = {svc: [] for svc in SERVICES}

    print(f" Daemon ready. Polling every {POLL_INTERVAL}s | Lookback window: {lookback} steps")
    print(f" Decisions logged to: {DECISION_LOG_PATH}")
    print("-" * 60)

    # Infinite loop to continuously monitor and scale
    while True:
        try:
            start_time = time.time()

            # Fetch fresh data from Prometheus
            df_current = fetch_current_metrics()

            # Guardrail 4: Prometheus timeout  skip the entire cycle
            if df_current is None:
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] Cycle skipped  Prometheus unavailable")
                time.sleep(POLL_INTERVAL)
                continue

            # Process each service independently
            for svc in SERVICES:
                svc_df = df_current[df_current['Service'] == svc]
                if svc_df.empty:
                    continue

                # Ensure all required feature columns exist, fill missing ones with 0.0
                for col in feature_cols:
                    if col not in svc_df.columns:
                        svc_df = svc_df.copy()
                        svc_df[col] = 0.0

                # Extract the raw feature row as a NumPy array
                feat = svc_df[feature_cols].values.astype(np.float32)[0]

                # Append the new data to the history buffer
                history_buffer[svc].append(feat)
                # Pad buffer for instant warmup
                if len(history_buffer[svc]) == 1:
                    history_buffer[svc] = history_buffer[svc] * lookback
                if len(history_buffer[svc]) > lookback:
                    history_buffer[svc].pop(0)

                # The model can only predict once the buffer is fully populated
                if len(history_buffer[svc]) < lookback:
                    remaining_steps = lookback - len(history_buffer[svc])
                    print(f" [{svc}] Warming up buffer: {remaining_steps} more steps needed")
                    continue

                # Build the input tensor
                window_feat = np.array(history_buffer[svc])
                svc_id    = np.full((lookback, 1), service_ids.get(svc, 0), dtype=np.float32)
                is_const  = np.full((lookback, 1), 1.0 if svc in constant_services else 0.0, dtype=np.float32)
                feat_with_meta = np.hstack([window_feat, svc_id, is_const])

                # Normalize + inference
                feat_scaled  = x_scaler.transform(feat_with_meta).astype(np.float32)
                tensor_batch = torch.from_numpy(feat_scaled).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred_scaled = model(tensor_batch).cpu().numpy()

                # Inverse-transform and round
                raw_pred     = float(y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0])
                rounded_pred = apply_rounding_threshold(raw_pred, TAU)

                # Apply all guardrails and (if safe) execute kubectl scale
                actuate_kubernetes(svc, rounded_pred, raw_pred, rounded_pred)

            # Maintain a steady polling interval
            elapsed    = time.time() - start_time
            sleep_time = max(0, POLL_INTERVAL - elapsed)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n Daemon stopped manually.")
            break
        except Exception as e:
            print(f" Daemon Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()



