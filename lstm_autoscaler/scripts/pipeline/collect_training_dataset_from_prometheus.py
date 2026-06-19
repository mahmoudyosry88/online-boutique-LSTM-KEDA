
import sys
import os
# Ensure the v2 root is in the Python path so we can import from config, scripts, etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import sys
sys.stdout.reconfigure(encoding='utf-8')
# =============================================================================
# collect_training_dataset_from_prometheus.py
# =============================================================================
"""
LSTM AUTOSCALER DATA COLLECTION FROM PROMETHEUS

Inputs:
    - Prometheus TSDB: The script queries raw metrics (CPU, Memory, Latency, Target Replicas) via PromQL.
    - Locust users log: A JSON log mapping time segments to the number of simulated active users.
    - Configuration files: Reads API URLs, namespaces, dataset size (hours), and microservices list from config.yaml.

Outputs:
    - autoscaling_training_dataset.csv: A highly structured, clean, and merged Pandas DataFrame saved as a CSV file.
      It contains perfectly synchronized 30-second time buckets with all features required to train the LSTM model.

Process:
    1. Querying: Sends chunked HTTP requests to the Prometheus API to fetch raw telemetry for each microservice.
    2. Parsing & Bucketing: Flattens the JSON responses and aggregates the varying time-series data into unified 30-second intervals.
    3. Merging: Performs SQL-like inner/left joins to combine all isolated metrics into a single multi-dimensional dataset.
    4. Data Cleaning: Handles missing values using forward/backward filling and caps extreme outliers.
    5. Annotation: Merges the Locust user simulation log to provide the contextual "cause" (Users) for the "effect" (Metrics).
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import os
import json
import sys
from config.config_loader import get_config

cfg = get_config()
prom_cfg = cfg['prometheus']
pipeline_cfg = cfg['pipeline']
paths_cfg = cfg['paths']
data_cfg = cfg['data']

BASE_DIR = paths_cfg['base_dir']

PROMETHEUS_URL = prom_cfg['url']
NAMESPACE      = prom_cfg['namespace']
HOURS          = pipeline_cfg.get('total_duration_hours', 6)
STEP           = prom_cfg['step']
CHUNK_HOURS    = prom_cfg['chunk_hours']

OUTPUT_FILE = os.path.join(BASE_DIR, paths_cfg['dataset_csv'])
USERS_LOG_FILE = os.path.join(BASE_DIR, paths_cfg['users_log'])

SERVICES     = list(data_cfg['services'])
SERVICE_SET   = set(SERVICES)
SERVICE_REGEX = "|".join(SERVICES)
DEPLOYMENT_TO_SERVICE = dict(data_cfg.get('service_to_deployment', {}))

# =============================================================================
# PROMETHEUS QUERY HELPER
# =============================================================================

def query_range(query, start, end, step):
    """
    Purpose: Sends an HTTP request to the Prometheus API to fetch raw time series data for a specific PromQL query within a given time window.
    """
    url = f"{PROMETHEUS_URL}/api/v1/query_range"
    params = {
        "query": query,
        "start": start.timestamp(),
        "end":   end.timestamp(),
        "step":  step,
    }
    try:
        r = requests.get(url, params=params, timeout=180)
        if r.status_code != 200:
            raise RuntimeError(f"Prometheus HTTP {r.status_code}: {r.text}")
        data = r.json()
        if data.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {data}")
        return data["data"]["result"]
    except Exception as e:
        print(f"❌ Query error: {e}")
        return []

def query_range_chunked(query, start, end, step, chunk_hours=None):
    """
    Purpose: Breaks a long time range into smaller chunks (e.g., 2 hours) to avoid overloading the Prometheus server and causing timeout errors during large data extractions.
    """
    if chunk_hours is None:
        chunk_hours = CHUNK_HOURS
    all_results = []
    current_start = start
    while current_start < end:
        current_end = min(current_start + timedelta(hours=chunk_hours), end)
        # Fetch chunk
        result = query_range(query, current_start, current_end, step)
        if result:
            if not all_results:
                all_results = result
            else:
                for series in result:
                    metric = series.get("metric", {})
                    # Find matching series in all_results
                    match = next((s for s in all_results if s.get("metric") == metric), None)
                    if match:
                        match["values"].extend(series.get("values", []))
                    else:
                        all_results.append(series)
        current_start = current_end
        time.sleep(1) # Sleep to avoid spamming the server
    return all_results

def to_utc_naive(ts):
    """Convert Unix timestamp to UTC datetime."""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).replace(tzinfo=None)

# =============================================================================
# TIME BUCKETING FUNCTIONS
# =============================================================================

def bucket_df(df, value_col):
    """
    Purpose: Unifies the time series by rounding timestamps to the nearest 30 seconds and aggregating values (using sum, mean, or last) so that different metrics can be perfectly aligned and merged later.
    """
    if df.empty:
        return pd.DataFrame(columns=["Timestamp", "Service", value_col])

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Timestamp"] = df["Timestamp"].dt.floor("30s")
    df = df[df["Service"].isin(SERVICES)]

    if value_col in ("CPU", "Memory"):
        df = df.groupby(["Timestamp", "Service"], as_index=False)[value_col].sum()
    elif value_col == "target_replicas":
        df = df.groupby(["Timestamp", "Service"], as_index=False)[value_col].last()
    else:
        df = df.groupby(["Timestamp", "Service"], as_index=False)[value_col].mean()

    return df

def bucket_rps_df(rps_result):
    """
    Purpose: Similar to bucket_df, but specifically tailored to flatten and align the frontend's Requests Per Second (RPS) data into uniform 30-second intervals.
    """
    rows = []
    for series in rps_result:
        for ts, value in series.get("values", []):
            try:
                rows.append({
                    "Timestamp":    to_utc_naive(ts),
                    "RPS_frontend": float(value),
                })
            except:
                continue

    rps_df = pd.DataFrame(rows)
    if rps_df.empty:
        return rps_df

    rps_df["Timestamp"] = pd.to_datetime(rps_df["Timestamp"])
    rps_df["Timestamp"] = rps_df["Timestamp"].dt.floor("30s")
    rps_df = rps_df.groupby("Timestamp", as_index=False)["RPS_frontend"].mean()
    return rps_df

# =============================================================================
# SERIES PARSER
# =============================================================================

def series_to_df_from_label(result, value_name, service_from_label):
    """
    Purpose: Converts the deeply nested JSON responses from Prometheus into structured, easy-to-use Pandas DataFrames, automatically mapping obscure pod names to their clean microservice names.
    """
    rows = []
    for series in result:
        metric = series.get("metric", {})
        raw    = str(metric.get(service_from_label, ""))

        svc = DEPLOYMENT_TO_SERVICE.get(raw)
        if svc is None and raw in SERVICE_SET:
            svc = raw
        if svc is None:
            for s in SERVICES:
                if s in raw:
                    svc = s
                    break
        if not svc:
            continue

        for ts, value in series.get("values", []):
            try:
                rows.append({
                    "Timestamp": to_utc_naive(ts),
                    "Service":   svc,
                    value_name:  float(value),
                })
            except:
                continue

    return pd.DataFrame(rows)

# =============================================================================
# LOAD USERS TIMELINE
# =============================================================================

def load_users_timeline():
    """
    Purpose: Reads the generated Locust log to figure out exactly how many simulated users were active at any given 30-second window, providing the essential 'cause' (user load) for our dataset.
    """
    users_map = {}
    
    if not os.path.exists(USERS_LOG_FILE):
        print(f"⚠️  Users log file not found: {USERS_LOG_FILE}")
        return users_map
    
    try:
        with open(USERS_LOG_FILE, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    start_ts = datetime.fromisoformat(data["start_time"])
                    end_ts = datetime.fromisoformat(data["end_time"])
                    users = data["users"]
                    
                    # Map every 30-second bucket in this segment to user count
                    current = start_ts.replace(microsecond=0, second=0)
                    current = current.replace(second=(current.second // 30) * 30)
                    
                    while current <= end_ts:
                        users_map[current] = users
                        current += timedelta(seconds=30)
        
        print(f"✅ Loaded users timeline: {len(users_map)} timestamps")
        return users_map
    except Exception as e:
        print(f"⚠️  Error loading users log: {e}")
        return users_map

# =============================================================================
# MAIN COLLECTION PIPELINE
# =============================================================================

def main():
    """
    Purpose: The primary orchestrator function. It sequentially queries all metrics, merges them into a single comprehensive table, cleans missing data, attaches user counts, and finally saves everything as the master CSV dataset.
    """
    end   = datetime.now(timezone.utc)
    start = end - timedelta(hours=HOURS)

    print("=" * 80)
    print("📊 LSTM AUTOSCALER DATA COLLECTION FROM PROMETHEUS")
    print("=" * 80)
    print(f"Range UTC: {start.strftime('%Y-%m-%d %H:%M:%S')} -> {end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {HOURS} hours")
    print(f"Step: {STEP}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Step 1: CPU Query
    # Purpose: Collects the total CPU usage rate across all pods of each service.
    # This is a critical metric because KEDA uses it directly to trigger scaling decisions.
    # -------------------------------------------------------------------------
    cpu_query = f"""
sum by (service) (
  label_replace(
    rate(container_cpu_usage_seconds_total{{
      namespace="{NAMESPACE}",
      container!="",
      container!="POD",
      pod=~"({SERVICE_REGEX}).*"
    }}[1m]),
    "service", "$1", "pod", "^({SERVICE_REGEX}).*"
  )
)
"""
    print("\n[1/5] Querying CPU per service (Chunked)...")
    cpu_result = query_range_chunked(cpu_query, start, end, STEP)
    cpu_df     = series_to_df_from_label(cpu_result, "CPU", "service")
    cpu_df     = bucket_df(cpu_df, "CPU")
    print(f"   ✅ CPU rows: {len(cpu_df)}")
    time.sleep(1)

    # -------------------------------------------------------------------------
    # Step 2: Memory Query
    # Purpose: Collects the working set memory bytes per service.
    # While KEDA might not scale on memory in this setup, it's a valuable contextual feature for the LSTM.
    # -------------------------------------------------------------------------
    memory_query = f"""
sum by (service) (
  label_replace(
    container_memory_working_set_bytes{{
      namespace="{NAMESPACE}",
      container!="",
      container!="POD",
      image!="",
      pod=~"({SERVICE_REGEX}).*"
    }},
    "service", "$1", "pod", "^({SERVICE_REGEX}).*"
  )
)
"""
    print("\n[2/5] Querying Memory per service (Chunked)...")
    memory_result = query_range_chunked(memory_query, start, end, STEP)
    memory_df     = series_to_df_from_label(memory_result, "Memory", "service")
    memory_df     = bucket_df(memory_df, "Memory")
    print(f"   ✅ Memory rows: {len(memory_df)}")
    time.sleep(1)

    # -------------------------------------------------------------------------
    # Step 3: Latency p95 Query
    # Purpose: Computes the 95th percentile request duration (latency) from Istio.
    # Latency spikes often precede scaling events, serving as an early-warning signal for the model.
    # -------------------------------------------------------------------------
    latency_query = f"""
label_replace(
  histogram_quantile(
    0.95,
    sum by (le, destination_service_name) (
      rate(istio_request_duration_milliseconds_bucket{{
        reporter="source",
        destination_service_namespace="{NAMESPACE}",
        destination_service_name=~"({SERVICE_REGEX})"
      }}[1m])
    )
  ),
  "service", "$1", "destination_service_name", "(.*)"
)
"""
    print("\n[3/5] Querying Latency p95 per service (Chunked)...")
    latency_result = query_range_chunked(latency_query, start, end, STEP)
    latency_df     = series_to_df_from_label(latency_result, "Latency", "service")
    latency_df     = bucket_df(latency_df, "Latency")
    print(f"   ✅ Latency rows: {len(latency_df)}")
    time.sleep(1)

    # -------------------------------------------------------------------------
    # Step 4: Frontend RPS Query
    # Purpose: Captures the total Requests Per Second (RPS) hitting the entry point (frontend).
    # Represents the raw throughput of the application workload.
    # -------------------------------------------------------------------------
    rps_query = f"""
sum(
  rate(istio_requests_total{{
    reporter="source",
    destination_service_namespace="{NAMESPACE}",
    destination_service_name="frontend"
  }}[1m])
)
"""
    print("\n[4/5] Querying RPS_frontend (Chunked)...")
    rps_result = query_range_chunked(rps_query, start, end, STEP)
    rps_df     = bucket_rps_df(rps_result)
    print(f"   ✅ RPS rows: {len(rps_df)}")
    time.sleep(1)

    # -------------------------------------------------------------------------
    # Step 5: Target Replicas Query
    # Purpose: Extracts the Ground Truth (labels) for our supervised learning task.
    # Records the exact number of pods requested by the KEDA Horizontal Pod Autoscaler.
    # -------------------------------------------------------------------------
    target_query = f"""
kube_deployment_status_replicas_available{{
  namespace="{NAMESPACE}",
  deployment=~"({SERVICE_REGEX})"
}}
"""
    print("\n[5/5] Querying target replicas per service (Chunked)...")
    target_result = query_range_chunked(target_query, start, end, STEP)
    target_df     = series_to_df_from_label(target_result, "target_replicas", "deployment")
    target_df     = bucket_df(target_df, "target_replicas")
    print(f"   ✅ Target replicas rows: {len(target_df)}")
    time.sleep(1)

    # -------------------------------------------------------------------------
    # Step 6: Merge All Metrics
    # Purpose: Joins all 5 independent DataFrames into one cohesive table using Timestamp and Service name as the primary keys.
    # -------------------------------------------------------------------------
    print("\nMerging metrics...")
    df = cpu_df.merge(memory_df,  on=["Timestamp", "Service"], how="inner")
    df = df.merge(latency_df,     on=["Timestamp", "Service"], how="inner")
    df = df.merge(target_df,      on=["Timestamp", "Service"], how="inner")

    if not rps_df.empty:
        df = df.merge(rps_df, on="Timestamp", how="left")

    df = df.sort_values(["Timestamp", "Service"]).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Step 7: Data Cleaning & Validation
    # Purpose: Ensures data quality by converting types, filling gaps (ffill/bfill), dropping irreparable rows, and clipping extreme latency outliers.
    # -------------------------------------------------------------------------
    for col in ["CPU", "Memory", "Latency", "RPS_frontend", "target_replicas"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["RPS_frontend"] = df["RPS_frontend"].fillna(0)

    df[["CPU", "Memory", "Latency"]] = (
        df.groupby("Service")[["CPU", "Memory", "Latency"]]
          .ffill()
          .bfill()
    )

    df["target_replicas"] = (
        df["target_replicas"]
          .fillna(1)
          .clip(lower=1)
          .round()
    )

    df = df.dropna(subset=["CPU", "Memory", "Latency", "target_replicas"])

    latency_cap = df["Latency"].quantile(0.99)
    df["Latency"] = df["Latency"].clip(upper=latency_cap)

    # -------------------------------------------------------------------------
    # Step 8: Add Users Column
    # -------------------------------------------------------------------------
    print("\n[6/6] Adding Users column...")
    users_map = load_users_timeline()

    if users_map:
        df["Timestamp_30s"] = df["Timestamp"].dt.floor("30s")
        df["Users"] = df["Timestamp_30s"].apply(
            lambda ts: users_map.get(ts, 100)
        )
        df = df.drop("Timestamp_30s", axis=1)
        print(f"   ✅ Users column added. Range: {df['Users'].min()} - {df['Users'].max()}")
    else:
        print("   ⚠️  No users data available, defaulting to 100")
        df["Users"] = 100

    # -------------------------------------------------------------------------
    # Step 9: Save Output
    # -------------------------------------------------------------------------
    out = df[[
        "Timestamp", "Service", "CPU", "Memory", "Latency",
        "RPS_frontend", "target_replicas", "Users"
    ]].copy()

    out = out.sort_values(["Timestamp", "Service"]).reset_index(drop=True)
    out.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Done. Saved: {OUTPUT_FILE}")
    print(f"   Total rows: {len(out)}")

    print("\n📊 Target replicas distribution:")
    print(out["target_replicas"].value_counts().sort_index())

    print("\n📊 Target replicas per service:")
    print(out.groupby("Service")["target_replicas"]
            .agg(["min", "max", "mean", "nunique"])
            .round(2))

    print("\n📊 Users distribution:")
    print(out["Users"].value_counts().sort_index())

    print("\n📊 Users per service:")
    print(out.groupby("Service")["Users"]
            .agg(["min", "max", "mean"])
            .round(2))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)