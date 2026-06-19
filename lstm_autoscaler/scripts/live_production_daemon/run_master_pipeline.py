import os
import sys
import subprocess
import time
import json
from datetime import datetime, timezone, timedelta

BASE_DIR = r"c:\ex1\microservices-demo"
KUBE_MANIFESTS = os.path.join(BASE_DIR, "kubernetes-manifests")
HPA_YAML = os.path.join(KUBE_MANIFESTS, "hpa-baseline.yaml")
KEDA_YAML = os.path.join(KUBE_MANIFESTS, "scaled_objects_all_services.yaml")
MODEL_CHECKPOINT = os.path.join(BASE_DIR, "lstm_autoscaler", "models", "best_lstm_final.pt")

LIVE_DIR = os.path.join(BASE_DIR, "lstm_autoscaler", "scripts", "live_production_daemon")
LOCUST_FILE = os.path.join(LIVE_DIR, "locustfile.py")
COLLECT_FILE = os.path.join(LIVE_DIR, "collect_metrics_live.py")
PREDICTOR_FILE = os.path.join(LIVE_DIR, "live_predictor.py")

OUTPUT_DIR = os.path.join(BASE_DIR, "lstm_autoscaler", "outputs", "live_comparison_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HPA_CSV = os.path.join(OUTPUT_DIR, "hpa_live_dataset.csv")
LSTM_CSV = os.path.join(OUTPUT_DIR, "lstm_live_dataset.csv")
HPA_JSON = os.path.join(OUTPUT_DIR, "hpa_users_log.json")
LSTM_JSON = os.path.join(OUTPUT_DIR, "lstm_users_log.json")
DEFAULT_CSV = os.path.join(BASE_DIR, "lstm_autoscaler", "data", "processed", "autoscaling_training_dataset.csv")

def start_port_forward():
    print("Starting persistent Prometheus port-forward on port 9090...", flush=True)
    cmd = 'powershell -Command "while ($true) { kubectl port-forward svc/monitoring-kube-prometheus-prometheus 9090:9090 -n monitoring; Start-Sleep -Seconds 2 }"'
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def start_frontend_port_forward():
    print("Starting persistent Frontend port-forward on port 8080...", flush=True)
    cmd = 'powershell -Command "while ($true) { kubectl port-forward svc/frontend 8080:80 -n default; Start-Sleep -Seconds 2 }"'
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

WORKLOAD_SEGMENTS = [
    (40, 5),    # Steady
    (800, 10),  # Burst
    (40, 7)     # Recovery
]

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def run_cmd(cmd, wait=True):
    log(f"CMD: {cmd}")
    proc = subprocess.Popen(cmd, shell=True)
    if wait:
        proc.wait()
    return proc

def run_locust_segment(users, mins):
    SPAWN_RATE = 10 # increased spawn rate for 800 users so it bursts faster
    log(f"Starting Locust segment: {users} users for {mins} minutes...")
    cmd = f"locust -f {LOCUST_FILE} --headless -u {users} -r {SPAWN_RATE} -H http://localhost:8080 -t {mins}m"
    proc = subprocess.Popen(cmd, shell=True)
    try:
        proc.wait(timeout=(mins * 60) + 10)
    except subprocess.TimeoutExpired:
        log("Locust timed out, forcefully killing...")
        subprocess.run("taskkill /F /T /PID " + str(proc.pid), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log("Locust segment complete.")

def run_locust_segments(log_file):
    timeline = []
    current_time = datetime.now(timezone.utc).replace(microsecond=0)
    for users, duration_mins in WORKLOAD_SEGMENTS:
        end_time = current_time + timedelta(minutes=duration_mins)
        timeline.append({
            "start_time": current_time.isoformat(),
            "end_time": end_time.isoformat(),
            "users": users
        })
        current_time = end_time
    
    with open(log_file, "w", encoding="utf-8") as f:
        for t in timeline:
            f.write(json.dumps(t) + "\n")
            
    for users, duration_mins in WORKLOAD_SEGMENTS:
        run_locust_segment(users, duration_mins)

def collect_and_save(target_path, json_path):
    log("Configuring collection script for this run...")
    with open(COLLECT_FILE, 'r', encoding='utf-8') as f:
        code = f.read()
    import re
    # Update collect script to use the correct target file and json log, and fetch last 0.4 hours
    code = re.sub(r"OUTPUT_FILE\s*=\s*r'.*'", lambda m: f"OUTPUT_FILE = r'{target_path}'", code)
    code = re.sub(r"USERS_LOG_FILE\s*=\s*r'.*'", lambda m: f"USERS_LOG_FILE = r'{json_path}'", code)
    code = re.sub(r"HOURS\s*=\s*.*", "HOURS = 0.5", code)
    with open(COLLECT_FILE, 'w', encoding='utf-8') as f:
        f.write(code)

    log("Collecting metrics from Prometheus...")
    run_cmd(f"python {COLLECT_FILE}")
    if os.path.exists(target_path):
        log(f"Saved dataset to {target_path}")
    else:
        log("ERROR: Data collection failed, CSV not found!")

def main():
    log("=== STARTING THE 800-USER LIVE EXECUTION PIPELINE ===")
    
    run_cmd("Stop-Process -Name locust -Force -ErrorAction SilentlyContinue")

    pf_proc = start_port_forward()
    frontend_pf_proc = start_frontend_port_forward()
    time.sleep(5)
    
    # STEP 1: Apply HPA Baseline
    log("\n[STEP 1] Applying HPA Baseline...")
    run_cmd("kubectl delete hpa --all -n default")
    run_cmd("kubectl delete scaledobject --all -n default")
    run_cmd(f"kubectl apply -f {HPA_YAML}")
    log("Waiting 30 seconds for HPA to initialize...")
    time.sleep(30)
    
    # STEP 2: Run HPA Phase
    log("\n[STEP 2] Running HPA Workload (22 mins)...")
    run_locust_segments(HPA_JSON)
    collect_and_save(HPA_CSV, HPA_JSON)
    
    # STEP 3: The Purge & Stabilization
    log("\n[STEP 3] The Purge: Deleting HPA...")
    run_cmd("kubectl delete hpa --all -n default")
    log("Stabilization Phase: Waiting 5 minutes for cluster CPU to cool down...")
    time.sleep(300)
    
    # STEP 4: Live LSTM Phase
    log("\n[STEP 4] Starting Live LSTM Daemon Phase...")
    run_cmd(f"kubectl apply -f {KEDA_YAML}")
    
    log("Starting live_predictor.py in the background...")
    daemon_proc = subprocess.Popen(f"python -u {PREDICTOR_FILE} > lstm_keda_debug.log 2>&1", shell=True)
    time.sleep(15)
    
    log("Running LSTM Workload (22 mins)...")
    run_locust_segments(LSTM_JSON)
    
    log("Stopping Live LSTM Daemon...")
    subprocess.run("taskkill /F /T /PID " + str(daemon_proc.pid), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    collect_and_save(LSTM_CSV, LSTM_JSON)
    
    # STEP 5: Analysis
    log("\n[STEP 5] Generating Comparison Plots...")
    run_cmd(f"python plot_all_services_twinx.py")
    
    log("Stopping port-forwards...")
    subprocess.run("taskkill /F /T /PID " + str(pf_proc.pid), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run("taskkill /F /T /PID " + str(frontend_pf_proc.pid), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    log("=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    main()
