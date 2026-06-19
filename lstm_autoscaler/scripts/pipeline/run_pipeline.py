import os
import sys
import subprocess
import time
import json
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.config_loader import get_config

cfg = get_config()
pipeline_cfg = cfg['pipeline']
paths_cfg = cfg['paths']

BASE_DIR = paths_cfg['base_dir']
LOCUST_FILE = os.path.join(BASE_DIR, paths_cfg['locust_file'])
COLLECT_FILE = os.path.join(BASE_DIR, paths_cfg['collect_script'])
USERS_LOG_FILE = os.path.join(BASE_DIR, paths_cfg['users_log'])

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def run_cmd(cmd, wait=True):
    log(f"CMD: {cmd}")
    proc = subprocess.Popen(cmd, shell=True)
    if wait:
        proc.wait()
    return proc

def get_users_for_segment(segment):
    PATTERN = [40, 50, 70, 100, 130, 160, 180, 200, 200, 180, 150, 120, 90, 60, 40, 50, 80, 100, 140, 180, 200, 160, 120, 80, 50, 40, 60, 90, 130, 150, 180, 200, 150, 100, 70, 50]
    return PATTERN[segment % 36]

def start_port_forward():
    log("Starting persistent Prometheus port-forward on port 9090...")
    cmd = 'powershell -Command "while ($true) { kubectl port-forward svc/monitoring-kube-prometheus-prometheus 9090:9090 -n monitoring; Start-Sleep -Seconds 2 }"'
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def start_frontend_port_forward():
    log("Starting persistent Frontend port-forward on port 8080...")
    cmd = 'powershell -Command "while ($true) { kubectl port-forward svc/frontend 8080:80 -n default; Start-Sleep -Seconds 2 }"'
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_locust_segment(segment, users, mins):
    SPAWN_RATE = pipeline_cfg.get('locust_spawn_rate', 20)
    log(f"Starting segment {segment}: {users} users for {mins} minutes...")
    
    start_time = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    cmd = f"locust -f {LOCUST_FILE} --headless -u {users} -r {SPAWN_RATE} -H http://localhost:8080 -t {mins}m"
    proc = subprocess.Popen(cmd, shell=True)
    try:
        proc.wait(timeout=(mins * 60) + pipeline_cfg.get('segment_timeout_buffer_seconds', 180))
    except subprocess.TimeoutExpired:
        log("Locust timed out, forcefully killing...")
        subprocess.run("taskkill /F /T /PID " + str(proc.pid), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    end_time = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    
    # Write to users log
    with open(USERS_LOG_FILE, 'a') as f:
        json.dump({
            'segment': segment,
            'users': users,
            'start_time': start_time,
            'end_time': end_time
        }, f)
        f.write('\n')
    
    log(f"Locust segment {segment} complete.")

def main():
    log("=== STARTING 6-HOUR DATA COLLECTION PIPELINE ===")
    
    # Clear the users log file before starting
    os.makedirs(os.path.dirname(USERS_LOG_FILE), exist_ok=True)
    if os.path.exists(USERS_LOG_FILE):
        os.remove(USERS_LOG_FILE)
        
    pf_proc = start_port_forward()
    frontend_pf_proc = start_frontend_port_forward()
    time.sleep(5)
    
    total_hours = pipeline_cfg.get('total_duration_hours', 6)
    segment_minutes = pipeline_cfg.get('segment_duration_minutes', 10)
    total_seconds = total_hours * 3600
    segment_seconds = segment_minutes * 60
    total_segments = int(total_seconds // segment_seconds)
    
    for i in range(total_segments):
        users = get_users_for_segment(i)
        run_locust_segment(i, users, segment_minutes)
        
    log("Data generation complete. Starting Prometheus metric collection...")
    run_cmd(f"python {COLLECT_FILE}")
    
    log("Metric collection complete. Starting LSTM Training...")
    TRAIN_FILE = os.path.join(BASE_DIR, paths_cfg['train_script'])
    run_cmd(f"python {TRAIN_FILE}")
    
    log("LSTM Training complete. Starting LSTM Prediction and Evaluation...")
    PREDICT_FILE = os.path.join(BASE_DIR, paths_cfg['predict_script'])
    run_cmd(f"python {PREDICT_FILE}")
    
    log("Stopping port-forwards...")
    subprocess.run("taskkill /F /T /PID " + str(pf_proc.pid), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run("taskkill /F /T /PID " + str(frontend_pf_proc.pid), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    log("=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    main()
