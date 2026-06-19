import pandas as pd
import os

OUTPUT_DIR = r"c:\ex1\microservices-demo\lstm_autoscaler\outputs\live_comparison_results"
HPA_CSV = os.path.join(OUTPUT_DIR, "hpa_live_dataset.csv")
LSTM_CSV = os.path.join(OUTPUT_DIR, "lstm_live_dataset.csv")

hpa_df = pd.read_csv(HPA_CSV)
lstm_df = pd.read_csv(LSTM_CSV)

hpa_frontend = hpa_df[hpa_df['Service'] == 'frontend']
lstm_frontend = lstm_df[lstm_df['Service'] == 'frontend']

print("=== FRONTEND COMPARISON ===")
print("HPA Max Replicas:", hpa_frontend['target_replicas'].max())
print("LSTM Max Replicas:", lstm_frontend['target_replicas'].max())
print("HPA Avg Replicas:", round(hpa_frontend['target_replicas'].mean(), 2))
print("LSTM Avg Replicas:", round(lstm_frontend['target_replicas'].mean(), 2))

print("\nHPA Max Latency (ms):", round(hpa_frontend['Latency'].max(), 2))
print("LSTM Max Latency (ms):", round(lstm_frontend['Latency'].max(), 2))
print("HPA Avg Latency (ms):", round(hpa_frontend['Latency'].mean(), 2))
print("LSTM Avg Latency (ms):", round(lstm_frontend['Latency'].mean(), 2))

print("\n=== CLUSTER WIDE REPLICAS ===")
print("HPA Total Replicas Avg:", round(hpa_df.groupby('Timestamp')['target_replicas'].sum().mean(), 2))
print("LSTM Total Replicas Avg:", round(lstm_df.groupby('Timestamp')['target_replicas'].sum().mean(), 2))
