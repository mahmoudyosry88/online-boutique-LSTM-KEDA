import pandas as pd
import os

OUTPUT_DIR = r"c:\ex1\microservices-demo\lstm_autoscaler\outputs\live_comparison_results"
HPA_CSV = os.path.join(OUTPUT_DIR, "hpa_live_dataset.csv")
LSTM_CSV = os.path.join(OUTPUT_DIR, "lstm_live_dataset.csv")

hpa_df = pd.read_csv(HPA_CSV)
lstm_df = pd.read_csv(LSTM_CSV)

hpa_frontend = hpa_df[hpa_df['Service'] == 'frontend']
lstm_frontend = lstm_df[lstm_df['Service'] == 'frontend']

# Let's check SLO violation rate assuming SLO is 500ms
hpa_violation = (hpa_frontend['Latency'] > 500).mean() * 100
lstm_violation = (lstm_frontend['Latency'] > 500).mean() * 100

print(f"HPA SLO Violation (>500ms): {hpa_violation:.1f}%")
print(f"LSTM SLO Violation (>500ms): {lstm_violation:.1f}%")

# Let's check assuming SLO is 300ms
hpa_violation_300 = (hpa_frontend['Latency'] > 300).mean() * 100
lstm_violation_300 = (lstm_frontend['Latency'] > 300).mean() * 100

print(f"HPA SLO Violation (>300ms): {hpa_violation_300:.1f}%")
print(f"LSTM SLO Violation (>300ms): {lstm_violation_300:.1f}%")
