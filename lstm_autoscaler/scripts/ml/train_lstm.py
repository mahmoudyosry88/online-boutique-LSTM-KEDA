"""
LSTM Training Script

Inputs:
    - autoscaling_training_dataset.csv: The master dataset containing normalized metrics and target_replicas per service.
    - config.yaml: Training hyperparameters (epochs, learning rate, batch size, etc.).

Outputs:
    - best_lstm_final.pt: A saved PyTorch model checkpoint containing the trained model weights and data scalers.

Process:
    1. Loads and splits the dataset chronologically into Train (70%), Validation (15%), and Test (15%).
    2. Fits standard scalers strictly on the training set to prevent data leakage.
    3. Generates sliding window sequences (e.g., 10-minute lookback) for each microservice independently.
    4. Trains the PyTorch LSTM model using MSE loss, Adam optimizer, and an adaptive learning rate scheduler.
    5. Evaluates on the validation set after each epoch, applying early stopping to prevent overfitting, and saves the best model.
"""

import sys
import os
# Ensure the v2 root is in the Python path so we can import from config, scripts, etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os
from config.config_loader import get_config

cfg = get_config()
train_cfg = cfg['training']
data_cfg = cfg['data']
paths_cfg = cfg['paths']
base_dir = paths_cfg['base_dir']

csv_path = os.path.join(base_dir, paths_cfg['dataset_csv'])
SEQ_LEN = train_cfg['seq_len']
BATCH_SIZE = train_cfg['batch_size']
EPOCHS = train_cfg['epochs']
HIDDEN_SIZE = train_cfg['hidden_size']
NUM_LAYERS = train_cfg['num_layers']
DROPOUT = train_cfg['dropout']
LR = train_cfg['lr']
PATIENCE_EARLY = train_cfg['early_stopping_patience']
LR_PATIENCE = train_cfg['lr_scheduler_patience']
LR_FACTOR = train_cfg['lr_scheduler_factor']
feature_cols = list(data_cfg['feature_cols'])
target_col = data_cfg['target_col']
service_col = data_cfg['service_col']

print(f"Loading CSV: {csv_path}")
df = pd.read_csv(csv_path)
df = df.sort_values(['Service', 'Timestamp']).reset_index(drop=True)

df['service_id'] = df[service_col].astype('category').cat.codes
constant_services = df.groupby(service_col)[target_col].nunique().eq(1)
df['is_constant'] = df[service_col].map(lambda x: 1.0 if constant_services[x] else 0.0)

# ---------------------------------------------------------
# Academic Train/Val/Test Chronological Split (70-15-15)
# ---------------------------------------------------------
df["ts_int"] = pd.to_datetime(df["Timestamp"], errors="coerce").astype("int64") // 10**9
t_min, t_max = df["ts_int"].min(), df["ts_int"].max()

train_frac = train_cfg['train_frac']
val_frac = train_cfg['val_frac']
# The remaining 0.15 is implicitly for test

t_train_end = t_min + (t_max - t_min) * train_frac
t_val_end = t_min + (t_max - t_min) * (train_frac + val_frac)

train_mask = df["ts_int"] <= t_train_end
val_mask = (df["ts_int"] > t_train_end) & (df["ts_int"] <= t_val_end)

print(f"Split sizes -> Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {(~train_mask & ~val_mask).sum()}")

x_scaler = StandardScaler()
y_scaler = StandardScaler()

# ---------------------------------------------------------
# Prevent Data Leakage: Fit scaler ONLY on the train set
# ---------------------------------------------------------
train_df = df[train_mask]
x_scaler.fit(train_df[feature_cols + ['service_id', 'is_constant']].values)
y_scaler.fit(train_df[target_col].values.reshape(-1, 1))

# Transform all data using the train-fitted scalers
X_all = df[feature_cols + ['service_id', 'is_constant']].values
Y_all = df[target_col].values

X_scaled = x_scaler.transform(X_all)
Y_scaled = y_scaler.transform(Y_all.reshape(-1, 1)).flatten()

def create_sequences_split(mask, X, Y, services, seq_len):
    """
    Purpose: Transforms flat time-series arrays into sliding-window sequences for the LSTM, ensuring that sequences do not cross between different microservices or disjoint chronological splits.
    """
    X_seq, Y_seq = [], []
    for svc in services:
        svc_mask = df[service_col].values == svc
        # Combine valid time split and service
        valid_mask = mask.values & svc_mask
        indices = np.where(valid_mask)[0]
        
        if len(indices) <= seq_len:
            continue
            
        x_svc = X[valid_mask]
        y_svc = Y[valid_mask]
        
        for i in range(len(x_svc) - seq_len):
            X_seq.append(x_svc[i:i + seq_len])
            Y_seq.append(y_svc[i + seq_len])
    return np.array(X_seq), np.array(Y_seq)

services = df[service_col].unique()

print("Building Train and Validation sequences...")
X_train, Y_train = create_sequences_split(train_mask, X_scaled, Y_scaled, services, SEQ_LEN)
X_val, Y_val = create_sequences_split(val_mask, X_scaled, Y_scaled, services, SEQ_LEN)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

if len(X_val) > 0:
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
else:
    val_loader = []

from scripts.ml.model import GlobalLSTMRegressor


num_features = X_train.shape[2] if len(X_train) > 0 else 7
model = GlobalLSTMRegressor(num_features, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE)

print("Starting Training Loop...")
best_val_loss = float('inf')
best_state = None
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        
    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb)
            loss = criterion(out, yb)
            val_loss += loss.item()
            
    train_loss_avg = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    val_loss_avg = val_loss / len(val_loader) if len(val_loader) > 0 else 0
    scheduler.step(val_loss_avg)
    print(f'Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss_avg:.6f} | Val Loss: {val_loss_avg:.6f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

    # Early stopping check
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        best_state = model.state_dict()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE_EARLY:
            print(f'Early stopping triggered after {epoch+1} epochs (no improvement for {PATIENCE_EARLY} epochs)')
            break

# Restore best weights
if best_state is not None:
    model.load_state_dict(best_state)

print("Saving complete model checkpoint for predict_lstm.py...")
torch.save({
    "model_state_dict": model.state_dict(),
    "params": {
        "hidden_size": HIDDEN_SIZE, 
        "num_layers": NUM_LAYERS, 
        "dropout": DROPOUT,
        "train_frac": train_frac,
        "val_frac": val_frac
    },
    "num_features": num_features,
    "x_scaler_mean": x_scaler.mean_,
    "x_scaler_scale": x_scaler.scale_,
    "y_scaler_mean": y_scaler.mean_,
    "y_scaler_scale": y_scaler.scale_,
    "feature_cols": feature_cols,
    "lookback": SEQ_LEN,
    "service_ids": dict(enumerate(df[service_col].astype('category').cat.categories)),
    "target_col": target_col,
    "constant_services": set(df[df['is_constant'] == 1.0][service_col].unique())
}, os.path.join(base_dir, paths_cfg['checkpoint']))

print('Done. Model checkpoint saved successfully. Ready for inference.')
