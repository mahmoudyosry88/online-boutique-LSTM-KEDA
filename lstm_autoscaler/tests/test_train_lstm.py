import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pandas as pd
import tempfile
import unittest

from torch.utils.data import DataLoader, TensorDataset


def create_sequences(X, Y, services, service_col, df, seq_len):
    X_seq, Y_seq = [], []
    for svc in services:
        mask = df[service_col].values == svc
        x_svc = X[mask]
        y_svc = Y[mask]
        if len(x_svc) <= seq_len:
            continue
        for i in range(len(x_svc) - seq_len):
            X_seq.append(x_svc[i:i + seq_len])
            Y_seq.append(y_svc[i + seq_len])
    return np.array(X_seq), np.array(Y_seq)


class TestTrainLSTM(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'Service': ['svc_a', 'svc_a', 'svc_a', 'svc_a', 'svc_a',
                        'svc_b', 'svc_b', 'svc_b', 'svc_b', 'svc_b'],
            'Timestamp': pd.date_range('2026-01-01', periods=10, freq='30s'),
            'CPU': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'Memory': [1e8] * 10,
            'Latency': [10.0] * 10,
            'RPS_frontend': [2.0] * 10,
            'Users': [100] * 10,
            'target_replicas': [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
        })
        self.seq_len = 3

    def test_create_sequences_shape(self):
        df = self.df.copy()
        service_col = 'Service'
        feature_cols = ['CPU', 'Memory', 'Latency', 'RPS_frontend', 'Users']
        target_col = 'target_replicas'

        df['service_id'] = df[service_col].astype('category').cat.codes
        df['is_constant'] = 0.0

        X = df[feature_cols + ['service_id', 'is_constant']].values
        Y = df[target_col].values
        services = df[service_col].unique()
        svc_mask = df[service_col].values

        X_seq, Y_seq = create_sequences(X, Y, services, service_col, df, self.seq_len)

        expected_service_a = len(df[df['Service'] == 'svc_a']) - self.seq_len
        expected_service_b = len(df[df['Service'] == 'svc_b']) - self.seq_len
        expected_total = expected_service_a + expected_service_b

        self.assertEqual(len(X_seq), expected_total)
        self.assertEqual(len(Y_seq), expected_total)
        self.assertEqual(X_seq.shape[1], self.seq_len)
        self.assertEqual(X_seq.shape[2], 7)

    def test_create_sequences_value_alignment(self):
        df = self.df.copy()
        service_col = 'Service'
        feature_cols = ['CPU']
        df['service_id'] = 0
        df['is_constant'] = 0.0
        X = df[feature_cols + ['service_id', 'is_constant']].values
        Y = df['target_replicas'].values
        services = df[service_col].unique()

        X_seq, Y_seq = create_sequences(X, Y, services, service_col, df, self.seq_len)

        svc_a_mask = df[service_col].values == 'svc_a'
        y_svc_a = Y[svc_a_mask]
        self.assertEqual(Y_seq[0], y_svc_a[self.seq_len])
        self.assertEqual(Y_seq[1], y_svc_a[self.seq_len + 1])

    def test_checkpoint_keys(self):
        model = torch.nn.Linear(10, 1)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'num_features': 7,
            'params': {
                'hidden_size': 64,
                'num_layers': 1,
                'dropout': 0.3,
                'train_frac': 0.70,
                'val_frac': 0.15,
            },
            'x_scaler_mean': np.array([0.0] * 7),
            'x_scaler_scale': np.array([1.0] * 7),
            'y_scaler_mean': np.array([0.0]),
            'y_scaler_scale': np.array([1.0]),
            'feature_cols': ['CPU', 'Memory', 'Latency', 'RPS_frontend', 'Users'],
            'target_col': 'target_replicas',
            'lookback': 20,
            'service_ids': {0: 'svc_a', 1: 'svc_b'},
            'constant_services': set(),
        }

        expected_keys = {
            'model_state_dict', 'num_features', 'params', 'x_scaler_mean',
            'x_scaler_scale', 'y_scaler_mean', 'y_scaler_scale',
            'feature_cols', 'target_col', 'lookback', 'service_ids',
            'constant_services'
        }
        self.assertEqual(set(checkpoint.keys()), expected_keys)

    def test_data_loader_batching(self):
        X = np.random.randn(100, 20, 7).astype(np.float32)
        Y = np.random.randn(100).astype(np.float32)
        dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        batches = list(loader)
        total = sum(xb.shape[0] for xb, yb in batches)
        self.assertEqual(total, 100)
        for xb, yb in batches:
            self.assertEqual(xb.shape[1], 20)
            self.assertEqual(xb.shape[2], 7)


if __name__ == '__main__':
    unittest.main()
