"""
MODEL DEFINITION

Inputs:
    - Time-series tensor data with shape (batch_size, sequence_length, num_features).
    - Features include normalized CPU, Memory, Latency, RPS, and Users.

Outputs:
    - A predicted continuous value representing the required target_replicas for the next time step.

Process:
    1. Uses a Long Short-Term Memory (LSTM) network to capture temporal dependencies in the microservices metrics.
    2. Extracts the hidden state of the final time step to summarize the sequence.
    3. Applies Layer Normalization to stabilize training.
    4. Passes the output through a fully connected regression head (Linear -> ReLU -> Dropout -> Linear) to predict the final replica count.
"""

import torch
import torch.nn as nn

class GlobalLSTMRegressor(nn.Module):
    def __init__(self, num_features: int, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.3):
        """
        Purpose: Initializes the neural network layers including the LSTM core, Layer Normalization, and the multi-layer regression head.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        Purpose: Defines the forward pass logic: feeds data through the LSTM, extracts the final temporal state, normalizes it, and passes it to the regression head to output the final prediction.
        """
        lstm_out, _ = self.lstm(x)
        last = self.norm(lstm_out[:, -1, :])
        return self.head(last).squeeze(-1)
