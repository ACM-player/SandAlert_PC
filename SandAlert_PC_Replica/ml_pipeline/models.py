from __future__ import annotations

import torch
from torch import nn


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        padding = 2 * dilation
        self.network = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, inputs):
        return self.activation(inputs + self.network(inputs))


class SandAlertTCN(nn.Module):
    """Compact CPU-friendly multi-task 1-D temporal convolutional network."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=9, stride=4, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.temporal = nn.Sequential(
            ResidualTemporalBlock(64, dilation=1),
            ResidualTemporalBlock(64, dilation=2),
            ResidualTemporalBlock(64, dilation=4),
            ResidualTemporalBlock(64, dilation=8),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.event_head = nn.Linear(64, 1)
        self.ips_head = nn.Linear(64, 1)
        self.energy_head = nn.Linear(64, 1)

    def forward(self, inputs):
        features = self.pool(self.temporal(self.stem(inputs))).squeeze(-1)
        return (
            self.event_head(features).squeeze(-1),
            self.ips_head(features).squeeze(-1),
            self.energy_head(features).squeeze(-1),
        )


class SandAlertONNXWrapper(nn.Module):
    def __init__(self, model: SandAlertTCN):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        event_logit, log_ips, energy_db = self.model(inputs)
        return (
            torch.sigmoid(event_logit),
            torch.exp(torch.clamp_min(log_ips, 0.0)) - 1.0,
            energy_db,
        )
