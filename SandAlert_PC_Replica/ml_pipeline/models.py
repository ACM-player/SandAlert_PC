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

    def __init__(self, aux_feature_count: int = 0):
        super().__init__()
        self.aux_feature_count = int(aux_feature_count)
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
        if self.aux_feature_count:
            self.auxiliary = nn.Sequential(
                nn.Linear(self.aux_feature_count, 32),
                nn.ReLU(),
                nn.LayerNorm(32),
            )
            self.fusion = nn.Sequential(
                nn.Linear(64 + 32, 64),
                nn.ReLU(),
            )
        else:
            self.auxiliary = None
            self.fusion = nn.Identity()
        self.event_head = nn.Linear(64, 1)
        self.ips_head = nn.Linear(64, 1)
        self.energy_head = nn.Linear(64, 1)

    def forward(self, inputs, aux_features=None):
        features = self.pool(self.temporal(self.stem(inputs))).squeeze(-1)
        if self.auxiliary is not None:
            if aux_features is None:
                raise ValueError("auxiliary features are required by this model")
            features = self.fusion(
                torch.cat((features, self.auxiliary(aux_features)), dim=1)
            )
        return (
            self.event_head(features).squeeze(-1),
            self.ips_head(features).squeeze(-1),
            self.energy_head(features).squeeze(-1),
        )


class SandAlertONNXWrapper(nn.Module):
    def __init__(
        self,
        model: SandAlertTCN,
        aux_mean,
        aux_std,
        analytic_energy_index: int,
        energy_residual_scale_db: float,
    ):
        super().__init__()
        self.model = model
        self.register_buffer(
            "aux_mean", torch.as_tensor(aux_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "aux_std", torch.as_tensor(aux_std, dtype=torch.float32)
        )
        self.analytic_energy_index = int(analytic_energy_index)
        self.energy_residual_scale_db = float(energy_residual_scale_db)

    def forward(self, inputs, aux_features):
        normalized_aux = (aux_features - self.aux_mean) / self.aux_std
        event_logit, log_ips, energy_residual = self.model(
            inputs, normalized_aux
        )
        analytic_energy = aux_features[:, self.analytic_energy_index]
        energy_db = (
            analytic_energy
            + energy_residual * self.energy_residual_scale_db
        )
        return (
            torch.sigmoid(event_logit),
            torch.exp(torch.clamp_min(log_ips, 0.0)) - 1.0,
            energy_db,
        )
