"""
AI Speed & Vibration Filter.

A compact 1-D CNN that maps a short window of mounting-invariant IMU features
to the vehicle's instantaneous forward speed. Vehicle speed is encoded in the
*vibration energy* of the signal (engine/road/wheel harmonics), so we pool both
the mean and the standard deviation across time -- exposing that energy directly
to the regression head. The network is deliberately small (~50k params) to stay
lightweight for on-device / edge deployment (ONNX / TFLite export).
"""
import torch
import torch.nn as nn


class SpeedCNN(nn.Module):
    def __init__(self, in_ch=6, width=32, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=5, padding=2),
            nn.BatchNorm1d(width), nn.ReLU(inplace=True),
            nn.Conv1d(width, width * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(width * 2), nn.ReLU(inplace=True),
            nn.Conv1d(width * 2, width * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(width * 2), nn.ReLU(inplace=True),
        )
        # mean + std pooling -> 2 * (width*2) features
        self.head = nn.Sequential(
            nn.Linear(width * 2 * 2, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        h = self.features(x)                     # [B, F, T]
        mean = h.mean(dim=2)
        std = h.std(dim=2)
        z = torch.cat([mean, std], dim=1)        # [B, 2F]
        out = self.head(z).squeeze(-1)           # [B]
        return out


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
