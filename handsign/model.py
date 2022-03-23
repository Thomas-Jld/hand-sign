import torch
import torch.nn as nn

class signclassifier(nn.Module):
    def __init__(self) -> None:
        super(signclassifier, self).__init__()

        self.func = nn.Sequential(
            nn.Linear(21*2, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),

            nn.Linear(48, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),

            nn.Linear(48, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),

            nn.Linear(48, 16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = x.view(-1, 21*2)
        z = self.func(y)
        # Sigmoid to reduce to [0, 1]
        # return torch.sigmoid(z)
        return z
