import torch
import torch.nn as nn

class signclassifier(nn.Module):
    def __init__(self) -> None:
        super(signclassifier, self).__init__()

        self.func = nn.Sequential(
            nn.Linear(21*2, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = x.view(-1, 21*2)
        z = self.func(y)

        return z
