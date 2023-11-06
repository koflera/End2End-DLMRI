import torch
import torch.nn as nn


class ApproxSoftShrinkAct(nn.Module):
    def __init__(self, b=0.001):
        super(ApproxSoftShrinkAct, self).__init__()

        self.b = b

    def approx_softshrink(self, x, lambd):
        return x + torch.tensor(1.0 / 2) * (
            torch.sqrt(torch.pow(x - lambd, 2) + self.b)
            - torch.sqrt(torch.pow(x + lambd, 2) + self.b)
        )

    def forward(self, x, threshold):
        return self.approx_softshrink(x, threshold)
