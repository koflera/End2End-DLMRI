import torch
import torch.nn as nn


class SingleCoilDC(nn.Module):
    def __init__(self, norm="ortho"):
        super(SingleCoilDC, self).__init__()

        self.norm = norm

    def forward(self, xreg, k, mask, reg_param):
        kreg = torch.fft.fftn(xreg, dim=(-3, -2), norm=self.norm)

        kest = (
            mask * (reg_param / (1.0 + reg_param)) * kreg
            + (1.0 / (1.0 + reg_param)) * k
            + (~mask) * kreg
        )

        x = torch.fft.ifftn(kest, dim=(-3, -2), norm=self.norm)

        return x
