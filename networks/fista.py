import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from networks.soft_thresholding import ApproxSoftShrinkAct
from utils.linalg import power_iteration


class FISTA(nn.Module):
    def __init__(self, show_progress=False):
        super(FISTA, self).__init__()

        # soft thresholding operator
        self.soft_thresholding = ApproxSoftShrinkAct()
        self.show_progress = show_progress

    def forward(
        self, dico, x, threshold, niter_fista=32, niter_pow_it=16, dico_norm=None
    ):
        """Sparse approximation of the patches in x with respect
        to the dictionary dico using FISTA"""

        # initialize sparse codes and other variables
        N, device, dtype = x.shape[0], x.device, x.dtype
        d, K = dico.shape

        gamma = torch.zeros(N, K, dtype=dtype, device=device)
        z = torch.zeros(N, K, dtype=dtype, device=device)
        beta = torch.ones(N, K, dtype=dtype, device=device)

        if dico_norm is None:

            def dTd(s):
                return F.linear(F.linear(s, dico), dico.transpose(0, 1))

            # estimate the norm of the current dictionary
            dico_norm_sq = power_iteration(
                dTd, torch.rand(1, K, dtype=dtype, device=device), niter=niter_pow_it
            )

        # calculate d^T x once and for all
        dTx = F.linear(x, dico.transpose(0, 1))

        threshold = threshold / dico_norm_sq
        for k in tqdm(range(niter_fista), desc="fista", disable=not self.show_progress):
            gamma_new = self.soft_thresholding(
                z - 1.0 / dico_norm_sq * (dTd(z) - dTx), threshold
            )
            beta_new = (1.0 + torch.sqrt(1 + 4 * beta**2)) / 2.0
            z = gamma_new + (beta - 1.0) / beta_new * (gamma_new - gamma)

            if k != niter_fista - 1:
                gamma = gamma_new
                beta = beta_new

        return gamma
