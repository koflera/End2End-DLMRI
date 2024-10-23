import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from networks.single_coil_dc import SingleCoilDC
from networks.fista import FISTA
from utils.patches import extract_patches_3d, combine_patches_3d
from einops import rearrange


def project_dico(dico):
    with torch.no_grad():
        for k in range(dico.shape[1]):
            dico[:, k].div_(torch.norm(dico[:, k].flatten(), p=2, keepdim=True))
    return dico


class DLMRI(nn.Module):
    def __init__(
        self,
        T=4,
        niter_fista=16,
        patches_size=(4, 4, 4),
        strides=(1, 1, 1),
        overcompleteness_factor=2,
        norm="ortho",
        show_progress=False,
    ):
        super(DLMRI, self).__init__()

        # hyper-params of the network
        self.T = T
        self.niter_fista = niter_fista

        self.patches_size = patches_size
        self.strides = strides

        self.npad = (
            self.patches_size[2] - 1,
            self.patches_size[2] - 1,
            self.patches_size[1] - 1,
            self.patches_size[1] - 1,
            self.patches_size[0] - 1,
            self.patches_size[0] - 1,
        )

        # the dictionary D
        self.d = torch.prod(torch.tensor(patches_size))
        self.K = int(overcompleteness_factor * self.d)
        self.dico = project_dico(nn.Parameter(torch.rand(self.d, self.K)))

        # FFT norm
        self.norm = norm

        # data-consistency block
        self.dc = SingleCoilDC(norm=self.norm)

        # FISTA block
        self.fista = FISTA()

        # regularization weights (to be activated with softplus)
        self.lambda_reg = nn.Parameter(torch.tensor(-2.5))
        self.beta_reg = nn.Parameter(torch.tensor(-2.5))

        # show progress bars during loops
        self.show_progress = show_progress

    def extract_patches(self, x):
        x = F.pad(x, self.npad, mode="circular")

        x_shape = x.shape

        x = torch.view_as_real(x)

        x = rearrange(x, "b x y t c -> b c x y t")

        x = extract_patches_3d(x, kernel_size=self.patches_size, stride=self.strides)

        x = rearrange(x, "b c dx dy dt -> (b c) (dx dy dt)")

        return x, x_shape

    def combine_patches(self, p, output_shape):
        p = rearrange(
            p,
            "(b c) (dx dy dt) -> b c dx dy dt",
            b=int(p.shape[0] / 2),
            c=2,
            dx=self.patches_size[0],
            dy=self.patches_size[1],
            dt=self.patches_size[2],
        )

        p = combine_patches_3d(
            p,
            output_shape=output_shape,
            kernel_size=self.patches_size,
            stride=self.strides,
        )

        p = rearrange(p, "b c x y t -> b x y t c")

        p = torch.view_as_complex(p.contiguous())

        p = F.pad(p, tuple([-pad for pad in self.npad]))
        factor = torch.prod(
            torch.tensor(self.patches_size) - torch.tensor(self.strides)
        )
        return p / factor

    def dl_reg(self, x, reg_param, niter_fista=32, niter_pow_it=16, dico_norm=None):
        # extract patches
        xp, output_shape = self.extract_patches(x)

        # subtract mean
        mu = torch.mean(xp, dim=1, keepdim=True)
        xp -= mu

        # sparse coding
        gamma = self.fista(
            self.dico,
            xp,
            reg_param,
            niter_fista=niter_fista,
            niter_pow_it=niter_pow_it,
            dico_norm=dico_norm,
        )

        # get regularized image patches
        Dgamma = F.linear(gamma, self.dico) + mu

        # reassemble patches to obtain a regularized image
        xreg = self.combine_patches(Dgamma, output_shape)

        return xreg

    def forward(self, y, mask, x=None):
        if x is None:
            # initial reconstruction
            x = torch.fft.ifftn(y, dim=(-3, -2), norm=self.norm)

        # activated (i.e. >0) regularization parameters
        lambda_reg = F.softplus(self.lambda_reg)
        beta_reg = F.softplus(self.beta_reg)

        threshold = beta_reg / lambda_reg

        for k in tqdm(range(self.T), desc="iteration", disable=not self.show_progress):
            # regularized image by sparse coding
            xreg = self.dl_reg(
                x,
                threshold,
                niter_fista=self.niter_fista,
            )

            # image update
            x = self.dc(xreg, y, mask, lambda_reg)

        return x
