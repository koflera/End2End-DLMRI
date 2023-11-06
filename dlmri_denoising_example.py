# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from networks.dlmri import DLMRI
from utils.mask import cine_cartesian_mask

# specify parameters for dictionary
dico_type = "nn"  # or "lasso"
patches_size = (4, 1, 16)  # (4, 4, 6), (4, 1, 16) or (1, 1, 30)
strides = (1, 1, 1)
Kfact = 1  # 1, 2 or 4
dx, dy, dt = patches_size
K = int(torch.prod(torch.tensor(patches_size))
params = torch.load(
    "dicos/{}_d{}x{}x{}_Kfact{}.pt".format(dico_type, dx, dy, dt, Kfact)
)
dico, _, _ = params["dico"], params["lambda_reg"], params["beta_reg"]

# load complex-valued cine MR image
x = torch.load("data/xtrue.pt")

# noisy image
sigma = 0.2
x0 = x + sigma * x.abs().max() * torch.randn(x.shape, device=x.device)

# set number of iterations
dlmri = DLMRI(
    T=1,
    niter_fista=64,
    patches_size=patches_size,
    strides=strides,
    overcompleteness_factor=Kfact,
    show_progress=False,
)

# assign loaded values
dlmri.dico = torch.nn.Parameter(dico)


if torch.cuda.is_available():
    x0 = x0.cuda()
    dlmri = dlmri.cuda()
    x = x.cuda()

mse_list = []

# denoise images;
# we solve the two sub-problems to be able to track the mse between
# the intermediate reconstructions and the target image
T = 16
lambda_reg = F.softplus(torch.tensor([9.0], device=x0.device))
beta_reg = F.softplus(torch.tensor([-2.0], device=x0.device))
with torch.no_grad():
    xreco = x0.clone()
    mse = F.mse_loss(torch.view_as_real(x), torch.view_as_real(xreco))
    mse_list.append(mse.item())
    print("denoise image")
    for kt in range(T):
        # sparse approximation of all image patches
        xDL = dlmri.dl_reg(
            xreco,
            beta_reg / lambda_reg,
            niter_fista=32,
            niter_pow_it=16,
        )

        # data consistency
        xreco = 1.0 / (1.0 + lambda_reg) * x0 + lambda_reg / (1.0 + lambda_reg) * xDL
        mse = F.mse_loss(torch.view_as_real(x), torch.view_as_real(xreco))
        mse_list.append(mse.item())


# generate plots
arrs_list = [x0.cpu(), xDL.cpu(), x.cpu()]
errs_list = [arr - x.cpu() for arr in arrs_list]
titles_list = ["Noisy", "DL-Denoising", "Target"]
fig, ax = plt.subplots(2, 3)
plt.subplots_adjust(wspace=0.05, hspace=-0.55)
for i, (arr, err, title) in enumerate(zip(arrs_list, errs_list, titles_list)):
    ax[0, i].set_title(title)
    ax[0, i].imshow(arr[0, ..., 0].abs(), clim=[0, 0.8], cmap=plt.cm.Greys_r)
    ax[1, i].imshow(3 * err[0, ..., 0].abs(), clim=[0, 0.8], cmap=plt.cm.viridis)
plt.setp(ax, xticks=[], yticks=[])

# %%
