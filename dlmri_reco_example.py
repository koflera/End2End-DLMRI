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
K = int(torch.prod(torch.tensor(patches_size)))
params = torch.load(
    "dicos/{}_d{}x{}x{}_Kfact{}.pt".format(dico_type, dx, dy, dt, Kfact)
)
dico, lambda_reg, beta_reg = params["dico"], params["lambda_reg"], params["beta_reg"]

# load complex-valued cine MR image
x = torch.load("data/xtrue.pt")

# define undersampling mask
R = 4
mask = cine_cartesian_mask(x.shape[1:], acc_factor=R).unsqueeze(0)

# retrospectively generated undersampled k-space data and add noise
norm = "ortho"
y = mask * torch.fft.fftn(x, dim=(-3, -2), norm=norm)
sigma = 0.05
y = y + mask * sigma * y.abs().max() * torch.randn(y.shape)

# undersampled (zero-filled) reco
xu = torch.fft.ifftn(y, dim=(-3, -2), norm=norm)

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
dlmri.lambda_reg = torch.nn.Parameter(lambda_reg)
dlmri.beta_reg = torch.nn.Parameter(beta_reg)

if torch.cuda.is_available():
    xu = xu.cuda()
    y = y.cuda()
    mask = mask.cuda()
    dlmri = dlmri.cuda()
    x = x.cuda()

mse_list = []

# reconstruct images; here, instead of performing xreco = dlmri(y, mask, x=xu),
# we perform only one iteration at the time to be able to track the mse
# between the iterate and the target imgage
T = 16
with torch.no_grad():
    xreco = xu.clone()
    mse = F.mse_loss(torch.view_as_real(x), torch.view_as_real(xreco))
    mse_list.append(mse.item())
    print("reconstruct images")
    for kt in range(T):
        xreco = dlmri(y, mask, x=xreco)
        mse = F.mse_loss(torch.view_as_real(x), torch.view_as_real(xreco))
        mse_list.append(mse.item())

# generate plots
arrs_list = [xu.cpu(), xreco.cpu(), x.cpu()]
errs_list = [arr - x.cpu() for arr in arrs_list]
titles_list = ["Zero-filled", "DLMRI", "Target"]
fig, ax = plt.subplots(2, 3)
plt.subplots_adjust(wspace=0.05, hspace=-0.55)
for i, (arr, err, title) in enumerate(zip(arrs_list, errs_list, titles_list)):
    ax[0, i].set_title(title)
    ax[0, i].imshow(arr[0, ..., 0].abs(), clim=[0, 1], cmap=plt.cm.Greys_r)
    ax[1, i].imshow(3 * err[0, ..., 0].abs(), clim=[0, 1], cmap=plt.cm.viridis)
plt.setp(ax, xticks=[], yticks=[])

# %%
