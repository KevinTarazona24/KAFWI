import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use("Agg")
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AcousticOperator.acoustic_operator import *
from inversion_network import *
from torchmetrics.functional import total_variation
import torch.nn.functional as F
#import lpips
from tqdm import tqdm
import random
from torch.nn import MSELoss
import torch.optim as optim
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import torch.nn as nn
from unet import UNet
from utils import set_all_seeds, rescale_to_uint8_like_grad, AnisotropicTVLoss, TOTAL_LOSS, Freq_reg, rescale_using_meanstd, SNR, MAE, mse_metric

# set deterministic seeds
set_all_seeds(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ground truth (leave on CPU, we'll move it into metrics calls)
#ground_truth = np.load("Data/model/CurveFault_A.npy")[498:499]
#np.save("Data/model/FlatVel_A_Model01.npy", ground_truth)
#ground_truth = torch.from_numpy(ground_truth)
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AcousticOperator.acoustic_operator import *
from inversion_network import *
from torchmetrics.functional import total_variation
import torch.nn.functional as F
from tqdm import tqdm
import random
from torch.nn import MSELoss
import torch.optim as optim
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import torch.nn as nn
from unet import UNet
from newnet import FeatureFusionDense

# import utilities from parent folder
from utils import (
    set_all_seeds,
    rescale_to_uint8_like_grad,
    AnisotropicTVLoss,
    TOTAL_LOSS,
    Freq_reg,
    rescale_using_meanstd,
    SNR,
    MAE,
    mse_metric,
)

# set deterministic seeds
set_all_seeds(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# base directories (place results/checkpoints next to this script)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'CurveFault_B')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Load ground truth (leave on CPU, we'll move it into metrics calls)
ground_truth = np.load('Data/model/CurveVel_A.npy')[498:499]
ground_truth = torch.from_numpy(ground_truth)

# Seismic acquisition parameters (unchanged)
nbc = 120; dx = 10; sx = 175; nx = 70; nz = nx
coord_gx = torch.arange(0, nx * dx, dx)
coord_sx = torch.arange(0, nx * dx + sx, sx)
nt = 1000; dt = 1e-3; freq = 15
s = ricker(freq, torch.linspace(0, 1, steps=nt))
isFS = False; movie = False

# forward modelling of observed data
x = acoustic_operator(ground_truth.to(device), nbc, dx, nt, dt, s, isFS, movie, nx, nz, coord_sx, coord_gx)

# prepare visualization input
sx = nn.functional.interpolate(x, size=(512, 64), mode='bicubic', align_corners=False).permute(1,0,2,3)
sx = rescale_to_uint8_like_grad(sx)

ground_truth = normalize_data(ground_truth).to(device)

# networks
block_transform   = TransformNet2().to(device)
inversion_network = FeatureFusionDense(base=32, in_ch=2,out_ch=1).to(device)

optimizer = optim.AdamW([
    {'params': block_transform.parameters(), 'lr': 1e-3}, 
    {'params': inversion_network.parameters(), 'lr': 1e-3}
    ],betas=(0.9, 0.999),
    weight_decay=1e-4)

# scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=20,
    threshold=1e-4,
    min_lr=1e-6
)

# metrics
mse_loss_fn = MSELoss().to(device)
PSNR = PeakSignalNoiseRatio(data_range=1).to(device)
SSIM = StructuralSimilarityIndexMeasure(data_range=1).to(device)

########################################################################
# training setup
total_loss = []
epochs = 1000
rg_l = AnisotropicTVLoss(weight=1e-3, reduction='mean')

# initial preprocessing
x = rescale_using_meanstd(x)

start = time.time()
for i in tqdm(range(epochs), desc="Training Epochs", colour='magenta'):

    print('Epoch:', i+1)
    block_transform.train()
    inversion_network.train()
    
    optimizer.zero_grad()
    
    # Compute inversion
    V_block = block_transform(x)
    V  = inversion_network(V_block)

    # Compute forward modelling
    print('Computing forward modeling')
    Pm = acoustic_operator(denormalize_labels(V), nbc, dx, nt, dt, s, isFS, movie, nx, nz, coord_sx, coord_gx)

    freq_bands1, freq_bands2, freq_bands3 = Freq_reg(Pm, fs=200.0, bands=((0.0,15.0),(15.0,30.0),(30.0,45.0)), device=device)
    sd = nn.functional.interpolate(Pm, size=(512, 64), mode='bilinear', align_corners=False).permute(1,0,2,3)
    sd = rescale_to_uint8_like_grad(sd)
    Pm = rescale_using_meanstd(Pm)

    #freq_eg = F.mse_loss(freq_bands1, xfreq_bands1) + F.mse_loss(freq_bands2, xfreq_bands2) + F.mse_loss(freq_bands3, xfreq_bands3)
    loss = TOTAL_LOSS(Pm, x) + rg_l(V)

    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    total_loss.append(loss.item())
    
    # Compute metrics
    ssim = SSIM(V.detach(), ground_truth)
    mse_val = mse_metric(V.detach(), ground_truth)
    psnr = PSNR(V.detach(), ground_truth)
    psnr_sh = PSNR(Pm.detach(), x.detach())
    snr = SNR(V.detach(), ground_truth)
    mae = MAE(V.detach(), ground_truth)
    
    metrics = {
        "epoch":    i,
        "ssim":     ssim.item(),
        "mse":      mse_val.item() if isinstance(mse_val, torch.Tensor) else float(mse_val),
        "psnr":     psnr.item(),
        "snr":      snr.item(),
        "mae":      mae.item(),
        "total":    loss.item()
    }
    print(metrics)
    
    
    block_transform.eval()
    inversion_network.eval()
    
    with torch.no_grad():
        block_estimated = block_transform(x)
        fig, ax = plt.subplots(4,4,figsize=(10,10))
        ax[0,0].imshow(x[0,0,100:,:].detach().cpu().numpy(),aspect='auto',cmap='seismic',vmin=0,vmax=1)
        ax[0,0].set_title('Original shot')
        ax[0,1].imshow(block_estimated[0,0].detach().cpu().numpy(),aspect='auto',cmap='seismic')
        ax[0,1].set_title('Channel 1')
        ax[0,2].imshow(block_estimated[0,1].detach().cpu().numpy(),aspect='auto',cmap='seismic')
        ax[0,2].set_title('Channel 2')

        #ax[1,1].imshow(freq_bands1[0,0].detach().cpu().numpy(),aspect='auto',cmap='jet')
        #ax[1,1].set_title('0-15 Hz')
        #ax[1,2].imshow(freq_bands2[0,0].detach().cpu().numpy(),aspect='auto',cmap='jet')
        #ax[1,2].set_title('15-30 Hz')
        #ax[1,3].imshow(freq_bands3[0,0].detach().cpu().numpy(),aspect='auto',cmap='jet')
        #ax[1,3].set_title('30-45 Hz')

        #ax[2,1].imshow(xfreq_bands1[0,0].detach().cpu().numpy(),aspect='auto',cmap='jet')
        #ax[2,1].set_title('0-15 Hz')
        #ax[2,2].imshow(xfreq_bands2[0,0].detach().cpu().numpy(),aspect='auto',cmap='jet')
        #ax[2,2].set_title('15-30 Hz')
        #ax[2,3].imshow(xfreq_bands3[0,0].detach().cpu().numpy(),aspect='auto',cmap='jet')
        #ax[2,3].set_title('30-45 Hz')

        ax[3,0].imshow(Pm[0,0].detach().cpu().numpy(),aspect='auto',cmap='seismic',vmin=0,vmax=1)
        ax[3,0].set_title('Estimated shots')
        ax[3,0].text(0.5, -0.1,
                f"PSNR: {psnr_sh.item():.2f}",
                transform=ax[3,0].transAxes,
                fontsize=10,
                ha='center',
                fontweight='bold')     
        ax[3,1].imshow(ground_truth[0,0].detach().cpu().numpy(),aspect='auto',cmap='jet',vmin=0,vmax=1)
        ax[3,1].set_title('GT model')      
        ax[3,2].imshow(V[0,0].detach().cpu().numpy(),aspect='auto',cmap='jet',vmin=0,vmax=1)
        ax[3,2].set_title('Estimated model')   
        ax[3,2].text(0.5, -0.1,
                f"SSIM: {ssim.item():.2f}  MSE: {mse_val.item() if isinstance(mse_val, torch.Tensor) else mse_val:.3f} MAE: {mae.item():.3f}",
                transform=ax[3,2].transAxes,
                fontsize=10,
                ha='center',
                fontweight='bold') 
    
        ax[3,3].plot(total_loss, color='blue')
        ax[3,3].set_title('Losses')               
        plt.savefig(os.path.join(RESULTS_DIR, f"prediction_{i}_epoch.jpg"), bbox_inches="tight")
        vvv = V.clone()
        vvv = vvv.detach().cpu().numpy()
        np.save(os.path.join(RESULTS_DIR, "model.npy"), vvv)

    torch.save({
        'epoch': i,
        'block_transform_state_dict': block_transform.state_dict(),
        'inversion_network_state_dict': inversion_network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item()}, os.path.join(CHECKPOINTS_DIR, 'CurveFault_B_1.pt'))

    plt.close()
end = time.time()
print("Total time", end-start)
    

	

















