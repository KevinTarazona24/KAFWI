import sys
import os
#sys.path.append("paul_edit/")
sys.path.append("Denoiser/paul_edit/")

from module import TransformNet2, DeTransformNet2
from NetworkPaul import  AttU_Net

#sys.path.append("copy/")
sys.path.append("Denoiser/")
from degradationOperator import *
from degradeFunctions import *


import torch 

import torch.optim as optim

from tqdm import tqdm
import numpy as np
import math
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt


from inversion_network import *
from acoustic_operator import *

from torch.nn import MSELoss
import torch.optim as optim
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from tqdm import tqdm
import random
import wandb
from torch.distributions import Normal, kl_divergence

wandb.login(key="d4d6be3e67d35616aec5ede205c72296825c4db1")
wandb.init(project="ANA_CAMSAP", name="GAUSSIAN_CURVEFAULT_B_35dB")

#device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

#set seed
def set_all_seeds(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
set_all_seeds(0)

# Seismic acquisition parameters (unchanged)
nbc = 120; dx = 10; sx = 175; nx = 70; nz = nx
coord_gx = torch.arange(0, nx * dx, dx)
coord_sx = torch.arange(0, nx * dx + sx, sx)
nt = 1000; dt = 1e-3; freq = 15
s = ricker(freq, torch.linspace(0, 1, steps=nt))
isFS = False; movie = False

#metrics
PSNR = PeakSignalNoiseRatio().to(device)
SSIM = StructuralSimilarityIndexMeasure().to(device)
MSE = torch.nn.MSELoss().to(device)
l1 = torch.nn.L1Loss().to(device)
def SNR(x, x_hat, eps=1e-8):
    """
    Calcula el SNR entre x y x_hat en decibelios (dB).
    x, x_hat: tensores de forma [N, C, H, W] o [N, H, W]
    """
    signal_power = torch.sum(x ** 2, dim=list(range(1, x.ndim)))
    noise_power = torch.sum((x - x_hat) ** 2, dim=list(range(1, x.ndim)))
    snr = 10 * torch.log10((signal_power + eps) / (noise_power + eps))
    return snr.mean()

def MAE(x, x_hat):
    """
    Calcula el MAE entre x y x_hat
    """
    return torch.mean(torch.abs(x - x_hat))

def tv_norm(x, reduction='mean'):
    batch_size, channels, height, width = x.size()
    horizontal_diff = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
    vertical_diff = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    
    tv = torch.sum(horizontal_diff) + torch.sum(vertical_diff)
    
    return tv

def add_gaussian_noise_db(signal, snr_db):
    signal_power = torch.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    std = torch.sqrt(noise_power)
    noise = torch.normal(mean=0.0, std=std, size=signal.shape, device=signal.device)
    return signal + noise

    
def kl_gaussians_fixed_sigma(output, sigma, sigma0, reduction='mean'):

    sigma = torch.tensor(sigma, dtype=output.dtype, device=output.device)
    sigma0 = torch.tensor(sigma0, dtype=output.dtype, device=output.device)
    mean = output.mean()
    
    # Apply the KL formula
    kl = torch.log(sigma0 / sigma) + (sigma**2 + mean**2) / (2 * sigma0**2) - 0.5
    
    # Reducción
    if reduction == 'mean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    elif reduction == 'none':
        return kl
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


if __name__ =="__main__":
    
    # Load data
    x = np.load("Data/seismic/CurveFault_B.npy")
    x = torch.from_numpy(x[88:89])

    x = normalize_data(x)
    #x = (degradeImage(x).gaussianNoise(2,0.01)).to(device)
    db_SNR = 35
    
    x1 = add_gaussian_noise_db(x, db_SNR).to(device)
    x2 = x1-x
    
    fig, ax = plt.subplots(1,3)# Muestra las imágenes y guarda las referencias a cada una
    im0 = ax[0].imshow(x[0, 0].numpy(), aspect='auto', cmap='seismic')
    im1 = ax[1].imshow(x1[0, 0].numpy(), aspect='auto', cmap='seismic')
    im2 = ax[2].imshow(x2[0, 0].numpy(), aspect='auto', cmap='seismic')

# Agrega una colorbar para cada imagen
fig.colorbar(im0, ax=ax[0])
fig.colorbar(im1, ax=ax[1])
fig.colorbar(im2, ax=ax[2])
    x = add_gaussian_noise_db(x, db_SNR).to(device)
    #x2 = add_gaussian_noise_db(x, 30)
    #x3 = add_gaussian_noise_db(x, 25)
    
    '''
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(x1[0,0,:,:],aspect='auto',cmap='seismic')
    ax[0].set_title('35 SNR')
    ax[1].imshow(x2[0,0,:,:],aspect='auto',cmap='seismic')
    ax[1].set_title('30 SNR')
    ax[2].imshow(x3[0,0,:,:],aspect='auto',cmap='seismic')
    ax[2].set_title('25 SNR')
    plt.show()
    
    '''
    
    # Load ground truth (leave on CPU, we'll move it into metrics calls)
    ground_truth = np.load("Data/model/CurveFault_B.npy")
    ground_truth = torch.from_numpy(ground_truth[88:89]).to(device)
    ground_truth = normalize_data(ground_truth)
    
    
    #load networks and optimizer
    #block_transform = TransformNet2().to(device)
    
    inversion_network = AttU_NetFWI(img_ch=5, output_ch=1).to(device)
    
    #checkpoint = torch.load('Baseline/new/results_CurveFault_A/baseline_CurveFault_A.pth', map_location=device)
    #inversion_network.load_state_dict(checkpoint["inversion_network_state_dict"])
    
    denoiser_network = AttU_Net(5,5).to(device)
    encod = TransformNet2(ch_out=5).to(device)

    #check=torch.load("copy/autoencoder_training.pth")
    #check=torch.load("Denoiser/autoencoder_training.pth")
    
    #denoiser_network.load_state_dict(check["network_state_dict"])
    #encod.load_state_dict(check["encod_state_dict"])
    
    '''
    den_net.eval()
    with torch.no_grad():
        new_denoised, new_noise = den_net(x)
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(x[0,0].cpu().numpy(),aspect='auto',cmap='seismic')
        ax[1].imshow(new_denoised[0,0].cpu().numpy(),aspect='auto',cmap='seismic')
        ax[2].imshow(new_noise[0,0].cpu().numpy(),aspect='auto',cmap='seismic')
        
    '''
    
    optimizer = optim.Adam([
        {'params': inversion_network.parameters(), 'lr': 1e-4}, #1e-3 antes
        {'params': denoiser_network.parameters(), 'lr': 1e-4} #1e-5 antes
    ])
   

    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Definición del scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',              # o 'max' si usas ssim/psnr/snr como métrica
    factor=0.5,
    patience=20,
    threshold=1e-4,
    min_lr=1e-5,
    verbose=True) 

    loss_lis = []
    loss_total_1 = []
    loss_total_2 = []
    loss_total_3 = []
    loss_total_4 = []
    loss_total_5 = []
    
    epochs =  10000
    
    x = F.interpolate(x, (256, 128), mode='bilinear', align_corners=True)

    for i in tqdm(range(epochs), desc="Training Epochs", colour='green'):
        print('Epoch: ', i+1)
        
        inversion_network.train()
        denoiser_network.train()
        encod.train()
            
        optimizer.zero_grad()

        #denoise shot
        
        den_shot = denoiser_network(x) #1,5,256,128
        noise_shot = den_shot - x #1,5,256,128
        
        mean_true = 0
        std_true = 1
        
        mean_pred = noise_shot.mean()
        std_pred = noise_shot.std()        
        
        # Compute the inversion
        v1, layer1 = inversion_network(den_shot) # [1,1,70,70]
        
        # Forward operator constraints
        print('Forward first velocity model')
        p_v1 = acoustic_operator(denormalize_labels(v1), nbc, dx, nt, dt, s, isFS, movie, nx, nz, coord_sx, coord_gx)
        
        p_v1 = F.interpolate(p_v1, (256, 128), mode='bilinear', align_corners=True)

       # Loss terms
        #rec = noise_shot + den_shot
        #loss_rec = MSE(rec, x)
        #loss_noise = MSE(den_shot, (x-noise_shot))
        #loss_noise2 = MSE(noise_shot, (x-den_shot))
        loss_modeled = 2*(MSE(den_shot, p_v1) + l1(den_shot, p_v1)) #+ SSIM(den_shot, p_v1)
        
        predicted_shot_noise = p_v1 + noise_shot
        predicted_shot_noise = predicted_shot_noise - predicted_shot_noise.min()
        predicted_shot_noise = predicted_shot_noise/predicted_shot_noise.max()
        
        loss_velocities = MSE((predicted_shot_noise), x) + l1((predicted_shot_noise), x) #+ SSIM((predicted_shot_noise), x)
        loss_tv = 1e-4*tv_norm(v1)
        #loss_tv = (1e-4 * (0.95 ** i))*tv_norm(v1)
        
        kl_loss = kl_gaussians_fixed_sigma(noise_shot, std_pred, std_true)*2.8e-3
        #print('Grad KL loss', kl_loss.requires_grad)

        #loss = MSE((p_v1+ noise_shot), x) + l1((p_v1+ noise_shot), x) + 0.05*loss_rec + 0.05*loss_noise + 0.05*loss_noise2
        loss = loss_velocities + loss_modeled + kl_loss + loss_tv

        loss.backward()
        optimizer.step()
        #scheduler.step(loss)
        
        loss_lis.append(loss.item())
        loss_total_1.append(loss_modeled.item())
        loss_total_2.append(loss_velocities.item())
        loss_total_3.append(kl_loss.item())
        loss_total_4.append(loss_tv.item())
        #loss_total_5.append((7*loss_noise2).item())

        tqdm.write(f"Loss total: {loss:.4f}")

        # Compute PSNR and SSIM vs. ground truth
        ssim = SSIM(v1, ground_truth)
        mse = MSE(v1, ground_truth)
        psnr = PSNR(v1, ground_truth)
        snr = SNR(v1, ground_truth)
        mae = MAE(v1, ground_truth)
        
        
        metrics = {
        "epoch":    i,
        "ssim":     ssim.item(),
        "mse":      mse.item(),
        "psnr":     psnr.item(),
        "snr":      snr.item(),
        "mae":      mae.item(),
        "total":    loss.item()
            }
            
        print(metrics)
        wandb.log(metrics, step=i)

        
        inversion_network.eval()
        denoiser_network.eval()
        encod.eval()
        
        with torch.no_grad():
        
            den_estimated = denoiser_network(x) #1,5,256,128
            noise_estimated = den_estimated - x 
            v_estimated, layer1 = inversion_network(den_estimated) # [1,1,70,70]
            
            v_estimated = v_estimated - v_estimated.min()
            v_estimated = v_estimated/v_estimated.max()
                    
            print('Plotting results')
            # === plotting & saving (unchanged) ===
            fig, ax = plt.subplots(2, 4, figsize=(12, 9), constrained_layout=True)
            
            im = ax[0,0].imshow(x[0,0].detach().cpu().numpy(), cmap="jet", aspect="auto")
            ax[0,0].set_title("Input shot")
            fig.colorbar(im, ax=ax[0,0])
            
            im = ax[0,1].imshow(noise_estimated[0, 0].detach().cpu().numpy(), cmap="jet", aspect="auto")
            ax[0,1].set_title("Estimated noise")
            fig.colorbar(im, ax=ax[0,1])
            
            im = ax[0,2].imshow(den_estimated[0, 0].detach().cpu().numpy(), cmap="jet", aspect="auto")
            ax[0,2].set_title("Denoised shot")
            fig.colorbar(im, ax=ax[0,2])
            
            im = ax[0,3].imshow(p_v1[0,0].detach().cpu().numpy(), cmap="jet", aspect="auto")
            ax[0,3].set_title("Modeled shot")
            fig.colorbar(im, ax=ax[0,3])
            
            im = ax[1, 0].imshow((predicted_shot_noise)[0, 0].detach().cpu().numpy(), cmap="jet", aspect="auto")
            ax[1, 0].set_title("Estimated shot with noise")
            fig.colorbar(im, ax=ax[1,0])
            
            im = ax[1,1].imshow(v_estimated[0,0].detach().cpu().numpy(), cmap="jet", aspect="auto", vmin=0, vmax=1)
            ax[1,1].set_title("Estimated model")
            fig.colorbar(im, ax=ax[1,1])
            ax[1,1].text(0.5, -0.1,
     		f"SSIM: {ssim.item():.2f}  MSE: {mse.item():.3f}",
     		transform=ax[1,1].transAxes,
     		fontsize=10,
     		ha='center',
     		fontweight='bold') 
     	
            im = ax[1,2].imshow(ground_truth[0,0].detach().cpu().numpy(), cmap="jet", aspect="auto", vmin=0, vmax=1)
            ax[1,2].set_title("Ground truth")
            fig.colorbar(im, ax=ax[1,2])
            
            ax[1,3].plot(loss_lis, label='Total')
            ax[1,3].plot(loss_total_1, label='Loss modeled')
            ax[1,3].plot(loss_total_2, label='Loss velocities')
            ax[1, 3].plot(loss_total_3, label='Loss KL')
            ax[1, 3].plot(loss_total_4, label='Loss TV')
            #ax[1, 3].plot(loss_total_5, label='Loss 5')
            ax[1,3].legend()


            wandb.log({"Prediction Results": wandb.Image(fig)})
            #plt.savefig(f"results/nafwi_{i}_epoch_Gaussian.jpg", dpi=300)
            plt.close()
            if (i + 1) % 10 == 0:
                torch.save({
                'epoch': i,
                'denoiser_network_state_dict': denoiser_network.state_dict(),
                'inversion_network_state_dict': inversion_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'encod_state_dict': encod.state_dict(),
                'loss': loss.item()}, f'results/nafwi_CurveFault_A_{db_SNR}.pth')
