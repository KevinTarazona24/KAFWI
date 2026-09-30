import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


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


def mse_metric(x, x_hat):
    """Mean squared error metric (simple implementation)."""
    return torch.mean((x - x_hat) ** 2)


def set_all_seeds(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def rescale_to_uint8_like_grad(x, p_low=15, p_high=90, beta=50.0):
    x = x.float()
    lo = torch.quantile(x, p_low/100.)
    hi = torch.quantile(x, p_high/100.)
    hi = torch.where(hi <= lo, lo + 1.0, hi)
    t = (hi - lo) / beta
    y = lo + t*F.softplus((x - lo)/t) - t*F.softplus((x - hi)/t)
    y = (y - lo) / (hi - lo)
    return y


def anisotropic_tv(x, weight=1.0, reduction='mean'):
    B, C, H, W = x.shape
    diff_v = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    diff_h = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    tv = diff_v.sum() + diff_h.sum()
    if reduction == 'mean':
        tv = tv / (B * C * H * W)
    elif reduction == 'sum':
        pass
    else:
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")
    return weight * tv


class AnisotropicTVLoss(nn.Module):
    def __init__(self, weight=1.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, x):
        return anisotropic_tv(x, self.weight, self.reduction)


def TOTAL_LOSS(estimated, target):
    l1_loss_fn = nn.L1Loss()
    l2_loss_fn = nn.MSELoss()
    l1 = l1_loss_fn(estimated, target)
    l2 = l2_loss_fn(estimated, target)
    return l1 + l2


def Freq_reg(x, fs=200.0, bands=((0.0,10.0),(10.0,20.0),(20.0,30.0)), device=None):
    if device is None:
        device = x.device
    orig_shape = x.shape
    collapsed = False
    if x.dim() == 2:
        x_proc = x.unsqueeze(0).unsqueeze(0)
        collapsed = True
    elif x.dim() == 3:
        x_proc = x.unsqueeze(0)
    elif x.dim() == 4:
        x_proc = x
    else:
        raise ValueError("Input must be 2D, 3D or 4D tensor (H,W), (C,H,W) or (B,C,H,W).")

    B, C, H, W = x_proc.shape
    x_proc = x_proc.to(device).float()
    Xf = torch.fft.fftshift(torch.fft.fft2(x_proc), dim=(-2, -1))
    fy = torch.fft.fftshift(torch.fft.fftfreq(H, d=1.0/fs).to(device))
    fx = torch.fft.fftshift(torch.fft.fftfreq(W, d=1.0/fs).to(device))
    fy_grid = fy.view(H, 1).repeat(1, W)
    fx_grid = fx.view(1, W).repeat(H, 1)
    freq_mag = torch.sqrt(fx_grid**2 + fy_grid**2)
    masks = []
    for (low, high) in bands:
        mask = ((freq_mag > low) & (freq_mag <= high)).to(dtype=x_proc.dtype, device=device)
        if low == 0.0:
            mask = ((freq_mag >= 0.0) & (freq_mag <= high)).to(dtype=x_proc.dtype, device=device)
        masks.append(mask)
    masks = [m.view(1, 1, H, W) for m in masks]
    Xf_band = [Xf * m for m in masks]
    x_bands = []
    for Xfb in Xf_band:
        Xfb_ishift = torch.fft.ifftshift(Xfb, dim=(-2, -1))
        xb_complex = torch.fft.ifft2(Xfb_ishift)
        xb = xb_complex.real
        x_bands.append(xb)
    x_out = []
    for xb in x_bands:
        if collapsed:
            xb = xb.squeeze(0).squeeze(0)
        elif len(orig_shape) == 3:
            xb = xb.squeeze(0)
        x_out.append(xb)
    return x_out


def rescale_using_meanstd(x, clip_std=2.0):
    x = x.float()
    mu = x.mean()
    sigma = x.std(unbiased=False).clamp(min=1e-6)
    lo = mu - clip_std * sigma
    hi = mu + clip_std * sigma
    t = (hi - lo) / 50.0
    y = lo + t * F.softplus((x - lo) / t) - t * F.softplus((x - hi) / t)
    y = (y - lo) / (hi - lo)
    return y



class SeismicDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

