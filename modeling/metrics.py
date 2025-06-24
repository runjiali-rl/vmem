from math import exp

import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable



from torcheval.metrics import PeakSignalNoiseRatio

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)





def calculate_score(pil_img1, pil_img2, lpips_loss_fn, device="cpu"):
    """
    Calculate the PSNR and SSIM between two PIL images.

    Args:
        pil_img1 (PIL.Image): First image.
        pil_img2 (PIL.Image): Second image.
        win_size (int, optional): The side length of the sliding window used in SSIM.
                                  Must be odd and <= the smallest spatial dimension.
                                  If not provided, defaults to 7 or the largest odd number
                                  that does not exceed the smallest image dimension.

    Returns:
        tuple: (psnr_value, ssim_value)
    
    Raises:
        ValueError: if the images have different shapes, are too small for SSIM, or unsupported dtype.
    """
    # Convert PIL images to numpy arrays
    img1 = np.array(pil_img1)
    img2 = np.array(pil_img2)
    
    # Ensure images have the same dimensions
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions.")
    
    # Squeeze singleton dimensions (e.g., single-channel)
    if img1.ndim == 3 and img1.shape[-1] == 1:
        img1 = img1.squeeze(axis=-1)
        img2 = img2.squeeze(axis=-1)
    
    # Determine data_range based on dtype
    if img1.dtype == np.uint8:
        data_range = 255.0
    elif img1.dtype == np.uint16:
        data_range = 65535.0
    elif np.issubdtype(img1.dtype, np.floating):
        data_range = 1.0
    else:
        raise ValueError("Unsupported dtype. Use uint8, uint16, or float.")
    
    # # # Compute PSNR
    # mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    # psnr_value_ = 20 * np.log10(data_range / np.sqrt(mse))
    img1_tensor = torch.from_numpy(img1)
    img2_tensor = torch.from_numpy(img2)
    psnr_metric = PeakSignalNoiseRatio(data_range=data_range)
    psnr_metric.update(img2_tensor.to(torch.float32), img1_tensor.to(torch.float32))
    psnr_value = psnr_metric.compute()
    # psnr_metric.reset()
    
    
    img1 = img1_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    img2 = img2_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    # resize the images to 64x64
    
    img1 = (img1.to(torch.float32)/255.0) * 2.0 - 1.0
    img2 = (img2.to(torch.float32)/255.0) * 2.0 - 1.0
    img1 = F.interpolate(img1, size=(64, 64), mode='bilinear', align_corners=False)
    img2 = F.interpolate(img2, size=(64, 64), mode='bilinear', align_corners=False)
    lpips_value = lpips_loss_fn(img1, img2).item()
    ssim_value = ssim(img1, img2).item()
    
   
    
    return (psnr_value, lpips_value, ssim_value)