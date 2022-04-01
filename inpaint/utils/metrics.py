import torch
from skimage.metrics import structural_similarity as ssim


def psnr(pred, target, pixel_max_cnt=255.0):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    psnr = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return psnr


def ssim(pred, target):
    pred = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target[0]
    pred = pred[0]
    ssim = ssim(target, pred, multichannel=True)
    return ssim
