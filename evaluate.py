import torch
import os
import csv
import numpy as np
from models.stegopnet import HidingNet, RevealNet
from utils.helpers import get_loaders, calculate_metrics
import torchvision.utils as vutils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
test_dir = 'results/test_steganalysis'
os.makedirs(test_dir, exist_ok=True)
os.makedirs(os.path.join(test_dir, 'clean'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'stego'), exist_ok=True)

def evaluate_model(model_path_h, model_path_r, use_ppm):
    h_net = HidingNet(use_ppm=use_ppm).to(device)
    r_net = RevealNet(use_ppm=use_ppm).to(device)
    h_net.load_state_dict(torch.load(model_path_h))
    r_net.load_state_dict(torch.load(model_path_r))
    h_net.eval()
    r_net.eval()

    _, _, test_loader = get_loaders('data', batch_size)
    
    total_psnr_h, total_ssim_h = [], []
    total_psnr_r = []

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if images.size(0) != batch_size: continue
            
            cover = images[:8].to(device)
            secret = images[8:].to(device)
            
            stego = h_net(torch.cat([cover, secret], dim=1))
            revealed = r_net(stego)
            
            psnr_h, ssim_h = calculate_metrics(cover, stego)
            psnr_r, _ = calculate_metrics(secret, revealed)
            
            total_psnr_h.append(psnr_h)
            total_ssim_h.append(ssim_h)
            total_psnr_r.append(psnr_r)
            
            for j in range(stego.size(0)):
                vutils.save_image(cover[j], os.path.join(test_dir, 'clean', f'img_{i}_{j}.png'))
                vutils.save_image(stego[j], os.path.join(test_dir, 'stego', f'img_{i}_{j}.png'))

    print(f"Hasil Evaluasi (PPM={use_ppm}):")
    print(f"Rata-rata PSNR Hiding: {np.mean(total_psnr_h):.2f} dB")
    print(f"Rata-rata SSIM Hiding: {np.mean(total_ssim_h):.4f}")
    print(f"Rata-rata PSNR Reveal: {np.mean(total_psnr_r):.2f} dB")

if __name__ == '__main__':
    evaluate_model('checkpoints/h_net_final.pth', 'checkpoints/r_net_final.pth', use_ppm=True)