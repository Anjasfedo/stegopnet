import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

def get_loaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    train_set = datasets.ImageFolder(data_dir + '/train', transform)
    val_set = datasets.ImageFolder(data_dir + '/val', transform)
    test_set = datasets.ImageFolder(data_dir + '/test', transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader

def calculate_metrics(img1, img2):
    img1 = img1.detach().cpu().permute(0, 2, 3, 1).numpy()
    img2 = img2.detach().cpu().permute(0, 2, 3, 1).numpy()
    
    psnrs = []
    ssims = []
    
    for i in range(img1.shape[0]):
        p = psnr_func(img1[i], img2[i], data_range=1.0)
        s = ssim_func(img1[i], img2[i], multichannel=True, data_range=1.0)
        psnrs.append(p)
        ssims.append(s)
        
    return np.mean(psnrs), np.mean(ssims)