import torch
import torch.optim as optim
import torch.nn as nn
from models.stegopnet import HidingNet, RevealNet
from utils.helpers import get_loaders, calculate_metrics
import torchvision.utils as vutils
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alpha = 0.6
lr = 0.001
epochs = 150
batch_size = 16

h_net = HidingNet(use_ppm=True, use_attn=True).to(device)
r_net = RevealNet(use_ppm=True, use_attn=True).to(device)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

h_net.apply(weights_init)
r_net.apply(weights_init)

optimizer = optim.Adam(list(h_net.parameters()) + list(r_net.parameters()), lr=lr)
mse_loss = nn.MSELoss()

train_loader, val_loader, _ = get_loaders('data', batch_size)

with open('results/log_train.csv', mode='w') as f:
    writer = csv.writer(f)
    writer.writerow(['Iteration', 'Total_Loss', 'Hiding_Loss', 'Reveal_Loss'])

global_step = 0
for epoch in range(epochs):
    h_net.train()
    r_net.train()
    
    for i, (images, _) in enumerate(train_loader):
        if images.size(0) != batch_size: continue
        
        cover = images[:8].to(device)
        secret = images[8:].to(device)
        
        optimizer.zero_grad()
        stego = h_net(torch.cat([cover, secret], dim=1))
        revealed = r_net(stego)
        
        l_h = mse_loss(stego, cover)
        l_r = mse_loss(revealed, secret)
        loss = l_h + alpha * l_r
        
        loss.backward()
        optimizer.step()
        
        if global_step % 100 == 0:
            with open('results/log_train.csv', mode='a') as f:
                writer = csv.writer(f)
                writer.writerow([global_step, loss.item(), l_h.item(), l_r.item()])
        
        if global_step % 500 == 0:
            vutils.save_image(torch.cat([cover, stego, secret, revealed], dim=0), 
                              f'results/sample_{global_step}.png', nrow=8)
            
        global_step += 1
    
    torch.save(h_net.state_dict(), f'checkpoints/h_net_epoch_{epoch}.pth')
    print(f"Epoch {epoch} selesai.")