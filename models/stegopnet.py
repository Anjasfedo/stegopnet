import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, use_sigmoid=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = use_sigmoid
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_sigmoid: return self.sigmoid(x)
        return self.relu(x)

class PPMModule(nn.Module):
    def __init__(self, in_ch=256, out_ch=64):
        super(PPMModule, self).__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d((s, s)),
                nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True)
            ) for s in [32, 16, 8, 4, 2]
        ])
    def forward(self, x):
        size = x.shape[2:]
        out = [x]
        for b in self.branches:
            out.append(F.interpolate(b(x), size=size, mode='bilinear', align_corners=False))
        return torch.cat(out, dim=1)

class HidingNet(nn.Module):
    def __init__(self, use_ppm=True):
        super(HidingNet, self).__init__()
        self.use_ppm = use_ppm
        self.e1 = ConvBlock(6, 32, stride=2)
        self.e2 = ConvBlock(32, 64, stride=2)
        self.e3 = ConvBlock(64, 128, stride=2)
        self.e4 = ConvBlock(128, 256, stride=1)
        
        d1_in = 576 if use_ppm else 256
        self.ppm = PPMModule() if use_ppm else nn.Identity()
            
        self.d1 = ConvBlock(d1_in, 128, stride=1) 
        self.d2 = nn.ConvTranspose2d(256, 64, 4, 2, 1)
        self.d3 = nn.ConvTranspose2d(128, 32, 4, 2, 1)
        self.d4 = nn.ConvTranspose2d(64, 16, 4, 2, 1)
        self.final = nn.Conv2d(16, 3, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h1 = self.e1(x)
        h2 = self.e2(h1)
        h3 = self.e3(h2)
        h4 = self.e4(h3)
        
        x = self.ppm(h4)
        x = self.d1(x)
        
        x = F.relu(self.d2(torch.cat([x, h3], dim=1)))
        x = F.relu(self.d3(torch.cat([x, h2], dim=1)))
        x = F.relu(self.d4(torch.cat([x, h1], dim=1)))
        return self.sigmoid(self.final(x))

class RevealNet(nn.Module):
    def __init__(self, use_ppm=True):
        super(RevealNet, self).__init__()
        self.use_ppm = use_ppm
        self.e1 = ConvBlock(3, 32, stride=2)
        self.e2 = ConvBlock(32, 64, stride=2)
        self.e3 = ConvBlock(64, 128, stride=2)
        self.e4 = ConvBlock(128, 256, stride=1)
        
        d1_in = 576 if use_ppm else 256
        self.ppm = PPMModule() if use_ppm else nn.Identity()
            
        self.d1 = ConvBlock(d1_in, 128, stride=1)
        self.d2 = nn.ConvTranspose2d(256, 64, 4, 2, 1)
        self.d3 = nn.ConvTranspose2d(128, 32, 4, 2, 1)
        self.d4 = nn.ConvTranspose2d(64, 16, 4, 2, 1)
        self.final = nn.Conv2d(16, 3, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h1 = self.e1(x)
        h2 = self.e2(h1)
        h3 = self.e3(h2)
        h4 = self.e4(h3)
        
        x = self.ppm(h4)
        x = self.d1(x)
        
        x = F.relu(self.d2(torch.cat([x, h3], dim=1)))
        x = F.relu(self.d3(torch.cat([x, h2], dim=1)))
        x = F.relu(self.d4(torch.cat([x, h1], dim=1)))
        return self.sigmoid(self.final(x))