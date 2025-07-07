import torch
import torch.nn as nn
import torch.nn.functional as F

# Generator module
class Generator(nn.Module):
    '''
    Generates new images from random noise
    in: latent_dim 256*8*8
    out: 32x32
    '''
    def __init__(self, z_dim):
        super().__init__()
        self.gen = nn.Sequential(
        nn.Linear(z_dim, 8*8*64), # [32, 256] -> [32, 4096]
        nn.ReLU(),
        nn.Unflatten(1, (64, 8, 8)), # [32, 4096] -> [32, 64, 8, 8]
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding = 1),  # [32, 64, 8, 8] -> [32, 32, 16, 16]
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding = 1),  # [32, 32, 16, 16] -> [32, 16, 32, 32]
        nn.ReLU(),
        )
        self.conv = nn.Conv2d(16, 3, kernel_size = 3, padding = 1) # [32, 16, 32, 32] -> [32, 3, 32, 32]
    
    def forward(self, x):
        return self.conv(self.gen(x))
    
# Discriminator Module
class Discriminator(nn.Module):
    def __init__(self, in_channels = 3):
        super().__init__()
        # Simple CNN
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(256 * 4 * 4, 1)

  
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, inplace = True)
        x = F.leaky_relu(self.bn1(self.conv2(x)), 0.2, inplace = True)
        x = F.leaky_relu(self.bn2(self.conv3(x)), 0.2, inplace = True)
        # Flatten the tensor so it can be fed into the FC layers
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return torch.sigmoid(x)
