import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

####################################################################
# ------------------------- Generator -------------------------------
####################################################################
class generator(nn.Module):
    '''
    Generates 64x64 RGB images from random noise
    in: latent_dim
    out: 3x64x64
    '''
    def __init__(self, z_dim):
        super(generator, self).__init__()


        self.gen = nn.Sequential(
            # Initial dense layer
            nn.Linear(z_dim, 4*4*512),
            nn.BatchNorm1d(4*4*512),
            nn.ReLU(True),
            
            # Reshape and start transposed convolutions
            nn.Unflatten(1, (512, 4, 4)),
            
            # Block 1: 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Block 2: 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Block 3: 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Block 4: 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        
        # Final convolution to get to 3 channels
        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    # Add gaussian weight initialisation    
    def weight_init(self):
        for m in self._modules:
            gaussian_weights_init(self._modules[m])
    
    def forward(self, x):
        x = self.gen(x)
        x = self.final_conv(x)
        return self.tanh(x)

####################################################################
#------------------------- Discriminators --------------------------
####################################################################
class discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        self.model = nn.Sequential(
            # Input: 3x64x64
            spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x32x32
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x16x16
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256x8x8
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 512x4x4
            nn.Flatten(),
            spectral_norm(nn.Linear(512*4*4, 1))
        )

    # Add gaussian weight initialisation    
    def weight_init(self):
        for m in self._modules:
            gaussian_weights_init(self._modules[m])
    
    def forward(self, x):
        return self.model(x)
    

####################################################################
#------------------------- Basic Functions -------------------------
####################################################################
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)