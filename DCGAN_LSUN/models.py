import networks
from config import *

import torch
import torch.nn as nn
import torch.optim as optim
config = get_config()

class DCGAN(nn.Module):
    def __init__(self, config):
        super(DCGAN, self).__init__()

        self.lr = config['LR']
        self.gen = networks.generator(config['LATENT_DIM'])
        self.disc = networks.discriminator()

        self.gen_opt = optim.Adam(self.gen.parameters(), lr = self.lr, betas = (0.5, 0.999), weight_decay = 0.0001)
        self.disc_opt = optim.Adam(self.disc.parameters(), lr = self.lr, betas = (0.5, 0.999), weight_decay = 0.0001)

        self.BCE_loss = torch.nn.BCEWithLogitsLoss()

    def initialize(self):
        self.gen.weight_init()
        self.disc.weight_init()

    def get_z_random(self, batch_size, nz):
        z = torch.randn(batch_size, nz)
        return z.cuda() if torch.cuda.is_available() else z

    def forward(self):
        self.z_random = self.get_z_random(self.real_image.size(0), self.nz)
        self.fake_image = self.gen.forward(self.z_random)

    def update_D(self, image):
        self.real_image = image
        self.forward()

        self.disc_opt.zero_grad()
        self.loss_D = self.backward_D(self.disc, self.real_image, self.fake_image)
        self.loss_D.backward()
        self.disc_opt.step()

    def update_G(self):
        self.gen_opt.zero_grad()
        self.loss_G = self.backward_G(self.disc, self.fake_image)
        self.loss_G.backward()
        self.gen_opt.step()

    def backward_D(self, disc, real, fake):
        pred_fake = disc.forward(fake.detach())
        pred_real = disc.forward(real)
        lossD_real = self.BCE_loss(pred_real, torch.ones_like(pred_real))
        lossD_fake = self.BCE_loss(pred_fake, torch.zeros_like(pred_fake))
        lossD = (lossD_real + lossD_fake) / 2
        return lossD
    
    def backward_G(self, disc, fake):
        pred_fake = disc.forward(fake)
        loss_G = self.BCE_loss(pred_fake, torch.ones_like(pred_fake))
        return loss_G