import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR10
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  

import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from dataset import *
from models import *
from config import get_config, get_weights_file_path

config = get_config()
lr = config["lr"]
z_dim = config["latent_dim"]
batch_size = config["batch_size"]
num_epochs = config["num_epochs"]

# Create directories for weights of both the discriminator and generator
os.makedirs(config["weight_path"], exist_ok=True)
os.makedirs(os.path.join(config["weight_path"], "discriminator"), exist_ok=True)
os.makedirs(os.path.join(config["weight_path"], "generator"), exist_ok=True)

disc = Discriminator().to(device)
gen = Generator(z_dim).to(device)

fixed_noise = torch.randn((batch_size, z_dim)).to(device)

opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

writer_fake = SummaryWriter(f"cifar_logs/fake")
writer_real = SummaryWriter(f"cifar_logs/real")
step = 0

train_dataset = CIFARDataset()
train_dataloader = train_dataset.get_dataloader()

for epoch in range(num_epochs):
    for batch_idx, (real) in enumerate(tqdm(train_dataloader)):
        # real is a batched tensor of the shape (batch_size, *img_dims) *img_dims = 1, 28, 28 for MNIST
        real = real.to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        ## The Discriminator must always be trained first to provide meaningful gradients for the generator
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        # torch.ones_like function creates a tensor of the same shape as the input tensor filled with ones
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        ## Here only the fake samples are passed to the discriminator as only those matter in measuring the quality of the generator
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad() # clears old gradients
        lossG.backward() # computes new gradients
        opt_gen.step() # updates weights based on gradients

        if batch_idx == 0:
            
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(train_dataloader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                data = real
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "CIFAR Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "CIFAR Real Images", img_grid_real, global_step=step
                )
                step += 1

    # save model weights with epoch as suffix 
    torch.save(disc.state_dict(), get_weights_file_path(config, "discriminator", f"{epoch}"))
    torch.save(gen.state_dict(), get_weights_file_path(config, "generator", f"{epoch}"))