import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from PIL import Image
from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import spectral_norm
from tqdm import tqdm

from config import get_config, get_weights_file_path
from model import DCGAN
from dataset import LSUNDataset
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from saver import Saver

import os
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# writer = SummaryWriter("logs")

def main():
    config = get_config()
    BATCH_SIZE = config['BATCH_SIZE']
    LR = config['LR']
    Z_DIM = config['LATENT_DIM']
    NUM_EPOCHS = config['NUM_EPOCHS']
    WEIGHT_PATH = config['WEIGHT_PATH']

    ds_train = load_dataset("nzm97/lsun_bedroom_64x64", split = 'train')
    train_dataset = LSUNDataset(
        dataset = ds_train,
        batch_size = BATCH_SIZE,
        im_size = 64
    )

    train_loader = train_dataset.get_dataloader()
    print('\n--- load model ---')
    model = DCGAN(config)
    model.setgpu(device)
    if config['resume'] is None:
        model.initialize()
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(config['resume'])

    saver = Saver(config)

    ep0 += 1
    print('Start the training at epoch %d'%(ep0))

    print('\n--- train ---')

    fixed_noise = torch.randn((BATCH_SIZE, Z_DIM)).to(device)

    for ep in range(ep0, NUM_EPOCHS):
        for it, images in enumerate(tqdm(train_loader)):
            if images.size(0) != BATCH_SIZE:
                continue
            images = images.to(device)
            model.update_D(images)
            model.update_G()

            saver.write_display(total_it, model)
            # if it % 20 == 0:
            #     print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
            total_it += 1
        # save result image
        saver.write_img(ep, model)
        print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
        # Save network weights
        saver.write_model(ep, total_it, model)
    return


if __name__ == '__main__':
    main()
