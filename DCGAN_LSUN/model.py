import networks

import torch
import torch.nn as nn
import torch.optim as optim

class DCGAN(nn.Module):
    def __init__(self, config):
        super(DCGAN, self).__init__()
        self.nz = config['LATENT_DIM']
        self.lr = config['LR']
        self.gen = networks.generator(self.nz)
        self.dis = networks.discriminator()

        self.gen_opt = optim.Adam(self.gen.parameters(), lr = self.lr, betas = (0.5, 0.999), weight_decay = 0.0001)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr = self.lr, betas = (0.5, 0.999), weight_decay = 0.0001)

        self.BCE_loss = torch.nn.BCEWithLogitsLoss()

    def initialize(self):
        self.gen.weight_init()
        self.dis.weight_init()

    def get_z_random(self, batch_size, nz):
        z = torch.randn(batch_size, nz)
        return z.cuda() if torch.cuda.is_available() else z

    def forward(self):
        self.z_random = self.get_z_random(self.real_image.size(0), self.nz)
        self.fake_image = self.gen.forward(self.z_random)
        self.image_display = torch.cat((self.real_image.detach().cpu(), self.fake_image.detach().cpu()), dim = 2)

    def update_D(self, image):
        self.real_image = image
        self.forward()

        self.dis_opt.zero_grad()
        self.loss_D = self.backward_D(self.dis, self.real_image, self.fake_image)
        self.loss_D.backward()
        self.dis_opt.step()

    def update_G(self):
        self.gen_opt.zero_grad()
        self.loss_G = self.backward_G(self.dis, self.fake_image)
        self.loss_G.backward()
        self.gen_opt.step()

    def backward_D(self, dis, real, fake):
        pred_fake = dis.forward(fake.detach())
        pred_real = dis.forward(real)
        lossD_real = self.BCE_loss(pred_real, torch.ones_like(pred_real))
        lossD_fake = self.BCE_loss(pred_fake, torch.zeros_like(pred_fake))
        lossD = (lossD_real + lossD_fake) / 2
        return lossD
    
    def backward_G(self, dis, fake):
        pred_fake = dis.forward(fake)
        loss_G = self.BCE_loss(pred_fake, torch.ones_like(pred_fake))
        return loss_G
    
    def update_lr(self):
        self.dis_sch.step()
        self.gen_sch.step()

    def resume(self, model_dir, train = True):
        checkpoint = torch.load(model_dir)

        if train:
            self.dis.load_state_dict(checkpoint['dis'])
        self.gen.load_state_dict(checkpoint['gen'])

        if train:
            self.dis_opt.load_state_dict(checkpoint['disc_opt'])
            self.gen_opt.load_state_dict(checkpoint['gen_opt'])
        return checkpoint['ep'], checkpoint['total_it']
    
    def save(self, filename, ep, total_it):
        state = {
            'dis': self.dis.state_dict(),
            'gen': self.gen.state_dict(),
            'disc_opt': self.dis_opt.state_dict(),
            'gen_opt': self.gen_opt.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        torch.save(state, filename)
        return
    
    def assemble_outputs(self):
        image_real = self.normalize_image(self.real_image).detach()
        image_fake = self.normalize_image(self.fake_image).detach()
        return torch.cat((image_real, image_fake), 2)
    
    def normalize_image(self, x):
        return x[:,0:3,:,:]
    
    def test_forward(self):
        z_random = self.get_z_random(self.real_image.size(0), self.nz)
        outputs = self.gen.forward(z_random)
        return outputs
    def setgpu(self, device):
        self.gen = self.gen.to(device)
        self.dis = self.dis.to(device)