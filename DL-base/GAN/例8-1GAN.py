import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
#from dataloader1 import dataloader
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class generator(nn.Module):
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)
        return x

class discriminator(nn.Module):
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )

        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)
        return x

def dataloader(dataset, input_size, batch_size, split='train'):
    #transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, ), std=(0.5, ))])
    data_loader = DataLoader(
            datasets.FashionMNIST('./data', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)

    return data_loader
class GAN(object):

    def __init__(self, args):
        # parameters
        self.epoch = 50
        self.sample_num = 100
        self.batch_size = 64
        self.save_dir = r'C:\Users\Administrator\Desktop\PyTorch\运行的程序\BOOK\models' #args.save_dir #
        self.result_dir = r'C:\Users\Administrator\Desktop\PyTorch\运行的程序\BOOK\results' #args.result_dir
        self.dataset = 'fashion-mnist' #args.dataset
        self.log_dir = r'C:\Users\Administrator\Desktop\PyTorch\运行的程序\BOOK\logs'#args.log_dir
        self.gpu_mode = True #args.gpu_mode
        self.model_name = GAN
        self.input_size = 32
        self.z_dim = 62
        # load dataset
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        data = self.data_loader.__iter__().__next__()[0]
        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=0.002, betas=(0.5, 0.999))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=0.002, betas=(0.5, 0.999))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise

        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, _) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break
                z_ = torch.rand((self.batch_size, self.z_dim))
                if self.gpu_mode:
                    x_, z_ = x_.cuda(), z_.cuda()

                # update D network
                self.D_optimizer.zero_grad()
                D_real = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)
                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)
                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()
                # update G network
                self.G_optimizer.zero_grad()
                G_ = self.G(z_)
                D_fake = self.D(G_)
                G_loss = self.BCE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.item())
                G_loss.backward()
                self.G_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1))
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

    def visualize_results(self, epoch, fix=True):
        self.G.eval()
        if not os.path.exists(r'C:\Users\Administrator\Desktop\PyTorch\运行的程序\BOOK\results\fashion-mnist\GAN'):
            os.makedirs(r'C:\Users\Administrator\Desktop\PyTorch\运行的程序\BOOK\results\fashion-mnist\GAN')
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()
            samples = self.G(sample_z_)
        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)
        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          r'C:\Users\Administrator\Desktop\PyTorch\运行的程序\BOOK\results\fashion-mnist\GAN' + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = r'C:\Users\Administrator\Desktop\PyTorch\运行的程序\BOOK\results\fashion-mnist\GAN'
        torch.save(self.G.state_dict(), os.path.join(r'C:\Users\Administrator\Desktop\PyTorch\运行的程序\BOOK\results\fashion-mnist\GAN' + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(r'C:\Users\Administrator\Desktop\PyTorch\运行的程序\BOOK\results\fashion-mnist\GAN' + '_D.pkl'))

        with open(os.path.join(r'C:\Users\Administrator\Desktop\PyTorch\运行的程序\BOOK\results\fashion-mnist\GAN' + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = r'C:\Users\Administrator\Desktop\PyTorch\运行的程序\BOOK\results\fashion-mnist\GAN'
        self.G.load_state_dict(torch.load(r'C:\Users\Administrator\Desktop\PyTorch\运行的程序\BOOK\results\fashion-mnist\GAN' + '_G.pkl'))
        self.D.load_state_dict(torch.load(r'C:\Users\Administrator\Desktop\PyTorch\运行的程序\BOOK\results\fashion-mnist\GAN' + '_D.pkl'))

gan = GAN(object)
gan.train()
print(" [*] Training finished!")
# visualize learned generator
gan.visualize_results(2)
print(" [*] Testing finished!")