import os
import torch
from torch import nn
from legacyMTT import wilks_map_tensor
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cpu'


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.shallow_cnn = nn.Sequential(
            nn.Conv2d(1,1,7,1,'same',device=device)
        )
    def forward(self, x):
        map = self.shallow_cnn(x)
        return map


class DetectionNet(nn.Module):
    def __init__(self):
        super(DetectionNet, self).__init__()
        self.flatten = nn.Flatten()
        self.soft_max = nn.Softmax(dim=0)
        self.deep_cnn = nn.Sequential(
            nn.Conv2d(1, 3, 7, 1, 'same', device=device),
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, 1, 'same', device=device),
            nn.ReLU(),
            nn.Conv2d(3, 1, 3, 1, 'same', device=device),
            nn.Sigmoid()
        )
    def forward(self, x):
        map_out = self.deep_cnn(x)
        #map_size = map_out.size()
        #map_out = self.soft_max(map_out.view(-1)).view(map_size)
        return map_out


class DetectionNetTruncating(nn.Module):
    def __init__(self):
        super(DetectionNetTruncating, self).__init__()
        self.flatten = nn.Flatten()
        self.soft_max = nn.Softmax(dim=0)
        self.deep_cnn = nn.Sequential(
            nn.Conv2d(1, 3, 7, 1, 'valid', device=device),
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, 1, 'valid', device=device),
            nn.ReLU(),
            nn.Conv2d(3, 1, 3, 1, 'valid', device=device),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = nn.functional.pad(x,(5, 5, 5, 5)) #pad the input with zeros to account for the truncation
        map_out = self.deep_cnn(x)
        #map_size = map_out.size()
        #map_out = self.soft_max(map_out.view(-1)).view(map_size)
        #map_out = nn.functional.pad(map_out,(5, 5, 5, 5))
        return map_out


class WilksNet(nn.Module):
    def __init__(self):
        super(WilksNet, self).__init__()
        self.wilks_map = wilks_map_tensor
        self.deep_cnn = nn.Sequential(
            nn.Conv2d(7, 4, 5, 1, 'valid', device = device), #takes in the original image, three gaussian blurred images, and three wilks map images
            nn.ReLU(),
            nn.Conv2d(4, 1, 3, 1, 'valid', device = device),
            nn.Sigmoid()
        )
    def forward(self, x):
        #As written, my wilks_map function works on images of size (r,c) not (N, C, r, c) as is the default for torch.
        #I cannot run this on batched data.
        _, blur1, wilks1 = self.wilks_map(x.squeeze(), 1.1, 5) #I could probably learn the radius parameters...
        _, blur2, wilks2 = self.wilks_map(x.squeeze(), 1.3, 5)
        _, blur3, wilks3 = self.wilks_map(x.squeeze(), 0.9, 5)
        input = torch.cat((x, blur1.unsqueeze(0).unsqueeze(0), wilks1.unsqueeze(0).unsqueeze(0),
                           blur2.unsqueeze(0).unsqueeze(0), wilks2.unsqueeze(0).unsqueeze(0),
                           blur3.unsqueeze(0).unsqueeze(0), wilks3.unsqueeze(0).unsqueeze(0)), dim=1)
        map = self.deep_cnn(input)
        map = nn.functional.pad(map, (3, 3, 3, 3)) #padding zeros to replace border lost to convolutions.
        return map


class BabyUNet(nn.Module):
    def __init__(self):
        super(BabyUNet, self).__init__()
        self.cnn_module1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1, 'valid', device=device),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 'valid', device=device),
            nn.ReLU()
        )

        self.cnn_module2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, 'valid', device=device),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, 1, 'valid', device=device),
            nn.ReLU()
        )

        self.cnn_module3 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 'valid', device=device),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 'valid', device=device),
            nn.ReLU()
        )

        self.cnn_module4 = nn.Sequential(
            nn.Conv2d(24, 8, 3, 1, 'valid', device=device),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, 1, 'valid', device=device),
            nn.ReLU()
        )

        self.cnn_module5 = nn.Sequential(
            nn.Conv2d(12, 4, 3, 1, 'valid', device=device),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 'valid', device=device),
            nn.ReLU()
        )

        self.down = nn.MaxPool2d(2, stride=2)
        def up_trim_and_merge(x1, x2):
            up = nn.Upsample(scale_factor=2, mode='bilinear')
            x1 = up(x1)
            r1, c1 = x1.size()[2], x1.size()[3]
            r2, c2 = x2.size()[2], x2.size()[3]
            delta_r = torch.floor(torch.tensor(r2 - r1)/2).long()
            delta_c = torch.floor(torch.tensor(c2 - c1)/2).long()
            out = torch.cat((x1, x2[:, :, delta_r:-delta_r, delta_r:-delta_r]), dim=1)
            return out

        self.up = up_trim_and_merge

        self.detection_module = nn.Sequential(
            nn.Conv2d(4, 4, 3, 1, 'valid', device=device),
            nn.ReLU(),
            nn.Conv2d(4, 1, 1, 1, 'valid', device=device),
            nn.Sigmoid()
        )

    def pad_input(self, x):
        # I should add a 21 pixel ring around the entire image to compensate for the loss of size during convolutions.
        # Also, each dimension must be divisible after padding to ensure each dimension always has an even number of
        # pixels before 2x2 max pooling

        x = nn.functional.pad(x, (21, 21, 21, 21))
        _, _, m, n = x.size()
        if np.mod(m, 2) != 0:
            x = nn.functional.pad(x, (0, 0, 1, 0))
        if np.mod(n, 2) != 0:
            x = nn.functional.pad(x, (1, 0, 0, 0))
        _, _, m, n = x.size()
        if np.mod(m, 4) != 0:
            x = nn.functional.pad(x, (0, 0, 1, 1))
        if np.mod(n, 4) != 0:
            x = nn.functional.pad(x, (1, 1, 0, 0))
        return x

    def trim_output(self, x, m0, n0):
        _, _, m, n = x.size()
        if m - m0 == 1:
            x = x[:, :, 1:, :]
        elif m - m0 == 2:
            x = x[:, :, 1:-1, :]
        elif m - m0 == 3:
            x = x[:, :, 2:-1, :]

        if n - n0 == 1:
            x = x[:, :, :, 1:]
        elif n - n0 == 2:
            x = x[:, :, :, 1:-1]
        elif n - n0 == 3:
            x = x[:, :, :, 2:-1]

        return x


    def forward(self, x):
        # How does the dimension of the tensor change with the convolutions, pooling and upsampling?
        # n, n-4, n/2 - 2, n/2 - 6, n/4 - 3, n/4 - 7, n/2 - 14, n/2 - 18, n - 36, n- 40, n - 42
        # I should pad by image by at least 21 all around. The input dimension should be divisible by 4

        _, _, m0, n0 = x.size()
        x = self.pad_input(x)
        x1 = self.cnn_module1(x) # 128 -> 124
        x = self.down(x1) # 124 -> 62
        x2 = self.cnn_module2(x) # 62 -> 58
        x = self.down(x2) # 58 -> 29
        x = self.cnn_module3(x) #29 -> 25
        x = self.up(x, x2) #25 -> 50
        x = self.cnn_module4(x) #50 -> 46
        x = self.up(x, x1) #46 -> 92
        x = self.cnn_module5(x) #92 -> 88
        x = self.detection_module(x) #88 -> 86
        x = self.trim_output(x, m0, n0)
        #x = nn.functional.pad(x, (21, 21, 21, 21))
        return x


class ShallowDECODE(nn.Module):
    def __init__(self):
        super(ShallowDECODE, self).__init__()

        self.wilks_map = wilks_map_tensor

        self.detection_module = nn.Sequential(
            nn.Conv2d(4, 4, 3, 1, 'valid', device=device),
            nn.ReLU(),
            nn.Conv2d(4, 1, 1, 1, 'valid', device=device),
            nn.Sigmoid()
        )

        self.position_module = nn.Sequential(
            nn.Conv2d(4, 4, 3, 1, 'valid', device=device),
            nn.ReLU(),
            nn.Conv2d(4, 2, 1, 1, 'valid', device=device),
            nn.Tanh()
        )

    def forward(self, x):
        _, _, x1 = self.wilks_map(x.squeeze(), 1.1, 5)
        _, _, x2 = self.wilks_map(x.squeeze(), 0.9, 5)
        _, _, x3 = self.wilks_map(x.squeeze(), 1.3, 5)
        x = torch.cat((x, x1.unsqueeze(0).unsqueeze(1), x2.unsqueeze(0).unsqueeze(1), x3.unsqueeze(0).unsqueeze(1)), dim=1)
        p_map = self.detection_module(x)
        shift_map = self.position_module(x)

        return p_map, shift_map


#model = NeuralNetwork().to(device)
#print(model)

#X = torch.rand(1, 1, 10, 10, device=device)
#map = model(X)

#print(X)
#print(map)

