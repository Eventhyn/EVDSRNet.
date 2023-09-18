from PIL.Image import NONE
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import spectral_norm
from torch.nn import functional as F   

class InputCvBlock_Y(nn.Module):
    def __init__(self, num_in_frames, out_ch):
        super(InputCvBlock_Y, self).__init__()
        self.interm_ch = 8
        self.conv1 = nn.Conv2d(9, 32 ,kernel_size=3,padding=1,  bias=False)
        self.conv2 = nn.Conv2d(32, out_ch, kernel_size=3, padding=1, bias=False)
        self.activation = nn.ReLU()
    def forward(self, x):
        x0 = self.activation(self.conv1(x))
        x1 = self.conv2(x0)
        x2 = self.activation(x1)
        return x2
               
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.downconv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU()
        )
    def forward(self, x):
        x0 = self.downconv(x)
        x1 = self.convblock(x0)
        return x1


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(out_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.PixelShuffle(2)
        )
    def forward(self, x):
        x0 = self.convblock(x)
        return x0



class OutputCvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )
    def forward(self, x):
        return self.convblock(x)

class DenBlock(nn.Module):
    def __init__(self, num_input_frames=3):
        super(DenBlock, self).__init__()
        self.chs_lyr0 = 32
        self.chs_lyr1 = 64
        self.chs_lyr2 = 128
        self.chs_lyr3 = 256
        self.downsample = nn.PixelUnshuffle(2)
        self.inc = InputCvBlock_Y(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
        self.down1 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.down2 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
        self.down3 = DownBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr3)
        self.up1 = UpBlock(in_ch=self.chs_lyr3, out_ch=self.chs_lyr2)
        self.up2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        self.up3 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=9)
        self.gamma_1 = nn.Parameter(torch.full(size=(1, 1, 1, 1), fill_value=1.0))
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self,in0,in1,in2,noise_map,percent):
        # Input convolution block
        x0 = self.inc((torch.cat((in0, in1, in2), dim=1)))
        # Downsampling
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.up1(x3)
        x5 = self.up2(x4+x2)
        x6 = self.up3(x5+x1)   
        # Estimation
        x7 = self.outc(x6+x0)

        x = torch.cat((in0, in1, in2), dim=1)-percent*self.gamma_1*x7
        return x
   
class DenBlock3(nn.Module):
    def __init__(self, num_input_frames=3):
        super(DenBlock3, self).__init__()
        self.chs_lyr0 = 64
        self.chs_lyr1 = 128
        self.chs_lyr2 = 256
        self.downsample = nn.PixelUnshuffle(2)
        self.inc = InputCvBlock_Y(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
        self.down1 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.down2 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
        self.up2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        self.up3 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=128)
        self.gamma_1 = nn.Parameter(torch.full(size=(1, 1, 1, 1), fill_value=1.0))
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self,in0,in1,in2,noise_map,percent):
        # Input convolution block
        x0 = self.inc((torch.cat((in0, in1, in2), dim=1)))
        # Downsampling
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x5 = self.up2(x2)
        x6 = self.up3(x5+x1)   
        # Estimation
        x7 = self.outc(x6+x0)

        x = x7
        return x
class DenBlock_2(nn.Module):
    def __init__(self, num_input_frames=3):
        super(DenBlock_2, self).__init__()
        self.chs_lyr0 = 16
        self.chs_lyr1 = 32
        self.chs_lyr2 = 64
        self.chs_lyr3 = 128
        self.downsample = nn.PixelUnshuffle(2)
        self.inc = InputCvBlock_Y(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
        self.down1 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.down2 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
        self.down3 = DownBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr3)
        self.up1 = UpBlock(in_ch=self.chs_lyr3, out_ch=self.chs_lyr2)
        self.up2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        self.up3 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3)
        self.gamma_1 = nn.Parameter(torch.full(size=(1, 1, 1, 1), fill_value=1.0))
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self,in0,in1,in2,noise_map,percent):
        # Input convolution block
        x0 = self.inc((torch.cat((in0, in1, in2), dim=1)))
        # Downsampling
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.up1(x3)
        x5 = self.up2(x4+x2)
        x6 = self.up3(x5+x1)   
        # Estimation
        x7 = self.outc(x6+x0)

        x = in2 - percent*self.gamma_1*x7
        return x
class DenBlock_1(nn.Module):
    def __init__(self, num_input_frames=3):
        super(DenBlock_1, self).__init__()
        self.chs_lyr0 = 16
        self.chs_lyr1 = 32
        self.chs_lyr2 = 64
        self.chs_lyr3 = 128
        self.downsample = nn.PixelUnshuffle(2)
        self.inc = InputCvBlock_Y(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
        self.down1 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.down2 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
        self.down3 = DownBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr3)
        self.up1 = UpBlock(in_ch=self.chs_lyr3, out_ch=self.chs_lyr2)
        self.up2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        self.up3 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3)
        self.gamma_1 = nn.Parameter(torch.full(size=(1, 1, 1, 1), fill_value=1.0))
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self,in0,in1,in2,noise_map,percent):
        # Input convolution block
        x0 = self.inc((torch.cat((in0, in1, in2), dim=1)))
        # Downsampling
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.up1(x3)
        x5 = self.up2(x4+x2)
        x6 = self.up3(x5+x1)   
        # Estimation
        x7 = self.outc(x6+x0)

        x = in1 - percent*self.gamma_1*x7
        return x
class DenBlock_0(nn.Module):
    def __init__(self, num_input_frames=3):
        super(DenBlock_0, self).__init__()
        self.chs_lyr0 = 16
        self.chs_lyr1 = 32
        self.chs_lyr2 = 64
        self.chs_lyr3 = 128
        self.downsample = nn.PixelUnshuffle(2)
        self.inc = InputCvBlock_Y(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
        self.down1 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.down2 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
        self.down3 = DownBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr3)
        self.up1 = UpBlock(in_ch=self.chs_lyr3, out_ch=self.chs_lyr2)
        self.up2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        self.up3 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3)
        self.gamma_1 = nn.Parameter(torch.full(size=(1, 1, 1, 1), fill_value=1.0))
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self,in0,in1,in2,noise_map,percent):
        # Input convolution block
        x0 = self.inc((torch.cat((in0, in1, in2), dim=1)))
        # Downsampling
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.up1(x3)
        x5 = self.up2(x4+x2)
        x6 = self.up3(x5+x1)   
        # Estimation
        x7 = self.outc(x6+x0)

        x = in0 - percent*self.gamma_1*x7
        return x

class Yuningnet_onestage(nn.Module):
    def __init__(self, num_input_frames=5,is_train=False):
        super(Yuningnet_onestage, self).__init__()
        self.num_input_frames = num_input_frames
        self.temp1 = DenBlock(num_input_frames=3)
        self.reset_params()
        self.is_train = is_train
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, percent):
        # Unpack inputs
        frames, map = torch.split(x, 3, dim=1)
        b, N, c, h, w = frames.size()
        map=map.squeeze(1)
        map=map[:,0,...].reshape((b,1,h,w))
        x0 = frames[:,0,...].view(b,-1,h,w)
        x1 = frames[:,1,...].view(b,-1,h,w)
        x2 = frames[:,2,...].view(b,-1,h,w)
        if self.is_train == True:
            percent = 1
        else:
            percent = percent
        x = self.temp1(x0, x1, x2, map, percent)
        return x