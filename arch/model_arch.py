from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torch.nn import init
import functools
from torch.nn.utils import weight_norm
from arch.model_onestage import *


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class neuro(nn.Module):
    def __init__(self, n_c, n_b, scale):
        super(neuro,self).__init__()
        pad = (1,1)
        block = []
        self.conv_1 = nn.Conv2d(scale**2*3 + n_c + 3*3, n_c, (3,3), stride=(1,1), padding=pad)
        basic_block = functools.partial(ResidualBlock_noBN, nf=n_c)
        self.recon_trunk = make_layer(basic_block, n_b)
        self.conv_h = nn.Conv2d(n_c, n_c, (3,3), stride=(1,1), padding=pad)
        self.conv_o = nn.Conv2d(n_c, scale**2*3, (3,3), stride=(1,1), padding=pad)
        initialize_weights([self.conv_1, self.conv_h, self.conv_o], 0.1)

    def forward(self, x, h, o):
        x = torch.cat((x, h, o), dim=1)
        #x = F.leaky_relu(self.conv_1(x))
        x = self.conv_1(x)
        x = self.recon_trunk(x)
        #x_h = F.leaky_relu(self.conv_h(x))
        x_h = self.conv_h(x)
        x_o = self.conv_o(x)
        return x_h, x_o

class neuro2(nn.Module):
    def __init__(self, n_c, n_b, scale):
        super(neuro2,self).__init__()
        pad = (1,1)
        block = []
        self.conv_1 = nn.Conv2d(scale**2*3 + n_c + 128, n_c, (3,3), stride=(1,1), padding=pad)
        basic_block = functools.partial(ResidualBlock_noBN, nf=n_c)
        self.recon_trunk = make_layer(basic_block, n_b)
        self.conv_h = nn.Conv2d(n_c, n_c, (3,3), stride=(1,1), padding=pad)
        self.conv_o = nn.Conv2d(n_c, scale**2*3, (3,3), stride=(1,1), padding=pad)
        initialize_weights([self.conv_1, self.conv_h, self.conv_o], 0.1)

    def forward(self, x, h, o):
        x = torch.cat((x, h, o), dim=1)
        #x = F.leaky_relu(self.conv_1(x))
        x = self.conv_1(x)
        x = self.recon_trunk(x)
        #x_h = F.leaky_relu(self.conv_h(x))
        x_h = self.conv_h(x)
        x_o = self.conv_o(x)
        return x_h, x_o

class VSRorg(nn.Module):
    def __init__(self, scale, n_c, n_b):
        super(VSRorg, self).__init__()
        self.neuro = neuro(n_c, n_b, scale)
        self.scale = scale
        self.down = PixelUnShuffle(scale)
        self.n_c = n_c
        self.denoise_b = DenBlock(num_input_frames=3)
        #self.denoise_b0 = DenBlock_0(num_input_frames=3)
        #self.denoise_b1 = DenBlock_1(num_input_frames=3)
        #self.denoise_b2 = DenBlock_2(num_input_frames=3)
        

    def forward(self, f1, f2, f3, x_h, x_o):
    
        #f1_clean = self.denoise_b0(f1,f2,f3,None,1)
        #f2_clean = self.denoise_b1(f1,f2,f3,None,1)
        #f3_clean = self.denoise_b2(f1,f2,f3,None,1)
        #x_input = torch.cat((f1_clean, f2_clean, f3_clean), dim=1)
        x_input = self.denoise_b(f1,f2,f3,None,1)
        x_o = self.down(x_o)
        x_h, x_o = self.neuro(x_input, x_h, x_o)
        x_o = F.pixel_shuffle(x_o, self.scale) + F.interpolate(f3, scale_factor=self.scale, mode='bicubic', align_corners=False)
        return x_h, x_o#,x_input[:,0:3,:,:],x_input[:,3:6,:,:],x_input[:,6:9,:,:]

class VSRorg_sr(nn.Module):
    def __init__(self, scale, n_c, n_b):
        super(VSRorg_sr, self).__init__()
        self.neuro = neuro(n_c, n_b, scale)
        self.scale = scale
        self.down = PixelUnShuffle(scale)
        self.n_c = n_c
        self.denoise_b = DenBlock(num_input_frames=3)
        #self.denoise_b0 = DenBlock_0(num_input_frames=3)
        #self.denoise_b1 = DenBlock_1(num_input_frames=3)
        #self.denoise_b2 = DenBlock_2(num_input_frames=3)
        

    def forward(self, f1, f2, f3, x_h, x_o):
    
        #f1_clean = self.denoise_b0(f1,f2,f3,None,1)
        #f2_clean = self.denoise_b1(f1,f2,f3,None,1)
        #f3_clean = self.denoise_b2(f1,f2,f3,None,1)
        x_input = torch.cat((f1, f2, f3), dim=1)
        #x_input = self.denoise_b(f1,f2,f3,None,1)
        x_o = self.down(x_o)
        x_h, x_o = self.neuro(x_input, x_h, x_o)
        x_o = F.pixel_shuffle(x_o, self.scale) #+ F.interpolate(f3, scale_factor=self.scale, mode='bicubic', align_corners=False)
        return x_h, x_o#,x_input[:,0:3,:,:],x_input[:,3:6,:,:],x_input[:,6:9,:,:]

class VSRorg3(nn.Module):
    def __init__(self, scale, n_c, n_b):
        super(VSRorg3, self).__init__()
        self.neuro = neuro2(n_c, n_b, scale)
        self.scale = scale
        self.down = PixelUnShuffle(scale)
        self.n_c = n_c
        self.denoise_b = DenBlock3(num_input_frames=3)
        #self.denoise_b0 = DenBlock_0(num_input_frames=3)
        #self.denoise_b1 = DenBlock_1(num_input_frames=3)
        #self.denoise_b2 = DenBlock_2(num_input_frames=3)
        

    def forward(self, f1, f2, f3, x_h, x_o):
    
        #f1_clean = self.denoise_b0(f1,f2,f3,None,1)
        #f2_clean = self.denoise_b1(f1,f2,f3,None,1)
        #f3_clean = self.denoise_b2(f1,f2,f3,None,1)
        #x_input = torch.cat((f1_clean, f2_clean, f3_clean), dim=1)
        x_input = self.denoise_b(f1,f2,f3,None,1)
        x_o = self.down(x_o)
        x_h, x_o = self.neuro(x_input, x_h, x_o)
        x_o = F.pixel_shuffle(x_o, self.scale) #+ F.interpolate(f3, scale_factor=self.scale, mode='bicubic', align_corners=False)
        return x_h, x_o#,x_input[:,0:3,:,:],x_input[:,3:6,:,:],x_input[:,6:9,:,:]
def pixel_unshuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class PixelUnShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor
    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
