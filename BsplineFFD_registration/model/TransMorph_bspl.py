'''
B-Spline TransMorph with Diffeomorphism
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''

import math
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import model.transformation as transformation
import model.configs_bspline as configs


def convNd(ndim,
           in_channels,
           out_channels,
           kernel_size=3,
           stride=1,
           padding=1,
           a=0.):
    """
    Convolution of generic dimension
    Args:
        in_channels: (int) number of input channels
        out_channels: (int) number of output channels
        kernel_size: (int) size of the convolution kernel
        stride: (int) convolution stride (step size)
        padding: (int) outer padding
        ndim: (int) model dimension
        a: (float) leaky-relu negative slope for He initialisation
    Returns:
        (nn.Module instance) Instance of convolution module of the specified dimension
    """
    conv_nd = getattr(nn, f"Conv{ndim}d")(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding)
    nn.init.kaiming_uniform_(conv_nd.weight, a=a)
    return conv_nd


def interpolate_(x, scale_factor=None, size=None, mode=None):
    """ Wrapper for torch.nn.functional.interpolate """
    if mode == 'nearest':
        mode = mode
    else:
        ndim = x.ndim - 2
        if ndim == 1:
            mode = 'linear'
        elif ndim == 2:
            mode = 'bilinear'
        elif ndim == 3:
            mode = 'trilinear'
        else:
            raise ValueError(f'Data dimension ({ndim}) must be 2 or 3')
    y = F.interpolate(x,
                      scale_factor=scale_factor,
                      size=size,
                      mode=mode,
                      )
    return y

# 生成网格图像以获得可视化形变场
def create_grid(size, batch_size):
    num1, num2 = (size[0] + 10) // 10, (size[1] + 10) // 10  # 改变除数（10），即可改变网格的密度
    x, y = np.meshgrid(np.linspace(-2, 2, num1), np.linspace(-2, 2, num2))
    grid = np.stack([x, y], axis=0)  # 将x和y堆叠在一起形成二维网格
    grid = np.expand_dims(grid, axis=0)  # 扩展为三维张量，形状为[1, 2, num1, num2]
    grid = np.repeat(grid, batch_size, axis=0)  # 在第一个维度上重复多次以匹配输入的batch大小，形状变为[B, 2, num1, num2]
    return grid

# -------------------BsplineNet from Transmorph-------------------------------
'''
class TranMorphBSplineNet(nn.Module):
    def __init__(self, config):
        super(TranMorphBSplineNet, self).__init__()

        # Determine and set output control point sizes from image size and control point spacing
        ndim = 2
        img_size = config.img_size
        cps = config.cps
        resize_channels = config.resize_channels
        for i, c in enumerate(cps):
            if c > 8 or c < 2:
                raise ValueError(f"Control point spacing ({c}) at dim ({i}) not supported, must be within [1, 8]")
        self.output_size = tuple([int(math.ceil((imsz-1) / c) + 1 + 2)
                                  for imsz, c in zip(img_size, cps)])

        # Add convolutional layers for feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Add upsampling layers
        self.decoder = nn.ModuleList()
        for i in range(len(resize_channels)):
            if i == 0:
                in_ch = 64
            else:
                in_ch = resize_channels[i-1]
            out_ch = resize_channels[i]
            self.decoder.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))

        # Final prediction layers for flow and displacement
        self.out_layers = nn.Conv2d(resize_channels[-1], ndim, kernel_size=3, padding=1)
        self.transform = transformation.CubicBSplineFFDTransform(ndim=2, svf=True, cps=cps)
    
    def forward(self, src, tar):

        # Feature extraction
        x = torch.cat((src, tar), dim=1)
        features = self.encoder(x)
        #print('enc_in features shape: ', features.shape)
        # Upsampling
        for upsample_layer in self.decoder:
            features = upsample_layer(features)
        #print('dec_out features shape: ', features.shape)
        
        # Final predictions
        cp = self.out_layers(features)
        #print('control points shape: ', cp.shape)
        flow, disp = self.transform(cp)
        y = transformation.warp(src, disp)

        return y, flow, disp
'''
'''
# Unet模块
class UNet(nn.Module):
    def __init__(self,
                 ndim,
                 enc_channels=(16, 32, 32, 32, 32),
                 dec_channels=(32, 32, 32, 32),
                 out_channels=(16, 16),
                 conv_before_out=True
                 ):
        super(UNet, self).__init__()

        self.ndim = ndim

        # encoder layers
        self.enc = nn.ModuleList()
        for i in range(len(enc_channels)):
            in_ch = 2 if i == 0 else enc_channels[i - 1]
            stride = 1 if i == 0 else 2
            self.enc.append(
                nn.Sequential(
                    convNd(ndim, in_ch, enc_channels[i], stride=stride, a=0.2),
                    nn.LeakyReLU(0.2)
                )
            )

        # decoder layers
        self.dec = nn.ModuleList()
        for i in range(len(dec_channels)):
            in_ch = enc_channels[-1] if i == 0 else dec_channels[i-1] + enc_channels[-i-1]
            self.dec.append(
                nn.Sequential(
                    convNd(ndim, in_ch, dec_channels[i], a=0.2),
                    nn.LeakyReLU(0.2)
                )
            )

        # upsampler
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # (optional) conv layers before prediction
        if conv_before_out:
            self.out_layers = nn.ModuleList()
            for i in range(len(out_channels)):
                in_ch = dec_channels[-1] + enc_channels[0] if i == 0 else out_channels[i-1]
                self.out_layers.append(
                    nn.Sequential(
                        convNd(ndim, in_ch, out_channels[i], a=0.2),  # stride=1
                        nn.LeakyReLU(0.2)
                    )
                )

            # final prediction layer with additional conv layers
            self.out_layers.append(
                convNd(ndim, out_channels[-1], ndim)
            )

        else:

            # final prediction layer without additional conv layers
            self.out_layers = nn.ModuleList()
            self.out_layers.append(
                convNd(ndim, dec_channels[-1] + enc_channels[0], ndim)
            )

    def forward(self, tar, src):
        x = torch.cat((tar, src), dim=1)

        # encoder
        fm_enc = [x]
        for enc in self.enc:
            fm_enc.append(enc(fm_enc[-1]))

        # decoder: conv + upsample + concatenate skip-connections (to full resolution)
        dec_out = fm_enc[-1]
        for i, dec in enumerate(self.dec):
            dec_out = dec(dec_out)
            dec_out = self.upsample(dec_out)
            dec_out = torch.cat([dec_out, fm_enc[-2-i]], dim=1)

        # further convs and prediction
        y = dec_out
        for out_layer in self.out_layers:
            y = out_layer(y)
        return y


# Bspline Unet网络
class CubicBSplineNet(UNet):
    def __init__(self,
                 config,
                 ):
        """
        Network to parameterise Cubic B-spline transformation
        """
        super(CubicBSplineNet, self).__init__()

        # determine and set output control point sizes from image size and control point spacing
        ndim = 2
        img_size = config.img_size
        cps = config.cps
        enc_channels = config.enc_channels
        dec_channels = config.dec_channels
        resize_channels = config.resize_channels
        for i, c in enumerate(cps):
            if c > 8 or c < 2:
                raise ValueError(f"Control point spacing ({c}) at dim ({i}) not supported, must be within [1, 8]")
        self.output_size = tuple([int(math.ceil((imsz-1) / c) + 1 + 2)  # (128-1)/4 = 32, +1向上取整确保覆盖整个图像区域；+2是避免边界效应（重复性？）而丢失信息
                                  for imsz, c in zip(img_size, cps)])

        # Network:
        # encoder: same u-net encoder
        # decoder: number of decoder layers / times of upsampling by 2 is decided by cps
        num_dec_layers = 4 - int(math.ceil(math.log2(min(cps))))
        self.dec = self.dec[:num_dec_layers]

        # conv layers following resizing
        self.resize_conv = nn.ModuleList()
        for i in range(len(resize_channels)):
            if i == 0:
                if num_dec_layers > 0:
                    in_ch = dec_channels[num_dec_layers-1] + enc_channels[-num_dec_layers]
                else:
                    in_ch = enc_channels[-1]
            else:
                in_ch = resize_channels[i-1]
            out_ch = resize_channels[i]
            self.resize_conv.append(nn.Sequential(convNd(ndim, in_ch, out_ch, a=0.2),
                                                  nn.LeakyReLU(0.2)))

        # final prediction layer
        delattr(self, 'out_layers')  # remove u-net output layers
        self.out_layer = convNd(ndim, resize_channels[-1], ndim)
        self.transform = transformation.CubicBSplineFFDTransform(ndim=2, svf=True, cps=cps)

    def forward(self, src, tar):
        x = torch.cat((tar, src), dim=1)
        # encoder
        fm_enc = [x]
        for enc in self.enc:
            fm_enc.append(enc(fm_enc[-1]))

        # decoder: conv + upsample + concatenate skip-connections
        if len(self.dec) > 0:
            dec_out = fm_enc[-1]
            for i, dec in enumerate(self.dec):
                dec_out = dec(dec_out)
                dec_out = self.upsample(dec_out)
                dec_out = torch.cat([dec_out, fm_enc[-2-i]], dim=1)
        else:
            dec_out = fm_enc

        # resize output of encoder-decoder
        x = interpolate_(dec_out, size=self.output_size)

        # layers after resize
        for resize_layer in self.resize_conv:
            x = resize_layer(x)
        y = self.out_layer(x)
        flow, disp = self.transform(y)
        warped = transformation.warp(src, disp)
        return warped, flow, disp
'''

# ------------------------without bspline---------------------------
class MorphNet(nn.Module):
    def __init__(self, config):
        super(MorphNet, self).__init__()

        # Add convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Add upsampling layers
        self.upsample_layers = nn.ModuleList()
        for i in range(len(config.resize_channels)):
            if i == 0:
                in_ch = 64
            else:
                in_ch = config.resize_channels[i-1]
            out_ch = config.resize_channels[i]
            self.upsample_layers.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))

        # Final prediction layers for flow and displacement
        self.flow_prediction = nn.Conv2d(config.resize_channels[-1], 2, kernel_size=3, padding=1)
        self.displacement_prediction = nn.Conv2d(config.resize_channels[-1], 2, kernel_size=3, padding=1)

    def forward(self, src, tar):

        # Feature extraction
        x = torch.cat((src, tar), dim=1)
        features = self.conv_layers(x)

        # Upsampling
        for upsample_layer in self.upsample_layers:
            features = upsample_layer(features)

        # Final predictions
        flow = self.flow_prediction(features)
        displacement = self.displacement_prediction(features)
        y = transformation.warp(src, displacement)
        return y, flow, displacement

# -----------------------Cubic Bspline-----------------------
class CubicBSplineNet(nn.Module):
    def __init__(self, config):
        super(CubicBSplineNet, self).__init__()

        ndim = 2
        img_size = config.img_size
        cps = config.cps
        enc_channels = config.enc_channels
        dec_channels = config.dec_channels
        resize_channels = config.resize_channels

        for i, c in enumerate(cps):
            if c > 8 or c < 2:
                raise ValueError(f"Control point spacing ({c}) at dim ({i}) not supported, must be within [1, 8]")

        self.output_size = tuple([int(math.ceil((imsz - 1) / c) + 1 + 2) 
                                  for imsz, c in zip(img_size, cps)])
        # Network:
        # encoder layers
        self.enc = nn.ModuleList()
        for i in range(len(enc_channels)):
            in_ch = 2 if i == 0 else enc_channels[i - 1]
            stride = 1 if i == 0 else 2
            self.enc.append(
                nn.Sequential(
                    convNd(ndim, in_ch, enc_channels[i], stride=stride, a=0.2),
                    nn.LeakyReLU(0.2)
                )
            )

        # decoder layers
        num_dec_layers = 4 - int(math.ceil(math.log2(min(cps))))
        #print('decoder layers:', num_dec_layers)
        self.dec = nn.ModuleList()
        for i in range(num_dec_layers):
            in_ch = enc_channels[-1] if i == 0 else dec_channels[i - 1] + enc_channels[-i - 1]
            self.dec.append(
                nn.Sequential(
                    convNd(ndim, in_ch, dec_channels[i], a=0.2),
                    nn.LeakyReLU(0.2)
                )
            )

        # upsampler
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        # conv layers following resizing
        self.resize_conv = nn.ModuleList()
        for i in range(len(resize_channels)):
            if i == 0:
                if num_dec_layers > 0:
                    in_ch = dec_channels[num_dec_layers - 1] + enc_channels[-num_dec_layers]
                else:
                    in_ch = enc_channels[-1]
            else:
                in_ch = resize_channels[i - 1]
            out_ch = resize_channels[i]
            self.resize_conv.append(nn.Sequential(convNd(ndim, in_ch, out_ch, a=0.2),
                                                  nn.LeakyReLU(0.2)))

        # final prediction layer
        self.out_layer = convNd(ndim, resize_channels[-1], ndim)
        self.transform = transformation.CubicBSplineFFDTransform(ndim=2, svf=True, cps=cps)
        
    def forward(self, src, tar):
        x = torch.cat((tar, src), dim=1)
        
        # encoder
        fm_enc = [x]
        for enc in self.enc:
            fm_enc.append(enc(fm_enc[-1]))

        # decoder: conv + upsample + concatenate skip-connections (to full resolution)
        dec_out = fm_enc[-1]
        for i, dec in enumerate(self.dec):
            dec_out = dec(dec_out)
            dec_out = self.upsample(dec_out)
            dec_out = torch.cat([dec_out, fm_enc[-2 - i]], dim=1)

        # resize output of encoder-decoder
        x = interpolate_(dec_out, size=self.output_size)

        # layers after resize
        for resize_layer in self.resize_conv:
            x = resize_layer(x)

        y = self.out_layer(x)
        flow, disp = self.transform(y)
        warped = transformation.warp(src, disp)
        return warped, flow, disp
    
    

CONFIGS = {
    'BSplineParam': configs.get_Bspl_config(),
}    