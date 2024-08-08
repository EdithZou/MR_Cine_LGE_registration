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
import model.configs_bspline_ACDC as configs
import model.unet as unet
import model.MIND_descriptor as MIND
import model.SSC_descriptor as SSC

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



class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        
        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        #print('new_locs shape',new_locs.shape)
        return F.grid_sample(src, new_locs, mode=self.mode)

class batch_SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(batch_SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, flow):
        warped = [len(flow[0]), 2, 256, 256]
        for i in range(len(flow[0])):
            new_locs = self.grid + flow[i,:,:,:]
            shape = flow.shape[2:]
            
            # Need to normalize grid values to [-1, 1] for resampler
            for i in range(len(shape)):
                new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
            print("new_locs values:", new_locs)
            if len(shape) == 2:
                new_locs = new_locs.permute(0, 2, 3, 1)
                new_locs = new_locs[..., [1, 0]]
            elif len(shape) == 3:
                new_locs = new_locs.permute(0, 2, 3, 4, 1)
                new_locs = new_locs[..., [2, 1, 0]]
            warped[i,:,:,:] = F.grid_sample(src[i,:,:,:], new_locs, mode=self.mode)
        return warped



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
        self.warp = SpatialTransformer([config.img_size[0],config.img_size[1]])
        
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
        y = self.warp(src, displacement)
        return y, flow, displacement

'''
class CubicBSplineNet(nn.Module):
    def __init__(self, config):
        super(CubicBSplineNet, self).__init__()

        self.ndim = config.ndim
        self.n = config.n
        img_size = config.img_size
        cps = config.cps
        depth = config.depth
        initial_channels = config.initial_channels
        normalization = config.normalization
        self.dim = config.ndim
        self.scale = config.scale
        self.unet = unet.UNet(in_channels = self.n, out_channels = self.ndim, dim = self.ndim, depth = depth, initial_channels = initial_channels, normalization = normalization)

        for i, c in enumerate(cps):
            if c > 8 or c < 2:
                raise ValueError(f"Control point spacing ({c}) at dim ({i}) not supported, must be within [1, 8]")

        self.output_size = tuple([int(math.ceil((imsz) / c)) 
                                  for imsz, c in zip(img_size, cps)])
        self.out_layer = convNd(self.dim, 1, self.dim)        
        self.transform = transformation.BSplineFFDTransformer(ndim=2, img_size=img_size, cps=cps)
        self.warp = SpatialTransformer([256,256])
        
    def forward(self, src, tar):
        input_image = torch.cat((src, tar), dim=1)
        ori_image_shape = input_image[2:]
        if self.scale < 1:
            scaled_image = F.interpolate(input_image, 
                                         scale_factor = self.scale, align_corners = True, 
                                         mode = 'bilinear' if self.dim == 2 else 'trilinear', recompute_scale_factor = False) # (1, n, h, w) or (1, n, d, h, w)
        else:
            scaled_image = input_image
        scaled_disp_t2i = self.unet(scaled_image)
        
        
        if self.scale < 1: 
            disp_t2i = F.interpolate(scaled_disp_t2i, 
                                                size = self.output_size, 
                                                mode = 'bilinear' if self.dim == 2 else 'trilinear', align_corners = True)           
        else:
            disp_t2i = scaled_disp_t2i
            
        flow_bspl = self.transform(disp_t2i)
        #print('flowbspl shape: ', flow.shape)
        flow = flow_bspl
        warped = self.warp(src, flow)
        #print('warp image shape: ', warped.shape)
        return warped, flow
'''
# -----------------------Cubic Bspline-----------------------
class CubicBSplineNet(nn.Module):
    def __init__(self, config):
        super(CubicBSplineNet, self).__init__()

        self.ndim = config.ndim
        self.n = config.n
        img_size = config.img_size
        self.img_size = img_size
        cps = config.cps
        depth = config.depth
        initial_channels = config.initial_channels
        normalization = config.normalization
        self.dim = config.ndim
        self.scale = config.scale
        self.feats_channels = 4
        self.batch_size = config.batch_size
        self.unet = unet.UNet(in_channels = self.n, out_channels = self.ndim, dim = self.ndim, depth = depth, initial_channels = initial_channels, normalization = normalization)
        self.extract = unet.UNet(in_channels = 1, out_channels = self.feats_channels, dim = self.ndim, depth = depth, initial_channels = initial_channels, normalization = normalization)
        self.feats_squeeze1 = nn.Sequential(
            nn.Conv1d(4*self.feats_channels, 4, kernel_size=1)
        )
        self.feats_squeeze2 = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=3, padding=1)
        )

        for i, c in enumerate(cps):
            if c > 8 or c < 2:
                raise ValueError(f"Control point spacing ({c}) at dim ({i}) not supported, must be within [1, 8]")

        self.output_size = tuple([int(math.ceil((imsz) / c)) 
                                  for imsz, c in zip(img_size, cps)])
        self.out_layer = convNd(self.dim, 1, self.dim)       
        self.transform = transformation.BSplineFFDTransformer(ndim=2, img_size=img_size, cps=cps)
        self.warp = SpatialTransformer([128,128])
        
        # 用于计算SSC特征描述
        self.r = config.r
        self.sigma = config.sigma
    
    def calculate_ssc(self, img):
        # Function to calculate MIND descriptors
        mind = MIND.compute_ssc_batch(img.clone(), self.r, self.sigma)
        # Reshape to match input shape (batch_size, channels, height, width)
        return mind
    
    def squeeze_feats(self, img, mode, shape):
        if mode == 'train':
            img_ssc = self.feats_squeeze1(img.view(self.batch_size, 4*self.feats_channels, shape*shape))
            img_ssc = img_ssc.view(self.batch_size, 4, shape, shape)
        else: 
            img_ssc = self.feats_squeeze1(img.view(1, 4*self.feats_channels, shape*shape))
            img_ssc = img_ssc.view(1, 4, shape, shape)            
        ssc = self.feats_squeeze2(img_ssc)
        return ssc
        
            
    def forward(self, src, tar, mode):
        # 提取描述符
        shape = int(src.shape[-1])
        '''
        tar_feats = self.extract(tar)
        tar_ssc = self.calculate_ssc(tar_feats).to('cuda')
        tar_ssc = self.squeeze_feats(tar_ssc, mode, shape)
        '''
        # 预测形变场
        input_image = torch.cat((src, tar), dim=1)
        if self.scale < 1:
            scaled_image = F.interpolate(input_image, 
                                         scale_factor = self.scale, align_corners = True, 
                                         mode = 'bilinear' if self.dim == 2 else 'trilinear', recompute_scale_factor = False) # (1, n, h, w) or (1, n, d, h, w)
        else:
            scaled_image = input_image
        scaled_disp_t2i = self.unet(scaled_image)
        
        
        if self.scale < 1: 
            disp_t2i = F.interpolate(scaled_disp_t2i, 
                                        size = self.output_size,
                                        mode = 'bilinear' if self.dim == 2 else 'trilinear', align_corners = True)           
        else:
            disp_t2i = scaled_disp_t2i

        flow_bspl = self.transform(disp_t2i)
        flow = flow_bspl
        warped = self.warp(src, flow)
        '''
        feats_warped = self.extract(warped)
        ssc_warped = self.calculate_ssc(feats_warped).to('cuda')
        warped_ssc = self.squeeze_feats(ssc_warped, mode, shape)
        '''
        return warped, flow


# -----------------------Cubic Bspline-----------------------
class PixelMorphNet(nn.Module):
    def __init__(self, config):
        super(PixelMorphNet, self).__init__()

        self.ndim = config.ndim
        self.n = config.n
        img_size = config.img_size
        self.img_size = img_size
        cps = config.cps
        depth = config.depth
        initial_channels = config.initial_channels
        normalization = config.normalization
        self.dim = config.ndim
        self.scale = config.scale
        self.feats_channels = 4
        self.batch_size = config.batch_size
        self.unet = unet.UNet(in_channels = self.n, out_channels = self.ndim, dim = self.ndim, depth = depth, initial_channels = initial_channels, normalization = normalization)
        self.extract = unet.UNet(in_channels = 1, out_channels = self.feats_channels, dim = self.ndim, depth = depth, initial_channels = initial_channels, normalization = normalization)
        self.feats_squeeze1 = nn.Sequential(
            nn.Conv1d(4*self.feats_channels, 4, kernel_size=1)
        )
        self.feats_squeeze2 = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=3, padding=1)
        )

        for i, c in enumerate(cps):
            if c > 8 or c < 2:
                raise ValueError(f"Control point spacing ({c}) at dim ({i}) not supported, must be within [1, 8]")

        self.output_size = tuple([int(math.ceil((imsz) / c)) 
                                  for imsz, c in zip(img_size, cps)])
        self.out_layer = convNd(self.dim, 1, self.dim)       
        #self.transform = transformation.BSplineFFDTransformer(ndim=2, img_size=img_size, cps=cps)
        self.warp = SpatialTransformer([128,128])
        
        # 用于计算SSC特征描述
        self.r = config.r
        self.sigma = config.sigma
    
    def calculate_ssc(self, img):
        # Function to calculate MIND descriptors
        mind = MIND.compute_ssc_batch(img.clone(), self.r, self.sigma)
        # Reshape to match input shape (batch_size, channels, height, width)
        return mind
    
    def squeeze_feats(self, img, mode, shape):
        if mode == 'train':
            img_ssc = self.feats_squeeze1(img.view(self.batch_size, 4*self.feats_channels, shape*shape))
            img_ssc = img_ssc.view(self.batch_size, 4, shape, shape)
        else: 
            img_ssc = self.feats_squeeze1(img.view(1, 4*self.feats_channels, shape*shape))
            img_ssc = img_ssc.view(1, 4, shape, shape)            
        ssc = self.feats_squeeze2(img_ssc)
        return ssc
        
            
    def forward(self, src, tar, mode):
        # 提取描述符
        '''
        shape = int(src.shape[-1])
        tar_feats = self.extract(tar)
        tar_ssc = self.calculate_ssc(tar_feats).to('cuda')
        tar_ssc = self.squeeze_feats(tar_ssc, mode, shape)
        '''
        # 预测形变场
        input_image = torch.cat((src, tar), dim=1)
        if self.scale < 1:
            scaled_image = F.interpolate(input_image, 
                                         scale_factor = self.scale, align_corners = True, 
                                         mode = 'bilinear' if self.dim == 2 else 'trilinear', recompute_scale_factor = False) # (1, n, h, w) or (1, n, d, h, w)
        else:
            scaled_image = input_image
        scaled_disp_t2i = self.unet(scaled_image)
        
        
        if self.scale < 1: 
            disp_t2i = F.interpolate(scaled_disp_t2i, 
                                        size = self.img_size,
                                        mode = 'bilinear' if self.dim == 2 else 'trilinear', align_corners = True)           
        else:
            disp_t2i = scaled_disp_t2i

        flow = disp_t2i
        warped = self.warp(src, flow)
        '''
        feats_warped = self.extract(warped)
        ssc_warped = self.calculate_ssc(feats_warped).to('cuda')
        warped_ssc = self.squeeze_feats(ssc_warped, mode, shape)
        '''
        return warped, flow

    

CONFIGS = {
    'BSplineParam': configs.get_Bspl_config(),
}    