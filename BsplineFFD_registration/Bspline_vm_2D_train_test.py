# python imports
import time
import csv
import os
import warnings
import argparse
import glob
import math
# external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import nibabel as nib
import skimage.metrics as metrics  # 这个里面包含了很多评估指标的计算方法 PSNR SSIM等
from skimage.transform import resize
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()

# 文件路径
parser.add_argument("--train_dir", type=str, help="data folder with training vols", dest="train_dir", default="/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Xiehe2D_resize")
parser.add_argument("--model_dir", type=str, help="models folder", dest="model_dir", default="/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Checkpoint_Xiehe")
parser.add_argument("--log_dir", type=str, help="logs folder", dest="log_dir", default="/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Log_Xiehe")
parser.add_argument("--result_dir", type=str, help="results folder", dest="result_dir", default="/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Result_Xiehe")

# network parameters
parser.add_argument("--pattern", type=str, help="select train or test", dest="pattern", default="train")

# training parameters
parser.add_argument("--lr", type=float, help="learning rate", dest="lr", default=1e-4)
parser.add_argument("--n_iter", type=int, help="number of iterations", dest="n_iter", default=1500)
parser.add_argument("--bsp_iter", type=int, help="number of bspline iterations", dest="bsp_iter", default=750)
parser.add_argument("--sim_loss", type=str, help="image similarity loss: mse or ncc", dest="sim_loss", default="ncc")
parser.add_argument("--alpha", type=float, help="regularization parameter", dest="alpha", default=0.1)  # recommend 1.0 for ncc, 0.01 for mse
parser.add_argument("--batch_size", type=int, help="batch_size", dest="batch_size", default=1)

# testing parameters
parser.add_argument("--test_dir", type=str, help="test data directory", dest="test_dir", default="/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Xiehe2D_resized_test")
parser.add_argument("--checkpoint_path_bsp", type=str, help="bspmodel weight file", dest="checkpoint_path_bsp", default="/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Checkpoint_Xiehe/trained_model.pth")
parser.add_argument("--checkpoint_path_vm", type=str, help="vmmodel weight file", dest="checkpoint_path_vm", default="/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Checkpoint_Xiehe/trained_model.pth")
parser.add_argument("--checkpoint_path_bspstn", type=str, help="bspmodel weight file", dest="checkpoint_path_bspstn", default="/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Checkpoint_Xiehe/trained_model.pth")

parser.add_argument("--dice", type=bool, help="if compute dice", dest="dice", default=False)
parser.add_argument("--psnr", type=bool, help="if compute psnr", dest="psnr", default=True)
parser.add_argument("--ssim", type=bool, help="if compute ssim", dest="ssim", default=True)
parser.add_argument("--jacobian", type=bool, help="if compute jacobian", dest="jac", default=True)

args = parser.parse_args()

# mse loss
def compute_mse(tensor_x, tensor_y):
    mse = torch.mean((tensor_x - tensor_y) ** 2)
    return mse


# gradient loss
def compute_gradient(tensor_x):
    dims = tensor_x.ndim
    gradient = 0.0
    if dims == 4:
        dx = (tensor_x[:, :, 1:, :] - tensor_x[:, :, :-1, :]) ** 2
        dy = (tensor_x[:, :, :, 1:] - tensor_x[:, :, :, :-1]) ** 2
        gradient = (dx.mean() + dy.mean()) / 2
    elif dims == 5:
        dx = (tensor_x[:, :, 1:, :, :] - tensor_x[:, :, :-1, :, :]) ** 2
        dy = (tensor_x[:, :, :, 1:, :] - tensor_x[:, :, :, :-1, :]) ** 2
        dz = (tensor_x[:, :, :, :, :] - tensor_x[:, :, :, :, :]) ** 2
        gradient = (dx.mean() + dy.mean() + dz.mean()) / 2
    return gradient

# 形变场平滑损失
def deformation_smooth_loss(deformation_field):
    # 计算两个通道的平均
    deformation_field_gray = torch.mean(deformation_field, dim=1, keepdim=True)
    # 计算形变场梯度
    gradients = F.conv2d(deformation_field_gray, torch.ones(1,1,3,3).to(deformation_field.device), padding=1)
    # 计算平滑损失
    smooth_loss = torch.mean(gradients**2)
    return smooth_loss
    
class LNCCLoss(nn.Module):
    """
    Local Normalized Cross Correlation loss
    Adapted from VoxelMorph implementation:
    https://github.com/voxelmorph/voxelmorph/blob/5273132227c4a41f793903f1ae7e27c5829485c8/voxelmorph/torch/losses.py#L7
    """
    def __init__(self, window_size=7):
        super(LNCCLoss, self).__init__()
        self.window_size = window_size

    def forward(self, x, y):
        # products and squares
        xsq = x * x
        ysq = y * y
        xy = x * y

        # set window size
        ndim = x.dim() - 2
        window_size = [7,7]

        # summation filter for convolution
        sum_filt = torch.ones(1, 1, *window_size).type_as(x)

        # set stride and padding
        stride = (1,) * ndim
        padding = tuple([math.floor(window_size[i]/2) for i in range(ndim)])

        # get convolution function of the correct dimension
        conv_fn = getattr(F, f'conv{ndim}d')

        # summing over window by convolution
        x_sum = conv_fn(x, sum_filt, stride=stride, padding=padding)
        y_sum = conv_fn(y, sum_filt, stride=stride, padding=padding)
        xsq_sum = conv_fn(xsq, sum_filt, stride=stride, padding=padding)
        ysq_sum = conv_fn(ysq, sum_filt, stride=stride, padding=padding)
        xy_sum = conv_fn(xy, sum_filt, stride=stride, padding=padding)

        window_num_points = np.prod(window_size)
        x_mu = x_sum / window_num_points
        y_mu = y_sum / window_num_points

        cov = xy_sum - y_mu * x_sum - x_mu * y_sum + x_mu * y_mu * window_num_points
        x_var = xsq_sum - 2 * x_mu * x_sum + x_mu * x_mu * window_num_points
        y_var = ysq_sum - 2 * y_mu * y_sum + y_mu * y_mu * window_num_points

        lncc = cov * cov / (x_var * y_var + 1e-5)

        return -torch.mean(lncc)

def compute_local_sums(x, y, filt, stride, padding, win):
    x2, y2, xy = x * x, y * y, x * y
    x_sum = F.conv2d(x, filt, stride=stride, padding=padding)
    y_sum = F.conv2d(y, filt, stride=stride, padding=padding)
    x2_sum = F.conv2d(x2, filt, stride=stride, padding=padding)
    y2_sum = F.conv2d(y2, filt, stride=stride, padding=padding)
    xy_sum = F.conv2d(xy, filt, stride=stride, padding=padding)
    win_size = np.prod(win)
    x_windowed = x_sum / win_size
    y_windowed = y_sum / win_size
    cross = xy_sum - y_windowed * x_sum - x_windowed * y_sum + x_windowed * y_windowed * win_size
    x_var = x2_sum - 2 * x_windowed * x_sum + x_windowed * x_windowed * win_size
    y_var = y2_sum - 2 * y_windowed * y_sum + y_windowed * y_windowed * win_size
    return x_var, y_var, cross


# ncc损失
def ncc_loss(x, y, win=None):
    """
    输入大小是[B,C,W,H]格式的，在计算ncc时用卷积来实现指定窗口内求和
    """
    ndims = len(list(x.size())) - 2
    assert ndims == 2, "Input volumes should be 2 dimensions. Found: %d" % ndims
    if win is None:
        win = [9, 9]  # 默认窗口大小为 9x9
    sum_filt = torch.ones([1, 1, *win]).cuda()
    #pad_no = np.floor(win[0] / 2)
    stride = [1] * ndims
    padding = [2] * ndims
    x_var, y_var, cross = compute_local_sums(x, y, sum_filt, stride=tuple(stride), padding=tuple(padding), win=win)
    cc = cross * cross / (x_var * y_var + 1e-5)
    return -1 * torch.mean(cc)


# count parameters in model
def countParameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


# 计算Dice B*C*H*W*D  多标签返回均值
def compute_Dice(tensor_pred, tensor_targ):
    smooth = 1e-5
    labels = tensor_pred.unique()
    if labels[0] == 0:
        labels = labels[1:]
    dice_list = torch.zeros([len(labels)])
    for _num in range(len(labels)):
        tensor_x = torch.where(tensor_pred == labels[_num], 1, 0).flatten()
        tensor_y = torch.where(tensor_targ == labels[_num], 1, 0).flatten()
        dice_list[_num] = (2.0 * (tensor_x * tensor_y).sum() + smooth) / (tensor_x.sum() + tensor_y.sum() + smooth)
    dice = torch.mean(dice_list).item()
    return dice


# compute the peak signal noise ratio //tensor
def compute_PSNR(tensor_x, tensor_y):
    mse = compute_mse(tensor_x, tensor_y)
    psnr = (-10 * torch.log10(mse)).item()
    return psnr


# compute structure similarity //tensor
def compute_SSIM(tensor_x, tensor_y):
    np_x = tensor_x.cpu().detach().numpy()[0, 0, ...]
    np_y = tensor_y.cpu().detach().numpy()[0, 0, ...]
    ssim = metrics.structural_similarity(np_x, np_y, data_range=1)
    return ssim


# compute Jacobian determinant
def compute_Jacobian(flow):
    Dx = (flow[:, 0, 1:, :-1] - flow[:, 0, :-1, :-1])
    Dy = (flow[:, 1, :-1, 1:] - flow[:, 1, :-1, :-1])

    D = Dx[:, None, ...] * Dy[:, None, ...]
    return D



class Jacobian:
    def __init__(self, flow):
        self.determinant = compute_Jacobian(flow)

    def count_minus_ratio(self):
        size = 1
        for dim in self.determinant.shape:
            size *= dim
        x = torch.where(self.determinant <= 0, 1, 0)
        ratio = (torch.sum(x) / size).item()
        return ratio


# ---------------------------------------Bspline-------------------------------------
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
                 ndim,
                 enc_channels=(16, 32, 32, 32, 32),
                 dec_channels=(32, 32, 32, 32),
                 resize_channels=(32, 32),
                 cps=(2, 2),
                 img_size=(128,128)
                 ):
        """
        Network to parameterise Cubic B-spline transformation
        """
        super(CubicBSplineNet, self).__init__(ndim=2,
                                              enc_channels=enc_channels,
                                              conv_before_out=False)

        # determine and set output control point sizes from image size and control point spacing
        img_size = [128,128]
        cps = [2,2]
        for i, c in enumerate(cps):
            if c > 8 or c < 2:
                raise ValueError(f"Control point spacing ({c}) at dim ({i}) not supported, must be within [1, 8]")
        self.output_size = tuple([int(math.ceil((imsz-1) / c) + 1 + 2)
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

    def forward(self, x):
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
        return y


from torch import Tensor
# Transformation
class SpatialTransform_bsp(nn.Module):
    def __init__(self, ndim, img_size, cps, svf=False, svf_steps=7, svf_scale=1):
        super(SpatialTransform_bsp, self).__init__()

        self.svf = svf
        self.svf_steps = svf_steps
        self.svf_scale = svf_scale
        self.ndim = ndim
        self.img_size = img_size
        self.stride = [cps] if isinstance(cps, int) else cps
        self.derivative_weights = [
            [0.5,0.2,0.3],
            [0.5,0.2,0.3],
            [0.5,0.2,0.3],
            [0.5,0.2,0.3],
            [0.5,0.2,0.3],
            [0.5,0.2,0.3],
            [0.5,0.2,0.3],
            [0.5,0.2,0.3]
        ]
        self.kernels = self.set_kernel(self.derivative_weights)
        self.padding = [(len(k) - 1) // 2 for k in self.kernels]

    def set_kernel(self, derivative_weights):
        kernels = list()
        for s in self.stride:
            kernels += [cubic_bspline1d(s, derivative_weights)]
        return kernels

    def forward(self, x):
        flow = x

        for i, (k, s, p) in enumerate(zip(self.kernels, self.stride, self.padding)):
            k = k.to(dtype=x.dtype, device=x.device)
            flow = conv1d(flow, dim=i + 2, kernel=k, stride=s, padding=p, transpose=True)
        
        # 将输出裁剪成输入图像的尺寸
        slicer = (slice(0, flow.shape[0]), slice(0, flow.shape[1])) \
                 + tuple(slice(s, s + self.img_size[i]) for i, s in enumerate(self.stride))
        flow = flow[slicer]

        if self.svf:
            disp = self.svf_exp(flow)
            return flow, disp
        else:
            disp = flow
            return disp

    def svf_exp(self, flow):
        disp = flow * (self.svf_scale / (2 ** self.svf_steps))
        for _ in range(self.svf_steps):
            disp = disp + self.warp(x=disp, disp=disp)
        return disp


class _Transform(object):
    """ Transformation base class """
    def __init__(self,
                 svf=False,
                 svf_steps=7,
                 svf_scale=1):
        self.svf = svf
        self.svf_steps = svf_steps
        self.svf_scale = svf_scale

    def compute_flow(self, x):
        raise NotImplementedError

    def __call__(self, x):
        flow = self.compute_flow(x)
        if self.svf:
            disp = svf_exp(flow,
                           scale=self.svf_scale,
                           steps=self.svf_steps)
            return flow, disp
        else:
            disp = flow
            return disp


class CubicBSplineFFDTransform(_Transform):
    def __init__(self,
                 ndim,
                 img_size=[128,128],
                 cps=[2,2],
                 svf=False,
                 svf_steps=7,
                 svf_scale=1):
        """
        Compute dense displacement field of Cubic B-spline FFD transformation model
        from input control point parameters.

        Args:
            ndim: (int) image dimension
            img_size: (int or tuple) size of the image
            cps: (int or tuple) control point spacing in number of intervals between pixel/voxel centres
            svf: (bool) stationary velocity field formulation if True
        """
        super(CubicBSplineFFDTransform, self).__init__(svf=svf,
                                                       svf_steps=svf_steps,
                                                       svf_scale=svf_scale)
        self.ndim = ndim
        self.img_size = img_size
        self.stride = [cps] if isinstance(cps, int) else cps

        self.kernels = self.set_kernel()
        self.padding = [(len(k) - 1) // 2
                        for k in self.kernels]  # the size of the kernel is always odd number

    def set_kernel(self):
        kernels = list()
        for s in self.stride:
            # 1d cubic b-spline kernels
            kernels += [cubic_bspline1d(s)]
        return kernels

    def compute_flow(self, x):
        """
        Args:
            x: (N, dim, *(sizes)) Control point parameters
        Returns:
            y: (N, dim, *(img_sizes)) The dense flow field of the transformation
        """
        # separable 1d transposed convolution
        flow = x
        for i, (k, s, p) in enumerate(zip(self.kernels, self.stride, self.padding)):
            # 在每个维度上应用Bspline核
            k = k.to(dtype=x.dtype, device=x.device)
            flow = conv1d(flow, dim=i + 2, kernel=k, stride=s, padding=p, transpose=True)
        # 将输出裁剪成输入图像的尺寸
        slicer = (slice(0, flow.shape[0]), slice(0, flow.shape[1])) \
                 + tuple(slice(s, s + self.img_size[i]) for i, s in enumerate(self.stride))
        flow = flow[slicer]
        
        return flow


def normalise_disp(disp):
    """
    Spatially normalise DVF to [-1, 1] coordinate system used by Pytorch `grid_sample()`
    Assumes disp size is the same as the corresponding image.

    Args:
        disp: (numpy.ndarray or torch.Tensor, shape (N, ndim, *size)) Displacement field

    Returns:
        disp: (normalised disp)
    """

    ndim = disp.ndim - 2

    if type(disp) is np.ndarray:
        norm_factors = 2. / np.array(disp.shape[2:])
        norm_factors = norm_factors.reshape(1, ndim, *(1,) * ndim)

    elif type(disp) is torch.Tensor:
        norm_factors = torch.tensor(2.) / torch.tensor(disp.size()[2:], dtype=disp.dtype, device=disp.device)
        norm_factors = norm_factors.view(1, ndim, *(1,)*ndim)

    else:
        raise RuntimeError("Input data type not recognised, expect numpy.ndarray or torch.Tensor")
    return disp * norm_factors


def svf_exp(flow, scale=1, steps=7, sampling='bilinear'):
    """ Exponential of velocity field by Scaling and Squaring"""
    disp = flow * (scale / (2 ** steps))
    for _ in range(steps):
        disp = disp + warp(x=disp, disp=disp,
                           interp_mode=sampling)
    return disp


def cubic_bspline_value(x, derivative) -> float:
    r"""Evaluate 1-dimensional cubic B-spline."""
    t = abs(x)
    # outside local support region
    if t > 2:
        return 0
    # 0-th order derivative
    if derivative == 0:
        if t < 1:
            return 2 / 3 + (0.5 * t - 1) * t ** 2
        return ((2 - t) ** 3) / 6
    # 1st order derivative
    if derivative == 1:
        if t < 1:
            return (1.5 * t - 2.0) * x
        if x < 0:
            return 0.5 * (t - 2) ** 2
        return 0.5 * (2 - t) ** 2
    # 2nd oder derivative
    if derivative == 2:
        if t < 1:
            return 3 * t - 2
        return -t + 2
    
def three_order_bspline_value(x, derivative) -> float:
    r"""Evaluate 1-dimensional cubic B-spline."""
    t = abs(x)
    # outside local support region
    if t > 3:
        return 0
    # 0-th order derivative
    if derivative == 0:
        if t <= 1:
            return (t ** 3) / 6.0
        elif 1 < t <= 2:
            return (1 / 6.0) * (4 - 6 * t + 3 * t ** 2)
        elif 2 < t <= 3:
            return (1 / 6.0) * (4 - t) ** 3
    # 1st order derivative
    elif derivative == 1:
        if t <= 1:
            return 0.5 * x ** 2
        elif 1 < t <= 2:
            return (1 / 6.0) * (-6 + 6 * t)
        elif 2 < t <= 3:
            return 0.5 * (3 - t) ** 2
    # 2nd order derivative
    elif derivative == 2:
        if t <= 1:
            return x
        elif 1 < t <= 2:
            return 1 - t
        elif 2 < t <= 3:
            return t - 3
    else:
        raise ValueError("Unsupported derivative order")


def cubic_bspline1d(stride, derivative_weights, dtype=None, device= None) -> torch.Tensor:
    r"""Cubic B-spline kernel for specified control point spacing.

    Args:
        stride: Spacing between control points with respect to original (upsampled) image grid.
        derivative: Order of cubic B-spline derivative.

    Returns:
        Cubic B-spline convolution kernel.

    """
    if dtype is None:
        dtype = torch.float
    if not isinstance(stride, int):
        (stride,) = stride
    kernel = torch.ones(4 * stride - 1, dtype=dtype) # Bspline核定义 4个控制点所以是4*stride
    radius = kernel.shape[0] // 2 # kernel的半径
    '''
    for i in range(kernel.shape[0]):
        #d = abs(abs(abs(i) - 4) - 2)
        #d = abs(abs(i) - 4)
        #kernel[i] = cubic_bspline_value((i - radius) / stride, derivative=d) # (i - radius) / stride 实际上是插值位置的坐标，确保了在进行卷积操作时，涉及到足够的邻近控制点
        for j, derivative_weight in enumerate(derivative_weights):
            for derivative, weights in enumerate(derivative_weights[j]):
                kernel[i] = weights * three_order_bspline_value((i - radius), derivative=derivative) # (i - radius) / stride 实际上是插值位置的坐标，确保了在进行卷积操作时，涉及到足够的邻近控制点
    '''
    for i in range(kernel.shape[0]):
        if i < len(derivative_weights):
            weights = derivative_weights[i]
            for derivative, weight in enumerate(weights):
                kernel[i] = weight * three_order_bspline_value((i - radius) / stride, derivative=derivative)

    if device is None:
        device = kernel.device
    return kernel.to(device)


def conv1d(
        data: Tensor,
        kernel: Tensor,
        dim: int = -1,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        transpose: bool = False
) -> Tensor:
    r"""Convolve data with 1-dimensional kernel along specified dimension."""
    result = data.type(kernel.dtype)  # (n, ndim, h, w, d)
    result = result.transpose(dim, -1)  # (n, ndim, ..., shape[dim])
    shape_ = result.size()

    groups = int(torch.prod(torch.tensor(shape_[1:-1])))
    # groups = numel(shape_[1:-1])  # (n, nidim * shape[not dim], shape[dim])
    weight = kernel.expand(groups, 1, kernel.shape[-1])  # 3*w*d, 1, kernel_size
    result = result.reshape(shape_[0], groups, shape_[-1])  # n, 3*w*d, shape[dim]
    conv_fn = F.conv_transpose1d if transpose else F.conv1d
    result = conv_fn(
        result,
        weight,
        stride=stride,
        dilation=dilation,
        padding=padding,
        groups=groups,
    )
    result = result.reshape(shape_[0:-1] + result.shape[-1:])
    result = result.transpose(-1, dim)
    return result


def warp(x, disp, interp_mode="bilinear"):
    """
    Spatially transform an image by sampling at transformed locations (2D and 3D)

    Args:
        x: (Tensor float, shape (N, ndim, *sizes)) input image
        disp: (Tensor float, shape (N, ndim, *sizes)) dense disp field in i-j-k order (NOT spatially normalised)
        interp_mode: (string) mode of interpolation in grid_sample()

    Returns:
        deformed x, Tensor of the same shape as input
    """
    ndim = x.ndim - 2
    size = x.size()[2:]
    disp = disp.type_as(x)
    #print('disp size:', disp.shape)
    # normalise disp to [-1, 1]
    disp = normalise_disp(disp)

    # generate standard mesh grid
    grid = torch.meshgrid([torch.linspace(-1, 1, size[i]).type_as(disp) for i in range(ndim)])
    grid = [grid[i].requires_grad_(False) for i in range(ndim)]

    # apply displacements to each direction (N, *size)

    warped_grid = [grid[i] + disp[:, i, ...] for i in range(ndim)]

    # swapping i-j-k order to x-y-z (k-j-i) order for grid_sample()
    warped_grid = [warped_grid[ndim - 1 - i] for i in range(ndim)]
    warped_grid = torch.stack(warped_grid, -1)  # (N, *size, dim)

    return F.grid_sample(x, warped_grid, mode=interp_mode, align_corners=False)

# ----------------------------------------------voxelmorph-------------------------------------
class UnetBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, img_in):
        img_out = self.conv1(img_in)
        img_out = self.bn1(img_out)
        img_out = self.relu(img_out)

        img_out = self.conv2(img_out)
        img_out = self.bn2(img_out)
        img_out = self.relu(img_out)

        return img_out

# Unet网络
class UNetvm(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        # nb_filter = [32, 64, 128, 256, 512]
        nb_filter = (16, 32, 32, 32, 32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.enc1 = UnetBlock(input_channels, nb_filter[0], nb_filter[0])
        self.enc2 = UnetBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.enc3 = UnetBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.enc4 = UnetBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.enc5 = UnetBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.dec1 = UnetBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.dec2 = UnetBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.dec3 = UnetBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.dec4 = UnetBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.dec5 = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)

    def forward(self, img_in):
        img_enc1 = self.enc1(img_in)
        img_enc2 = self.enc2(self.pool(img_enc1))
        img_enc3 = self.enc3(self.pool(img_enc2))
        img_enc4 = self.enc4(self.pool(img_enc3))
        img_enc5 = self.enc5(self.pool(img_enc4))
        img_dec1 = self.dec1(torch.cat([img_enc4, self.up(img_enc5)], dim=1))
        img_dec2 = self.dec2(torch.cat([img_enc3, self.up(img_dec1)], dim=1))
        img_dec3 = self.dec3(torch.cat([img_enc2, self.up(img_dec2)], dim=1))
        img_dec4 = self.dec4(torch.cat([img_enc1, self.up(img_dec3)], dim=1))
        img_out = self.dec5(img_dec4)
        return img_out

# STN空间变换网络
class SpatialTransformer(nn.Module):
    def __init__(self, img_shape):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid  B*C*H*W*D
        vectors = [torch.arange(0, s) for s in img_shape]
        grid = torch.stack(torch.meshgrid(vectors)).unsqueeze(0).type(torch.float32)
        self.register_buffer('grid', grid)

    def forward(self, img_moving, flow, mode='bilinear'):
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

        img_warped = F.grid_sample(img_moving, new_locs, align_corners=True, mode=mode)
        return img_warped

def train():
    # 准备工作
    # 创建文件夹
    print("----Make directory----")
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # 创建log文件 命名：迭代次数_学习率_正则化系数
    print("----Make log file----")
    log_name_bsp = "%d_%lf_%f_bsp.csv" % (args.n_iter, args.lr, args.alpha)
    log_name_vm = "%d_%lf_%f_vm.csv" % (args.n_iter, args.lr, args.alpha)
    print("BSP_log_name: ", log_name_bsp)
    
    file_log_bsp = open(os.path.join(args.log_dir, log_name_bsp), "w")
    file_log_vm = open(os.path.join(args.log_dir, log_name_vm), "w")
    print("BSP iter,train_loss,sim_loss,grad_loss,valid_loss,sim_loss,grad_loss,valid_dice", file=file_log_bsp)
    print("VM iter,train_loss,sim_loss,grad_loss,valid_loss,sim_loss,grad_loss,valid_dice", file=file_log_vm)


    # 创建配准网络 unet+stn
    print("----Build registration network----")
    img_size = [128,128]
    unet_bsp = CubicBSplineNet(ndim=2, img_size=img_size).cuda()
    #stn_bsp = CubicBSplineFFDTransform(ndim=2, img_size=img_size)  # 创建stn需要shape
    stn_bsp = SpatialTransform_bsp(ndim=2, cps=[4,4], img_size=img_size).cuda()
    unet_vm = UNetvm(input_channels=2, output_channels=len(img_size)).cuda()
    stn_vm = SpatialTransformer(img_size).cuda()
    
    # 模型参数个数
    print("unet: ", countParameters(unet_bsp)+countParameters(unet_vm))

    # 设置优化器和loss函数
    print("----Set initial parameters----")
    param_bsp = list(unet_bsp.parameters()) + list(stn_bsp.parameters()) 
    opt_bsp = Adam(param_bsp, lr=args.lr)
    opt_vm = Adam(unet_vm.parameters(), lr=args.lr)
    if args.sim_loss == "mse":
        sim_loss_fn = compute_mse
    elif args.sim_loss == "ncc":
        sim_loss_fn = ncc_loss
    else:
        sim_loss_fn = ncc_loss
    grad_loss_fn = compute_gradient
    LNCC_loss = LNCCLoss(window_size=[7,7])
    # 数据处理
    print("----Process data----")
    
    
    train_list = np.arange(0, 60)
    valid_list = np.arange(60, 76)             

    dataset_train_img = torch.zeros([2, 60, 128, 128], dtype=torch.float32)
    dataset_valid_img = torch.zeros([2, 16, 128, 128], dtype=torch.float32)
    

    subject_forms = ["CINE", "DE"]
    # CINE或DE
    for _form in range(2):
        # 训练集
        for _num in range(len(train_list)):
            subject = train_list[_num] + 1
            file_pattern = os.path.join(args.train_dir, f"EP{subject}_{subject_forms[_form]}*.nii.gz")
            file_list = glob.glob(file_pattern)
            # img
            if len(file_list) > 0:
                file_path = file_list[0]
                data = torch.from_numpy(nib.load(file_path).get_fdata()).type(torch.float32)
                dataset_train_img[_form, _num, :, :] = (data - data.min()) / (data.max() - data.min())
        # 验证集
        for _num in range(len(valid_list)):
            subject = valid_list[_num] + 1
            file_pattern = os.path.join(args.train_dir, f"EP{subject}_{subject_forms[_form]}*.nii.gz")
            file_list = glob.glob(file_pattern)
            if len(file_list) > 0:
                file_path = file_list[0]
                data = torch.from_numpy(nib.load(file_path).get_fdata()).type(torch.float32)
                dataset_valid_img[_form, _num, :, :] = (data - data.min()) / (data.max() - data.min())
                

    # 开始训练
    print("----Start training----")
    # 观察损失曲线
    train_losses_vm = []
    valid_losses_vm = []
    train_losses_bsp = []
    valid_losses_bsp = []
    # 计时
    start_time = float(time.time())

    vm_best_valid_loss = 0.0
    vm_final_train_loss = 0.0
    vm_final_valid_loss = 0.0
    bsp_best_valid_loss = 0.0
    bsp_final_train_loss = 0.0
    bsp_final_valid_loss = 0.0
    
    for _iter in range(1, args.n_iter + 1):
        # 将train_data_list进行随机排序
        train_list_permuted = np.random.permutation(train_list)
        # ------------------------------训练bspline------------------------------
        if _iter <= args.bsp_iter:
            # ----------------------训练部分----------------------
            sim_loss_train = 0.0
            grad_loss_train = 0.0
            loss_train = 0.0
            unet_bsp.train()
            stn_bsp.train()
            opt_bsp.zero_grad()
            
            # 以batch_size为步长批量读取数据
            steps = len(train_list_permuted) // args.batch_size
            for _step in range(steps):
                # 预先定义fixed 和 moving 张量 batch_size*C*H*W*D
                img_fixed = torch.zeros([args.batch_size, 1, 128, 128], dtype=torch.float32)
                img_moving = torch.zeros([args.batch_size, 1, 128, 128], dtype=torch.float32)

                # 迭代读取fixed 和 moving图像
                for _batch in range(args.batch_size):
                    subject = _step * args.batch_size + _batch
                    img_moving[_batch, 0, :, :] = dataset_train_img[0, subject, :, :]
                    img_fixed[_batch, 0, :, :] = dataset_train_img[1, subject, :, :]

                img_fixed = img_fixed.cuda()
                img_moving = img_moving.cuda()
                input_image = torch.cat([img_fixed, img_moving], dim=1)
                input_image_1 = torch.cat([img_moving, img_fixed], dim=1)
                
                # ---------------输入Bspline网络-------------------
                BspParam = unet_bsp(input_image) # 得到预测的网络参数
                disp = stn_bsp(BspParam)
                img_warped = warp(img_moving, disp)
                
                BspParam_bw = unet_bsp(input_image_1)
                disp_bw = stn_bsp(BspParam_bw)
                    
                # 计算loss
                #sim_loss = sim_loss_fn(img_warped, img_fixed)
                sim_loss = LNCC_loss(img_warped,img_fixed)
                grad_loss = (grad_loss_fn(disp) + grad_loss_fn(disp_bw)) / 2 # 加上反向形变场损失
                #smooth_loss = deformation_smooth_loss(disp) + deformation_smooth_loss(disp_bw)      
                loss = args.alpha * grad_loss + sim_loss
                #loss = 1 * sim_loss + args.alpha * grad_loss


                # Backwards and optimize
                loss.backward()
                opt_bsp.step()
                
                sim_loss_train += sim_loss.item()
                grad_loss_train += grad_loss.item()
                loss_train += loss.item()

            sim_loss_train /= steps
            grad_loss_train /= steps
            loss_train /= steps

            # --------------------------验证部分-------------------------
            sim_loss_valid = 0.0
            grad_loss_valid = 0.0
            loss_valid = 0.0
            
            #dice_valid = 0.0
            unet_bsp.eval()
            stn_bsp.eval()
            with torch.no_grad():
                img_fixed = torch.zeros([1, 1, 128, 128], dtype=torch.float32)
                img_moving = torch.zeros([1, 1, 128, 128], dtype=torch.float32)
                

                for _num in range(len(valid_list)):
                    # img & label
                    img_moving[0, 0, :, :] = dataset_valid_img[0, _num, :, :]
                    img_fixed[0, 0, :, :] = dataset_valid_img[1, _num, :, :]
                    
                    img_fixed = img_fixed.cuda()
                    img_moving = img_moving.cuda()

                    # 做拼接后输入网络
                    input_image = torch.cat([img_fixed, img_moving], dim=1)
                    input_image_1 = torch.cat([img_moving, img_fixed], dim=1)
                    # 输入网络
                    BspParam = unet_bsp(input_image) #得到预测的网络参数
                    disp = stn_bsp(BspParam)
                    img_warped = warp(img_moving, disp)
                    
                    BspParam_bw = unet_bsp(input_image_1)
                    disp_bw = stn_bsp(BspParam_bw)
                        
                    # 计算loss
                    #sim_loss = sim_loss_fn(img_warped, img_fixed)
                    sim_loss = LNCC_loss(img_warped,img_fixed)
                    grad_loss = (grad_loss_fn(disp) + grad_loss_fn(disp_bw)) / 2 # 加上反向形变场损失    
                    #smooth_loss = deformation_smooth_loss(disp) + deformation_smooth_loss(disp_bw)      
                    loss = args.alpha * grad_loss +  sim_loss
                    

                    sim_loss_valid += sim_loss.item()
                    grad_loss_valid += grad_loss.item()
                    loss_valid += loss.item()
                    

            sim_loss_valid /= len(valid_list)
            grad_loss_valid /= len(valid_list)
            loss_valid /= len(valid_list)
            
            # 记录损失
            train_losses_bsp.append(loss_train)
            valid_losses_bsp.append(loss_valid)
            
            print("BSP_epoch: %d  train_loss: %f  sim_loss: %f  grad_loss: %f" % (_iter, loss_train, sim_loss_train, grad_loss_train), flush=True)
            print("BSP_epoch: %d  valid_loss: %f  sim_loss: %f  grad_loss: %f" % (_iter, loss_valid, sim_loss_valid, grad_loss_valid), flush=True)
            print("%d,%f,%f,%f,%f,%f,%f" % (_iter, loss_train, sim_loss_train, grad_loss_train, loss_valid, sim_loss_valid, grad_loss_valid), file=file_log_bsp)

            # 计时
            if _iter % 10 == 0:
                print("----time_used: %f" % float(time.time() - start_time), flush=True)
                print("----time_used: %f" % float(time.time() - start_time), file=file_log_bsp)

            # 保存最佳模型参数
            if loss_valid < bsp_best_valid_loss:
                bsp_best_valid_loss = loss_valid
                bsp_final_train_loss = loss_train 
                bsp_final_valid_loss = loss_valid
                # Save model checkpoint
                save_file_dir = os.path.join(args.model_dir, f"bspline_stg1_{_iter}.pth")
                save_stn_dir = os.path.join(args.model_dir, "stn_bsp.pth")
                torch.save(unet_bsp.state_dict(), save_file_dir)
                torch.save(stn_bsp.state_dict(), save_stn_dir)

             
        # ---------------------------------------训练vm--------------------------------------
        else:
            # 冻结bsp网络参数
            for param in unet_bsp.parameters():
                param.requires_grad = False
            for param in stn_bsp.parameters():
                param.requires_grad = False
            # ---------------------------训练部分------------------------------
            sim_loss_train = 0.0
            grad_loss_train = 0.0
            loss_train = 0.0
            unet_vm.train()
            stn_vm.train()
            opt_vm.zero_grad()
            
            # 以batch_size为步长批量读取数据
            steps = len(train_list_permuted) // args.batch_size
            for _step in range(steps):
                # 预先定义fixed 和 moving 张量 batch_size*C*H*W*D
                img_fixed = torch.zeros([args.batch_size, 1, 128, 128], dtype=torch.float32)
                img_moving = torch.zeros([args.batch_size, 1, 128, 128], dtype=torch.float32)

                # 迭代读取fixed 和 moving图像
                for _batch in range(args.batch_size):
                    subject = _step * args.batch_size + _batch
                    img_moving[_batch, 0, :, :] = dataset_train_img[0, subject, :, :]
                    img_fixed[_batch, 0, :, :] = dataset_train_img[1, subject, :, :]

                img_fixed = img_fixed.cuda()
                img_moving = img_moving.cuda()
                BspParam_vm = unet_bsp(torch.cat([img_fixed, img_moving], dim=1))
                disp_vm = stn_bsp(BspParam_vm)
                img_trans = warp(img_moving, disp_vm)
                input_image = torch.cat([img_fixed, img_trans], dim=1)
                input_image_1 = torch.cat([img_trans, img_fixed], dim=1)
                # --------------------输入vm网络-------------------
                flow = unet_vm(input_image) # 得到预测的网络参数
                img_warped = stn_vm(img_trans, flow, mode='bilinear')
                flow_bw = unet_vm(input_image_1)
                    
                # 计算loss
                sim_loss = sim_loss_fn(img_warped, img_fixed)
                grad_loss = (grad_loss_fn(flow) + grad_loss_fn(flow_bw)) / 2 # 加上反向形变场损失
                loss = args.alpha * grad_loss + sim_loss


                # Backwards and optimize
                loss.backward()
                opt_vm.step()

                sim_loss_train += sim_loss.item()
                grad_loss_train += grad_loss.item()
                loss_train += loss.item()

            sim_loss_train /= steps
            loss_train /= steps

            # -----------------------------------验证部分---------------------------------
            sim_loss_valid = 0.0
            grad_loss_valid = 0.0
            loss_valid = 0.0
            
            #dice_valid = 0.0
            unet_vm.eval()
            stn_vm.eval()
            with torch.no_grad():
                img_fixed = torch.zeros([1, 1, 128, 128], dtype=torch.float32)
                img_moving = torch.zeros([1, 1, 128, 128], dtype=torch.float32)
                

                for _num in range(len(valid_list)):
                    # img & label
                    img_moving[0, 0, :, :] = dataset_valid_img[0, _num, :, :]
                    img_fixed[0, 0, :, :] = dataset_valid_img[1, _num, :, :]
                    
                    img_fixed = img_fixed.cuda()
                    img_moving = img_moving.cuda()
                    
                    # 做拼接后输入网络
                    BspParam_vm = unet_bsp(torch.cat([img_fixed, img_moving], dim=1))
                    disp_vm = stn_bsp(BspParam_vm)
                    img_trans = warp(img_moving, disp_vm)
                    input_image = torch.cat([img_fixed, img_trans], dim=1)
                    input_image_1 = torch.cat([img_trans, img_fixed], dim=1)
                    
                    # --------------------输入vm网络-------------------
                    flow = unet_vm(input_image) # 得到预测的网络参数
                    img_warped = stn_vm(img_trans, flow, mode='bilinear')
                    flow_bw = unet_vm(input_image_1)
                        
                    # 计算loss
                    sim_loss = sim_loss_fn(img_warped, img_fixed)
                    grad_loss = (grad_loss_fn(flow) + grad_loss_fn(flow_bw)) /2 # 加上反向形变场损失
                    loss = args.alpha * grad_loss + sim_loss

                    sim_loss_valid += sim_loss.item()
                    grad_loss_valid += grad_loss.item()
                    loss_valid += loss.item()
                        

            sim_loss_valid /= len(valid_list)
            grad_loss_valid /= len(valid_list)
            loss_valid /= len(valid_list)

            # 记录损失
            train_losses_vm.append(loss_train)
            valid_losses_vm.append(loss_valid)
            
            print("VM_epoch: %d  train_loss: %f  sim_loss: %f  grad_loss: %f" % (_iter, loss_train, sim_loss_train, grad_loss_train), flush=True)
            print("VM_epoch: %d  valid_loss: %f  sim_loss: %f  grad_loss: %f" % (_iter, loss_valid, sim_loss_valid, grad_loss_valid), flush=True)
            print("%d,%f,%f,%f,%f,%f,%f" % (_iter, loss_train, sim_loss_train, grad_loss_train, loss_valid, sim_loss_valid, grad_loss_valid), file=file_log_vm)

            # 计时
            if _iter % 10 == 0:
                print("----time_used: %f" % float(time.time() - start_time), flush=True)
                print("----time_used: %f" % float(time.time() - start_time), file=file_log_vm)

            # 保存最佳模型参数
            if loss_valid < vm_best_valid_loss:
                vm_best_valid_loss = loss_valid
                vm_final_train_loss = loss_train 
                vm_final_valid_loss = loss_valid
                # Save model checkpoint
                save_file_dir = os.path.join(args.model_dir,  f"vm_stg2_{_iter}.pth")
                torch.save(unet_vm.state_dict(), save_file_dir)
                
        
                # 解冻Bspline网络参数
            for param in unet_bsp.parameters():
                param.requires_grad = True
            for param in stn_bsp.parameters():
                param.requires_grad = True

     # 保存损失数据到文件
    with open('train_losses_bsp.txt', 'w') as f:
        for loss in train_losses_bsp:
            f.write(str(loss) + '\n')

    with open('valid_losses_bsp.txt', 'w') as f:
        for loss in valid_losses_bsp:
            f.write(str(loss) + '\n')
            
    with open('train_losses_vm.txt', 'w') as f:
        for loss in train_losses_vm:
            f.write(str(loss) + '\n')

    with open('valid_losses_vm.txt', 'w') as f:
        for loss in valid_losses_vm:
            f.write(str(loss) + '\n')
    
    # 打印Log
    print("final_train_loss = %f,final_valid_loss = %f" % (bsp_final_train_loss, bsp_final_valid_loss), flush=True)
    print("final_train_loss = %f,final_valid_loss = %f" % (bsp_final_train_loss, bsp_final_valid_loss), file=file_log_bsp)
    file_log_bsp.close()    
    print("final_train_loss = %f,final_valid_loss = %f" % (vm_final_train_loss, vm_final_valid_loss), flush=True)
    print("final_train_loss = %f,final_valid_loss = %f" % (vm_final_train_loss, vm_final_valid_loss), file=file_log_vm)
    file_log_vm.close()


def test():
    # 创建文件夹
    print("----Make directory----")
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # 创建配准网络 unet+stn
    print("----Build registration network----")
    img_size=[128,128]
    unet_bsp = CubicBSplineNet(ndim=2, img_size=[128,128]).cuda()
    stn_bsp = SpatialTransform_bsp(ndim=2, cps=[4,4], img_size=img_size).cuda()
    #stn_bsp = CubicBSplineFFDTransform(ndim=2, img_size=img_size)  # 创建stn需要shape
    unet_vm = UNetvm(input_channels=2, output_channels=len(img_size)).cuda()
    stn_vm = SpatialTransformer(img_size).cuda()
    unet_bsp.load_state_dict(torch.load(args.checkpoint_path_bsp))
    unet_vm.load_state_dict(torch.load(args.checkpoint_path_vm))
    stn_bsp.load_state_dict(torch.load(args.checkpoint_path_bspstn))
    unet_bsp.eval()
    unet_vm.eval()
    stn_bsp.eval()
    stn_vm.eval()

    # 数据处理
    print("----Process data----")

    # 测试序列
    test_list = np.arange(76, 92)

    # 读取图像数据
    dataset_img = torch.zeros([2, 16, 128, 128], dtype=torch.float32)
    subject_forms = ["CINE", "DE"]

    for _form in range(2):
        # 测试集
        for _num in range(len(test_list)):
            subject = test_list[_num] + 1
            file_pattern = os.path.join(args.test_dir, f"EP{subject}_{subject_forms[_form]}*.nii.gz")
            file_list = glob.glob(file_pattern)
            if len(file_list) > 0:
                file_path = file_list[0]
                data = torch.from_numpy(nib.load(file_path).get_fdata()).type(torch.float32)
                dataset_img[_form, _num, :, :] = (data - data.min()) / (data.max() - data.min())
    # 开始测试
    print("----Start testing----")
    # 计时
    time_list = []
    psnr_list = []
    ssim_list = []
    jac_list = []

    img_fixed = torch.zeros([1, 1, 128, 128], dtype=torch.float32)
    img_moving = torch.zeros([1, 1, 128, 128], dtype=torch.float32)

    for _num in range(len(test_list)):
        # 创建subject文件目录 
        subject = test_list[_num] + 1
        subject_dir = os.path.join(args.result_dir, "EP%d" % subject)
        if not os.path.exists(subject_dir):
            os.mkdir(subject_dir)

        # img & label
        img_moving[0, 0, :, :] = dataset_img[0, _num, :, :]
        img_fixed[0, 0, :, :] = dataset_img[1, _num, :, :]
        img_fixed = img_fixed.cuda()
        img_moving = img_moving.cuda()
        # 做拼接后输入网络 计时
        start_time = time.time()
        BspParam = unet_bsp(torch.cat([img_fixed, img_moving], dim=1))
        disp = stn_bsp(BspParam)
        img_trans = warp(img_moving, disp)
        
        flow = unet_vm(torch.cat([img_fixed, img_trans], dim=1))
        img_warped = stn_vm(img_trans, flow, mode='bilinear')

        time_list.append([float(time.time() - start_time)])

        # 计算psnr
        if args.psnr:
            psnr_list.append([compute_PSNR(img_fixed, img_moving), compute_PSNR(img_fixed, img_warped)])
        # 计算ssim
        if args.ssim:
            ssim_list.append([compute_SSIM(img_fixed, img_moving), compute_SSIM(img_fixed, img_warped)])
        # 计算雅克比行列式分数
        if args.jac:
            jac = Jacobian(flow)
            jac_list.append([jac.count_minus_ratio()])

        # 保存图像
        # img & label
        img = nib.Nifti1Image(img_fixed[0, 0, :, :].cpu().detach().numpy(), None)
        nib.save(img, os.path.join(subject_dir, "fixed.nii.gz"))

        img = nib.Nifti1Image(img_moving[0, 0, :, :].cpu().detach().numpy(), None)
        nib.save(img, os.path.join(subject_dir, "moving.nii.gz"))

        img = nib.Nifti1Image(img_warped[0, 0, :, :].cpu().detach().numpy(), None)
        nib.save(img, os.path.join(subject_dir, "warped.nii.gz"))
        
        DVF = nib.Nifti1Image(flow[0, :, :, :].cpu().detach().numpy(), None)
        nib.save(DVF, os.path.join(subject_dir, "flow.nii.gz"))

    print("time_used = %f" % np.sum(time_list))

    # 保存结果
    with open(os.path.join(args.result_dir, "result.csv"), "w") as f:
        writer = csv.writer(f)
        header = ["time"]
        data = np.array(time_list)
        if args.psnr:
            header.append("psnr_pre")
            header.append("psnr_done")
            psnr_list = np.array(psnr_list)
            data = np.append(data, psnr_list, axis=1)
        if args.ssim:
            header.append("ssim_pre")
            header.append("ssim_done")
            ssim_list = np.array(ssim_list)
            data = np.append(data, ssim_list, axis=1)
        if args.jac:
            header.append("jac")
            jac_list = np.array(jac_list)
            data = np.append(data, jac_list, axis=1)
        writer.writerow(header)
        writer.writerows(data)


if __name__ == "__main__":
    if args.pattern == "train":
        train()
    else:
        test()
    print("end")
