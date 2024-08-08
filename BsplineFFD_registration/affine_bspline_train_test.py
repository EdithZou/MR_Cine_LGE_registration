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
from torch.autograd import gradcheck
import numpy as np
import nibabel as nib
import skimage.metrics as metrics  # 这个里面包含了很多评估指标的计算方法 PSNR SSIM等
from skimage.transform import resize
import matplotlib.pyplot as plt
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()

# 文件路径
parser.add_argument("--train_dir", type=str, help="data folder with training vols", dest="train_dir", default="/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Xiehe2D_resize")
parser.add_argument("--model_dir", type=str, help="models folder", dest="model_dir", default="/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Checkpoint_Xiehe")
parser.add_argument("--log_dir", type=str, help="logs folder", dest="log_dir", default="/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Log_Xiehe")
parser.add_argument("--result_dir", type=str, help="results folder", dest="result_dir", default="/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Result_Affine_Xiehe")

# network parameters
parser.add_argument("--pattern", type=str, help="select train or test", dest="pattern", default="train")

# training parameters
parser.add_argument("--lr", type=float, help="learning rate", dest="lr", default=3*1e-4)
parser.add_argument("--n_iter", type=int, help="number of iterations", dest="n_iter", default=300)
parser.add_argument("--aff_iter", type=int, help="number of iterations", dest="aff_iter", default=200)
parser.add_argument("--sim_loss", type=str, help="image similarity loss: mse or ncc", dest="sim_loss", default="ncc")
parser.add_argument("--alpha", type=float, help="regularization parameter", dest="alpha", default=0.5)  # recommend 1.0 for ncc, 0.01 for mse
parser.add_argument("--beta", type=float, help="regularization parameter for b-spline", dest="beta", default=0.1)  # recommend 1.0 for ncc, 0.01 for mse
parser.add_argument("--batch_size", type=int, help="batch_size", dest="batch_size", default=1)

# testing parameters
parser.add_argument("--test_dir", type=str, help="test data directory", dest="test_dir", default="/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Xiehe2D_resized_test")
parser.add_argument("--checkpoint_path", type=str, help="model weight file", dest="checkpoint_path", default="/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Checkpoint_Xiehe/trained_model.pth")
parser.add_argument("--checkpoint_path_bsp", type=str, help="bspmodel weight file", dest="checkpoint_path_bsp", default="/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Checkpoint_Xiehe/trained_model.pth")
parser.add_argument("--checkpoint_path_bspstn", type=str, help="bspmodel weight file", dest="checkpoint_path_bspstn", default="/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Checkpoint_Xiehe/trained_model.pth")

parser.add_argument("--dice", type=bool, help="if compute dice", dest="dice", default=False)
parser.add_argument("--psnr", type=bool, help="if compute psnr", dest="psnr", default=True)
parser.add_argument("--ssim", type=bool, help="if compute ssim", dest="ssim", default=True)
parser.add_argument("--jacobian", type=bool, help="if compute jacobian", dest="jac", default=False)

args = parser.parse_args()

# 计算反向形变场
def invert_deformation_field(forward_deformation_field):
    # 获取输入形变场的维度信息
    B, C, H, W = forward_deformation_field.size()

    # 生成标准网格
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
    grid = torch.stack([grid_x, grid_y], dim=-1).float()

    # 将网格坐标转化为[-1, 1]的规范化坐标
    normalized_grid = (grid.unsqueeze(0).repeat(B, 1, 1, 1).float() / torch.tensor([W - 1, H - 1], dtype=torch.float32).view(1, 1, 1, 2) * 2.0) - 1.0
    # 获取正向形变场的偏移量
    deformation_offsets = forward_deformation_field.permute(0, 2, 3, 1)

    # 计算反向形变场的规范化坐标
    normalized_grid = normalized_grid.to(deformation_offsets.device)
    inverse_normalized_grid = normalized_grid - deformation_offsets
    device = inverse_normalized_grid.device
    # 将反向形变场的规范化坐标映射回[0, H-1]和[0, W-1]的坐标
    device_tensor = torch.tensor([W - 1, H - 1], dtype=torch.float32, device=device)
    inverse_grid = ((inverse_normalized_grid + 1.0) / 2.0) * device_tensor
    #inverse_grid = ((inverse_normalized_grid + 1.0) / 2.0) * torch.tensor([W - 1, H - 1], dtype=torch.float32).view(1, 1, 1, 2)

    # 反向形变场为反向网格坐标与标准网格的差异
    inverse_deformation_field = inverse_grid - grid.unsqueeze(0).unsqueeze(-1).float()

    return inverse_deformation_field.permute(0, 3, 1, 2)  # 将维度调整为[B, 2, H, W]

# symmetric loss
def scale_sym_reg_loss(affine_param, sched='l2'):
        """
        in symmetric forward, compute regularization loss of  affine parameters,
        l2: compute the l2 loss between the affine parameter and the identity parameter
        det: compute the determinant of the affine parameter, which prefers to rigid transformation

        :param sched: 'l2' , 'det'
        :return: the regularization loss on batch
        """
        
        loss = scale_multi_step_reg_loss(affine_param,sched)
        return loss
    
def scale_multi_step_reg_loss(affine_param, sched='l2'):
        """
        compute regularization loss of  affine parameters,
        l2: compute the l2 loss between the affine parameter and the identity parameter
        det: compute the determinant of the affine parameter, which prefers to rigid transformation

        :param sched: 'l2' , 'det'
        :return: the regularization loss on batch
        """
        affine_identity = torch.zeros(3, 3).cuda()
        affine_identity[0, 0] = 1.  
        affine_identity[1, 1] = 1.  
        affine_identity[2, 2] = 1.  
        
        affineparam = affine_param.view(-1,3,3)
        weight_mask = torch.ones(3,3).cuda()
        bias_factor = 1.0
        weight_mask[1,:]=bias_factor
        weight_mask = weight_mask.view(-1)
        if sched == 'l2':
            return torch.sum((affine_identity - affineparam) ** 2)
        elif sched == 'det':
            mean_det = 0.
            for i in range(affine_param.shape[0]):
                affine_matrix = affine_param[i, :9].contiguous().view(3, 3)
                mean_det += torch.det(affine_matrix)
            return mean_det / affine_param.shape[0]

# mse loss
def compute_mse(tensor_x, tensor_y):
    mse = torch.mean((tensor_x - tensor_y) ** 2)
    return mse

# 归一化
def normalize_tensor(tensor):
    mean = tensor.mean()
    std = tensor.std()
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor

# gradient loss for [B,H,W,C]
def compute_gradient_aff(tensor_x):
    dims = tensor_x.ndim
    gradient = 0.0
    epsilon = 1e-8
    if dims == 4:
        dx = (tensor_x[:, 1:, :, :] - tensor_x[:, :-1, :, :]) ** 2
        dy = (tensor_x[:, :, 1:, :] - tensor_x[:, :, :-1, :]) ** 2
        gradient = (dx.mean() + dy.mean()) / 2
        
    elif dims == 5:
        dx = (tensor_x[:, :, 1:, :, :] - tensor_x[:, :, :-1, :, :]) ** 2
        dy = (tensor_x[:, :, :, 1:, :] - tensor_x[:, :, :, :-1, :]) ** 2
        dz = (tensor_x[:, :, :, :, :] - tensor_x[:, :, :, :, :]) ** 2
        gradient = (dx.mean() + dy.mean() + dz.mean()) / 2
    
    return gradient

# gradient loss for [B,C,H,W]
def compute_gradient(tensor_x):
    dims = tensor_x.ndim
    gradient = 0.0
    epsilon = 1e-8
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

# Regularize loss
class DisplacementRegularizer2D(nn.Module):
    def __init__(self, energy_type):
        super().__init__()
        self.energy_type = energy_type

    def gradient_dx(self, fv):
        return (fv[:, :, 1:, :] - fv[:, :, :-1, :]) / 2

    def gradient_dy(self, fv):
        return (fv[:, :, :, 1:] - fv[:, :, :, :-1]) / 2

    def gradient_txy(self, Txy, fn):
        return torch.stack([fn(Txy[:, i, ...]) for i in [0, 1]], dim=1)

    def compute_gradient_norm(self, displacement, flag_l1=False):
        dTdx = self.gradient_txy(displacement, self.gradient_dx)
        dTdy = self.gradient_txy(displacement, self.gradient_dy)
        if flag_l1:
            norms = torch.abs(dTdx) + torch.abs(dTdy)
        else:
            norms = dTdx**2 + dTdy**2
        return torch.mean(norms) / 2.0

    def compute_bending_energy(self, displacement):
        dTdx = self.gradient_txy(displacement, self.gradient_dx)
        dTdy = self.gradient_txy(displacement, self.gradient_dy)
        dTdxx = self.gradient_txy(dTdx, self.gradient_dx)
        dTdyy = self.gradient_txy(dTdy, self.gradient_dy)
        return torch.mean(dTdxx**2 + dTdyy**2)

    def forward(self, disp):
        if self.energy_type == 'bending':
            energy = self.compute_bending_energy(disp)
        elif self.energy_type == 'gradient-l2':
            energy = self.compute_gradient_norm(disp)
        elif self.energy_type == 'gradient-l1':
            energy = self.compute_gradient_norm(disp, flag_l1=True)
        else:
            raise Exception('Not recognised local regulariser!')
        return energy


# Local NCC loss
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
        win = [7, 7]  # 默认窗口大小为 9x9
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

# 仿射参数更新
def update_affine_param(cur_af, last_af): # A2(A1*x+b1) + b2 = A2A1*x + A2*b1+b2
        """
        update the current affine parameter A2 based on last affine parameter A1
         A2(A1*x+b1) + b2 = A2A1*x + A2*b1+b2, results in the composed affine parameter A3=(A2A1, A2*b1+b2)

        :param cur_af: current affine parameter
        :param last_af: last affine parameter
        :return: composed affine parameter A3
        """
        '''
        cur_af = cur_af.view(cur_af.shape[0], 2, 3)
        last_af = last_af.view(last_af.shape[0], 2, 3)
        # 初始化更新后的仿射参数
        updated_af = torch.zeros_like(cur_af.data).cuda()
        # 更新旋转、错切和缩放部分
        updated_af[:,:,:2] = torch.matmul(cur_af[:,:,:2],last_af[:,:,:2])
        # 更新平移部分
        updated_af[:,:,2] = cur_af[:,:,2] + torch.squeeze(torch.matmul(cur_af[:,:,:2], torch.transpose(last_af[:,:,2:],1,2)),2) # 选择 cur_af 中的前两列（旋转和缩放部分）：cur_af[:,:2,:]
                                                                                                                                # 选择 last_af 中的最后一列（错切部分）：last_af[:,2:,:]
                                                                                                                                # 转置相乘
        '''
        cur_af = cur_af.view(cur_af.shape[0], 3, 2)
        last_af = last_af.view(last_af.shape[0],3,2)
        updated_af = torch.zeros_like(cur_af.data).cuda()
        updated_af[:,:2,:] = torch.matmul(cur_af[:,:2,:],last_af[:,:2,:])
        updated_af[:,2,:] = cur_af[:,2,:] + torch.squeeze(torch.matmul(cur_af[:,:2,:], torch.transpose(last_af[:,2:,:],1,2)),2)
        # 将参数形状调整回原始大小
        updated_af = updated_af.contiguous().view(cur_af.shape[0],-1)
        return updated_af
    

# ----------------------------------------------粗仿射模块----------------------------------------------------
dim = 2
Conv = nn.Conv2d if dim == 2 else nn.Conv3d
MaxPool = nn.MaxPool2d if dim == 2 else nn.MaxPool3d
ConvTranspose = nn.ConvTranspose2d if dim == 2 else nn.ConvTranspose3d
BatchNorm = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
conv = F.conv2d if dim == 2 else F.conv3d


class conv_bn_rel(nn.Module):
    """
    conv + bn (optional) + relu

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, active_unit='relu', same_padding=False,
                 bn=False, reverse=False, group=1, dilation=1):
        super(conv_bn_rel, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        if not reverse:
            self.conv = Conv(in_channels, out_channels, kernel_size, stride, padding=padding, groups=1, dilation=1)
        else:
            self.conv = ConvTranspose(in_channels, out_channels, kernel_size, stride, padding=padding, groups=1,
                                      dilation=1)

        self.bn = BatchNorm(out_channels) if bn else None #, eps=0.0001, momentum=0, affine=True
        if active_unit == 'relu':
            self.active_unit = nn.ReLU(inplace=True)
        elif active_unit == 'elu':
            self.active_unit = nn.ELU(inplace=True)
        elif active_unit == 'leaky_relu':
            self.active_unit = nn.LeakyReLU(inplace=True)
        else:
            self.active_unit = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.active_unit is not None:
            x = self.active_unit(x)
        return x


class FcRel(nn.Module):
    """
    fc+ relu(option)
    """
    def __init__(self, in_features, out_features, active_unit='relu'):
        super(FcRel, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        if active_unit == 'relu':
            self.active_unit = nn.ReLU(inplace=True)
        elif active_unit == 'elu':
            self.active_unit = nn.ELU(inplace=True)
        else:
            self.active_unit = None

    def forward(self, x):
        x = self.fc(x)
        if self.active_unit is not None:
            x = self.active_unit(x)
        return x

# ----------------------------------------------unet模块--------------------------------------
class Affine_unet(nn.Module):

    def __init__(self):
        super(Affine_unet,self).__init__()
        #(W−F+2P)/S+1, W - input size, F - filter size, P - padding size, S - stride.
        # self.down_path_1 = conv_bn_rel(2, 16, 3, stride=1,active_unit='relu', same_padding=True, bn=False)
        # self.down_path_2 = conv_bn_rel(16, 32, 3, stride=2, active_unit='relu', same_padding=True, bn=False)
        # self.down_path_4 = conv_bn_rel(32, 32, 3, stride=2, active_unit='relu', same_padding=True, bn=False)
        # self.down_path_8 = conv_bn_rel(32, 32, 3, stride=2, active_unit='relu', same_padding=True, bn=False)
        # self.down_path_16 = conv_bn_rel(32, 16, 3, stride=2, active_unit='relu', same_padding=True, bn=False)
        # self.fc_1 = FcRel(16*5*12*12,144,active_unit='relu')
        # self.fc_2 = FcRel(144,12,active_unit = 'None')

        self.down_path_1 = conv_bn_rel(1, 16, 3, stride=1, active_unit='relu', same_padding=True, bn=False)
        self.down_path_2 = MaxPool(2,2)
        self.down_path_4 = conv_bn_rel(32, 16, 3, stride=2, active_unit='relu', same_padding=True, bn=False)
        self.down_path_8 = MaxPool(2,2)
        self.down_path_16 = conv_bn_rel(16, 4, 3, stride=2, active_unit='relu', same_padding=True, bn=False)
        self.down_path_32 = MaxPool(2,2)

        self.fc_1 = FcRel(4*4*4, 32, active_unit='relu') # 沿Batchsize展开为C*H*W
        self.fc_2 = FcRel(32, 7, active_unit='None')

    def forward(self, m,t):
        d1_m = self.down_path_1(m)
        d1_t = self.down_path_1(t)
        d1 = torch.cat((d1_m,d1_t),1)
        d2 = self.down_path_2(d1)
        d4 = self.down_path_4(d2)
        d8 = self.down_path_8(d4)
        d16 = self.down_path_16(d8)
        d32 = self.down_path_32(d16)
        fc1 = self.fc_1(d32.view(d32.shape[0],-1))
        fc2 = self.fc_2(fc1).view((d32.shape[0],-1))
        return fc2



class AffineTransform(nn.Module):
    """
    2-D Affine Transformer with 7 parameters
    """

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, fc2):
        """
        Perform 2-D affine transformation.

        :param src: source image
        :param affine: rotation angle
        :param scale: scaling factors along x and y axes
        :param translate: translation along x and y axes
        :param shear: shearing parameters
        :return: transformed image, forward transformation matrix, inverse transformation matrix
        """
        output_size = src.size()[2:]
        # Extracting parameters
        affine = fc2[:, 0:1] * 0.3
        affine = torch.clamp(affine, min=-1, max=1) * np.pi
        scale = fc2[:, 1:3] * 0.3
        scale = scale + 1
        scale = torch.clamp(scale, min=0, max=5)
        translate = fc2[:, 3:5] * 0.3  # Assuming transl has 2 values (x, y)
        shear = fc2[:, 5:7] * 0.3
        shear = torch.clamp(shear, min=-1, max=1) * np.pi  # shr = torch.tanh(shr) * np.pi
        
        # 分配参数
        theta = affine[:, 0]  # Rotation angle
        scale_x, scale_y = scale[:, 0], scale[:, 1]  # Scaling factors
        trans_x, trans_y = translate[:, 0], translate[:, 1]  # Translation
        shear_xy, shear_yx = shear[:, 0], shear[:, 1]  # Shearing parameters

        # Constructing transformation matrices
        rot_mat = torch.stack([torch.stack([torch.cos(theta), -torch.sin(theta), torch.zeros_like(torch.cos(theta))], dim=1),
                               torch.stack([torch.sin(theta), torch.cos(theta),torch.zeros_like(torch.cos(theta))], dim=1),
                               torch.stack([torch.zeros_like(torch.cos(theta)), torch.zeros_like(torch.cos(theta)), torch.ones_like(torch.cos(theta))], dim=1)], dim=2).cuda()
        
        scale_mat = torch.stack([torch.stack([scale_x, torch.zeros_like(scale_x), torch.zeros_like(scale_x)], dim=1),
                                torch.stack([torch.zeros_like(scale_y), scale_y, torch.zeros_like(scale_y)], dim=1),
                                torch.stack([torch.zeros_like(scale_y), torch.zeros_like(scale_y), torch.ones_like(scale_y)], dim=1)], dim=2).cuda()

        shear_mat = torch.stack([torch.stack([torch.ones_like(shear_xy), shear_xy, torch.zeros_like(shear_xy)], dim=1),
                                torch.stack([shear_yx, torch.ones_like(shear_yx), torch.zeros_like(shear_yx)], dim=1),
                                torch.stack([torch.zeros_like(shear_yx), torch.zeros_like(shear_yx), torch.ones_like(shear_yx)], dim=1)], dim=2).cuda()
        
        trans_mat = torch.stack([torch.stack([torch.ones_like(trans_x), torch.zeros_like(trans_x), trans_x], dim=1),
                                 torch.stack([torch.zeros_like(trans_y), torch.ones_like(trans_y), trans_y], dim=1),
                                 torch.stack([torch.zeros_like(trans_x), torch.zeros_like(trans_y), torch.ones_like(trans_y)], dim=1)], dim=2).cuda()
        
        # 取转置
        rot_mat = torch.transpose(rot_mat,1,2)
        scale_mat = torch.transpose(scale_mat,1,2)
        shear_mat = torch.transpose(shear_mat,1,2)
        trans_mat = torch.transpose(trans_mat,1,2)
        
        # Combining all transformations into a single matrix
        mat = torch.bmm(trans_mat, torch.bmm(shear_mat, torch.bmm(scale_mat, rot_mat)))

        # Inverse transformation matrix
        inv_mat = torch.inverse(mat)
        
        # Applying affine transformation using grid sampling
        grid = F.affine_grid(mat[:, :2, :], torch.Size([mat.shape[0],1] + list(output_size))) # 生成[B,C=1,H,W]的grid
        inv_grid = F.affine_grid(inv_mat[:, :2, :], torch.Size([inv_mat.shape[0],1] + list(output_size))) # 生成[B,C=1,H,W]的grid
        #grid = F.affine_grid(mat[:, :, :2], [src.shape[0], 2, src.shape[2], src.shape[3]], align_corners=False
        transformed = F.grid_sample(src, grid, align_corners=False, mode=self.mode)

        return transformed, mat, inv_mat, grid, inv_grid



# -----------------------------------Bspline---------------------------------------
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
                 cps=(4, 4),
                 img_size=(128,128)
                 ):
        """
        Network to parameterise Cubic B-spline transformation
        """
        super(CubicBSplineNet, self).__init__(ndim=2,
                                              enc_channels=enc_channels,
                                              conv_before_out=False)

        # determine and set output control point sizes from image size and control point spacing
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
            [0.5,0.3,0.2],
            [0.5,0.3,0.2],
            [0.5,0.3,0.2],
            [0.5,0.3,0.2],
            [0.5,0.3,0.2],
            [0.5,0.3,0.2],
            [0.5,0.3,0.2],
            [0.5,0.3,0.2]
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
            disp = disp + warp(x=disp, disp=disp)
        return disp

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
    for i in range(kernel.shape[0]):
        if i < len(derivative_weights):
            weights = derivative_weights[i]
            for derivative, weight in enumerate(weights):
                kernel[i] = weight * cubic_bspline_value((i - radius) / stride, derivative=derivative)
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
    disp = normalise_disp(disp)

    # generate standard mesh grid
    grid = torch.meshgrid([torch.linspace(-1, 1, size[i]).type_as(disp) for i in range(ndim)])
    grid = [grid[i].requires_grad_(False) for i in range(ndim)]

    # apply displacements to each direction (N, *size)

    warped_grid = [grid[i] + disp[:, i, ...] for i in range(ndim)]

    # swapping i-j-k order to x-y-z (k-j-i) order for grid_sample()
    warped_grid = [warped_grid[ndim - 1 - i] for i in range(ndim)]
    warped_grid = torch.stack(warped_grid, -1)  # (N, *size, dim)
    
    return F.grid_sample(x, warped_grid, mode=interp_mode, align_corners=False), warped_grid



# -------------------------------------------------------训练------------------------------------------------
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
    log_name = "%d_%lf_%f_aff.csv" % (args.n_iter, args.lr, args.alpha)
    log_name_bsp = "%d_%lf_%f_bsp.csv" % (args.n_iter, args.lr, args.alpha)
    print("log_name: ", log_name)
    file_log = open(os.path.join(args.log_dir, log_name), "w")
    file_log_bsp = open(os.path.join(args.log_dir, log_name_bsp), "w")
    print("iter,train_loss,sim_loss,sym_loss,valid_loss,sim_loss,sym_loss", file=file_log)

    # 创建配准网络 unet+stn
    print("----Build registration network----")
    unet = Affine_unet().cuda()
    aft = AffineTransform().cuda()  # 创建aft需要dim
    img_size = [128,128]
    unet_bsp = CubicBSplineNet(ndim=2, img_size=img_size).cuda()
    stn_bsp = SpatialTransform_bsp(ndim=2, cps=[4,4], img_size=img_size).cuda()
    
    # 模型参数个数
    print("unet: ", countParameters(unet) + countParameters(unet_bsp) + countParameters(stn_bsp))

    # 设置优化器和loss函数
    print("----Set initial parameters----")
    opt = Adam(unet.parameters(), lr=args.lr)
    param_bsp = list(unet_bsp.parameters()) + list(stn_bsp.parameters()) 
    opt_bsp = Adam(param_bsp, lr=args.lr)
    
    if args.sim_loss == "mse":
        sim_loss_fn = compute_mse
    elif args.sim_loss == "ncc":
        sim_loss_fn = ncc_loss
    else:
        sim_loss_fn = ncc_loss
    grad_loss_fn_bsp = compute_gradient
    grad_loss_fn_aff = compute_gradient_aff
    
    LNCC_loss = LNCCLoss(window_size=[7,7])
    
    # --------------------------------------------数据处理-----------------------------------------------------
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
    # 计时
    start_time = float(time.time())
    
    # 观察损失曲线
    train_losses_aff = []
    valid_losses_aff = []
    train_losses_bsp = []
    valid_losses_bsp = []
    
    # 打印损失并保存模型参数
    best_valid_loss = 1.0
    final_train_loss = 1.0
    final_valid_loss = 1.0
    bsp_best_valid_loss = 0.0
    bsp_final_train_loss = 0.0
    bsp_final_valid_loss = 0.0
    # Affine网络参数
    last_ap = torch.zeros([args.batch_size,6],requires_grad=True).cuda()
    last_ap = last_ap.view(last_ap.shape[0],2,3)
    last_ap[:,0,0] = 1
    last_ap[:,1,1] = 1
    last_ap = last_ap.view(last_ap.shape[0],6)
    max_grad_norm = 0.5
    
    # 迭代训练
    for _iter in range(1, args.n_iter + 1):
        # 将train_data_list进行随机排序
        train_list_permuted = np.random.permutation(train_list)
        # --------------------------------------训练affine部分--------------------------------------      
        if _iter <= args.aff_iter:
            sim_loss_train = 0.0
            sym_loss_train = 0.0
            grad_loss_train = 0.0
            loss_train = 0.0
            unet.train()
            aft.train()
            opt.zero_grad()
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
                
                # 将仿射参数更新为新的值
                ap = unet(img_moving,img_fixed)
                
                # 计算新的仿射参数
                af, mat, inv_mat, grid, inv_grid = aft(img_moving, ap)
                sym_loss = scale_sym_reg_loss(mat) + scale_sym_reg_loss(inv_mat)
                grad_loss = grad_loss_fn_aff(grid) + grad_loss_fn_aff(inv_grid)
                sim_loss = sim_loss_fn(af, img_fixed)
                loss = 1 * sim_loss + args.alpha * sym_loss + args.beta * grad_loss
                
                # 反向传播
                loss.backward()
                
                # Perform gradient check
                #check = gradcheck(scale_multi_step_reg_loss, (mat,), eps=1e-6, atol=1e-3)
                
                # 优化器更新参数
                opt.step()
                
                sim_loss_train += sim_loss.item()
                sym_loss_train += sym_loss.item()
                grad_loss_train += grad_loss.item()
                loss_train += loss.item()

            sim_loss_train /= steps
            sym_loss_train /= steps
            grad_loss_train /= steps
            loss_train /= steps

            # ------------------------------------验证部分--------------------------------------
            sim_loss_valid = 0.0
            sym_loss_valid = 0.0
            grad_loss_valid = 0.0
            loss_valid = 0.0
                        
            unet.eval()
            aft.eval()
            with torch.no_grad():
                img_fixed = torch.zeros([1, 1, 128, 128], dtype=torch.float32)
                img_moving = torch.zeros([1, 1, 128, 128], dtype=torch.float32)

                for _num in range(len(valid_list)):
                    # img & label
                    img_moving[0, 0, :, :] = dataset_valid_img[0, _num, :, :]
                    img_fixed[0, 0, :, :] = dataset_valid_img[1, _num, :, :]

                    img_fixed = img_fixed.cuda()
                    img_moving = img_moving.cuda()

                    # 将仿射参数更新为新的值
                    ap = unet(img_moving,img_fixed)
                    
                    # 计算新的仿射参数
                    af, mat, inv_mat, grid, inv_grid = aft(img_moving, ap)
                    sym_loss = scale_sym_reg_loss(mat) + scale_sym_reg_loss(inv_mat)
                    grad_loss = grad_loss_fn_aff(grid) + grad_loss_fn_aff(inv_grid)
                    sim_loss = sim_loss_fn(af, img_fixed)
                    loss = 1 * sim_loss + args.alpha * sym_loss + args.beta * grad_loss
                    
                    sim_loss_valid += sim_loss.item()
                    sym_loss_valid += sym_loss.item()
                    grad_loss_valid += grad_loss.item()
                    loss_valid += loss.item()
                    

            sim_loss_valid /= len(valid_list)
            sym_loss_valid /= len(valid_list)
            grad_loss_valid /= len(valid_list)
            loss_valid /= len(valid_list)
            
            # 记录损失
            train_losses_aff.append(loss_train)
            valid_losses_aff.append(loss_valid)
            
            # 打印损失
            print("Aff_epoch: %d  train_loss: %f  sim_loss: %f  sym_loss: %f  grad_loss: %f" % (_iter, loss_train, sim_loss_train, sym_loss_train, grad_loss_train), flush=True)
            print("Aff_epoch: %d  valid_loss: %f  sim_loss: %f  sym_loss: %f  grad_loss: %f" % (_iter, loss_valid, sim_loss_valid, sym_loss_valid, grad_loss_valid), flush=True)
            print("%d,%f,%f,%f,%f,%f,%f,%f,%f" % (_iter, loss_train, sim_loss_train, sym_loss_train, grad_loss_train, loss_valid, sim_loss_valid, sym_loss_valid, grad_loss_valid), file=file_log)

            # 计时
            if _iter % 10 == 0:
                print("----time_used: %f" % float(time.time() - start_time), flush=True)
                print("----time_used: %f" % float(time.time() - start_time), file=file_log)

            # 保存最佳模型参数
            if loss_valid < best_valid_loss:
                best_valid_loss = loss_valid
                final_train_loss = loss_train 
                final_valid_loss = loss_valid
                # Save model checkpoint
                save_file_dir_aff = os.path.join(args.model_dir, f"affine_unet_{_iter}.pth")
                torch.save(unet.state_dict(), save_file_dir_aff)
        
        # ---------------------------------------训练bspline部分----------------------------------
        else:  
            sim_loss_train = 0.0
            grad_loss_train = 0.0
            loss_train = 0.0
            unet_bsp.train()
            stn_bsp.train()
            unet.load_state_dict(torch.load(save_file_dir_aff))
            unet.eval()
            aft.eval()
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
                             
                ap = unet(img_moving,img_fixed)
                img_affined, mat, inv_mat, grid, inv_grid = aft(img_moving, ap)
                input_image = torch.cat([img_fixed, img_affined], dim=1)

                # ---------------输入Bspline网络-------------------
                bsp_param = unet_bsp(input_image) # 得到预测的网络参数
                print('bspparam shape is: ', bsp_param.shape)
                disp= stn_bsp(bsp_param)
                img_warped, grid = warp(img_affined, disp)
                
                # 计算loss
                sim_loss = LNCC_loss(img_warped,img_fixed)
                grad_loss = grad_loss_fn_bsp(disp)# 加上反向形变场损失
                dummy_model = nn.Parameter(grad_loss)
                nn.utils.clip_grad_norm_([dummy_model], max_grad_norm)
                loss = args.beta * grad_loss + sim_loss


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
                    
                    ap = unet(img_moving,img_fixed)
                    img_affined, mat, inv_mat, grid, inv_grid = aft(img_moving, ap)
                    input_image = torch.cat([img_fixed, img_affined], dim=1)
                    
                    # ---------------输入Bspline网络-------------------
                    bsp_param = unet_bsp(input_image) # 得到预测的网络参数
                    disp = stn_bsp(bsp_param)
                    img_warped, grid = warp(img_affined, disp)
                    # 计算loss
                    sim_loss = LNCC_loss(img_warped,img_fixed)
                    grad_loss = grad_loss_fn_bsp(disp)
                    dummy_model = nn.Parameter(grad_loss)
                    nn.utils.clip_grad_norm_([dummy_model], max_grad_norm)
                    loss = args.beta * grad_loss + sim_loss
                        

                    sim_loss_valid += sim_loss.item()
                    grad_loss_valid += grad_loss.item()
                    loss_valid += loss.item()
                    

            sim_loss_valid /= len(valid_list)
            grad_loss_valid /= len(valid_list)
            loss_valid /= len(valid_list)
            
            # 记录损失
            train_losses_bsp.append(loss_train)
            valid_losses_bsp.append(loss_valid)
            
            # 打印损失
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
                save_file_dir = os.path.join(args.model_dir, f"bspline_unet_stg1_{_iter}.pth")
                save_stn_dir = os.path.join(args.model_dir, f"bspline_stn_stg2_{_iter}.pth")
                torch.save(unet_bsp.state_dict(), save_file_dir)
                torch.save(stn_bsp.state_dict(), save_stn_dir)
            
    
    # 保存损失数据到文件
    with open('train_losses_aff.txt', 'w') as f:
        for loss in train_losses_aff:
            f.write(str(loss) + '\n')

    with open('valid_losses_aff.txt', 'w') as f:
        for loss in valid_losses_aff:
            f.write(str(loss) + '\n')
            
    with open('train_losses_bsp.txt', 'w') as f:
        for loss in train_losses_bsp:
            f.write(str(loss) + '\n')

    with open('valid_losses_bsp.txt', 'w') as f:
        for loss in valid_losses_bsp:
            f.write(str(loss) + '\n')
    
        
    # 打印Log参数
    print("final_train_loss = %f,final_valid_loss = %f" % (final_train_loss, final_valid_loss), flush=True)
    print("final_train_loss = %f,final_valid_loss = %f" % (final_train_loss, final_valid_loss), file=file_log)
    file_log.close()
    print("final_train_loss = %f,final_valid_loss = %f" % (bsp_final_train_loss, bsp_final_valid_loss), flush=True)
    print("final_train_loss = %f,final_valid_loss = %f" % (bsp_final_train_loss, bsp_final_valid_loss), file=file_log_bsp)
    file_log_bsp.close()
   


def test():
    # 创建文件夹
    print("----Make directory----")
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # 创建配准网络 unet+stn
    print("----Build registration network----")
    unet = Affine_unet().cuda()
    unet.load_state_dict(torch.load(args.checkpoint_path))
    aft = AffineTransform().cuda()
    unet.eval()
    aft.eval()
    img_size=[128,128]
    unet_bsp = CubicBSplineNet(ndim=2, img_size=[128,128]).cuda()
    stn_bsp = SpatialTransform_bsp(ndim=2, cps=[4,4], img_size=img_size).cuda()
    unet_bsp.load_state_dict(torch.load(args.checkpoint_path_bsp))
    stn_bsp.load_state_dict(torch.load(args.checkpoint_path_bspstn))
    unet_bsp.eval()
    stn_bsp.eval()

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
        ap = unet(img_moving,img_fixed)
        img_affined, mat, inv_mat = aft(img_moving, ap)
        input_image = torch.cat([img_fixed, img_affined], dim=1)
        bsp_param = unet_bsp(input_image)
        disp, inv_disp = stn_bsp(bsp_param)
        img_warped, grid = warp(img_affined, disp)
        
        time_list.append([float(time.time() - start_time)])

        # 计算psnr
        if args.psnr:
            psnr_list.append([compute_PSNR(img_fixed, img_moving), compute_PSNR(img_fixed, img_warped)])
        # 计算ssim
        if args.ssim:
            ssim_list.append([compute_SSIM(img_fixed, img_moving), compute_SSIM(img_fixed, img_warped)])
        # 计算雅克比行列式分数
        if args.jac:
            jac = Jacobian(disp)
            jac_list.append([jac.count_minus_ratio()])

        # 保存图像
        # img & label
        img = nib.Nifti1Image(img_fixed[0, 0, :, :].cpu().detach().numpy(), None)
        nib.save(img, os.path.join(subject_dir, "fixed.nii.gz"))

        img = nib.Nifti1Image(img_moving[0, 0, :, :].cpu().detach().numpy(), None)
        nib.save(img, os.path.join(subject_dir, "moving.nii.gz"))
        
        img = nib.Nifti1Image(img_affined[0, 0, :, :].cpu().detach().numpy(), None)
        nib.save(img, os.path.join(subject_dir, "affiend.nii.gz"))

        img = nib.Nifti1Image(img_warped[0, 0, :, :].cpu().detach().numpy(), None)
        nib.save(img, os.path.join(subject_dir, "warped.nii.gz"))
        
        DVF = nib.Nifti1Image(disp[0, :, :, :].cpu().detach().numpy(), None)
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
