# python imports
import time
import csv
import re
import os, random
import warnings
import argparse
import glob
import math
# external imports
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch import nn
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
from utils.Dataset import Tongji_Dataset_train, Tongji_Dataset_test
import skimage.metrics as metrics  # 这个里面包含了很多评估指标的计算方法 PSNR SSIM等
from model.TransMorph_bspl_ACDC import CONFIGS as CONFIGS_BSP
import model.TransMorph_morph_ACDC as bspl
from model.layers import CalcDisp
from model.layers import cal_spatial_transformer
from model.loss import MILossGaussian as NMI_Loss
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
parser = argparse.ArgumentParser()

# 文件路径
parser.add_argument("--train_dir", type=str, help="data folder with training vols", dest="train_dir", default="/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Xiehe2D_resize")
parser.add_argument("--model_dir", type=str, help="models folder", dest="model_dir", default="/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Checkpoint_Xiehe")
parser.add_argument("--log_dir", type=str, help="logs folder", dest="log_dir", default="/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Log_Xiehe")
parser.add_argument("--result_dir", type=str, help="results folder", dest="result_dir", default="/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Result_Xiehe")
parser.add_argument("--csv_train_dir", type=str, help="csv train", dest="csv_train", default="/home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/Tongji_CineLGE_2D/Tongji_CineLGE_2D_metadata.csv")
parser.add_argument("--csv_valid_dir", type=str, help="csv valid", dest="csv_valid", default="/home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/Tongji_CineLGE_2D/Tongji_CineLGE_2D_metadata.csv")
parser.add_argument("--csv_test_dir", type=str, help="csv test", dest="csv_test", default="/home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/Tongji_CineLGE_2D/Tongji_CineLGE_2D_metadata.csv")

# network parameters
parser.add_argument("--pattern", type=str, help="select train or test", dest="pattern", default="train")

# training parameters
parser.add_argument("--lr", type=float, help="learning rate", dest="lr", default=1e-4)
parser.add_argument("--n_iter", type=int, help="number of iterations", dest="n_iter", default=1000)
parser.add_argument("--sim_loss", type=str, help="image similarity loss: mse or ncc", dest="sim_loss", default="ncc")
parser.add_argument("--alpha", type=float, help="regularization parameter", dest="alpha", default=10.0)  # recommend 1.0 for ncc, 0.01 for mse
parser.add_argument("--beta", type=float, help="regularization parameter", dest="beta", default=1.0)  # recommend 1.0 for ncc, 0.01 for mse
parser.add_argument("--batch_size", type=int, help="batch_size", dest="batch_size", default=2)

# testing parameters
parser.add_argument("--test_dir", type=str, help="test data directory", dest="test_dir", default="/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Xiehe2D_resized_test")
parser.add_argument("--checkpoint_path", type=str, help="model weight file", dest="checkpoint_path", default="/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Checkpoint_Xiehe/trained_model.pth")
parser.add_argument("--dice", type=bool, help="if compute dice", dest="dice", default=True)
parser.add_argument("--psnr", type=bool, help="if compute psnr", dest="psnr", default=True)
parser.add_argument("--ssim", type=bool, help="if compute ssim", dest="ssim", default=True)
parser.add_argument("--jacobian", type=bool, help="if compute jacobian", dest="jac", default=True)

args = parser.parse_args()

# smooth loss
def smooth_loss(disp, image):
    '''
    Calculate the smooth loss. Return mean of absolute or squared of the forward difference of  flow field. 
    
    Parameters
    ----------
    disp : (n, 2, h, w) or (n, 3, d, h, w)
        displacement field
        
    image : (n, 1, d, h, w) or (1, 1, d, h, w)

    '''

    image_shape = disp.shape
    dim = len(image_shape[2:])
    
    d_disp = torch.zeros((image_shape[0], dim) + tuple(image_shape[1:]), dtype = disp.dtype, device = disp.device)
    d_image = torch.zeros((image_shape[0], dim) + tuple(image_shape[1:]), dtype = disp.dtype, device = disp.device)
    
    # forward difference
    if dim == 2:
        print(d_image.shape)
        d_disp[:, 1, :, :-1, :] = (disp[:, :, 1:, :] - disp[:, :, :-1, :])
        d_disp[:, 0, :, :, :-1] = (disp[:, :, :, 1:] - disp[:, :, :, :-1])
        d_image[:, 1, :, :-1, :] = (image[:, :, 1:, :] - image[:, :, :-1, :])
        d_image[:, 0, :, :, :-1] = (image[:, :, :, 1:] - image[:, :, :, :-1])
        
    elif dim == 3:
        d_disp[:, 2, :, :-1, :, :] = (disp[:, :, 1:, :, :] - disp[:, :, :-1, :, :])
        d_disp[:, 1, :, :, :-1, :] = (disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :])
        d_disp[:, 0, :, :, :, :-1] = (disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1])
        
        d_image[:, 2, :, :-1, :, :] = (image[:, :, 1:, :, :] - image[:, :, :-1, :, :])
        d_image[:, 1, :, :, :-1, :] = (image[:, :, :, 1:, :] - image[:, :, :, :-1, :])
        d_image[:, 0, :, :, :, :-1] = (image[:, :, :, :, 1:] - image[:, :, :, :, :-1])

    loss = torch.mean(torch.sum(torch.abs(d_disp), dim = 2, keepdims = True)*torch.exp(-torch.abs(d_image)))
    
    return loss


# mse loss
def compute_mse(tensor_x, tensor_y):
    mse = torch.mean((tensor_x - tensor_y) ** 2)
    return mse

# smooth loss
def smooth_loss(disp, image):
    '''
    Calculate the smooth loss. Return mean of absolute or squared of the forward difference of  flow field. 
    
    Parameters
    ----------
    disp : (n, 2, h, w) or (n, 3, d, h, w)
        displacement field
        
    image : (n, 1, d, h, w) or (1, 1, d, h, w)

    '''

    image_shape = disp.shape
    dim = len(image_shape[2:])
    
    d_disp = torch.zeros((image_shape[0], dim) + tuple(image_shape[1:]), dtype = disp.dtype, device = disp.device)
    d_image = torch.zeros((image_shape[0], dim) + tuple(image_shape[1:]), dtype = disp.dtype, device = disp.device)
    
    # forward difference
    if dim == 2:
        d_disp[:, 1, :, :-1, :] = (disp[:, :, 1:, :] - disp[:, :, :-1, :])
        d_disp[:, 0, :, :, :-1] = (disp[:, :, :, 1:] - disp[:, :, :, :-1])
        d_image[:, 1, :, :-1, :] = (image[:, :, 1:, :] - image[:, :, :-1, :])
        d_image[:, 0, :, :, :-1] = (image[:, :, :, 1:] - image[:, :, :, :-1])
        
    elif dim == 3:
        d_disp[:, 2, :, :-1, :, :] = (disp[:, :, 1:, :, :] - disp[:, :, :-1, :, :])
        d_disp[:, 1, :, :, :-1, :] = (disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :])
        d_disp[:, 0, :, :, :, :-1] = (disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1])
        
        d_image[:, 2, :, :-1, :, :] = (image[:, :, 1:, :, :] - image[:, :, :-1, :, :])
        d_image[:, 1, :, :, :-1, :] = (image[:, :, :, 1:, :] - image[:, :, :, :-1, :])
        d_image[:, 0, :, :, :, :-1] = (image[:, :, :, :, 1:] - image[:, :, :, :, :-1])

    loss = torch.mean(torch.sum(torch.abs(d_disp), dim = 2, keepdims = True)*torch.exp(-torch.abs(d_image)))
    
    return loss

# gradient loss
def compute_gradient(tensor_x):
    dims = tensor_x.ndim
    tensor_x = tensor_x / tensor_x.abs().max()
    gradient = 0.0
    if dims == 4:
        dx = (tensor_x[:, :, 1:, :] - tensor_x[:, :, :-1, :]) ** 2
        dy = (tensor_x[:, :, :, 1:] - tensor_x[:, :, :, :-1]) ** 2
        gradient = (dx.mean() + dy.mean()) / 2
    elif dims == 5:
        dx = (tensor_x[:, :, 1:, :, :] - tensor_x[:, :, :-1, :, :]) ** 2
        dy = (tensor_x[:, :, :, 1:, :] - tensor_x[:, :, :, :-1, :]) ** 2
        dz = (tensor_x[:, :, :, :, 1:] - tensor_x[:, :, :, :, :-1]) ** 2
        gradient = (dx.mean() + dy.mean() + dz.mean()) / 3
    return gradient


def compute_local_sums(x, y, filt, stride, padding, win):
    x2, y2, xy = x * x, y * y, x * y
    
    x_sum = F.conv2d(x, filt, stride=stride, padding=padding)
    y_sum = F.conv2d(y, filt, stride=stride, padding=padding)
    xy_sum = F.conv2d(xy, filt, stride=stride, padding=padding)
    
    win_size = np.prod(win)
    mean_x = x_sum / win_size
    mean_y = y_sum / win_size
    
    cross = xy_sum - mean_x * mean_y * win_size
    
    x_var = F.conv2d(x2, filt, stride=stride, padding=padding) - 2 * mean_x * x_sum + mean_x * mean_x * win_size
    y_var = F.conv2d(y2, filt, stride=stride, padding=padding) - 2 * mean_y * y_sum + mean_y * mean_y * win_size
    
    return x_var, y_var, cross


# ncc损失
def ncc_loss(x, y, win=None):
    """
    输入大小是[B,C,W,H]格式的，在计算ncc时用卷积来实现指定窗口内求和
    """
    ndims = len(list(x.size())) - 2
    assert ndims == 2, "Input volumes should be 2 dimensions. Found: %d" % ndims
    if win is None:
        win = [5, 5]  # 默认窗口大小为 9x9
    sum_filt = torch.ones([1, 1, *win]).cuda()
    #pad_no = np.floor(win[0] / 2)
    stride = [1] * ndims
    padding = [2] * ndims
    x_var, y_var, cross = compute_local_sums(x, y, sum_filt, stride=tuple(stride), padding=tuple(padding), win=win)
    cc = cross * cross / (x_var * y_var + 1e-5)
    return 1 - torch.mean(cc)


# 形变场扭曲损失
class Bend_Penalty(nn.Module):
    """
    Bending Penalty of the spatial transformation (2D)
    """
    def __init__(self):
        super(Bend_Penalty, self).__init__()
    
    def _diffs(self, y, dim):#y shape(bs, nfeat, vol_shape)
        ndims = y.ndimension() - 2
        d = dim + 2
        # permute dimensions to put the ith dimension first
#       r = [d, *range(d), *range(d + 1, ndims + 2)]
        y = y.permute(d, *range(d), *range(d + 1, ndims + 2))
        dfi = y[1:, ...] - y[:-1, ...]
        
        # permute back
        # note: this might not be necessary for this loss specifically,
        # since the results are just summed over anyway.
#       r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
        df = dfi.permute(*range(1, d + 1), 0, *range(d + 1, ndims + 2))
        
        return df
    
    def forward(self, pred):#shape(B,C,H,W)
        Ty = self._diffs(pred, dim=0)
        Tx = self._diffs(pred, dim=1)
        Tyy = self._diffs(Ty, dim=0)
        Txx = self._diffs(Tx, dim=1)
        Txy = self._diffs(Tx, dim=0)
        p = Tyy.pow(2).mean() + Txx.pow(2).mean() + 2 * Txy.pow(2).mean()
        
        return p


def ssim_loss(img1, img2, window_size=11, sigma=1.5):
    """
    计算两个图像的 SSIM 损失
    Args:
        img1 (torch.Tensor): 第一个图像，shape 为 [batch_size, channels, height, width]
        img2 (torch.Tensor): 第二个图像，shape 为 [batch_size, channels, height, width]
        window_size (int): SSIM 使用的高斯加权窗口的大小，默认为 11
        sigma (float): 高斯加权窗口的标准差，默认为 1.5
    Returns:
        torch.Tensor: 两个图像的 SSIM 损失，shape 为 [batch_size]
    """
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    print(img1.shape)
    window = torch.tensor([gaussian(window_size, sigma) for _ in range(img1.shape[1])]).to(img1.device)
    window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, window_size, window_size]
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.shape[1])
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.shape[1]) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    ssim_val = torch.mean(ssim_map, dim=(1, 2, 3))
    return 1 - ssim_val  # 返回的是 SSIM 损失，需要取反作为损失值

def gaussian(window_size, sigma):
    """
    生成一个一维的高斯加权窗口
    Args:
        window_size (int): 窗口大小
        sigma (float): 标准差
    Returns:
        torch.Tensor: 高斯加权窗口
    """
    gauss = torch.tensor([math.exp(-(x - window_size // 2)**2 / (2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


# count parameters in model
def countParameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


# 多标签计算dice，返回均值
class Dice(nn.Module):
    """
    N-D dice for segmentation
    """
    def __init__(self):
        super(Dice, self).__init__()

    def dice(self, y_true, y_pred):
        # 计算Dice系数
        vol_axes = [2, 3]  # 对于输入tensor[B,C,H,W]
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true.sum(dim=vol_axes) + y_pred.sum(dim=vol_axes)), min=1e-5) 
        dice = top / bottom
        return dice
    
    def forward(self, y_pred , y_true, labelorders):
        # 计算多值标签的Dice损失
        num_classes = len(labelorders)
        dicescore = 0
        for order in labelorders:
            order = int(order)
            y_true_onehot = y_true == order
            y_pred_onehot = y_pred == order
            dicescore += self.dice(y_true_onehot, y_pred_onehot)
            
        dicescore = dicescore / num_classes
        return dicescore


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


class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='nearest'):
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
        return F.grid_sample(src, new_locs, mode=self.mode, align_corners = True, padding_mode = 'border')

# 数据增强
def affine_aug(im, im_label=None, seed=10):
    # mode = 'bilinear' or 'nearest'
    with torch.no_grad():
        random.seed(seed)
        angle_range = 10
        trans_range = 0.1
        scale_range = 0.1

        angle = random.uniform(-angle_range, angle_range) * math.pi / 180
        scale = random.uniform(1 - scale_range, 1 + scale_range)
        tx = random.uniform(-trans_range, trans_range) * im.shape[3]
        ty = random.uniform(-trans_range, trans_range) * im.shape[2]
        '''
        # 生成仿射变换矩阵单批次
        theta = torch.tensor([
            [math.cos(angle) * scale, -math.sin(angle), tx],
            [math.sin(angle), math.cos(angle) * scale, ty]
        ], dtype=torch.float32).unsqueeze(0).cuda()
        '''
         # 生成仿射变换矩阵
        theta = torch.tensor([
            [math.cos(angle) * scale, -math.sin(angle), tx],
            [math.sin(angle), math.cos(angle) * scale, ty]
        ], dtype=torch.float32).unsqueeze(0).repeat(im.shape[0], 1, 1).cuda()
        
        # 对图像进行仿射变换
        grid = F.affine_grid(theta, im.shape)
        im = F.grid_sample(im, grid, mode='bilinear', padding_mode='border')

        if im_label is not None:
            im_label = F.grid_sample(im_label, grid, mode='nearest', padding_mode='border')
            return im, im_label
        else:
            return im

# 图像裁剪
def crop_image_128_train(file_path, crop_size=128):
    c_data = []
    for i in range(len(file_path)):
        data = torch.from_numpy(nib.load(file_path[i]).get_fdata()).type(torch.float32)
        h, w = data.shape[:2]  # 获取未裁剪图像的高度和宽度
        crop_start_h = (h - crop_size) // 2  # 计算裁剪起始点的高度
        crop_start_w = (w - crop_size) // 2  # 计算裁剪起始点的高度
        cropped_data = data[crop_start_h:crop_start_h+crop_size, crop_start_w:crop_start_w+crop_size,:]
        cropped_data = (cropped_data - cropped_data.min()) / (cropped_data.max() - cropped_data.min())
        c_data.append(cropped_data)
    return torch.stack(c_data)

# 图像裁剪
def crop_image_128_valid(file_path, crop_size=128):
    file_name = os.path.basename(file_path[0])
    directory, file_name = os.path.split(file_path[0])
    
    # 使用正则表达式匹配数字部分
    match = re.search(r'\d+', file_name)
    if match:
        number = match.group()
    else:
        number = None
    match = re.search(r'P\d{3}', directory)
    if match:
        subject = match.group()
    else:
        subject = None
    #print(subject,number)
    data = torch.from_numpy(nib.load(file_path[0]).get_fdata()).type(torch.float32)
    h, w = data.shape[:2]  # 获取未裁剪图像的高度和宽度
    crop_start_h = (h - crop_size) // 2  # 计算裁剪起始点的高度
    crop_start_w = (w - crop_size) // 2  # 计算裁剪起始点的高度
    cropped_data = data[crop_start_h:crop_start_h+crop_size, crop_start_w:crop_start_w+crop_size,:]
    cropped_data = (cropped_data - cropped_data.min()) / (cropped_data.max() - cropped_data.min())
        
    return cropped_data.unsqueeze(0), subject, number


# 图像裁剪
def crop_image_128_test(file_path, crop_size=128):
    file_name = os.path.basename(file_path[0])
    label_path = file_path[0].replace(r'.nii.gz','_gt.nii.gz')
    directory, file_name = os.path.split(file_path[0])
    
    # 使用正则表达式匹配数字部分
    match = re.search(r'\d+', file_name)
    if match:
        number = match.group()
    else:
        number = None
    match = re.search(r'P\d{3}', directory)
    if match:
        subject = match.group()
    else:
        subject = None
    #print(subject,number)
    data = torch.from_numpy(nib.load(file_path[0]).get_fdata()).type(torch.float32)
    label = torch.from_numpy(nib.load(label_path).get_fdata()).type(torch.float32)
    h, w = data.shape[:2]  # 获取未裁剪图像的高度和宽度
    crop_start_h = (h - crop_size) // 2  # 计算裁剪起始点的高度
    crop_start_w = (w - crop_size) // 2  # 计算裁剪起始点的高度
    cropped_data = data[crop_start_h:crop_start_h+crop_size, crop_start_w:crop_start_w+crop_size,:]
    cropped_label = label[crop_start_h:crop_start_h+crop_size, crop_start_w:crop_start_w+crop_size,:]
    cropped_data = (cropped_data - cropped_data.min()) / (cropped_data.max() - cropped_data.min())
    
    
    return cropped_data.unsqueeze(0), subject, number, cropped_label.unsqueeze(0)

# 动态调整学习率
def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)


from sklearn.model_selection import KFold
from torch.utils.data import Subset

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
    log_name = "%d_%lf_%f.csv" % (args.n_iter, args.lr, args.alpha)
    print("log_name: ", log_name)
    file_log = open(os.path.join(args.log_dir, log_name), "w")
    print("fold,iter,train_loss,sim_loss,reg_loss,valid_loss,sim_loss,reg_loss", file=file_log)

    # 创建配准网络 unet+stn
    print("----Build registration network----")
    config = CONFIGS_BSP['BSplineParam']
    model = bspl.PixelMorph_Temp_Net(config).cuda()
    #stn = SpatialTransformer([128,128]).cuda()
    bend_fn = Bend_Penalty().cuda()
    #dice_fn = Dice().cuda()
    #nmi_fn = NMI_Loss().cuda()
    calcdisp = CalcDisp(config)
    
    # 模型参数个数
    print("unet: ", countParameters(model))

    # 设置优化器和loss函数
    print("----Set initial parameters----")
    #opt = Adam(model.parameters(), lr=args.lr)
    if args.sim_loss == "mse":
        sim_loss_fn = compute_mse
    elif args.sim_loss == "ncc":
        sim_loss_fn = ncc_loss
    else:
        sim_loss_fn = ssim_loss

    # 交叉验证
    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    
    # 数据处理
    print("----Process data----")    
    dataset = Tongji_Dataset_train(args.csv_train)
    
    # 开始训练
    print("----Start training----")
    best_train_loss_overall = float('inf')  # 初始化为正无穷
    best_valid_loss_overall = float('inf')  # 初始化为正无穷
    best_model_state_overall = None
    # 计时
    start_time = float(time.time())
    
    for fold, (train_index, valid_index) in enumerate(kf.split(dataset), 1):
        # 划分训练集和验证集
        train_dataset = Subset(dataset, train_index)
        valid_dataset = Subset(dataset, valid_index)
        
        # 创建对应的数据加载器
        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        dataloader_valid = DataLoader(valid_dataset, batch_size=1, shuffle=True)
        
        # 初始化模型和优化器
        model = bspl.PixelMorph_Temp_Net(config).cuda()
        opt = Adam(model.parameters(), lr=args.lr)
        
        best_valid_loss = 10.0
        final_train_loss = 10.0
        final_valid_loss = 10.0
        best_model_state = None
        
        for _iter in range(1, args.n_iter + 1):
            # ---------------------------训练部分------------------------------
            adjust_learning_rate(opt, _iter, args.n_iter, args.lr)
            sim_loss_train = 0.0
            reg_loss_train = 0.0
            smth_loss_train = 0.0
            loss_train = 0.0
            model.train()
            opt.zero_grad()
            
            # 以batch_size为步长批量读取数据
            for batch in dataloader:
                img_moving= crop_image_128_train(batch['img_moving'])
                img_moving = img_moving.cuda()
                img_fixed= crop_image_128_train(batch['img_fixed'])
                img_fixed = img_fixed.cuda()
                
                img_moving = img_moving.permute(0,3,1,2)
                img_fixed = img_fixed.permute(0,3,1,2)
                #print(img_moving.shape, img_fixed.shape)
                # 输入网络
                res = model(img_moving, img_fixed, mode='train')
                
                # 计算loss
                sim_loss = 0.0
                reg_loss = 0.0
                smth_loss = 0.0
                for i in range(args.batch_size):
                    #print('scaled disp', res['scaled_disp_t2i'][i].shape, 'scaled template', res['scaled_template'].shape)
                    sim_loss += sim_loss_fn(res['warped'][i].to('cuda'), res['template'][i].to('cuda'))
                    reg_loss += bend_fn(res['scaled_disp_t2i'][i])
                    smth_loss += smooth_loss(res['scaled_disp_t2i'][i], res['scaled_template'][i].unsqueeze(0))
                loss = args.alpha * reg_loss + sim_loss + args.beta * smth_loss

                # Backwards and optimize
                loss.backward()
                
                opt.step()

                sim_loss_train += sim_loss.item()
                reg_loss_train += reg_loss.item()
                smth_loss_train += smth_loss.item()
                loss_train += loss.item()

            sim_loss_train /= len(dataloader)
            reg_loss_train /= len(dataloader)
            smth_loss_train /= len(dataloader)
            loss_train /= len(dataloader)
            
            # -----------------------------------验证部分---------------------------------
            sim_loss_valid = 0.0
            reg_loss_valid = 0.0
            smth_loss_valid = 0.0
            loss_valid = 0.0

            with torch.no_grad():
                model.eval()
                
                for batch in dataloader_valid:
                    img_moving, subject, number  = crop_image_128_valid(batch['img_moving'])
                    img_moving = img_moving.cuda()
                    img_fixed, subject, number  = crop_image_128_valid(batch['img_fixed'])
                    img_fixed = img_fixed.cuda()
                    
                    # 输入网络
                    img_moving = img_moving.permute(0,3,1,2)
                    img_fixed = img_fixed.permute(0,3,1,2)
                                                        
                    res = model(img_moving, img_fixed, mode='valid')
                    
                    # 计算dice
                    #label_warped = stn(label_moving, flow)
                    #dice_loss = dice_fn(label_fixed, label_warped, labelorders=[1,2,3])
                    
                    # 计算loss
                    sim_loss = 0.0
                    reg_loss = 0.0
                    smth_loss = 0.0
                    sim_loss = sim_loss_fn(res['warped'], res['template'])
                    reg_loss = bend_fn(res['scaled_disp_t2i'])
                    smth_loss = smooth_loss(res['scaled_disp_t2i'], res['scaled_template'])
                    loss = args.alpha * reg_loss + sim_loss + args.beta * smth_loss
                    
                    # 计算复合形变场
                    disp_i2t = calcdisp.inverse_disp(res['disp_t2i'][config.pair_disp_indexes])
                    composed_disp = calcdisp.compose_disp(disp_i2t, res['disp_t2i'][config.pair_disp_indexes], mode='all')
                    flow = composed_disp[1, 0].unsqueeze(0)
                    
                    sim_loss_valid += sim_loss.item()
                    reg_loss_valid += reg_loss.item()
                    smth_loss_valid += smth_loss.item()

                    loss_valid += loss.item()

            sim_loss_valid /= len(dataloader_valid)
            reg_loss_valid /= len(dataloader_valid)
            smth_loss_valid /= len(dataloader_valid)
            loss_valid /= len(dataloader_valid)

            print("fold: %d epoch: %d  train_loss: %f  sim_loss: %f  reg_loss: %f  smooth_loss: %f" % (fold, _iter, loss_train, sim_loss_train, reg_loss_train, smth_loss_train), flush=True)
            print("fold: %d epoch: %d  valid_loss: %f  sim_loss: %f  reg_loss: %f  smooth_loss: %f" % (fold, _iter, loss_valid, sim_loss_valid, reg_loss_valid, smth_loss_valid), flush=True)
            print("%d,%f,%f,%f,%f,%f,%f,%f,%f" % (_iter, loss_train, sim_loss_train, reg_loss_train, smth_loss_train, loss_valid, sim_loss_valid, reg_loss_valid, smth_loss_valid), file=file_log)

            # 计时
            if _iter % 10 == 0:
                print("----time_used: %f" % float(time.time() - start_time), flush=True)
                print("----time_used: %f" % float(time.time() - start_time), file=file_log)

            # 保存最佳模型参数
            if loss_valid <= best_valid_loss:
                best_valid_loss = loss_valid
                final_train_loss = loss_train 
                final_valid_loss = loss_valid
                best_model_state = model.state_dict()

        # Save model checkpoint
        save_file_dir = os.path.join(args.model_dir, "tongji_morph_fold{}.pth".format(fold))
        torch.save(best_model_state, save_file_dir)
        
        # 如果该折叠的验证损失比所有折叠中的最佳验证损失更低，则更新最佳模型参数
        if best_valid_loss <= best_valid_loss_overall:
            best_valid_loss_overall = best_valid_loss
            best_train_loss_overall = final_train_loss
            best_model_state_overall = best_model_state
            
        print("fold = %d,final_train_loss = %f,final_valid_loss = %f" % (fold, final_train_loss, final_valid_loss), flush=True)
        #print("fold = %d,final_train_loss = %f,final_valid_loss = %f" % (fold, final_train_loss, final_valid_loss), file=file_log)
    
    # 在所有fold结束后，保存最佳模型参数
    save_file_dir = os.path.join(args.model_dir, "morph_model.pth")
    torch.save(best_model_state_overall, save_file_dir)
    print("best_train_loss = %f,best_valid_loss = %f" % (final_train_loss, final_valid_loss), flush=True)
    print("best_train_loss = %f,best_valid_loss = %f" % (best_train_loss_overall, best_valid_loss_overall), file=file_log)    
    file_log.close()


    

def test():
    # 创建文件夹
    print("----Make directory----")
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # 创建配准网络 unet+stn
    print("----Build registration network----")
    config = CONFIGS_BSP['BSplineParam']
    model = bspl.PixelMorph_Temp_Net(config).cuda()
    stn_img = SpatialTransformer([128,128], mode='bilinear').cuda()
    stn = SpatialTransformer([128,128], mode='nearest').cuda()
    dice_fn = Dice().cuda()
    calcdisp = CalcDisp(config)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()

    # 数据处理
    print("----Process data----")
    dataset = Tongji_Dataset_test(args.csv_test)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 开始测试
    print("----Start testing----")
    # 计时
    time_list = []
    dice_list = []
    myo_dice_list = []
    lv_dice_list = []
    psnr_list = []
    ssim_list = []
    jac_list = []
    
    for batch in dataloader:
        img_moving, subject_mov, slicing_mov, label_moving = crop_image_128_test(batch['img_moving'])
        img_moving = img_moving.cuda()
        label_moving = label_moving.cuda()
        img_fixed, subject_fix, slicing_fix, label_fixed = crop_image_128_test(batch['img_fixed'])
        img_fixed = img_fixed.cuda()        
        label_fixed = label_fixed.cuda()
        
        img_moving = img_moving.permute(0,3,1,2)
        img_fixed = img_fixed.permute(0,3,1,2)
        label_moving = label_moving.permute(0,3,1,2)
        label_fixed = label_fixed.permute(0,3,1,2)
        
        # 输入网络计时
        start_time = time.time()
        res = model(img_moving, img_fixed, mode='test')
        time_list.append([float(time.time() - start_time)])
        
        # 计算复合形变场
        disp_i2t = calcdisp.inverse_disp(res['disp_t2i'][config.pair_disp_indexes])
        composed_disp = calcdisp.compose_disp(disp_i2t, res['disp_t2i'][config.pair_disp_indexes], mode='all')
        flow = composed_disp[1, 0].unsqueeze(0)
        img_warped = stn_img(img_moving, flow)
        label_warped = stn(label_moving, flow)
        
        # 计算dice
        if args.dice:
            dice_list.append([dice_fn(label_fixed, label_moving, labelorders =[1,2]).cpu().numpy().item(), dice_fn(label_fixed, label_warped, labelorders =[1,2]).cpu().numpy().item()])
            myo_dice_list.append([dice_fn(label_fixed, label_moving, labelorders =[1]).cpu().numpy().item(), dice_fn(label_fixed, label_warped, labelorders =[1]).cpu().numpy().item()])
            lv_dice_list.append([dice_fn(label_fixed, label_moving, labelorders =[2]).cpu().numpy().item(), dice_fn(label_fixed, label_warped, labelorders =[2]).cpu().numpy().item()])
        
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
        
        # 保存结果
        subject_dir = os.path.join(args.result_dir, f"{subject_fix}")
        if not os.path.exists(subject_dir):
            os.mkdir(subject_dir)
        # 保存图像
        img = nib.Nifti1Image(img_warped[0,:,:,:].cpu().detach().numpy(), None)
        nib.save(img, os.path.join(subject_dir,f"warped_{slicing_mov}.nii.gz"))
        img = nib.Nifti1Image(img_fixed[0,:,:,:].cpu().detach().numpy(), None)
        nib.save(img, os.path.join(subject_dir,f"fixed_{slicing_mov}.nii.gz"))
        img = nib.Nifti1Image(img_moving[0,:,:,:].cpu().detach().numpy(), None)
        nib.save(img, os.path.join(subject_dir,f"moving_{slicing_mov}.nii.gz"))
        
        label = nib.Nifti1Image(label_fixed[0, 0, :, :].cpu().detach().type(torch.int8).numpy(), None)
        nib.save(label, os.path.join(subject_dir, f"label_fixed_{slicing_mov}.nii.gz"))
        label = nib.Nifti1Image(label_moving[0, 0, :, :].cpu().detach().type(torch.int8).numpy(), None)
        nib.save(label, os.path.join(subject_dir, f"label_moving_{slicing_mov}.nii.gz"))        
        label = nib.Nifti1Image(label_warped[0, 0, :, :].cpu().detach().type(torch.int8).numpy(), None)
        nib.save(label, os.path.join(subject_dir, f"label_warped_{slicing_mov}.nii.gz"))
                
        flow = flow.permute(0,3,2,1)
        flow = flow[0]
        DVF = nib.Nifti1Image(flow[:, :, :].cpu().detach().numpy()[:,:,None,:][:,:,:,None,:], None) # 变为三维网格，以便使用ITK-snap的grid格式
        nib.save(DVF, os.path.join(subject_dir, f"flow_{slicing_mov}.nii.gz"))


    print("time_used = %f" % np.sum(time_list))

    # 保存结果
    with open(os.path.join(args.result_dir, "result.csv"), "w") as f:
        writer = csv.writer(f)
        header = ["time"]
        data = np.array(time_list)
        if args.dice:
            header.append("myo_dice_pre")
            header.append("myo_dice_done")
            myo_dice_list = np.array(myo_dice_list)
            data = np.append(data, myo_dice_list, axis=1)
        if args.dice:
            header.append("lv_dice_pre")
            header.append("lv_dice_done")
            lv_dice_list = np.array(lv_dice_list)
            data = np.append(data, lv_dice_list, axis=1)        
        if args.dice:
            header.append("dice_pre")
            header.append("dice_done")
            dice_list = np.array(dice_list)
            data = np.append(data, dice_list, axis=1)      
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
