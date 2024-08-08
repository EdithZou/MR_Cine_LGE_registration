# python imports
import time
import csv
import os, random
import warnings
import argparse
import glob
import math
# external imports
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch import nn
import numpy as np
import nibabel as nib
import skimage.metrics as metrics  # 这个里面包含了很多评估指标的计算方法 PSNR SSIM等
from model.TransMorph_bspl import CONFIGS as CONFIGS_BSP
import model.TransMorph_bspl as bspl
from model.regloss import l2reg_loss
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_fn
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


'''
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
'''
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


# 计算Dice B*C*H*W*D  多标签返回均值
def compute_Dice(tensor_pred, tensor_targ):
    smooth = 1e-5
    labels = tensor_pred.unique()
    if labels[0] == 0:
        labels = labels[1:]
    #print('labels number: ', len(labels))
    dice_list = torch.zeros([len(labels)])
    for _num in range(len(labels)):
        tensor_x = torch.where(tensor_pred == labels[_num], 1, 0).flatten()
        tensor_y = torch.where(tensor_targ == labels[_num], 1, 0).flatten()
        dice_list[_num] = (2.0 * (tensor_x * tensor_y).sum() + smooth) / (tensor_x.sum() + tensor_y.sum() + smooth)
    dice = torch.mean(dice_list).item()
    return dice



def dice_val(y_pred, y_true, num_clus):
    y_pred = nn.functional.one_hot(y_pred, num_classes=num_clus)
    y_pred = torch.squeeze(y_pred, 1)
    y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
    y_true = nn.functional.one_hot(y_true, num_classes=num_clus)
    y_true = torch.squeeze(y_true, 1)
    y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2.*intersection) / (union + 1e-5)
    return torch.mean(torch.mean(dsc, dim=1))


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

        return F.grid_sample(src, new_locs, mode=self.mode)

# 数据增强
def affine_aug(im, im_label, seed=10):
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



# 动态调整学习率
def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)


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
    print("iter,train_loss,sim_loss,reg_loss,valid_loss,sim_loss,reg_loss,dice_valid", file=file_log)

    # 创建配准网络 unet+stn
    print("----Build registration network----")
    config = CONFIGS_BSP['BSplineParam']
    #model = bspl.CubicBSplineNet(config).cuda()
    model = bspl.CubicBSplineNet(config).cuda()
    stn = SpatialTransformer([128,128]).cuda()
    bend_fn = Bend_Penalty().cuda()
    dice_fn = Dice().cuda()
    # 模型参数个数
    print("unet: ", countParameters(model))

    # 设置优化器和loss函数
    print("----Set initial parameters----")
    opt = Adam(model.parameters(), lr=args.lr)
    if args.sim_loss == "mse":
        sim_loss_fn = compute_mse
    elif args.sim_loss == "ncc":
        sim_loss_fn = ncc_loss
    else:
        sim_loss_fn = ssim_loss

    # 数据处理
    print("----Process data----")
    
    
    train_list = np.arange(0, 60)
    valid_list = np.arange(60, 76)             

    dataset_train_img = torch.zeros([2, 60, 128, 128], dtype=torch.float32)
    dataset_train_label = torch.zeros([2, 60, 128, 128], dtype=torch.int8)
    dataset_valid_img = torch.zeros([2, 16, 128, 128], dtype=torch.float32)
    dataset_valid_label = torch.zeros([2, 16, 128, 128], dtype=torch.int8)

    subject_forms = ["CINE", "DE"]
    # CINE或DE
    for _form in range(2):
        # 训练集
        for _num in range(len(train_list)):
            subject = train_list[_num] + 1
            file_pattern = os.path.join(args.train_dir, f"EP{subject}_{subject_forms[_form]}.nii.gz")
            file_list = glob.glob(file_pattern)
            # img
            if len(file_list) > 0:
                file_path = file_list[0]
                #print('train file path:', file_path)
                data = torch.from_numpy(nib.load(file_path).get_fdata()).type(torch.float32)
                dataset_train_img[_form, _num, :, :] = (data - data.min()) / (data.max() - data.min())
            # 读取对应的标签文件
            label_pattern = os.path.join(args.train_dir, f"EP{subject}_{subject_forms[_form]}_gt.nii.gz")
            label_list = glob.glob(label_pattern)
            if len(label_list) > 0:
                file_path_label = label_list[0]
                #print('valid label path:', file_path_label)
                data_label = torch.from_numpy(nib.load(file_path_label).get_fdata()).type(torch.int8)
                dataset_train_label[_form, _num, :, :] = data_label[:, :]
        # 验证集
        for _num in range(len(valid_list)):
            subject = valid_list[_num] + 1
            file_pattern = os.path.join(args.train_dir, f"EP{subject}_{subject_forms[_form]}.nii.gz")
            file_list = glob.glob(file_pattern)
            if len(file_list) > 0:
                file_path = file_list[0]
                #print('valid file path:', file_path)
                data = torch.from_numpy(nib.load(file_path).get_fdata()).type(torch.float32)
                dataset_valid_img[_form, _num, :, :] = (data - data.min()) / (data.max() - data.min())
            # 读取对应的标签文件
            label_pattern = os.path.join(args.train_dir, f"EP{subject}_{subject_forms[_form]}_gt.nii.gz")
            label_list = glob.glob(label_pattern)
            if len(label_list) > 0:
                file_path_label = label_list[0]
                #print('valid label path:', file_path_label)
                data_label = torch.from_numpy(nib.load(file_path_label).get_fdata()).type(torch.int8)
                dataset_valid_label[_form, _num, :, :] = data_label[:, :]
                
            

    # 开始训练
    print("----Start training----")
    # 计时
    start_time = float(time.time())

    best_valid_loss = 10.0
    final_train_loss = 10.0
    final_valid_loss = 10.0
    for _iter in range(1, args.n_iter + 1):
        # 将train_data_list进行随机排序
        train_list_permuted = np.random.permutation(train_list)
        # ---------------------------训练部分------------------------------
        adjust_learning_rate(opt, _iter, args.n_iter, args.lr)
        sim_loss_train = 0.0
        reg_loss_train = 0.0
        smth_loss_train = 0.0
        loss_train = 0.0
        model.train()
        opt.zero_grad()
        
        # 以batch_size为步长批量读取数据
        steps = len(train_list_permuted) // args.batch_size
        for _step in range(steps):
            # 预先定义fixed 和 moving 张量 batch_size*C*H*W*D
            img_fixed = torch.zeros([args.batch_size, 1, 128, 128], dtype=torch.float32)
            img_moving = torch.zeros([args.batch_size, 1, 128, 128], dtype=torch.float32)
            label_fixed = torch.zeros([args.batch_size, 1, 128, 128], dtype=torch.float32)
            label_moving = torch.zeros([args.batch_size, 1, 128, 128], dtype=torch.float32)
            # 迭代读取fixed 和 moving图像
            for _batch in range(args.batch_size):
                subject = _step * args.batch_size + _batch
                img_moving[_batch, 0, :, :] = dataset_train_img[0, subject, :, :]
                img_fixed[_batch, 0, :, :] = dataset_train_img[1, subject, :, :]

            
            img_fixed = img_fixed.cuda()
            img_moving = img_moving.cuda()
            # 输入网络
            img_warped, flow = model(img_moving, img_fixed)
            
            # 计算loss
            sim_loss = sim_loss_fn(img_warped, img_fixed)
            reg_loss = bend_fn(flow)
            smth_loss = smooth_loss(flow, img_warped)
            loss = args.alpha * reg_loss + sim_loss + args.beta * smth_loss

            # Backwards and optimize
            loss.backward()
            
            opt.step()

            sim_loss_train += sim_loss.item()
            reg_loss_train += reg_loss.item()
            smth_loss_train += smth_loss.item()
            loss_train += loss.item()

        sim_loss_train /= steps
        reg_loss_train /= steps
        smth_loss_train /= steps
        loss_train /= steps

        # -----------------------------------验证部分---------------------------------
        sim_loss_valid = 0.0
        reg_loss_valid = 0.0
        smth_loss_valid = 0.0
        loss_valid = 0.0
        dice_valid = 0.0
        
        with torch.no_grad():
            model.eval()
            img_fixed = torch.zeros([1, 1, 128, 128], dtype=torch.float32)
            img_moving = torch.zeros([1, 1, 128, 128], dtype=torch.float32)
            label_fixed = torch.zeros([1, 1, 128, 128], dtype=torch.float32)
            label_moving = torch.zeros([1, 1, 128, 128], dtype=torch.float32)

            for _num in range(len(valid_list)):
                # img & label
                img_moving[0, 0, :, :] = dataset_valid_img[0, _num, :, :]
                img_fixed[0, 0, :, :] = dataset_valid_img[1, _num, :, :]
                label_moving[0, 0, :, :] = dataset_valid_label[0, _num, :, :]
                label_fixed[0, 0, :, :] = dataset_valid_label[1, _num, :, :]
                img_fixed = img_fixed.cuda()
                img_moving = img_moving.cuda()


                # 输入网络
                img_warped, flow = model(img_moving, img_fixed)
                
                # 计算dice
                label_fixed = label_fixed.cuda()
                label_moving = label_moving.cuda()
                label_warped = stn(label_moving, flow)
                dice_loss = dice_fn(label_fixed, label_warped, labelorders=[2,3,4])
                                    
                # 计算loss
                sim_loss = sim_loss_fn(img_warped, img_fixed)
                reg_loss = bend_fn(flow)
                smth_loss = smooth_loss(flow, img_warped)
                loss = args.alpha *  reg_loss + sim_loss + args.beta * smth_loss

                sim_loss_valid += sim_loss.item()
                reg_loss_valid += reg_loss.item()
                smth_loss_valid += smth_loss.item()
                dice_valid += dice_loss
                loss_valid += loss.item()
                  
                

        sim_loss_valid /= len(valid_list)
        reg_loss_valid /= len(valid_list)
        smth_loss_valid /= len(valid_list)
        loss_valid /= len(valid_list)
        dice_valid /= len(valid_list)
        
        print("epoch: %d  train_loss: %f  sim_loss: %f  reg_loss: %f  smooth_loss: %f" % (_iter, loss_train, sim_loss_train, reg_loss_train, smth_loss_train), flush=True)
        print("epoch: %d  valid_loss: %f  sim_loss: %f  reg_loss: %f  smooth_loss: %f valid_dice: %f" % (_iter, loss_valid, sim_loss_valid, reg_loss_valid, smth_loss_valid, dice_valid), flush=True)
        print("%d,%f,%f,%f,%f,%f,%f,%f,%f,%f" % (_iter, loss_train, sim_loss_train, reg_loss_train, smth_loss_train, loss_valid, sim_loss_valid, reg_loss_valid, smth_loss_valid, dice_valid), file=file_log)

        # 计时
        if _iter % 10 == 0:
            print("----time_used: %f" % float(time.time() - start_time), flush=True)
            print("----time_used: %f" % float(time.time() - start_time), file=file_log)
        
        # 保存最佳模型参数
        if loss_valid <= best_valid_loss:
            best_valid_loss = loss_valid
            final_train_loss = loss_train 
            final_valid_loss = loss_valid
            # Save model checkpoint
            save_file_dir = os.path.join(args.model_dir, "Xiehe_bspline_5000_train.pth")
            torch.save(model.state_dict(), save_file_dir)

    print("final_train_loss = %f,final_valid_loss = %f" % (final_train_loss, final_valid_loss), flush=True)
    print("final_train_loss = %f,final_valid_loss = %f" % (final_train_loss, final_valid_loss), file=file_log)
    file_log.close()

def test():
    # 创建文件夹
    print("----Make directory----")
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # 创建配准网络 unet+stn
    print("----Build registration network----")
    config = CONFIGS_BSP['BSplineParam']
    model = bspl.CubicBSplineNet(config).cuda()
    stn = bspl.SpatialTransformer([128,128]).cuda()
    dice_fn = Dice().cuda()
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()


    # 数据处理
    print("----Process data----")

    # 测试序列
    test_list = np.arange(76, 92)

    # 读取图像数据
    dataset_img = torch.zeros([2, 16, 128, 128], dtype=torch.float32)
    dataset_label = torch.zeros([2, 16, 128, 128], dtype=torch.int8)
    subject_forms = ["CINE", "DE"]

    for _form in range(2):
        # 测试集
        for _num in range(len(test_list)):
            subject = test_list[_num] + 1
            file_pattern = os.path.join(args.test_dir, f"EP{subject}_{subject_forms[_form]}.nii.gz")
            file_list = glob.glob(file_pattern)
            if len(file_list) > 0:
                file_path = file_list[0]
                data = torch.from_numpy(nib.load(file_path).get_fdata()).type(torch.float32)
                dataset_img[_form, _num, :, :] = (data - data.min()) / (data.max() - data.min())
            # 读取对应的标签文件
            label_pattern = os.path.join(args.test_dir, f"EP{subject}_{subject_forms[_form]}_gt.nii.gz")
            label_list = glob.glob(label_pattern)
            if len(label_list) > 0:
                file_path_label = label_list[0]
                data_label = torch.from_numpy(nib.load(file_path_label).get_fdata()).type(torch.int8)
                dataset_label[_form, _num, :, :] = data_label[:, :]
    # 开始测试
    print("----Start testing----")
    # 计时
    time_list = []
    dice_list = []
    psnr_list = []
    ssim_list = []
    jac_list = []

    img_fixed = torch.zeros([1, 1, 128, 128], dtype=torch.float32)
    img_moving = torch.zeros([1, 1, 128, 128], dtype=torch.float32)
    label_fixed = torch.zeros([1, 1, 128, 128], dtype=torch.float32)
    label_moving = torch.zeros([1, 1, 128, 128], dtype=torch.float32)
    
    for _num in range(len(test_list)):
        # 创建subject文件目录
        subject = test_list[_num] + 1
        subject_dir = os.path.join(args.result_dir, "EP%d" % subject)
        if not os.path.exists(subject_dir):
            os.mkdir(subject_dir)

        # img & label
        img_moving[0, 0, :, :] = dataset_img[0, _num, :, :]
        img_fixed[0, 0, :, :] = dataset_img[1, _num, :, :]
        label_fixed[0, 0, :, :] = dataset_label[0, _num, :, :]
        label_moving[0, 0, :, :] = dataset_label[1, _num, :, :]
        img_fixed = img_fixed.cuda()
        img_moving = img_moving.cuda()
        label_fixed = label_fixed.cuda()
        label_moving = label_moving.cuda()
        
        # 输入网络 计时
        start_time = time.time()
        img_warped, flow = model(img_moving, img_fixed)
        time_list.append([float(time.time() - start_time)])

        label_warped = stn(label_moving, flow)
        
        # 计算dice
        if args.dice:
            dice_list.append([dice_fn(label_fixed, label_moving, labelorders=[2,3,4]), dice_fn(label_fixed, label_warped, labelorders=[2,3,4])])
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
        img = nib.Nifti1Image(img_fixed[0, :, :, :].permute(2,1,0).cpu().detach().numpy(), None)
        nib.save(img, os.path.join(subject_dir, "fixed.nii.gz"))
        label = nib.Nifti1Image(label_fixed[0, :, :, :].permute(2,1,0).cpu().detach().type(torch.int8).numpy(), None)
        nib.save(label, os.path.join(subject_dir, "label_fixed.nii.gz"))
        
        img = nib.Nifti1Image(img_moving[0, :, :, :].permute(2,1,0).cpu().detach().numpy(), None)
        nib.save(img, os.path.join(subject_dir, "moving.nii.gz"))
        label = nib.Nifti1Image(label_moving[0, :, :, :].permute(2,1,0).cpu().detach().type(torch.int8).numpy(), None)
        nib.save(label, os.path.join(subject_dir, "label_moving.nii.gz"))
        
        img = nib.Nifti1Image(img_warped[0, :, :, :].permute(2,1,0).cpu().detach().numpy(), None)
        nib.save(img, os.path.join(subject_dir, "warped.nii.gz"))
        label = nib.Nifti1Image(label_warped[0, :, :, :].permute(2,1,0).cpu().detach().type(torch.int8).numpy(), None)
        nib.save(label, os.path.join(subject_dir, "label_warped.nii.gz"))
        
        flow = flow.permute(0,3,2,1)
        DVF = nib.Nifti1Image(flow[0, :, :, :].cpu().detach().numpy()[:,:,None,:][:,:,:,None,:], None)
        nib.save(DVF, os.path.join(subject_dir, "flow.nii.gz"))

    print("time_used = %f" % np.sum(time_list))

    # 保存结果
    with open(os.path.join(args.result_dir, "result.csv"), "w") as f:
        writer = csv.writer(f)
        header = ["time"]
        data = np.array(time_list)
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
