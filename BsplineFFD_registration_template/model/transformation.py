import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class _Transform(object):
    """ Transformation base class """
    def __init__(self,
                 svf=True,
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
            return flow, disp


class DenseTransform(_Transform):
    """ Dense field transformation """
    def __init__(self,
                 svf=False,
                 svf_steps=7,
                 svf_scale=1):
        super(DenseTransform, self).__init__(svf=svf,
                                             svf_steps=svf_steps,
                                             svf_scale=svf_scale)

    def compute_flow(self, x):
        return x


class CubicBSplineFFDTransform(_Transform):
    def __init__(self,
                 ndim,
                 img_size=(128,128),
                 cps=(4, 4),
                 svf=True,
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
        self.stride = cps

        self.kernels = self.set_kernel()
        self.padding = [(len(k) - 1) // 2
                        for k in self.kernels]  # the size of the kernel is always odd number

    def set_kernel(self):
        kernels = list()
        for s in self.stride:
            # 1d cubic b-spline kernels
            kernels += [ffd_bspline1d(s)]
            #kernels += [bspline_kernel_1d(s)]
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
            k = k.to(dtype=x.dtype, device=x.device)
            flow = conv1d(flow, dim=i + 2, kernel=k, stride=s, padding=p, transpose=True)
            print('flow shape:', flow.shape)
        #  crop the output to image size
        slicer = (slice(0, flow.shape[0]), slice(0, flow.shape[1])) \
                 + tuple(slice(s, s + self.img_size[i]) for i, s in enumerate(self.stride))
        flow = flow[slicer]
        return flow


class BSplineFFDTransformer(nn.Module):
    def __init__(
        self,
        ndim,
        img_size=(128, 128),
        cps=(4, 4),
        order=3,
        trainable=False,
        apply_diffeomorphic_limits=False,
    ):
        super(BSplineFFDTransformer, self).__init__()
        self.ndim = ndim
        self.img_size = img_size
        self.cps = cps
        self.order = order
        self.trainable = trainable
        self.apply_diffeomorphic_limits = apply_diffeomorphic_limits

        self.upsampling_factors = cps  # Adjusted upsampling factors for input size
        bspline_kernel = self.make_bspline_kernel()
        kernel_size = bspline_kernel.shape
        # Calculate padding size for convolution
        self.crop_size = tuple(int(el * 3 / 8) if el != 5 else 2 for el in bspline_kernel.shape)

        # Transposed convolution layer
        self.upsampler = nn.ConvTranspose2d(
            in_channels=1,
            out_channels=1,
            kernel_size=bspline_kernel.shape,
            stride=self.upsampling_factors,
            padding=self.crop_size,
            bias=False,
        )
        self.upsampler.weight = nn.Parameter(bspline_kernel[None, None], requires_grad=self.trainable)

    def make_bspline_kernel(self, dtype=torch.float32):
        bspline_kernel = self.bspline_convolution_kernel(
            self.upsampling_factors, order=self.order, dtype=dtype
        )

        if (np.array(bspline_kernel.shape[::-1]) == 4).any() or (
            np.array(bspline_kernel.shape[::-1]) == 2
        ).any():  # hack to deal with 1 strides and kernel size of 4
            padding = list()
            for s in bspline_kernel.shape[::-1]:
                if s == 4 or s == 2:
                    padding.extend([1, 0])
                else:
                    padding.extend([0, 0])
            bspline_kernel = F.pad(bspline_kernel, padding, mode="constant")

        return bspline_kernel

    def bspline_kernel_nd(self, t):
        tpowers = t ** torch.arange(self.order, 0 - 1, -1, dtype=torch.float32)
        if self.order == 1:
            return tpowers @ torch.tensor(((-1, 1), (1, 0)), dtype=torch.float32)
        elif self.order == 2:
            return (
                tpowers
                @ torch.tensor(((1, -2, 1), (-2, 2, 0), (1, 1, 0)), dtype=torch.float32)
                / 2.0
            )
        elif self.order == 3:
            return (
                tpowers
                @ torch.tensor(
                    ((-1, 3, -3, 1), (3, -6, 3, 0), (-3, 0, 3, 0), (1, 4, 1, 0)),
                    dtype=torch.float32,
                )
                / 6.0
            )
    def bspline_convolution_kernel(self, upsampling_factors, order, dtype=float):
        ndim = len(upsampling_factors)
        for i, us_factor in enumerate(upsampling_factors):
            t = torch.linspace(1 - (1 / us_factor), 0, us_factor)
            ker1D = self.bspline_kernel_nd(t[:, None]).T.flatten()
            shape = (1,) * i + ker1D.shape + (1,) * (ndim - 1 - i)
            try:
                kernel = kernel * ker1D.view(shape)
            except NameError:
                kernel = ker1D.view(shape)
        return kernel
    
    def forward(self, input_tensor, batchsize):
        if batchsize >1:
            dvf = torch.zeros([batchsize, self.ndim, self.ndim, 128, 128], dtype=torch.float32)
            for i in range(batchsize):
                dvf[i,:,:,:,:] = self.create_dvf(input_tensor[i,:,:,:,:])
        else:
            dvf = self.create_dvf(input_tensor)
            # Perform additional operations if needed
        return dvf

    def create_dvf(self, bspline_parameters):
        # Perform any necessary processing on bspline_parameters
        assert bspline_parameters.shape[1] == self.ndim
        shape = bspline_parameters.shape
        dvf = self.upsampler(
            bspline_parameters.view((shape[0] * self.ndim, 1) + shape[2:]),
            output_size=self.img_size,
        )
        newshape = dvf.shape
        return dvf.view((shape[0], self.ndim) + newshape[2:])



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


def svf_exp(flow, scale=1, steps=2, sampling='bilinear'):
    """ Exponential of velocity field by Scaling and Squaring"""
    disp = flow * (scale / (2 ** steps))
    for _ in range(steps):
        disp = disp + warp(x=disp, disp=disp,
                           interp_mode=sampling)
    return disp


def cubic_bspline_value(x: float, derivative: int = 0) -> float:
    r"""Evaluate 1-dimensional cubic B-spline."""
    t = abs(x)
    # outside local support region
    if t >= 2:
        return 0
    # 0-th order derivative
    if derivative == 0:
        if t < 1:
            return 2 / 3 + (0.5 * t - 1) * t ** 2
        return -((t - 2) ** 3) / 6
    # 1st order derivative
    if derivative == 1:
        if t < 1:
            return (1.5 * t - 2.0) * x
        if x < 0:
            return 0.5 * (t - 2) ** 2
        return -0.5 * (t - 2) ** 2
    # 2nd oder derivative
    if derivative == 2:
        if t < 1:
            return 3 * t - 2
        return -t + 2


def cubic_bspline1d(stride, derivative: int = 1, dtype=None, device= None) -> torch.Tensor:
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
    kernel = torch.ones(4 * stride - 1, dtype=dtype)
    radius = kernel.shape[0] // 2
    for i in range(kernel.shape[0]):
        kernel[i] = cubic_bspline_value((i - radius) / stride, derivative=derivative)
    if device is None:
        device = kernel.device
    return kernel.to(device)


def ffd_bspline_value(x: float, stride: int) -> float: # 3阶bspline基函数
    r"""Evaluate 1-dimensional ffd B-spline."""
    t = abs(x)
    u = t / stride
    #  outside local support region
    if t >= 0 and t < 1:
        return ((1-u) ** 3) / 6
    elif t >= 1 and t < 2:
        #return -1 * u ** 3 / 2 + 2 * u **2 - 2 * u + 2 / 3
        return (3*u ** 3 - 6*u ** 2 + 4) / 6
    elif t >= 2 and t < 3:
        #return u ** 3 / 2 - 4 * u **2 + 10 * u + 11 / 3
        return (-3*u ** 3 + 3*u ** 2 + 3*u + 1) / 6
    elif t >= 3 and t < 4:
        #return ((4 - u) ** 3) / 6
        return ((u) ** 3) / 6
    elif t >= 4:
        return 0
    '''
    elif t >= 4 and t < 5:
        return (-3*u** 3 + 9*u**2 - 9*u + 3) / 6
    elif t >= 5 and t < 6:
        return (3*u** 3 - 12*u**2 + 12*u + 4) / 6
    elif t >= 6 and t < 7:
        return (-3*u** 3 + 15*u**2 - 21*u + 7) / 6
    elif t >= 7 and t < 8:
        return (u** 3 - 6*u**2 + 12*u - 8) / 6
    if t >= 8:
        return 0    
    '''



def ffd_bspline1d(stride, dtype=None, device= None) -> torch.Tensor:
    r"""Freeform deformation B-spline kernel for specified control point spacing.

    Args:
        stride: Spacing between control points with respect to original (upsampled) image grid.
        derivative: Order of cubic B-spline derivative.

    Returns:
        FFD B-spline convolution kernel.

    """
    if dtype is None:
        dtype = torch.float
    if not isinstance(stride, int):
        (stride,) = stride
    kernel = torch.ones(stride, dtype=dtype)
    radius = kernel.shape[0] // 2
    for i in range(kernel.shape[0]):
        kernel[i] = ffd_bspline_value((i - radius), stride)
    if device is None:
        device = kernel.device
    return kernel.to(device)

def bspline_kernel_1d(sigma, order=3, device= None) -> torch.Tensor:
    kernel_ones = torch.ones(1, 1, sigma)
    kernel = kernel_ones
    padding = sigma - 1
    for i in range(1, order + 1):
        kernel = torch.nn.functional.conv1d(kernel, kernel_ones, padding=padding) / sigma
    if device is None:
        device = kernel.device
    return kernel[0, 0, ...].to(device)

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
    # use native pytorch
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
    print('result shape: ', result.shape)
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

    # generate standard mesh grid
    grid = torch.meshgrid([torch.linspace(-1, 1, size[i]).type_as(disp) for i in range(ndim)])
    grid = [grid[i].requires_grad_(False) for i in range(ndim)]

    # apply displacements to each direction (N, *size)
    warped_grid = [grid[i] + disp[:, i, ...] for i in range(ndim)]
    

    # normalize the warped grid to [-1, 1]
    for i in range(ndim):
        warped_grid[i] = 2 * (warped_grid[i] / (size[i] - 1) - 0.5)
    
    # swapping i-j-k order to x-y-z (k-j-i) order for grid_sample()
    warped_grid = [warped_grid[ndim - 1 - i] for i in range(ndim)]
    warped_grid = torch.stack(warped_grid, -1)  # (N, *size, dim)

    return F.grid_sample(x, warped_grid, mode=interp_mode, align_corners=False)
