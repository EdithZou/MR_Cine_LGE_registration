# CPU版

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

'''
def search_region(r):
    if r > 0:
        """
        dense sampling with half-width r
        """
        xs, ys = torch.meshgrid(torch.arange(-r, r + 1), torch.arange(-r, r + 1))
        xs = xs.flatten()
        ys = ys.flatten()
        mid = int(len(xs) / 2)
        xs = torch.cat((xs[:mid], xs[mid + 1:]))
        ys = torch.cat((ys[:mid], ys[mid + 1:]))
    else:
        """
        four or six-neighbourhood
        右：(1, 0)
        左：(-1, 0)
        上：(0, 1)
        下：(0, -1)
        """
        # 2D: right, left, up, down
        xs = torch.tensor([1, -1, 0, 0])
        ys = torch.tensor([0, 0, 1, -1])
        # 3D
        #xs = torch.tensor([1, -1, 0, 0, 1, -1])
        #ys = torch.tensor([0, 0, 1, -1, 1, -1])
    return xs, ys


def imshift(im1, x, y, pad=0):
    device = im1.device
    m, n = im1.shape[1], im1.shape[2]

    x1s = torch.max(torch.tensor(0), x)
    x2s = torch.min(torch.tensor(n - 1), n + x - 1)

    y1s = torch.max(torch.tensor(0), y)
    y2s = torch.min(torch.tensor(m - 1), m - 1 + y)

    x1 = torch.max(torch.tensor(0), -x)
    x2 = torch.min(torch.tensor(n - 1), n - 1 - x)

    y1 = torch.max(torch.tensor(0), -y)
    y2 = torch.min(torch.tensor(m - 1), m - 1 - y)

    im1shift = torch.clone(im1)  # Clone the tensor
    im1shift[..., y1:y2 + 1, x1:x2 + 1] = im1[..., y1s:y2s + 1, x1s:x2s + 1]
    return im1shift.to(device)


def MIND_descriptor2D(I, r=0, sigma=0.5):
    """
    Calculation of MIND (modality independent neighbourhood descriptor)
    :param I:
    :param r:
    :param sigma:
    :return:
    """
    smooth = 1e-5
    xs, ys = search_region(r)
    xs0, ys0 = search_region(0)
    
    Dp = torch.zeros((len(xs0), *I.shape), device=I.device)
    for i in range(len(xs0)):
        Dp[i, ...] = torch.from_numpy(gaussian_filter(((I - imshift(I, xs0[i], ys0[i])) ** 2).cpu().detach().numpy(), sigma=sigma, truncate=3 / 2))
    
    V = torch.mean(Dp, dim=0)

    val1 = [0.001 * V.mean(), 1000. * V.mean()]
    V = (torch.clamp(V, min=val1[0], max=val1[1]))
    I1 = torch.zeros((len(xs0), *I.shape), device=I.device)
    for i in range(len(xs0)):
        I1[i, ...] = torch.exp(-Dp[i, ...] / (V + smooth))

    mind = torch.zeros((len(xs), *I.shape), device=I.device)
    if r > 0:
        for i in range(len(xs)):
            mind[i, :, :] = torch.exp(
                torch.from_numpy(gaussian_filter((I - imshift(I, xs[i], ys[i])) ** 2, sigma=sigma, truncate=3 / 2)).to(I.device) / (V + smooth))
    else:
        for i in range(len(xs0)):
            mind[i, ...] = I1[i, ...]
    max1 = torch.max(mind, dim=0)[0]
    # normalization
    for i in range(len(xs)):
        mind[i, ...] = mind[i, ...] / max1

    return mind


def compute_ssc(img, r=0, sigma=0.5):
    # 得到[4,C,H,W]的特征层
    mind = MIND_descriptor2D(img, r=r, sigma=sigma)
    return mind


def compute_ssc_batch(img, r=0, sigma=0.5):
    batch_size = img.shape[0]
    ssc_batch = []

    for i in range(batch_size):
        ssc = compute_ssc(img[i], r=r, sigma=sigma)
        ssc_batch.append(ssc)
    return torch.stack(ssc_batch)
'''

# GPU版
import numpy as np
import torch
import torch.nn.functional as F


def search_region(r):
    if r > 0:
        """
        dense sampling with half-width r
        """
        xs, ys = torch.meshgrid(torch.arange(-r, r + 1), torch.arange(-r, r + 1))
        xs = xs.flatten()
        ys = ys.flatten()
        mid = int(len(xs) / 2)
        xs = torch.cat((xs[:mid], xs[mid + 1:]))
        ys = torch.cat((ys[:mid], ys[mid + 1:]))
    else:
        """
        four or six-neighbourhood
        右：(1, 0)
        左：(-1, 0)
        上：(0, 1)
        下：(0, -1)
        """
        # 2D: right, left, up, down
        xs = torch.tensor([1, -1, 0, 0])
        ys = torch.tensor([0, 0, 1, -1])
        # 3D
        #xs = torch.tensor([1, -1, 0, 0, 1, -1])
        #ys = torch.tensor([0, 0, 1, -1, 1, -1])
    return xs, ys


def imshift(im1, x, y, pad=0):
    device = im1.device
    m, n = im1.shape[1], im1.shape[2]

    x1s = torch.max(torch.tensor(0), x)
    x2s = torch.min(torch.tensor(n - 1), n + x - 1)

    y1s = torch.max(torch.tensor(0), y)
    y2s = torch.min(torch.tensor(m - 1), m - 1 + y)

    x1 = torch.max(torch.tensor(0), -x)
    x2 = torch.min(torch.tensor(n - 1), n - 1 - x)

    y1 = torch.max(torch.tensor(0), -y)
    y2 = torch.min(torch.tensor(m - 1), m - 1 - y)

    im1shift = torch.clone(im1).detach()  # Clone the tensor
    imshift = torch.clone(im1).detach()  # Clone the tensor
    imshift[..., y1:y2 + 1, x1:x2 + 1] = im1shift[..., y1s:y2s + 1, x1s:x2s + 1]
    return imshift.to(device)


def MIND_descriptor2D(I, r=0, sigma=0.5):
    """
    Calculation of MIND (modality independent neighbourhood descriptor)
    :param I:
    :param r:
    :param sigma:
    :return:
    """
    smooth = 1e-5
    xs, ys = search_region(r)
    xs0, ys0 = search_region(0)
    
    Dp = torch.zeros((len(xs0), *I.shape), device=I.device)
    for i in range(len(xs0)):
        Dp[i, ...] = gaussian_filter_torch(((I - imshift(I, xs0[i], ys0[i])) ** 2), sigma=sigma)
    
    V = torch.mean(Dp, dim=0)

    val1 = [0.001 * V.mean(), 1000. * V.mean()]
    V = (torch.clamp(V.clone(), min=val1[0], max=val1[1]))  # Avoid inplace operation
    I1 = torch.zeros((len(xs0), *I.shape), device=I.device)
    for i in range(len(xs0)):
        I1[i, ...] = torch.exp(-Dp[i, ...] / (V + smooth))

    mind = torch.zeros((len(xs), *I.shape), device=I.device)
    if r > 0:
        for i in range(len(xs)):
            mind[i, :, :] = torch.exp(
                gaussian_filter_torch((I - imshift(I, xs[i], ys[i])) ** 2, sigma=sigma) / (V + smooth))
    else:
        for i in range(len(xs0)):
            mind[i, ...] = I1[i, ...]
    max1 = torch.max(mind, dim=0)[0]
    # normalization
    for i in range(len(xs)):
        mind[i, ...] = mind[i, ...] / max1

    return mind

def compute_ssc(img, r=0, sigma=0.5):
    # 得到[4,C,H,W]的特征层
    mind = MIND_descriptor2D(img, r=r, sigma=sigma)
    return mind

def compute_ssc_batch(img, r=0, sigma=0.5):
    batch_size = img.shape[0]
    ssc_batch = []

    for i in range(batch_size):
        ssc = compute_ssc(img[i], r=r, sigma=sigma)
        ssc_batch.append(ssc)
    return torch.stack(ssc_batch)



def gaussian_filter_torch(input_tensor, sigma=0.5):
    """
    Apply Gaussian filter to a PyTorch tensor along the spatial dimensions.
    """
    if sigma == 0:
        return input_tensor

    # Create Gaussian kernel
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.exp(-(np.arange(kernel_size) - kernel_size // 2) ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    # Create 2D Gaussian kernel
    kernel_2d = np.outer(kernel, kernel)
    # Convert to tensor
    kernel_2d = torch.tensor(kernel_2d, dtype=torch.float32, device=input_tensor.device).unsqueeze(0).unsqueeze(0)
    
    # Reshape input tensor to 4D: [1, C, H, W]
    channels = int(input_tensor.shape[0])
    input_tensor_reshaped = input_tensor.clone().detach().unsqueeze(0)
    # Apply filter along spatial dimensions for each channel
    filtered_tensor = torch.zeros_like(input_tensor_reshaped)
    # Apply filter along spatial dimensions
    for i in range(channels):
        filtered_tensor[:, i:i+1, :, :] = F.conv2d(input_tensor_reshaped[:, i:i+1, :, :], kernel_2d, padding=kernel_size // 2)
    return filtered_tensor.squeeze(0)
