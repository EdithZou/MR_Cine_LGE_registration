import torch
import torch.nn as nn
import torch.nn.functional as nnf


class SpatialTransformer(nn.Module):
    # 2D or 3d spatial transformer network to calculate the warped moving image
    
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.grid_dict = {}
        self.norm_coeff_dict = {}

    def forward(self, input_image, flow):   
        '''
        input_image: (n, 1, h, w) or (n, 1, d, h, w)
        flow: (n, 2, h, w) or (n, 3, d, h, w)
        
        return: 
            warped moving image, (n, 1, h, w) or (n, 1, d, h, w)
        '''
        img_shape = input_image.shape[2:]
        if img_shape in self.grid_dict:
            grid = self.grid_dict[img_shape]
            norm_coeff = self.norm_coeff_dict[img_shape]
        else:
            grids = torch.meshgrid([torch.arange(0, s) for s in img_shape]) 
            grid  = torch.stack(grids[::-1], dim = 0) # 2 x h x w or 3 x d x h x w, the data in second dimension is in the order of [w, h, d]
            grid  = torch.unsqueeze(grid, 0)
            grid  = grid.to(dtype = flow.dtype, device = flow.device)
            norm_coeff = 2./(torch.tensor(img_shape[::-1], dtype = flow.dtype, device = flow.device) - 1.) # the coefficients to map image coordinates to [-1, 1]
            self.grid_dict[img_shape] = grid
            self.norm_coeff_dict[img_shape] = norm_coeff
            logging.info(f'\nAdd grid shape {tuple(img_shape)}')
        new_grid = grid + flow 

        if self.dim == 2:
            new_grid = new_grid.permute(0, 2, 3, 1) # n x h x w x 2
        elif self.dim == 3:
            new_grid = new_grid.permute(0, 2, 3, 4, 1) # n x d x h x w x 3
            
        if len(input_image) != len(new_grid):
            # make the image shape compatable by broadcasting
            input_image += torch.zeros_like(new_grid)
            new_grid += torch.zeros_like(input_image)

        warped_input_img =  F.grid_sample(input_image, new_grid*norm_coeff - 1. , mode = 'bilinear', align_corners = True, padding_mode = 'border')
        return warped_input_img


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

class CalcDisp(object):
    def __init__(self, dim, calc_device = 'cuda'):
        self.device = torch.device(calc_device)
        self.dim = dim
        self.spatial_transformer = SpatialTransformer(dim = dim)
        
    def inverse_disp(self, disp, threshold = 0.01, max_iteration = 20):
        '''
        compute the inverse field. implementation of "A simple fixed‐point approach to invert a deformation field"
        # 计算逆向场
        disp : (n, 2, h, w) or (n, 3, d, h, w) or (2, h, w) or (3, d, h, w)
            displacement field
        '''
        forward_disp = disp.detach().to(device = self.device)
        if disp.ndim < self.dim + 2:
            forward_disp = torch.unsqueeze(forward_disp, 0)
        backward_disp = torch.zeros_like(forward_disp)
        backward_disp_old = backward_disp.clone()
        for i in range(max_iteration):
            backward_disp = - self.spatial_transformer(forward_disp, backward_disp)
            diff = torch.max(torch.abs(backward_disp - backward_disp_old)).item()
            if diff < threshold:
                break
            backward_disp_old = backward_disp.clone()
        if disp.ndim < self.dim + 2:
            backward_disp = torch.squeeze(backward_disp, 0)

        return backward_disp
        
    def compose_disp(self, disp_i2t, disp_t2i, mode = 'corr'):
        '''
        compute the composition field
        # 计算变形场
        disp_i2t: (n, 3, d, h, w)
            displacement field from the input image to the template
            
        disp_t2i: (n, 3, d, h, w)
            displacement field from the template to the input image
            
        mode: string, default 'corr'
            'corr' means generate composition of corresponding displacement field in the batch dimension only, the result shape is the same as input (n, 3, d, h, w)
            'all' means generate all pairs of composition displacement field. The result shape is (n, n, 3, d, h, w)
        '''
        # [n,2,h,w]
        disp_i2t_t = disp_i2t.detach().to(device = self.device)
        disp_t2i_t = disp_t2i.detach().to(device = self.device)
        if disp_i2t.ndim < self.dim + 2:
            disp_i2t_t = torch.unsqueeze(disp_i2t_t, 0)
        if disp_t2i.ndim < self.dim + 2:
            disp_t2i_t = torch.unsqueeze(disp_t2i_t, 0)
        
        if mode == 'corr':
            composed_disp = self.spatial_transformer(disp_t2i_t, disp_i2t_t) + disp_i2t_t # (n, 2, h, w) or (n, 3, d, h, w)
        elif mode == 'all':
            assert len(disp_i2t_t) == len(disp_t2i_t)
            n, _, *image_shape = disp_i2t.shape # n=2, _=2, image_shape =h,w
            # 以下得到n*n: [n,2,h,w]->[n,1,2,h,w],然后在第一个维度repeat n次，得到[n,n,2,h,w]
            disp_i2t_nxn = torch.repeat_interleave(torch.unsqueeze(disp_i2t_t, 1), n, 1) # (n, n, 2, h, w) or (n, n, 3, d, h, w)
            # 得到i2t nn: [n,n,2,h,w]->[n*n,2,h,w]
            disp_i2t_nn = disp_i2t_nxn.reshape(n*n, self.dim, *image_shape) # (n*n, 2, h, w) or (n*n, 3, d, h, w), the order in the first dimension is [0_T, 0_T, ..., 0_T, 1_T, 1_T, ..., 1_T, ..., n_T, n_T, ..., n_T]
            
            # 得到t2i nn: [n,n,2,h,w]->[1,n,2,h,w]->[n*n,2,h,w]
            disp_t2i_nn = torch.repeat_interleave(torch.unsqueeze(disp_t2i_t, 0), n, 0).reshape(n*n, self.dim, *image_shape) # (n*n, 2, h, w) or (n*n, 3, d, h, w), the order in the first dimension is [0_T, 1_T, ..., n_T, 0_T, 1_T, ..., n_T, ..., 0_T, 1_T, ..., n_T]
            # composed_disp = self.spatial_transformer(disp_t2i_nn, disp_i2t_nn).reshape(n, n, self.dim, *image_shape) + disp_i2t_nxn # (n, n, 2, h, w) or (n, n, 3, d, h, w) + disp_i2t_nxn
            # Ti2t(Tt2i)=Tm2f，也就是反向形变场*正向形变场=形变场，计算了从m2f的形变场
            composed_disp = self.spatial_transformer(disp_t2i_nn, disp_i2t_nn)
            composed_disp = composed_disp.reshape(n, n, self.dim, *image_shape)
            composed_disp = composed_disp + disp_i2t_nxn
        else:
            raise
        if disp_i2t.ndim < self.dim + 2 and disp_t2i.ndim < self.dim + 2:
            composed_disp = torch.squeeze(composed_disp)
        return composed_disp
    
    # def cal_warped_i2i(self, input_image, disp_i2t, disp_t2i, mode = 'corr'):
    def cal_warped_i2i(self, input_image, disp_t2i,  disp_i2t, mode = 'corr'):
        '''
        compute the composition field
        # 计算变形场
        disp_i2t: (n, 3, d, h, w)
            displacement field from the input image to the template
            
        disp_t2i: (n, 3, d, h, w)
            displacement field from the template to the input image
            
        mode: string, default 'corr'
            'corr' means generate composition of corresponding displacement field in the batch dimension only, the result shape is the same as input (n, 3, d, h, w)
            'all' means generate all pairs of composition displacement field. The result shape is (n, n, 3, d, h, w)
        '''
        disp_i2t_t = disp_i2t.detach().to(device = self.device)
        disp_t2i_t = disp_t2i.detach().to(device = self.device)
        if disp_i2t.ndim < self.dim + 2:
            disp_i2t_t = torch.unsqueeze(disp_i2t_t, 0)
        if disp_t2i.ndim < self.dim + 2:
            disp_t2i_t = torch.unsqueeze(disp_t2i_t, 0)
        
        if mode == 'corr':
            composed_disp = self.spatial_transformer(disp_t2i_t, disp_i2t_t) + disp_i2t_t # (n, 2, h, w) or (n, 3, d, h, w)
        elif mode == 'i2i':
            # assert len(disp_i2t_t) == len(disp_t2i_t)
            # n, _, *image_shape = disp_i2t.shape
            # disp_i2t_nxn = torch.repeat_interleave(torch.unsqueeze(disp_i2t_t, 1), n, 1) # (n, n, 2, h, w) or (n, n, 3, d, h, w)
            # disp_i2t_nn = disp_i2t_nxn.reshape(n*n, self.dim, *image_shape) # (n*n, 2, h, w) or (n*n, 3, d, h, w), the order in the first dimension is [0_T, 0_T, ..., 0_T, 1_T, 1_T, ..., 1_T, ..., n_T, n_T, ..., n_T]
            # disp_t2i_nn = torch.repeat_interleave(torch.unsqueeze(disp_t2i_t, 0), n, 0).reshape(n*n, self.dim, *image_shape) # (n*n, 2, h, w) or (n*n, 3, d, h, w), the order in the first dimension is [0_T, 1_T, ..., n_T, 0_T, 1_T, ..., n_T, ..., 0_T, 1_T, ..., n_T]
            # composed_disp = self.spatial_transformer(disp_t2i_nn, disp_i2t_nn)
            # composed_disp = composed_disp.reshape(n, n, self.dim, *image_shape)
            # composed_disp = composed_disp + disp_i2t_nxn
            
            # 这里的 t2i 叠加 i2t，如果不是n*n的话，很好理解，就是将一个时刻的到模板的变形场加上模板到另一个时刻的变形场
            # 也即两个变形场的叠加，相当于从 i2i的变形场，但此时如果要得出 warped-i2i，则需要以当前时刻的原始图叠加 i2i变形场
            # 所以总共分两部分：首先将双阶段的变形场进行合并，进行第一次transformer叠加，然后将结果当作总变形场进行第二次原始图像的transformer叠加
            t2i_last = torch.unsqueeze(disp_t2i_t[0], 0) # t2i5 → i2t1
            i2t_first = torch.unsqueeze(disp_i2t_t[1], 0)
            t_2_i_2_t = self.spatial_transformer(t2i_last, i2t_first) + i2t_first
            warped_i2i = self.spatial_transformer(torch.unsqueeze(input_image[0,...], 1), t_2_i_2_t) 
            
        else:
            raise
        if disp_i2t.ndim < self.dim + 2 and disp_t2i.ndim < self.dim + 2:
            composed_disp = torch.squeeze(composed_disp)
        # return composed_disp
        return warped_i2i, t_2_i_2_t
    
class cal_spatial_transformer():
    def __init__(self, dim = 3):
        self.spatial_transformer = SpatialTransformer(dim = dim)
    def SpatialTransformer_continuation(self, inputimage, flow):
        # 多个时序的顺序变形场叠加
        warped_image = self.spatial_transformer(inputimage, flow)
        return warped_image